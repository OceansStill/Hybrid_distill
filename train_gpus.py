#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多 GPU 蒸馏训练（老师=1 卡，学生=DDP 多卡）
- 学生：使用前 N 张可见 GPU（local_rank=0..N-1）
- 老师：独占最后一张可见 GPU（index = visible_count - 1）
- 各 rank 先 all_gather 输入到 rank0，rank0 在老师卡上前向，
  再按 rank 切片，用 broadcast 逐个发回对应 rank 的 teacher logits。

关键点：
1) 非 rank0 构造“骨架老师”（无权重，仅保持同结构），所有 rank 都执行层替换与 rotary 对齐，
   但“拷老师权重到学生”仅在 rank0 执行；随后由 DDP 广播 rank0 权重，保证各 rank 拓扑与参数一致。
2) DataLoader 建议 drop_last=True（见 data.py），保证各 rank batch 等长，便于 gather/broadcast。
3) 同时支持 CE + KL（软目标 KD，含 Hinton 的 T^2 缩放）；CE 分母仅统计有效标签；label 越界自动忽略。
   若只想 CE，传参 --kl_weight 0 即可（代码自动跳过老师前向与通信）。
"""

import os
import copy
import time
import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from arguments import parse_args
from data import build_dataloader
from model_utils import (
    apply_replacements,
    count_layer_parameters,
    filter_layer_indices,
    freeze_student_layers,
    freeze_student_mlps,
    get_backbone,
    get_embedding_module,
    get_student_model,
    get_teacher_model,
)
from training_utils import (
    build_wsd_scheduler,
    compute_distill_loss,
    ddp_cleanup,
    ddp_init,
    get_learning_rate,
    model_fingerprint,
    split_batch,
)
from utils import configure_environment, init_wandb, save_checkpoint, setup_logger


def main():
    configure_environment()
    args = parse_args()

    rank, world_size, local_rank = ddp_init()
    is_main = (rank == 0)

    visible_count = torch.cuda.device_count()
    if visible_count < world_size:
        raise RuntimeError(f"可见 GPU 数={visible_count}，不足以覆盖学生 world_size={world_size}。")

    student_device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(student_device)

    teacher_device_arg = getattr(args, "teacher_device", None)
    if teacher_device_arg:
        teacher_device = torch.device(teacher_device_arg)
        if teacher_device.type != "cuda":
            raise ValueError(f"--teacher_device 目前仅支持 CUDA 设备，收到：{teacher_device_arg}")
        if teacher_device.index is None:
            teacher_device = torch.device("cuda:0")
        if teacher_device.index < 0 or teacher_device.index >= visible_count:
            raise RuntimeError(
                f"--teacher_device={teacher_device_arg} 超出可见 GPU 范围 (0-{visible_count - 1})"
            )
    else:
        if visible_count < world_size + 1:
            raise RuntimeError(
                f"可见 GPU 数={visible_count}，需要 >= world_size+1 = {world_size+1} 才能实现 老师=1卡 + 学生={world_size}卡。"
            )
        teacher_device = torch.device(f"cuda:{visible_count - 1}")

    torch_dtype = getattr(torch, args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger(log_file=args.log_file, name="train.gpus")
    if not is_main:
        logger.setLevel(logging.WARNING)

    wandb_run = None
    if getattr(args, "wandb", False) and is_main:
        wandb_run = init_wandb(args, config=dict(vars(args)))

    if is_main:
        logger.info("===== 蒸馏训练（学生=DDP 多卡，老师=单卡）=====")
        logger.info(f"world_size={world_size}, rank={rank}, local_rank={local_rank}")
        logger.info(f"student_device={student_device}, teacher_device={teacher_device}")
        if teacher_device.index == student_device.index:
            logger.warning("老师和 rank0 学生位于同一 GPU，可能导致显存紧张。")

    cpu = torch.device("cpu")
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        teacher, _ = get_teacher_model(device=cpu, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    else:
        cfg = AutoConfig.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            local_files_only=args.local_files_only,
        )
        teacher = AutoModelForCausalLM.from_config(cfg)
        teacher.config.use_cache = False
        teacher.backbone = teacher.model
        for layer in teacher.backbone.layers:
            layer.layer_idx = layer.self_attn.layer_idx
            layer.mixer = layer.self_attn
            layer.mixer.out_proj = layer.mixer.o_proj

    student = get_student_model(device=cpu, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    student.tokenizer = tokenizer

    t_backbone = get_backbone(teacher)
    s_backbone = get_backbone(student)
    if t_backbone is not None and s_backbone is not None and hasattr(t_backbone, "rotary_emb"):
        try:
            s_backbone.rotary_emb = copy.deepcopy(t_backbone.rotary_emb).to(device=cpu)
            if is_main:
                logger.info("已将 teacher.backbone.rotary_emb 深拷贝到学生（CPU）。")
        except Exception as exc:
            if is_main:
                logger.warning(f"复制 rotary_emb 失败：{exc}")

    layer_indices = filter_layer_indices(getattr(args, "layers", []))
    if layer_indices:
        if is_main:
            logger.info(f"替换以下层为教师模型对应层: {layer_indices}")
        apply_replacements(student, teacher, layer_indices)
    replaced_stats = (
        count_layer_parameters(student, layer_indices) if layer_indices else {"total": 0, "mlp": 0, "non_mlp": 0}
    )

    frozen_mlp = freeze_student_mlps(student) if getattr(args, "freeze_mlp", False) else 0
    frozen_replaced = freeze_student_layers(student, layer_indices) if getattr(args, "freeze_replaced_layers", False) else 0

    if is_main:
        teacher.eval().to(teacher_device)
        for param in teacher.parameters():
            param.requires_grad_(False)
    else:
        teacher = None
        torch.cuda.empty_cache()

    student.to(student_device)

    fingerprint = model_fingerprint(student, student_device)
    fingerprint_list = [torch.zeros_like(fingerprint) for _ in range(world_size)]
    dist.all_gather(fingerprint_list, fingerprint)
    uniq = sorted({(int(x[0].item()), int(x[1].item())) for x in fingerprint_list})
    if is_main:
        logger.info(f"param_fingerprint across ranks: {uniq}")
    assert len(uniq) == 1, "各 rank 学生模型参数形状不一致，请检查层替换/rotary 对齐是否在所有 rank 均执行。"

    student = DDP(
        student,
        device_ids=[student_device.index],
        output_device=student_device.index,
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=False,
    )
    model_to_save = student.module

    if is_main:
        save_checkpoint(model_to_save, tokenizer, args.output_dir, 0)
        logger.info("已保存训练前学生模型 checkpoint-0。")
    dist.barrier()

    trainable_params = [p for p in model_to_save.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found; 请检查冻结设置。")

    total_params = sum(p.numel() for p in model_to_save.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    if is_main:
        logger.info(f"可训练参数: {trainable_param_count:,} / {total_params:,}")
        if layer_indices:
            logger.info(f"替换层: {layer_indices}")
            logger.info(
                "替换层参数统计: total=%s, mlp=%s, other=%s",
                f"{replaced_stats['total']:,}",
                f"{replaced_stats['mlp']:,}",
                f"{replaced_stats['non_mlp']:,}",
            )
        if frozen_mlp:
            logger.info(f"冻结 MLP 参数数: {frozen_mlp:,}")
        if getattr(args, "freeze_replaced_layers", False):
            logger.info(f"冻结替换层参数数: {frozen_replaced:,}")

        replaced_non_mlp = replaced_stats["non_mlp"] if layer_indices else 0
        covered_params = trainable_param_count + frozen_mlp + replaced_non_mlp
        uncovered = total_params - covered_params
        if layer_indices:
            logger.info(
                "参数覆盖检查: trainable(%s) + frozen_mlp(%s) + replaced_non_mlp(%s) = %s (差值 %s)",
                f"{trainable_param_count:,}",
                f"{frozen_mlp:,}",
                f"{replaced_non_mlp:,}",
                f"{covered_params:,}",
                f"{uncovered:,}",
            )
        else:
            logger.info(
                "参数覆盖检查: trainable(%s) + frozen_mlp(%s) = %s (差值 %s) —— 当前未替换任何层。",
                f"{trainable_param_count:,}",
                f"{frozen_mlp:,}",
                f"{covered_params:,}",
                f"{uncovered:,}",
            )

    if wandb_run:
        wandb_run.summary["ddp_world_size"] = world_size
        wandb_run.summary["trainable_params"] = trainable_param_count
        wandb_run.summary["total_params"] = total_params
        wandb_run.summary["frozen_mlp_params"] = frozen_mlp
        wandb_run.summary["frozen_replaced_layer_params"] = frozen_replaced
        if layer_indices:
            wandb_run.summary["replaced_layers"] = layer_indices
            wandb_run.summary["replaced_layers_total_params"] = replaced_stats["total"]
            wandb_run.summary["replaced_layers_mlp_params"] = replaced_stats["mlp"]
            wandb_run.summary["replaced_layers_non_mlp_params"] = replaced_stats["non_mlp"]

    from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO

    adamw_kwargs = dict(lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, foreach=False)
    optimizer = ZRO(
        params=trainable_params,
        optimizer_class=AdamW,
        parameters_as_bucket_view=True,
        overlap_with_ddp=False,
        **adamw_kwargs,
    )

    epochs = int(args.epochs) if args.epochs and args.epochs > 0 else 1
    max_optimizer_steps = max(0, int(getattr(args, "max_steps", 0)))
    total_steps_for_sched = max(max_optimizer_steps, 1) if max_optimizer_steps > 0 else max(epochs, 1)
    scheduler = build_wsd_scheduler(optimizer, args, total_steps_for_sched)

    dataset_kwargs = {"name": args.dataset_subset} if getattr(args, "dataset_subset", None) else None

    def _get_student_vocab():
        backbone = get_backbone(model_to_save)
        if backbone is not None and hasattr(backbone, "embedding"):
            return int(backbone.embedding.weight.shape[0])
        return None

    if is_main:
        from transformers import AutoConfig

        teacher_config = AutoConfig.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            local_files_only=args.local_files_only,
        )
        teacher_vocab = int(teacher_config.vocab_size)
    else:
        teacher_vocab = -1

    vocab_tensor = torch.tensor([teacher_vocab], device=student_device, dtype=torch.long)
    dist.broadcast(vocab_tensor, src=0)
    teacher_vocab = int(vocab_tensor.item())
    student_vocab = _get_student_vocab()
    min_vocab = min(student_vocab if student_vocab is not None else teacher_vocab, teacher_vocab)

    student_embedding = get_embedding_module(model_to_save)
    embedding_dtype = getattr(getattr(student_embedding, "weight", None), "dtype", torch.float32)
    loss_dtype = torch_dtype
    if is_main:
        logger.info(f"学生 embedding dtype={embedding_dtype}；min_vocab={min_vocab}；loss_dtype={loss_dtype}。")

    def fetch_teacher_logits_distributed(
        inputs_local: torch.Tensor,
        attn_local: Optional[torch.Tensor],
        seq_len: int,
        need_teacher: bool,
    ) -> Optional[torch.Tensor]:
        if not need_teacher:
            return None

        batch_local = inputs_local.size(0)

        gather_inputs = [torch.empty_like(inputs_local, device=student_device) for _ in range(world_size)]
        dist.all_gather(gather_inputs, inputs_local.to(student_device, non_blocking=True))

        gather_attention = None
        if attn_local is not None:
            gather_attention = [torch.empty_like(attn_local, device=student_device) for _ in range(world_size)]
            dist.all_gather(gather_attention, attn_local.to(student_device, non_blocking=True))

        if is_main:
            big_inputs = torch.cat(gather_inputs, dim=0).to(teacher_device, non_blocking=True)
            teacher_kwargs = {"input_ids": big_inputs}
            if gather_attention is not None:
                big_attention = torch.cat(gather_attention, dim=0).to(teacher_device, non_blocking=True)
                teacher_kwargs["attention_mask"] = big_attention

            with torch.no_grad():
                teacher_out = teacher(**teacher_kwargs)
                teacher_logits_all = (
                    teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out
                ).to(dtype=loss_dtype)
                if teacher_logits_all.size(-1) > min_vocab:
                    teacher_logits_all = teacher_logits_all[..., :min_vocab]

            slices = torch.chunk(teacher_logits_all, world_size, dim=0)
        else:
            slices = None

        recv_logits = torch.empty((batch_local, seq_len, min_vocab), dtype=loss_dtype, device=student_device)
        tmp_buf = torch.empty_like(recv_logits)
        for r in range(world_size):
            if is_main:
                tmp_buf.copy_(slices[r].to(student_device, non_blocking=True))
            dist.broadcast(tmp_buf, src=0)
            if r == rank:
                recv_logits.copy_(tmp_buf)
        return recv_logits

    global_step = int(getattr(args, "resume_step", 0))
    grad_accum = int(getattr(args, "grad_accumulation_steps", 1))
    temperature = float(getattr(args, "temperature", 1.0))
    save_interval = int(getattr(args, "save_interval", 0))

    tokens_accum = 0.0
    loss_accum = 0.0
    step_last_log = None
    start_time = time.time()

    need_teacher_kd = float(getattr(args, "kl_weight", 0.0)) > 0.0
    optimizer_steps_done = global_step // max(grad_accum, 1)

    for epoch in range(epochs):
        if is_main:
            logger.info(f"=== Epoch {epoch + 1}/{epochs} 开始 ===")

        dataloader = build_dataloader(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            text_field=args.dataset_text_field,
            batch_size=args.batch_size,
            streaming=not getattr(args, "no_dataset_streaming", False),
            shuffle_buffer=args.shuffle_buffer,
            revision=args.dataset_revision,
            local_files_only=args.local_files_only,
            world_size=world_size,
            rank=rank,
            dataset_kwargs=dataset_kwargs,
        )

        if is_main and epoch == 0:
            logger.info(f"DataLoader 构建完成，数据集 {args.dataset_name}，分片 rank={rank}。")
            logger.info("注意：请确保 DataLoader(drop_last=True)，以保证各 rank 每步 batch 等长。")

        for batch in dataloader:
            if max_optimizer_steps and optimizer_steps_done >= max_optimizer_steps:
                break

            input_ids_cpu, attention_mask_cpu, labels_cpu, loss_mask_cpu = split_batch(batch)
            # Step-wise preprocessing:
            # 1. split_batch may return only packed tokens; synthesize shifted labels/masks when absent.
            # 2. Trim the last token from inputs so each step predicts the next token (teacher forcing).
            # 3. Build attention/loss masks that mark valid prediction positions for CE / KL.
            # 4. Move tensors onto the local GPU before the forward pass.
            if labels_cpu is None:
                labels_cpu = input_ids_cpu[:, 1:].contiguous()
            else:
                labels_cpu = labels_cpu.contiguous()
            input_ids_cpu = input_ids_cpu[:, :-1].contiguous()
            if loss_mask_cpu is None:
                loss_mask_cpu = (
                    attention_mask_cpu[:, 1:].contiguous()
                    if attention_mask_cpu is not None
                    else torch.ones_like(labels_cpu, dtype=torch.long)
                )
            else:
                loss_mask_cpu = loss_mask_cpu.contiguous()

            # Track effective tokens for logging / throughput accounting (mask excludes padding).
            tokens_accum += float(loss_mask_cpu.sum())

            inputs_student = input_ids_cpu.to(student_device, non_blocking=True)
            labels = labels_cpu.to(student_device, non_blocking=True)
            loss_mask = loss_mask_cpu.to(student_device, non_blocking=True)

            student_kwargs = {"input_ids": inputs_student}
            if attention_mask_cpu is not None:
                student_kwargs["attention_mask"] = attention_mask_cpu[:, :-1].to(student_device, non_blocking=True)
            student_out = student(**student_kwargs)
            student_logits_full = student_out.logits if hasattr(student_out, "logits") else student_out

            seq_len = inputs_student.size(1)
            teacher_logits = fetch_teacher_logits_distributed(
                input_ids_cpu,
                attention_mask_cpu[:, :-1] if attention_mask_cpu is not None else None,
                seq_len,
                need_teacher_kd,
            )

            student_logits_raw = student_logits_full[..., :min_vocab].to(dtype=loss_dtype)
            student_logits = student_logits_raw
            if teacher_logits is None:
                teacher_logits = student_logits_raw.new_zeros(student_logits_raw.shape)

            ce_weight = float(getattr(args, "ce_weight", 1.0))
            kl_weight = float(getattr(args, "kl_weight", 1.0))
            total_loss, kl_loss, ce_loss = compute_distill_loss(
                student_logits,
                teacher_logits,
                student_logits_raw,
                labels[:, :seq_len],
                loss_mask[:, :seq_len],
                ce_weight,
                kl_weight,
                temperature,
            )

            micro_loss = total_loss / grad_accum
            if (global_step % grad_accum) != (grad_accum - 1):
                with student.no_sync():
                    micro_loss.backward()
            else:
                micro_loss.backward()

            if ((global_step + 1) % grad_accum) == 0:
                optimizer_steps_done += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step_loss = (loss_accum + float(total_loss.detach().cpu())) / float(grad_accum)
                loss_accum = 0.0

                if is_main:
                    now = time.time()
                    step_time = (now - step_last_log) if step_last_log is not None else None
                    step_last_log = now
                    lr = get_learning_rate(scheduler, optimizer)
                    msg = (
                        f"[train] micro_step {global_step+1} (opt {optimizer_steps_done}) | "
                        f"loss {step_loss:.6e} | kl {float(kl_loss):.6e} | ce {float(ce_loss):.6e} | lr {lr:.2e}"
                    )
                    if step_time is not None:
                        msg += f" | step_time {step_time:.2f}s"
                    logger.info(msg)
                    if wandb_run:
                        wandb_run.log(
                            {
                                "train/loss": float(step_loss),
                                "train/kl": float(kl_loss),
                                "train/ce": float(ce_loss),
                                "train/lr": float(lr),
                                "train/tokens": float(tokens_accum),
                                "train/optimizer_step": optimizer_steps_done,
                                "train/stage": "Stage-1",
                            },
                            step=optimizer_steps_done,
                        )
                tokens_accum = 0.0
            else:
                loss_accum += float(total_loss.detach().cpu())

            if save_interval > 0 and ((global_step + 1) % save_interval == 0) and is_main:
                save_checkpoint(model_to_save, tokenizer, args.output_dir, global_step + 1)

            global_step += 1
            if max_optimizer_steps and optimizer_steps_done >= max_optimizer_steps:
                break

        if max_optimizer_steps and optimizer_steps_done >= max_optimizer_steps:
            break

    if is_main:
        save_checkpoint(model_to_save, tokenizer, args.output_dir, global_step)
        elapsed = time.time() - start_time
        logger.info(f"训练完成，已保存最终模型。总时长 {elapsed/60.0:.1f} 分钟")
        if wandb_run:
            wandb_run.summary["final_step"] = global_step
            wandb_run.summary["final_optimizer_step"] = optimizer_steps_done
            wandb_run.log({"train/final_step": global_step}, step=global_step)
            wandb_run.finish()

    ddp_cleanup()


if __name__ == "__main__":
    main()
