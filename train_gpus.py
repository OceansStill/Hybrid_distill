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
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from pytorch_optimizer.lr_scheduler import get_wsd_schedule as get_wsd_schedule_pyo

# 你项目内模块
from arguments import parse_args
from data import build_dataloader
from model_utils import (
    apply_replacements,
    get_student_model,
    get_teacher_model,
    transfer_teacher_weights,
)
from utils import configure_environment, init_wandb, save_checkpoint, setup_logger


# -------------------------
# 小工具
# -------------------------

def _get(module, names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None

def _get_backbone(model):
    return _get(model, ["backbone", "model"])

def _get_embedding(model):
    bb = _get_backbone(model)
    if bb is None:
        return None
    return _get(bb, ["embedding", "embed_tokens"])

def _get_layers(model) -> List[nn.Module]:
    bb = _get_backbone(model)
    if bb is None:
        return []
    return list(getattr(bb, "layers", []))

def _filtered_layers(layers: Optional[List[int]]):
    if layers is None:
        return []
    return [li for li in layers if isinstance(li, int) and li >= 0]

def freeze_student_mlps(student) -> int:
    frozen = 0
    for layer in _get_layers(student):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for p in mlp.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen += p.numel()
    return frozen

def freeze_student_layers(student, indices: List[int]) -> int:
    frozen = 0
    layers = _get_layers(student)
    for idx in indices:
        if 0 <= idx < len(layers):
            for p in layers[idx].parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen += p.numel()
    return frozen

def build_wsd_scheduler(optimizer: AdamW, args, total_steps: int):
    warm_steps = max(1, int(total_steps * args.warmup_ratio))
    decay_steps = max(1, int(total_steps * args.decay_ratio))
    stable_steps = max(0, int(total_steps) - warm_steps - decay_steps)
    min_lr_ratio = float(args.min_lr) / float(args.max_lr) if args.max_lr > 0 else 0.0
    return get_wsd_schedule_pyo(
        optimizer,
        num_warmup_steps=warm_steps,
        num_stable_steps=stable_steps,
        num_decay_steps=decay_steps,
        min_lr_ratio=min_lr_ratio,
        num_cycles=0.5,
        cooldown_type="1-sqrt",
    )

def ddp_init():
    """从 torchrun 环境变量初始化分布式。"""
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用")
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank, world_size, local_rank

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def ddp_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def _get_lr(scheduler, optimizer):
    if hasattr(scheduler, "get_last_lr"):
        lr = scheduler.get_last_lr()
        return float(lr[0] if isinstance(lr, (list, tuple)) else lr)
    return float(optimizer.param_groups[0].get("lr", 0.0))

def _fingerprint(model: nn.Module, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        num_params = sum(1 for _ in model.parameters())
        num_elems  = sum(p.numel() for p in model.parameters())
        return torch.tensor([num_params, num_elems], device=device, dtype=torch.long)


# -------------------------
# main
# -------------------------

def main():
    configure_environment()
    args = parse_args()

    # === 分布式初始化 ===
    rank, world_size, local_rank = ddp_init()
    is_main = (rank == 0)

    # 设备映射：
    # - 学生：可见 0..world_size-1
    # - 老师：可见 visible_count-1（最后一张）
    visible_count = torch.cuda.device_count()
    if visible_count < world_size + 1:
        raise RuntimeError(
            f"可见 GPU 数={visible_count}，需要 >= world_size+1 = {world_size+1} 才能实现 老师=1卡 + 学生={world_size}卡。"
        )

    student_device = torch.device(f"cuda:{local_rank}")
    teacher_device = torch.device(f"cuda:{visible_count - 1}")
    torch.cuda.set_device(student_device)

    torch_dtype = getattr(torch, args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    # === 日志 / wandb ===
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

    # === Teacher / Student（先在 CPU，便于修改/同步） ===
    cpu = torch.device("cpu")
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    # tokenizer：所有 rank 都要
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        # rank0：真老师（有权重）
        teacher, _ = get_teacher_model(device=cpu, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    else:
        # 其他 rank：骨架老师（无权重，只要结构）
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

    # 学生（先在 CPU）
    student = get_student_model(device=cpu, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    student.tokenizer = tokenizer

    # --- rotary：所有 rank 都要把 teacher 的 rotary_emb 覆盖到学生，保持 buffer/结构一致 ---
    t_bb = _get_backbone(teacher)
    s_bb = _get_backbone(student)
    if t_bb is not None and s_bb is not None and hasattr(t_bb, "rotary_emb"):
        try:
            s_bb.rotary_emb = copy.deepcopy(t_bb.rotary_emb).to(device=cpu)
            if is_main:
                logger.info("已将 teacher.backbone.rotary_emb 深拷贝到学生（CPU）。")
        except Exception as e:
            if is_main:
                logger.warning(f"复制 rotary_emb 失败：{e}")

    # --- 层替换：所有 rank 都要替换，以保证拓扑一致；但只在 rank0 拷权重 ---
    layer_indices = _filtered_layers(getattr(args, "layers", []))
    if layer_indices:
        if is_main:
            logger.info(f"替换以下层为教师模型对应层: {layer_indices}")
        apply_replacements(student, teacher, layer_indices)   # 所有 rank：替换成老师同类模块（结构一致）
        if is_main:
            transfer_teacher_weights(student, teacher)        # 仅 rank0：把老师权重拷到学生

    # 冻结学生
    frozen_mlp = freeze_student_mlps(student) if getattr(args, "freeze_mlp", False) else 0
    frozen_replaced = freeze_student_layers(student, layer_indices) if getattr(args, "freeze_replaced_layers", False) else 0

    # 老师设备：只有 rank0 需要保留并放到 teacher_device
    if is_main:
        teacher.eval().to(teacher_device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        # 其他 rank：老师只用于结构替换，之后即可释放
        del teacher
        torch.cuda.empty_cache()

    # 学生到本 rank 设备，并包 DDP（DDP 会把 rank0 的参数广播到其他 rank）
    student.to(student_device)

    # DDP 前做一次“参数拓扑指纹”校验（同构性）
    fp = _fingerprint(student, student_device)
    all_fp = [torch.zeros_like(fp) for _ in range(world_size)]
    dist.all_gather(all_fp, fp)
    uniq = sorted({(int(x[0].item()), int(x[1].item())) for x in all_fp})
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

    # === 优化器 / 调度器 ===
    trainable_params = [p for p in model_to_save.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found; 请检查冻结设置。")

    total_params = sum(p.numel() for p in model_to_save.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    if is_main:
        logger.info(f"可训练参数: {trainable_param_count:,} / {total_params:,}")
        if layer_indices:
            logger.info(f"替换层: {layer_indices}")
        if frozen_mlp:
            logger.info(f"冻结 MLP 参数数: {frozen_mlp:,}")
        if getattr(args, "freeze_replaced_layers", False):
            logger.info(f"冻结替换层参数数: {frozen_replaced:,}")

    from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
    adamw_kwargs = dict(lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, foreach=False)
    optimizer = ZRO(
        params=trainable_params,
        optimizer_class=AdamW,
        parameters_as_bucket_view=True,
        overlap_with_ddp=False,
        **adamw_kwargs,
    )

    total_steps_for_sched = max(1, int(args.max_steps) * int(args.epochs or 1))
    scheduler = build_wsd_scheduler(optimizer, args, total_steps_for_sched)

    # === DataLoader（按 DDP 分片） ===
    dataset_kwargs = {"name": args.dataset_subset} if getattr(args, "dataset_subset", None) else None
    dataloader: DataLoader = build_dataloader(
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
    if is_main:
        logger.info(f"DataLoader 构建完成，数据集 {args.dataset_name}，分片 rank={rank}。")
        logger.info("注意：请确保 DataLoader(drop_last=True)，以保证各 rank 每步 batch 等长。")

    # === 词表对齐 / eos ===
    def _get_student_vocab():
        bb = _get_backbone(model_to_save)
        if bb is not None and hasattr(bb, "embedding"):
            return int(bb.embedding.weight.shape[0])
        return None

    # rank0 统计老师 vocab，并广播（这里以 tokenizer 的 vocab_size 为准）
    if is_main:
        from transformers import AutoConfig
        tcfg = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", local_files_only=args.local_files_only)
        t_vocab = int(tcfg.vocab_size)
    else:
        t_vocab = -1

    t_vocab_tensor = torch.tensor([t_vocab], device=student_device, dtype=torch.long)
    dist.broadcast(t_vocab_tensor, src=0)
    t_vocab = int(t_vocab_tensor.item())
    s_vocab = _get_student_vocab()
    min_vocab = min(s_vocab if s_vocab is not None else t_vocab, t_vocab)

    # 学生 embedding dtype（用于对齐 teacher logits）
    s_emb = _get_embedding(model_to_save)
    embedding_dtype = getattr(getattr(s_emb, "weight", None), "dtype", torch.float32)
    loss_dtype = torch.bfloat16  # 统一 16 位（损失内部 fp32 计算）
    if is_main:
        logger.info(f"学生 embedding dtype={embedding_dtype}；min_vocab={min_vocab}。")

    # === 教师 logits 的跨 rank 获取函数 ===
    def fetch_teacher_logits_distributed(inputs_local: torch.Tensor,
                                         attn_local: Optional[torch.Tensor],
                                         seq_len: int,
                                         need_teacher: bool) -> Optional[torch.Tensor]:
        """
        各 rank：把本地 [B, L] 输入 gather 到 rank0；rank0 在老师卡上前向 -> [B_total, L, V_t]
        -> 裁剪到 min_vocab -> 按 rank 切片 -> 逐个 broadcast 回各 rank。
        返回：本 rank 的 teacher logits [B, L, min_vocab]（dtype=loss_dtype，device=student_device）。
        若 need_teacher=False，则直接返回 None（跳过老师前向与通信）。
        """
        if not need_teacher:
            return None

        B_local = inputs_local.size(0)

        # 1) all_gather 输入
        gather_inputs = [torch.empty_like(inputs_local, device=student_device) for _ in range(world_size)]
        dist.all_gather(gather_inputs, inputs_local.to(student_device, non_blocking=True))

        gather_attn = None
        if attn_local is not None:
            gather_attn = [torch.empty_like(attn_local, device=student_device) for _ in range(world_size)]
            dist.all_gather(gather_attn, attn_local.to(student_device, non_blocking=True))

        # 2) rank0 老师卡前向
        if is_main:
            big_inputs = torch.cat(gather_inputs, dim=0).to(teacher_device, non_blocking=True)
            teacher_kwargs = {"input_ids": big_inputs}
            if gather_attn is not None:
                big_attn = torch.cat(gather_attn, dim=0).to(teacher_device, non_blocking=True)
                teacher_kwargs["attention_mask"] = big_attn

            with torch.no_grad():
                t_out = teacher(**teacher_kwargs)  # teacher 仅存在于 rank0
                t_all = (t_out.logits if hasattr(t_out, "logits") else t_out).to(dtype=loss_dtype)
                if t_all.size(-1) > min_vocab:
                    t_all = t_all[..., :min_vocab]

            # 按等份切回各 rank（要求每 rank 的 B_local 相等；建议 DataLoader(drop_last=True)）
            slices = torch.chunk(t_all, world_size, dim=0)
        else:
            slices = None

        # 3) broadcast 回各 rank（一次一个 rank 的切片）
        recv_logits = torch.empty((B_local, seq_len, min_vocab), dtype=loss_dtype, device=student_device)
        for r in range(world_size):
            if is_main:
                buf = slices[r]
            else:
                buf = torch.empty_like(recv_logits)

            dist.broadcast(buf, src=0)
            if r == rank:
                recv_logits.copy_(buf.to(student_device, non_blocking=True))

        return recv_logits

    # === 训练循环 ===
    global_step = int(getattr(args, "resume_step", 0))
    max_steps = int(getattr(args, "max_steps", 1000))
    grad_accum = int(getattr(args, "grad_accumulation_steps", 1))
    T = float(getattr(args, "temperature", 1.0))
    save_interval = int(getattr(args, "save_interval", 0))
    epochs = int(args.epochs) if args.epochs and args.epochs > 0 else 1

    if wandb_run:
        wandb_run.summary["ddp_world_size"] = world_size

    tokens_accum = 0.0
    t0 = time.time()
    for ep in range(epochs):
        if is_main:
            logger.info(f"=== Epoch {ep+1}/{epochs} 开始 ===")

        loss_accum = 0.0
        optimizer_steps = 0
        step_last_log = None

        for batch in dataloader:
            if max_steps and global_step >= max_steps:
                break

            input_ids_cpu, attention_mask_cpu, labels_cpu, loss_mask_cpu = _split_batch(batch)
            if labels_cpu is None:
                labels_cpu = input_ids_cpu[:, 1:].contiguous()
            else:
                labels_cpu = labels_cpu.contiguous()
            input_ids_cpu = input_ids_cpu[:, :-1].contiguous()
            if loss_mask_cpu is None:
                loss_mask_cpu = attention_mask_cpu[:, 1:].contiguous() if attention_mask_cpu is not None else torch.ones_like(labels_cpu, dtype=torch.long)
            else:
                loss_mask_cpu = loss_mask_cpu.contiguous()

            tokens_accum += float(loss_mask_cpu.sum())

            # === 学生前向（本 rank）
            inputs_stu = input_ids_cpu.to(student_device, non_blocking=True)
            labels = labels_cpu.to(student_device, non_blocking=True)
            loss_mask = loss_mask_cpu.to(student_device, non_blocking=True)
            stu_kwargs = {"input_ids": inputs_stu}
            if attention_mask_cpu is not None:
                stu_kwargs["attention_mask"] = attention_mask_cpu[:, :-1].to(student_device, non_blocking=True)
            s_out = student(**stu_kwargs)
            s_logits_full = s_out.logits if hasattr(s_out, "logits") else s_out

            # === 老师 logits（跨 rank 取回；若 kl_weight==0 则跳过）
            need_teacher = float(getattr(args, "kl_weight", 0.0)) > 0.0
            inputs_teh_local = input_ids_cpu  # 仍在 CPU
            attn_teh_local = attention_mask_cpu[:, :-1] if attention_mask_cpu is not None else None
            seq_len = inputs_teh_local.size(1)
            t_logits = fetch_teacher_logits_distributed(inputs_teh_local, attn_teh_local, seq_len, need_teacher)

            # === 对齐 vocab / 温度
            s_logits_raw = s_logits_full[..., :min_vocab].to(dtype=loss_dtype)
            #s_logits = (s_logits_raw / T) if T != 1.0 else s_logits_raw
            s_logits=s_logits_raw
            if t_logits is None:
                # 若无需 KD，构造占位张量以便传参（不会被用到，因为 kl_weight==0）
                t_logits = s_logits_raw.new_zeros((s_logits_raw.size(0), s_logits_raw.size(1), s_logits_raw.size(2)))

            # === 计算损失（CE + KL；若只想 CE，可把 kl_weight 设为 0）
            ce_w = float(getattr(args, "ce_weight", 0))
            kl_w = float(getattr(args, "kl_weight", 1.0))
            total_loss, kl_loss, ce_loss = _compute_distill_loss(
                s_logits,
                t_logits, #/ T if T != 1.0 else t_logits,
                s_logits_raw,
                labels[:, :seq_len],
                loss_mask[:, :seq_len],
                ce_w,
                kl_w,
                T,
            )

            # === 梯度累积 + no_sync
            micro_loss = total_loss / grad_accum
            if (global_step % grad_accum) != (grad_accum - 1):
                with student.no_sync():
                    micro_loss.backward()
            else:
                micro_loss.backward()

            # === 到达一个优化步
            if ((global_step + 1) % grad_accum) == 0:
                optimizer_steps += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step_loss = (loss_accum + float(total_loss.detach().cpu())) / float(grad_accum)
                loss_accum = 0.0

                if is_main:
                    now = time.time()
                    step_time = (now - step_last_log) if step_last_log is not None else None
                    step_last_log = now
                    lr = _get_lr(scheduler, optimizer)
                    msg = (
                        f"[train] step {global_step+1} (opt {optimizer_steps}) | "
                        f"loss {step_loss:.4f} | kl {float(kl_loss):.4f} | ce {float(ce_loss):.4f} | lr {lr:.2e}"
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
                                "train/stage": "Stage-1",
                            },
                            step=global_step + 1,
                        )
                tokens_accum = 0.0
            else:
                loss_accum += float(total_loss.detach().cpu())

            # === 周期保存（仅主进程）
            if save_interval > 0 and ((global_step + 1) % save_interval == 0) and is_main:
                save_checkpoint(model_to_save, tokenizer, args.output_dir, global_step + 1)

            global_step += 1

    # 结束保存（主进程）
    if is_main:
        save_checkpoint(model_to_save, tokenizer, args.output_dir, global_step)
        elapsed = time.time() - t0
        logger.info(f"训练完成，已保存最终模型。总时长 {elapsed/60.0:.1f} 分钟")
        if wandb_run:
            wandb_run.summary["final_step"] = global_step
            wandb_run.log({"train/final_step": global_step}, step=global_step)
            wandb_run.finish()

    ddp_cleanup()


def _split_batch(batch):
    if isinstance(batch, dict):
        return (
            batch["input_ids"],
            batch.get("attention_mask"),
            batch.get("labels"),
            batch.get("loss_mask"),
        )
    if isinstance(batch, (list, tuple)):
        items = list(batch) + [None] * (4 - len(batch))
        return tuple(items[:4])
    return batch, None, None, None


def _compute_distill_loss(
    student_logits: torch.Tensor,      # 学生用于 KD 的 logits [B, L, V]（可已/未除以 T；本函数内部会再次除以 T 以防万一）
    teacher_logits: torch.Tensor,      # 老师用于 KD 的 logits [B, L, V]
    student_logits_raw: torch.Tensor,  # 学生用于 CE 的“原始 logits”（不做温度软化）
    labels: torch.Tensor,              # [B, L]；-100 表示忽略
    loss_mask: torch.Tensor,           # [B, L]；1=计入损失
    ce_weight: float,
    kl_weight: float,
    temperature: float,
):
    """
    总损失 = kl_weight * KL(teacher || student) + ce_weight * CE(labels, student)

    - KD（软目标）：对 teacher / student logits 做温度软化（除以 T），
      用 KL(teacher || student) 的数值稳定实现（log_target=True），最后乘以 T^2（Hinton 规则）。
    - CE（硬标签）：使用学生“原始 logits”（不除以 T），与真实 labels 做交叉熵；支持 ignore_index=-100。
    - 掩码：KL 用 loss_mask 作为分母；CE 的分母只统计“标签有效 & mask=1”的位置。
    - 为健壮性，KD 分支内部会再次执行 /T 与 log_softmax（即使外部已 /T），避免重复使用时遗漏。
    """
    loss_mask = loss_mask.float()

    # ---------- KL (teacher || student) ----------
    if kl_weight > 0.0:
        s_logp = (student_logits.float() / temperature).log_softmax(dim=-1)          # log P_s^T
        t_logp = (teacher_logits.float() / temperature).log_softmax(dim=-1).detach() # log P_t^T（冻结老师）

        kl_per_token = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
        denom_kl = loss_mask.sum().clamp_min(1.0)
        kl = (kl_per_token * loss_mask).sum() / denom_kl
        kl = kl * (temperature ** 2)  # Hinton 的 T^2 缩放
    else:
        kl = student_logits.new_zeros([])

    # ---------- CE（兼容 label 越界 -> 忽略；分母仅统计有效标签） ----------
    if ce_weight > 0.0:
        logits_ce = student_logits_raw.float()  # [B, L, V]
        V = logits_ce.size(-1)

        labels_ce = labels.clone()
        labels_ce = torch.where(labels_ce < V, labels_ce, labels_ce.new_full(labels_ce.shape, -100))

        valid_mask = (loss_mask > 0) & (labels_ce != -100)
        denom_ce = valid_mask.float().sum().clamp_min(1.0)

        ce_flat = F.cross_entropy(
            logits_ce.view(-1, V),
            labels_ce.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        ce_per_token = ce_flat.view_as(labels_ce).float()
        ce_per_token = torch.nan_to_num(ce_per_token, nan=0.0)

        ce = (ce_per_token * valid_mask.float()).sum() / denom_ce
    else:
        ce = student_logits.new_zeros([])

    total = kl_weight * kl + ce_weight * ce
    return total, kl.detach(), ce.detach()


if __name__ == "__main__":
    main()
