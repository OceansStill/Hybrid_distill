#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP 蒸馏训练（学生在 2/3，老师在 0/1）
- 仅启 2 个进程做 DDP（local_rank=0/1 -> 学生分别在 可见cuda:0/1 => 物理 2/3）
- 同一进程内再各自加载一份 Teacher 在 可见cuda:2/3 => 物理 0/1，仅做 forward
- 老师 logits 通过 P2P 复制到本进程学生卡，再计算蒸馏 loss

依赖你的工程模块：
- arguments.parse_args
- data.build_dataloader
- model_utils.{get_teacher_model, get_student_model, apply_replacements, transfer_teacher_weights}
- utils.{configure_environment, init_wandb, save_checkpoint, setup_logger}

注意：
- 请用 torchrun 启动（见文件头注释中的命令）。
- 建议在 Linux 5.4 上导出 NCCL 稳定变量（如 NCCL_SHM_DISABLE=1 等），可在启动命令前设置。
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

def _get_final_norm(model):
    bb = _get_backbone(model)
    if bb is None:
        return None
    return _get(bb, ["final_layernorm", "norm", "ln_f"])

def _get_lm_head(model):
    return _get(model, ["lm_head", "output", "classifier", "head"])

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
    stable_steps = max(0, total_steps - warm_steps - decay_steps)
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
        # 可能被重入，直接复用
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank, world_size, local_rank

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # 后端固定 nccl
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    # 设定当前设备（映射到可见设备索引）
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


# -------------------------
# main
# -------------------------

def main():
    configure_environment()
    args = parse_args()

    # === 分布式初始化 ===
    rank, world_size, local_rank = ddp_init()
    is_main = (rank == 0)

    # 设备规划：
    # 可见顺序: CUDA_VISIBLE_DEVICES=2,3,0,1
    #   local_rank=0 -> 学生卡 = 可见cuda:0(物理2), 老师卡 = 可见cuda:2(物理0)
    #   local_rank=1 -> 学生卡 = 可见cuda:1(物理3), 老师卡 = 可见cuda:3(物理1)
    # 若未按上述设置可见卡顺序，将无法保证老师/学生分布到指定物理卡
    visible_count = torch.cuda.device_count()
    if visible_count < 4:
        raise RuntimeError(
            f"当前可见 GPU 数={visible_count}，需要 >=4（学生2/3 + 老师0/1）。"
            "请按文档设置 CUDA_VISIBLE_DEVICES=2,3,0,1 并只启 2 个进程。"
        )

    student_device = torch.device(f"cuda:{local_rank}")       # 可见 0 或 1 -> 物理 2 或 3
    teacher_device = torch.device(f"cuda:{local_rank + 2}")   # 可见 2 或 3 -> 物理 0 或 1
    torch.cuda.set_device(student_device)

    torch_dtype = getattr(torch, args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    # === 日志 / wandb ===
    logger = setup_logger(log_file=args.log_file, name="train.ddp")
    if not is_main:
        logger.setLevel(logging.WARNING)

    wandb_run = None
    if getattr(args, "wandb", False) and is_main:
        wandb_run = init_wandb(args, config=dict(vars(args)))

    if is_main:
        logger.info("===== 蒸馏训练（DDP：学生@2/3，老师@0/1）=====")
        logger.info(f"world_size={world_size}, rank={rank}, local_rank={local_rank}")
        logger.info(f"student_device={student_device}, teacher_device={teacher_device}")

    # === 构建 Teacher / Student（先在 CPU，便于共享/迁移 rotary） ===
    cpu = torch.device("cpu")
    teacher, tokenizer = get_teacher_model(device=cpu, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    student = get_student_model(device=cpu, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    student.tokenizer = tokenizer
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # rotary 迁移到学生（保持与单卡/Accelerate 逻辑一致）
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

    # 层替换 + 权重迁移（与单卡逻辑对齐）
    layer_indices = _filtered_layers(getattr(args, "layers", []))
    if layer_indices and is_main:
        logger.info(f"替换以下层为教师模型对应层: {layer_indices}")
    if layer_indices:
        apply_replacements(student, teacher, layer_indices)
    transfer_teacher_weights(student, teacher)

    # 冻结学生
    frozen_mlp = freeze_student_mlps(student) if getattr(args, "freeze_mlp", False) else 0
    frozen_replaced = freeze_student_layers(student, layer_indices) if getattr(args, "freeze_replaced_layers", False) else 0

    # 将 Teacher / Student 分配到对应设备
    teacher.eval().to(teacher_device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.to(student_device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found; 请检查冻结设置。")

    total_params = sum(p.numel() for p in student.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    if is_main:
        logger.info(f"可训练参数: {trainable_param_count:,} / {total_params:,}")
        if layer_indices:
            logger.info(f"替换层: {layer_indices}")
        if frozen_mlp:
            logger.info(f"冻结 MLP 参数数: {frozen_mlp:,}")
        if getattr(args, "freeze_replaced_layers", False):
            logger.info(f"冻结替换层参数数: {frozen_replaced:,}")

    # === DDP 包装学生 ===
    student = DDP(
        student,
        device_ids=[student_device.index],
        output_device=student_device.index,
        broadcast_buffers=False,
        find_unused_parameters=False,  # 若确有未用分支可设 True
        static_graph=False,
    )
    logger.info(f"DDP 包装学生模型，local_rank={local_rank}。")
    model_to_save = student.module
    from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO

    # === 优化器 / 调度器 ===
   # optimizer = AdamW(trainable_params, lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    adamw_kwargs = dict(lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, foreach=False)  # foreach=False 很关键
    optimizer = ZRO(
        params=trainable_params,
        optimizer_class=AdamW,
        parameters_as_bucket_view=True,
        overlap_with_ddp=False,
        **adamw_kwargs
    )

    if args.epochs and args.epochs > 0:
        total_steps_for_sched = max(1, int(args.max_steps) * int(args.epochs))
    else:
        total_steps_for_sched = max(1, int(args.max_steps))
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
    logger.info(f"DataLoader 构建完成，数据集 {args.dataset_name}，分片 rank={rank}。")
    # === 词表上界 / eos ===
    def _get_teacher_vocab():
        bb = _get_backbone(teacher)
        if bb is not None and hasattr(bb, "embed_tokens"):
            return int(bb.embed_tokens.weight.shape[0])
        if hasattr(teacher, "model") and hasattr(teacher.model, "embed_tokens"):
            return int(teacher.model.embed_tokens.weight.shape[0])
        return None

    def _get_student_vocab():
        bb = _get_backbone(model_to_save)
        if bb is not None and hasattr(bb, "embedding"):
            return int(bb.embedding.weight.shape[0])
        return None

    t_vocab = _get_teacher_vocab()
    s_vocab = _get_student_vocab()
    min_vocab = None
    if t_vocab is not None:
        min_vocab = t_vocab if min_vocab is None else min(min_vocab, t_vocab)
    if s_vocab is not None:
        min_vocab = s_vocab if min_vocab is None else min(min_vocab, s_vocab)
    eos_id = getattr(tokenizer, "eos_token_id", 0) or 0

    # 学生 embedding dtype（用于对齐 teacher logits）
    s_emb = _get_embedding(model_to_save)
    embedding_dtype = getattr(getattr(s_emb, "weight", None), "dtype", torch.float32)
    # 统一为 16 位（float16）。注意损失内部会 .float() 转为 fp32 计算，避免溢出。
    loss_dtype = torch.bfloat16
    logger.info(f"学生 embedding dtype={embedding_dtype}。")
    # === 训练循环 ===
    global_step = int(getattr(args, "resume_step", 0))
    max_steps = int(getattr(args, "max_steps", 1000))
    grad_accum = int(getattr(args, "grad_accumulation_steps", 1))
    T = float(getattr(args, "temperature", 1.0))
    save_interval = int(getattr(args, "save_interval", 0))

    epochs = int(args.epochs) if args.epochs and args.epochs > 0 else 1
    if wandb_run:
        wandb_run.summary["ddp_world_size"] = world_size

    t0 = time.time()
    for ep in range(epochs):
        if is_main:
            if args.epochs and args.epochs > 0:
                logger.info(f"=== Epoch {ep+1}/{epochs} 开始 ===")
            else:
                logger.info(f"=== 单轮（按 max_steps={max_steps}）开始 ===")

        tokens_accum = 0.0
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

            inputs_teh = input_ids_cpu.to(teacher_device, non_blocking=True)
            inputs_stu = input_ids_cpu.to(student_device, non_blocking=True)
            labels = labels_cpu.to(student_device, non_blocking=True)
            loss_mask = loss_mask_cpu.to(student_device, non_blocking=True)

            teacher_kwargs = {"input_ids": inputs_teh}
            if attention_mask_cpu is not None:
                teacher_kwargs["attention_mask"] = attention_mask_cpu[:, :-1].to(teacher_device, non_blocking=True)

            with torch.no_grad():
                t_out = teacher(**teacher_kwargs)
                t_logits = (t_out.logits if hasattr(t_out, "logits") else t_out).to(
                    device=student_device, dtype=loss_dtype, non_blocking=True
                )

            student_kwargs = {"input_ids": inputs_stu}
            if attention_mask_cpu is not None:
                student_kwargs["attention_mask"] = attention_mask_cpu[:, :-1].to(student_device, non_blocking=True)

            s_out = student(**student_kwargs)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
            seq_len = min(
                s_logits.size(1),
                t_logits.size(1),
                labels.size(1),
                loss_mask.size(1),
            )
            s_logits_raw = s_logits[:, :seq_len, :].to(dtype=loss_dtype)
            t_logits = t_logits[:, :seq_len, :].to(dtype=loss_dtype)
            labels = labels[:, :seq_len]
            loss_mask = loss_mask[:, :seq_len]
            #s_logits = (s_logits_raw / T) if T != 1.0 else s_logits_raw
            s_logits = s_logits_raw
            ce_w = float(getattr(args, "ce_weight", 1))
            kl_w = float(getattr(args, "kl_weight", 0))
            raw_loss, kl_loss, ce_loss = _compute_distill_loss(
                s_logits,
                t_logits ,#/ T if T != 1.0 else t_logits,
                s_logits_raw,
                labels,
                loss_mask,
                ce_w,
                kl_w,
                T,
            )

            # 梯度累积 + no_sync
            micro_loss = raw_loss / grad_accum
            if (global_step % grad_accum) != (grad_accum - 1):
                # 不是累积末步，避免 allreduce
                with student.no_sync():
                    micro_loss.backward()
            else:
                micro_loss.backward()

            # 到达一个优化步
            if ((global_step + 1) % grad_accum) == 0:
                optimizer_steps += 1
                # 可选：仅对可训练参数裁剪
                # torch.nn.utils.clip_grad_norm_( [p for p in model_to_save.parameters() if p.requires_grad], args.grad_clip )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 统计与日志
                step_loss = (loss_accum + float(raw_loss.detach().cpu())) / float(grad_accum)
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
                        payload = {
                            "train/loss": float(step_loss),
                            "train/lr": float(lr),
                            "train/tokens": float(tokens_accum),
                            "train/stage": "Stage-1",
                        }
                        wandb_run.log(payload, step=global_step + 1)
                tokens_accum = 0.0
            else:
                # 未到优化步，累计 raw_loss
                loss_accum += float(raw_loss.detach().cpu())

            # 周期保存（主进程）
            if save_interval > 0 and ((global_step + 1) % save_interval == 0) and is_main:
                save_checkpoint(model_to_save, tokenizer, args.output_dir, global_step + 1)

            global_step += 1

        # epoch 结束（可按需重建 dataloader 以改变shuffle/seed）
        # 这里保持简单不重建

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
    student_logits: torch.Tensor,      # 学生模型用于 KD 的 logits，形状 [B, L, V] 或 [B, V]
    teacher_logits: torch.Tensor,      # 老师模型用于 KD 的 logits，同形状
    student_logits_raw: torch.Tensor,  # 学生模型用于 CE 的“原始 logits”（不做温度软化），同形状
    labels: torch.Tensor,              # 监督标签 [B, L] 或 [B]；LM 训练常用 -100 表示忽略
    loss_mask: torch.Tensor,           # 有效 token 掩码 [B, L] 或 [B]，1=计入损失，0=忽略
    ce_weight: float,                  # 监督 CE 的权重；设为 0 可禁用 CE
    kl_weight: float,                  # KD(KL) 的权重；设为 0 可禁用 KD
    temperature: float,                # 蒸馏温度 T（>1 更“软”）
):
    """
    计算总损失 = kl_weight * KL(teacher || student) + ce_weight * CE(labels, student)

    设计要点：
    - KD：对 teacher / student 的 logits 先除以 T 做“温度软化”，
          使用 KL(teacher || student)，并在最后乘以 T^2（Hinton 规则，补偿梯度幅度）。
    - CE：使用 student 的“原始 logits”（不做温度软化），与真实 labels 做交叉熵；
          支持 ignore_index=-100，并与 loss_mask 联合控制有效位置。
    - 掩码归一：对 KL 和 CE 都是“加权平均”，分母为有效 token 数，避免长度偏差。
    - 数值稳定：在 fp32 中计算 softmax / log_softmax；teacher 分布 .detach()，避免梯度回传。
    """

    # --------- 统一 / 准备掩码与分母 ----------
    # 将 mask 转 float，便于加权平均；若你的上游保证 dtype 已是浮点，可省略这步。
    loss_mask = loss_mask.float()

    # 有效 token 的数量（分母）；clamp_min(1.0) 防止全零 mask 时除零。
    denom = loss_mask.sum().clamp_min(1.0)

    # ======================= KD（知识蒸馏，KL(teacher || student)） =======================
    # 1) 温度软化：将两侧 logits / T 后再做 (log_)softmax，能让分布更“平缓”，提供更丰富的暗知识。
    # 2) 全程用 float32 做 softmax/log_softmax 更稳，减少半精度下的下溢/NaN 风险。
    # 3) teacher 侧 .detach()：避免梯度回到老师。
    s_logp = (student_logits.float() / temperature).log_softmax(dim=-1)             # log P_s^T
    t_logp = (teacher_logits.float() / temperature).log_softmax(dim=-1).detach()    # log P_t^T（冻结）

    # 使用 log_target=True 的 KL：更稳更省（不用显式 exp）
    # 逐 token KL：在词表维（最后一维）求和；得到 [B, L] 或 [B] 的逐位 KL。
    kl_per_token = F.kl_div(
        s_logp, t_logp, reduction="none", log_target=True
    ).sum(dim=-1)

    # 基于 mask 的加权平均，得到 batch 级 KL
    kl = (kl_per_token * loss_mask).sum() / denom

    # 乘 T^2（Hinton 约定）：温度改变了梯度尺度，这个系数用于补偿
    kl = kl * (temperature ** 2)

    # ======================= CE（监督交叉熵） =======================
    if ce_weight > 0.0:
        # 用“原始 logits”做 CE（不除以 T），因为 CE 表达的是真实标签的对数似然
        logits_ce = student_logits_raw.float()
        V = logits_ce.size(-1)

        # reduction="none"：逐 token CE，稍后与 mask 相乘后按有效 token 平均；
        # ignore_index=-100：与 HF / 常见 LM 训练一致，labels 为 -100 的位置不计入 CE。
        ce_flat = F.cross_entropy(
            logits_ce.view(-1, V),           # [*, V]
            labels.view(-1),                 # [*]
            reduction="none",
            ignore_index=-100,
        )

        # 还原回与 mask 相同的形状，便于逐位相乘
        ce_per_token = ce_flat.view_as(loss_mask)

        # 保险：若上游偶发 NaN，这里清零以免污染总损失（正常情况下应为 0 次命中）
        ce_per_token = torch.nan_to_num(ce_per_token, nan=0.0)

        # 基于 mask 的加权平均，得到 batch 级 CE
        ce = (ce_per_token * loss_mask).sum() / denom
    else:
        # 若禁用 CE，返回一个与计算图兼容的 0 标量
        ce = student_logits.new_zeros([])

    # ======================= 总损失与可视化输出 =======================
    total_loss = kl_weight * kl + ce_weight * ce

    # 为了日志记录方便，将 KL/CE 的返参截断梯度，避免误参与后续图构建
    return total_loss, kl.detach(), ce.detach()


if __name__ == "__main__":
    main()
