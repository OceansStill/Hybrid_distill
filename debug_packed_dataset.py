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
)
from utils import configure_environment, init_wandb, save_checkpoint, setup_logger


def main():
    configure_environment()
    args = parse_args()
    logger = setup_logger(log_file=args.log_file, name="train.gpus")
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    epochs = int(args.epochs) if args.epochs and args.epochs > 0 else 1
    dataset_kwargs = {"name": args.dataset_subset} if getattr(args, "dataset_subset", None) else None
 
    dataset_previewed = False

    for epoch in range(epochs):
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
            world_size=1,
            rank=0,
            dataset_kwargs=dataset_kwargs,
        )

        if  not dataset_previewed:
            try:
                preview_loader = build_dataloader(
                    dataset_name=args.dataset_name,
                    split=args.dataset_split,
                    tokenizer=tokenizer,
                    seq_length=args.seq_length,
                    text_field=args.dataset_text_field,
                    batch_size=1,
                    streaming=not getattr(args, "no_dataset_streaming", False),
                    shuffle_buffer=args.shuffle_buffer,
                    revision=args.dataset_revision,
                    local_files_only=args.local_files_only,
                    world_size=1,
                    rank=0,
                    dataset_kwargs=dataset_kwargs,
                )
                preview_batch = next(iter(preview_loader))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[preview] 获取样本失败：{exc}")
            else:
                eos_id = tokenizer.eos_token_id
                preview_input = preview_batch["input_ids"][0]
                preview_mask = preview_batch["loss_mask"][0]
                valid_len = int(preview_mask.sum().item())
                valid_len = max(valid_len, 1)
                tokens = preview_input[:valid_len].tolist()
                eos_positions = [idx for idx, tid in enumerate(tokens) if tid == eos_id]
                input_tokens = tokens[:-1] if len(tokens) > 1 else tokens
                target_token = tokens[-1]
                logger.info(
                    "[preview] total_tokens=%d eos_token_id=%s eos_count=%d eos_positions=%s",
                    valid_len,
                    eos_id,
                    len(eos_positions),
                    eos_positions[:10],
                )
                logger.info("[preview] raw_input_ids=%s", tokens)
                logger.info(
                    "[preview] input_text=%r",
                    tokenizer.decode(input_tokens, skip_special_tokens=False),
                )
                logger.info(
                    "[preview] target_token_id=%d target_token_text=%r",
                    target_token,
                    tokenizer.decode([target_token], skip_special_tokens=False),
                )
            finally:
                dataset_previewed = True

        if epoch == 0:
            logger.info(f"DataLoader 构建完成，数据集 {args.dataset_name}。")



        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # 只处理前3个batch
                break
                
            batch_input = batch["input_ids"]
            loss_mask_cpu = batch.get("loss_mask")

            input_ids_cpu = batch_input[:, :-1].contiguous()
            labels_cpu = batch_input[:, 1:].contiguous()

            if loss_mask_cpu is not None:
                loss_mask_cpu = loss_mask_cpu[:, 1:].contiguous()
            else:
                loss_mask_cpu = torch.ones_like(labels_cpu, dtype=torch.long)

            inputs_student = input_ids_cpu
            labels = labels_cpu
            loss_mask = loss_mask_cpu

            # 打印当前batch的信息
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch {batch_idx + 1}/3 信息:")
            logger.info(f"{'='*60}")
            
            # 打印形状信息
            logger.info(f"inputs_student shape: {inputs_student.shape}")
            logger.info(f"labels shape: {labels.shape}")
            logger.info(f"loss_mask shape: {loss_mask.shape}")
            
            # 打印有效token数量
            logger.info(f"sum of loss_mask: {float(loss_mask.sum())}")
            logger.info(f"有效token比例: {float(loss_mask.sum()) / loss_mask.numel():.2%}")
            
            # 遍历batch中的每个样本
            batch_size = inputs_student.shape[0]
            for sample_idx in range(batch_size):
                logger.info(f"\n  --- Sample {sample_idx + 1}/{batch_size} ---")
                
                # 获取当前样本的数据
                sample_input = inputs_student[sample_idx]
                sample_label = labels[sample_idx]
                sample_mask = loss_mask[sample_idx]
                
                # 打印样本形状
                logger.info(f"  input shape: {sample_input.shape}")
                logger.info(f"  label shape: {sample_label.shape}")
                logger.info(f"  mask shape: {sample_mask.shape}")
                
                # 打印有效长度
                valid_len = int(sample_mask.sum().item())
                logger.info(f"  有效长度: {valid_len}/{sample_mask.shape[0]}")
                
                # 打印输入的前10个和后10个token
                logger.info(f"  input前10个token: {sample_input[:10].tolist()}")
                logger.info(f"  input后10个token: {sample_input[-10:].tolist()}")
                logger.info(f"  input所有token: {sample_input.tolist()}")
                # 打印标签的前10个和后10个token
                logger.info(f"  label前10个token: {sample_label[:10].tolist()}")
                logger.info(f"  label后10个token: {sample_label[-10:].tolist()}")
                logger.info(f"  label所有token: {sample_label.tolist()}")
                # 打印mask的前10个和后10个值
                logger.info(f"  mask前10个值: {sample_mask[:10].tolist()}")
                logger.info(f"  mask后10个值: {sample_mask[-10:].tolist()}")
                logger.info(f"  mask所有值: {sample_mask.tolist()}")
                # 解码有效部分的文本(前50个token)
                if valid_len > 0:
                    valid_input = sample_input[:min(valid_len, 50)]
                    decoded_text = tokenizer.decode(valid_input.tolist(), skip_special_tokens=False)
                    logger.info(f"  输入文本(前50 tokens): {decoded_text[:200]}...")
                    
                    valid_label = sample_label[:min(valid_len, 50)]
                    decoded_label = tokenizer.decode(valid_label.tolist(), skip_special_tokens=False)
                    logger.info(f"  标签文本(前50 tokens): {decoded_label[:200]}...")
            
            logger.info(f"\n{'='*60}\n")
        
        # 只处理一个epoch后退出
        break


if __name__ == "__main__":
    main()
