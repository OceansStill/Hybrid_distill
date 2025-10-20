import logging
import copy
import os
from typing import List

import torch
from torch.optim import AdamW

from arguments import parse_args
from data import build_dataloader
from model_utils import (
    apply_replacements,
    get_student_model,
    get_teacher_model,
    transfer_teacher_weights,
)
from pytorch_optimizer.lr_scheduler import get_wsd_schedule as get_wsd_schedule_pyo
from trainer import run_stage
from utils import configure_environment, init_wandb, save_checkpoint, setup_logger
from datasets import load_dataset


def filter_layers(layers: List[int]):
    if layers is None:
        return []
    return [li for li in layers if li >= 0]


def freeze_student_mlps(student):
    frozen_params = 0
    for layer in getattr(student.backbone, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for param in mlp.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
    return frozen_params


def freeze_student_layers(student, layer_indices):
    frozen_params = 0
    backbone_layers = getattr(student.backbone, "layers", [])
    for li in layer_indices:
        if li < 0 or li >= len(backbone_layers):
            continue
        for param in backbone_layers[li].parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
    return frozen_params


def main():
    configure_environment()
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch_dtype = getattr(torch, args.dtype)
    is_main_process = True

    wandb_run = None
    if args.wandb and is_main_process:
        wandb_run = init_wandb(args, config=dict(vars(args)))

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logger(log_file=args.log_file, name="finetune.train")
        if not is_main_process:
            logger.setLevel(logging.WARNING)
        logger.info("===== 蒸馏训练开始 =====")
        logger.info(f"运行设备: {device}")

        teacher_device = torch.device(args.teacher_device) if args.teacher_device else device

        teacher, tokenizer = get_teacher_model(
            device=teacher_device,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
        )
        student = get_student_model(
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
        )
        student.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Rotary embedding 放置到学生设备：
        # 同设备时可共享模块；跨设备场景需要深拷贝并迁移到学生设备，避免 device mismatch。
        if hasattr(teacher, "backbone") and hasattr(student, "backbone") and hasattr(teacher.backbone, "rotary_emb"):
            if teacher_device == device:
                student.backbone.rotary_emb = teacher.backbone.rotary_emb
            else:
                student.backbone.rotary_emb = copy.deepcopy(teacher.backbone.rotary_emb).to(device=device)

        layer_indices = filter_layers(args.layers)
        if layer_indices:
            logger.info(f"替换以下层为教师模型对应层: {layer_indices}")
            apply_replacements(student, teacher, layer_indices)
        else:
            logger.info("未进行层替换，完全依赖学生模型原始结构。")

        logger.info("执行教师权重迁移。")
        transfer_teacher_weights(student, teacher)

        frozen_params = 0
        if args.freeze_mlp:
            frozen_params = freeze_student_mlps(student)
            logger.info(f"冻结学生模型 MLP 参数，总数约为 {frozen_params:,}。")

        frozen_replaced_params = 0
        if args.freeze_replaced_layers and layer_indices:
            frozen_replaced_params = freeze_student_layers(student, layer_indices)
            logger.info(
                f"冻结已替换层参数，总数约为 {frozen_replaced_params:,}。"
            )

        trainable_params = [p for p in student.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("未找到可训练参数，请检查冻结设置。")

        total_params = sum(p.numel() for p in student.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        if is_main_process:
            logger.info(
                f"学生模型参数总数: {total_params:,}，其中可训练参数: {trainable_param_count:,}"
            )
        if wandb_run:
            wandb_run.summary["total_params"] = total_params
            wandb_run.summary["trainable_params"] = trainable_param_count
            wandb_run.summary["frozen_mlp_params"] = frozen_params
            wandb_run.summary["frozen_replaced_layer_params"] = frozen_replaced_params
            if layer_indices:
                wandb_run.summary["replaced_layers"] = layer_indices

        optimizer = AdamW(
            trainable_params,
            lr=args.max_lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

        model_to_save = student

        # Build WSD scheduler using pytorch-optimizer implementation
        def build_wsd_scheduler(total_steps: int):
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
                cooldown_type='1-sqrt',
            )

        # decide total steps for schedule
        if args.epochs and args.epochs > 0:
            est_per_epoch = None
            if hasattr(args, 'epoch_step_hint') and isinstance(getattr(args, 'epoch_step_hint'), int) and args.epoch_step_hint > 0:
                est_per_epoch = args.epoch_step_hint
            total_steps_for_sched = (est_per_epoch or args.max_steps) * args.epochs
        else:
            total_steps_for_sched = args.max_steps
        scheduler = build_wsd_scheduler(max(1, int(total_steps_for_sched)))

        logger.info("Stage 1: 加载 FineWeb-Edu 数据流。")
        dataset_kwargs = {"name": args.dataset_subset} if args.dataset_subset else None

        # 自动估算每个 epoch 的优化步数（仅在 --epochs>0 且未提供 epoch_step_hint 时）
        if args.epochs and args.epochs > 0:
            try:
                # 离线模式用环境变量而不是向 load_dataset 传递 local_files_only（某些 BuilderConfig 不接受该参数）
                if args.local_files_only:
                    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

                ds_est = load_dataset(
                    args.dataset_name,
                    split=args.dataset_split,
                    streaming=False,
                    revision=args.dataset_revision,
                    **(dataset_kwargs or {}),
                )
                num_rows = getattr(ds_est, "num_rows", None)
                if num_rows is None:
                    num_rows = len(ds_est)
                sample_n = min(2000, int(num_rows)) if num_rows is not None else 0
                avg_tokens = None
                if sample_n and sample_n > 0:
                    try:
                        ds_sample = ds_est.select(range(sample_n))
                        total_tokens = 0
                        for i in range(sample_n):
                            text = ds_sample[i][args.dataset_text_field]
                            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
                            total_tokens += (len(toks) + 1)
                        avg_tokens = total_tokens / sample_n
                        tokens_epoch = avg_tokens * num_rows
                        seq_tokens = args.seq_length + 1
                        seqs_epoch = max(1, int(tokens_epoch // seq_tokens))
                        batches_epoch = max(1, (seqs_epoch + args.batch_size - 1) // args.batch_size)
                        opt_steps = max(1, (batches_epoch + args.grad_accumulation_steps - 1) // args.grad_accumulation_steps)
                        args.epoch_step_hint = int(opt_steps)
                        logger.info(
                            f"自动估算每个 epoch 优化步数 ~ {args.epoch_step_hint} (avg_tokens≈{avg_tokens:.1f}, rows={num_rows})"
                        )
                    except Exception as e:
                        logger.warning(f"自动估算 epoch 步数失败（采样失败）: {e}")
                else:
                    logger.warning("自动估算 epoch 步数失败：无法获取数据行数。")
            except Exception as e:
                logger.warning(f"自动估算 epoch 步数失败（加载数据失败）: {e}")
        global_step = args.resume_step
        if args.epochs and args.epochs > 0:
            logger.info(f"使用 epochs 模式: 共 {args.epochs} 轮（忽略 --max_steps）")
            for ep in range(1, args.epochs + 1):
                stage1_loader = build_dataloader(
                    dataset_name=args.dataset_name,
                    split=args.dataset_split,
                    tokenizer=tokenizer,
                    seq_length=args.seq_length,
                    text_field=args.dataset_text_field,
                    batch_size=args.batch_size,
                    streaming=not args.no_dataset_streaming,
                    shuffle_buffer=args.shuffle_buffer,
                    revision=args.dataset_revision,
                    local_files_only=args.local_files_only,
                    world_size=1,
                    rank=0,
                    dataset_kwargs=dataset_kwargs,
                )
                global_step = run_stage(
                    logger=logger,
                    stage_name=f"Stage-1 (epoch {ep}/{args.epochs})",
                    dataloader=stage1_loader,
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    args=args,
                    teacher_device=teacher_device,
                    tokenizer=tokenizer,
                    start_step=global_step,
                    stage_steps=None,
                    trainable_params=trainable_params,
                    wandb_run=wandb_run if is_main_process else None,
                    should_save=is_main_process,
                    model_to_save=model_to_save,
                )
        else:
            stage1_loader = build_dataloader(
                dataset_name=args.dataset_name,
                split=args.dataset_split,
                tokenizer=tokenizer,
                seq_length=args.seq_length,
                text_field=args.dataset_text_field,
                batch_size=args.batch_size,
                streaming=not args.no_dataset_streaming,
                shuffle_buffer=args.shuffle_buffer,
                revision=args.dataset_revision,
                local_files_only=args.local_files_only,
                world_size=1,
                rank=0,
                dataset_kwargs=dataset_kwargs,
            )

            global_step = run_stage(
                logger=logger,
                stage_name="Stage-1",
                dataloader=stage1_loader,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                args=args,
                teacher_device=teacher_device,
                tokenizer=tokenizer,
                start_step=global_step,
                stage_steps=args.max_steps,
                trainable_params=trainable_params,
                wandb_run=wandb_run if is_main_process else None,
                should_save=is_main_process,
                model_to_save=model_to_save,
            )
        if wandb_run:
            wandb_run.summary["stage1_micro_steps"] = global_step

        if (not args.skip_stage2) and args.second_dataset_name:
            logger.info("Stage 2: 启动 OpenHermes-2.5 监督蒸馏。")
            tokens_per_step = args.batch_size * args.seq_length
            target_steps = max(1, args.second_dataset_tokens // max(1, tokens_per_step))
            total_stage2_steps = args.second_dataset_epochs * target_steps
            stage1_final_step = global_step

            stage2_loader = build_dataloader(
                dataset_name=args.second_dataset_name,
                split=args.second_dataset_split,
                tokenizer=tokenizer,
                seq_length=args.seq_length,
                text_field=args.second_dataset_text_field,
                batch_size=args.batch_size,
                streaming=True,
                shuffle_buffer=args.shuffle_buffer,
                revision=None,
                local_files_only=args.local_files_only,
                world_size=1,
                rank=0,
            )

            scheduler.config.total_steps = global_step + total_stage2_steps

            global_step = run_stage(
                logger=logger,
                stage_name="Stage-2",
                dataloader=stage2_loader,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                args=args,
                teacher_device=teacher_device,
                tokenizer=tokenizer,
                start_step=global_step,
                stage_steps=total_stage2_steps,
                trainable_params=trainable_params,
                wandb_run=wandb_run if is_main_process else None,
                should_save=is_main_process,
                model_to_save=model_to_save,
            )
            if wandb_run:
                stage2_micro_steps = global_step - stage1_final_step
                wandb_run.summary["stage2_micro_steps"] = stage2_micro_steps
                wandb_run.summary["stage2_target_steps"] = total_stage2_steps

        if is_main_process:
            logger.info("保存最终模型权重。")
            save_checkpoint(model_to_save, tokenizer, args.output_dir, global_step)
            logger.info("===== 蒸馏训练完成 =====")
        if wandb_run:
            wandb_run.summary["final_step"] = global_step
            wandb_run.log({"train/final_step": global_step}, step=global_step)
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
