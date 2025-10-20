from typing import Iterable, Optional
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import save_checkpoint


def distillation_loss(student_logits, teacher_logits, temperature: float = 1.0):
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    loss = -(teacher_probs * student_log_probs).sum(dim=-1)
    if temperature != 1.0:
        loss = loss * (temperature ** 2)
    return loss.mean()


def run_stage(
    logger,
    stage_name: str,
    dataloader: Iterable,
    student,
    teacher,
    optimizer,
    scheduler,
    device: torch.device,
    args,
    teacher_device: torch.device,
    tokenizer,
    start_step: int = 0,
    stage_steps: Optional[int] = None,
    trainable_params=None,
    wandb_run=None,
    should_save: bool = True,
    model_to_save=None,
):
    student.train()
    teacher.eval()
    global_step = start_step
    steps_completed = 0
    progress_bar = tqdm(total=stage_steps, desc=stage_name, initial=0, dynamic_ncols=True)

    if trainable_params is None:
        trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer.zero_grad()
    embedding_dtype = student.backbone.embedding.weight.dtype

    loss_accum = 0.0
    tokens_accum = 0
    optimizer_steps = 0

    epoch_t0 = time.time()
    last_log_t = None

    for batch in dataloader if dataloader is not None else []:
        if stage_steps is not None and steps_completed >= stage_steps:
            break

        # 保持批次在 CPU，随后分别拷到老师/学生设备，避免跨 GPU 拷贝
        inputs_cpu = batch[:, :-1].contiguous()
        tokens_accum += int(inputs_cpu.numel())

        # 运行时健壮性检查：确保 token id 落在老师/学生词表范围
        # 以老师词表为基准做一次清洗，保持师生输入一致
        with torch.no_grad():
            # 读取词表大小
            def _get_teacher_vocab():
                if hasattr(teacher, "backbone") and hasattr(teacher.backbone, "embed_tokens"):
                    return int(teacher.backbone.embed_tokens.weight.shape[0])
                if hasattr(teacher, "model") and hasattr(teacher.model, "embed_tokens"):
                    return int(teacher.model.embed_tokens.weight.shape[0])
                return None

            def _get_student_vocab():
                if hasattr(student, "backbone") and hasattr(student.backbone, "embedding"):
                    return int(student.backbone.embedding.weight.shape[0])
                return None

            t_vocab = _get_teacher_vocab()
            s_vocab = _get_student_vocab()
            # 以更小的词表上界为准在 CPU 上清洗 token
            min_vocab = None
            if t_vocab is not None:
                min_vocab = t_vocab if min_vocab is None else min(min_vocab, t_vocab)
            if s_vocab is not None:
                min_vocab = s_vocab if min_vocab is None else min(min_vocab, s_vocab)
            if min_vocab is not None:
                eos_id = getattr(tokenizer, "eos_token_id", 0) or 0
                bad_mask_cpu = (inputs_cpu < 0) | (inputs_cpu >= min_vocab)
                if bad_mask_cpu.any():
                    inputs_cpu = torch.where(bad_mask_cpu, torch.full_like(inputs_cpu, eos_id), inputs_cpu)

            # 分别拷到老师/学生设备
            if teacher_device != device:
                inputs_teh = inputs_cpu.to(teacher_device, non_blocking=True)
                inputs_stu = inputs_cpu.to(device, non_blocking=True)
            else:
                inputs_teh = inputs_cpu.to(device, non_blocking=True)
                inputs_stu = inputs_teh

            teacher_outputs = teacher(input_ids=inputs_teh)
            teacher_logits = teacher_outputs.logits
            # 将老师 logits 搬回学生设备并匹配 dtype
            if teacher_device != device:
                teacher_logits = teacher_logits.to(device=device, dtype=embedding_dtype, non_blocking=True)
            else:
                teacher_logits = teacher_logits.to(dtype=embedding_dtype)

        student_outputs = student(input_ids=inputs_stu)
        student_logits = student_outputs.logits

        raw_loss = distillation_loss(student_logits, teacher_logits, temperature=args.temperature)
        loss_value = raw_loss.detach().float().item()
        loss_accum += loss_value
        loss = raw_loss / args.grad_accumulation_steps

        loss.backward()

        if (global_step + 1) % args.grad_accumulation_steps == 0:
            optimizer_steps += 1

            step_loss = loss_accum / args.grad_accumulation_steps
            tokens_for_step = tokens_accum
            loss_accum = 0.0
            tokens_accum = 0

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            now_t = time.time()
            step_time = None
            avg_step_time = None
            eta_str = ""
            if last_log_t is not None:
                step_time = now_t - last_log_t
            last_log_t = now_t
            elapsed = now_t - epoch_t0
            if optimizer_steps > 0:
                avg_step_time = elapsed / optimizer_steps

            # ETA 估计：
            if args.epochs and args.epochs > 0:
                if getattr(args, "epoch_step_hint", 0) and avg_step_time is not None:
                    remaining = max(0, int(args.epoch_step_hint - optimizer_steps))
                    eta_seconds = remaining * avg_step_time
                    eta_str = f" | eta {eta_seconds/60:.1f}m"
            elif (args.epochs == 0) and (args.max_steps is not None) and avg_step_time is not None:
                remaining = max(0, int((args.max_steps or 0) - (global_step + 1)))
                eta_seconds = remaining * avg_step_time
                eta_str = f" | eta {eta_seconds/60:.1f}m"

            if wandb_run:
                current_lr = scheduler.get_lr()[0]
                log_payload = {
                    "train/loss": step_loss,
                    "train/lr": current_lr,
                    "train/stage": stage_name,
                    "train/tokens": tokens_for_step,
                    "train/optimizer_step": optimizer_steps,
                }
                if grad_norm is not None:
                    log_payload["train/grad_norm"] = float(grad_norm)
                if step_time is not None:
                    log_payload["train/step_time_s"] = float(step_time)
                if avg_step_time is not None:
                    log_payload["train/avg_step_time_s"] = float(avg_step_time)
                wandb_run.log(log_payload, step=global_step + 1)

            if optimizer_steps % args.log_interval == 0:
                learning_rate = scheduler.get_lr()[0]
                extra = ""
                if step_time is not None:
                    extra += f" | step_time {step_time:.2f}s"
                if avg_step_time is not None:
                    extra += f" | avg_step {avg_step_time:.2f}s"
                extra += eta_str
                logger.info(
                    f"[{stage_name}] step {global_step + 1} (opt {optimizer_steps}) | "
                    f"loss {step_loss:.4f} | lr {learning_rate:.2e}{extra}"
                )

        if should_save and (global_step + 1) % args.save_interval == 0:
            target_model = model_to_save or student
            save_checkpoint(target_model, tokenizer, args.output_dir, global_step + 1)

        global_step += 1
        steps_completed += 1
        progress_bar.update(1)

    progress_bar.close()
    return global_step
