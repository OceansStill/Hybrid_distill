import os
from typing import Dict, Iterable, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_optimizer.lr_scheduler import get_wsd_schedule as get_wsd_schedule_pyo

BatchType = Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Dict[str, torch.Tensor],
    torch.Tensor,
    Iterable[torch.Tensor],
]


def ddp_init() -> Tuple[int, int, int]:
    """Initialize torch.distributed from torchrun environment variables."""
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


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_wsd_scheduler(optimizer: Optimizer, args, total_steps: int):
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


def get_learning_rate(scheduler, optimizer: Optimizer) -> float:
    if hasattr(scheduler, "get_last_lr"):
        lr = scheduler.get_last_lr()
        return float(lr[0] if isinstance(lr, (list, tuple)) else lr)
    return float(optimizer.param_groups[0].get("lr", 0.0))


def model_fingerprint(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        num_params = sum(1 for _ in model.parameters())
        num_elems = sum(param.numel() for param in model.parameters())
        return torch.tensor([num_params, num_elems], device=device, dtype=torch.long)


def split_batch(batch: BatchType):
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


def compute_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_logits_raw: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    ce_weight: float,
    kl_weight: float,
    temperature: float,
):
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    if loss_mask is None:
        raise ValueError("loss_mask must not be None in compute_distill_loss.")

    loss_mask = loss_mask.to(device=student_logits.device).float()
    valid_mask = loss_mask > 0
    kl_mask = valid_mask.float()

    if kl_weight > 0.0:
        student_logits_float = torch.nan_to_num(
            student_logits.float(), nan=0.0, posinf=1e9, neginf=-1e9
        )
        teacher_logits_float = torch.nan_to_num(
            teacher_logits.float(), nan=0.0, posinf=1e9, neginf=-1e9
        )

        s_logp = (student_logits_float / temperature).log_softmax(dim=-1)
        t_logp = (teacher_logits_float / temperature).log_softmax(dim=-1).detach()

        kl_per_token = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0)

        denom_kl = kl_mask.sum().clamp_min(1.0)
        kl = (kl_per_token * kl_mask).sum() / denom_kl
        kl = kl * (temperature ** 2)
    else:
        kl = student_logits.new_zeros([])

    if ce_weight > 0.0:
        logits_ce = torch.nan_to_num(
            student_logits_raw.float(), nan=0.0, posinf=1e9, neginf=-1e9
        )
        vocab_size = logits_ce.size(-1)

        labels_ce = labels.clone()
        labels_ce = torch.where(labels_ce < vocab_size, labels_ce, labels_ce.new_full(labels_ce.shape, -100))

        valid_ce_mask = (loss_mask > 0) & (labels_ce != -100)
        denom_ce = valid_ce_mask.float().sum().clamp_min(1.0)

        ce_flat = F.cross_entropy(
            logits_ce.view(-1, vocab_size),
            labels_ce.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        ce_per_token = ce_flat.view_as(labels_ce).float()
        ce_per_token = torch.nan_to_num(ce_per_token, nan=0.0)

        ce = (ce_per_token * valid_ce_mask.float()).sum() / denom_ce
    else:
        ce = student_logits.new_zeros([])

    total = kl_weight * kl + ce_weight * ce
    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    return total, torch.nan_to_num(kl.detach(), nan=0.0), torch.nan_to_num(ce.detach(), nan=0.0)
