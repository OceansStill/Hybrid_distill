import copy
import os
import sys
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import wrap_model_generate


_GA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GA_ROOT not in sys.path:
    sys.path.append(_GA_ROOT)



def get_teacher_model(
    device: torch.device,
    torch_dtype: torch.dtype,
    local_files_only: bool = False,
):
    # 关闭老师侧 Flash-Attention，改用 SDPA 以降低显存压力并避免某些内核兼容性问题
    teacher = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_files_only=local_files_only,
    )
    teacher.config.use_cache = False
    teacher.backbone = teacher.model
    for layer in teacher.backbone.layers:
        layer.layer_idx = layer.self_attn.layer_idx
        layer.mixer = layer.self_attn
        layer.mixer.out_proj = layer.mixer.o_proj
    teacher = teacher.to(device=device)
    wrap_model_generate(teacher)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return teacher, tokenizer


def get_student_model(
    device: torch.device,
    torch_dtype: torch.dtype,
    local_files_only: bool = False,
):
    # 延迟导入，避免在仅查看 --help 时引入 mamba 依赖
    from models.llamba import LlambaLMHeadModel  # noqa: WPS433
    student = LlambaLMHeadModel.from_pretrained(
        "goombalab/Llamba-8B-untied-unaligned",
        strict=True,
        local_files_only=local_files_only,
    )
    student = student.to(dtype=torch_dtype)
    student = student.to(device=device)
    wrap_model_generate(student)
    return student


def apply_replacements(model, aux_model, layers_to_replace: Iterable[int]):
    if not layers_to_replace:
        return {}
    originals: Dict[int, nn.Module] = {}
    device = next(model.parameters()).device
    layer_count = len(model.backbone.layers)
    for li in layers_to_replace:
        if li < 0 or li >= layer_count:
            raise IndexError(f"Layer index {li} 超出学生模型层范围 0-{layer_count - 1}")
        originals[li] = model.backbone.layers[li]
        replacement = copy.deepcopy(aux_model.backbone.layers[li]).to(device=device)
        model.backbone.layers[li] = replacement
    torch.cuda.empty_cache()
    return originals


def revert_replacements(model, originals: Dict[int, nn.Module]):
    for idx, original in originals.items():
        model.backbone.layers[idx] = original
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def transfer_teacher_weights(student, teacher, layer_count: Optional[int] = None):
    """
    保留兼容接口：当前蒸馏流程仅通过 apply_replacements 深拷贝指定层，
    不再额外复制 embedding / lm_head 等权重。
    """
    return


def copy_module_parameters(target: nn.Module, source: nn.Module):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def get_first_attr(module, names: Iterable[str]):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def get_backbone(model) -> Optional[nn.Module]:
    return get_first_attr(model, ("backbone", "model"))


def get_embedding_module(model) -> Optional[nn.Module]:
    backbone = get_backbone(model)
    if backbone is None:
        return None
    return get_first_attr(backbone, ("embedding", "embed_tokens"))


def get_layers(model) -> List[nn.Module]:
    backbone = get_backbone(model)
    if backbone is None:
        return []
    return list(getattr(backbone, "layers", []))


def filter_layer_indices(indices: Optional[Iterable[int]]) -> List[int]:
    if indices is None:
        return []
    return [idx for idx in indices if isinstance(idx, int) and idx >= 0]


def freeze_student_mlps(student: nn.Module) -> int:
    frozen = 0
    for layer in get_layers(student):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for param in mlp.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
                frozen += param.numel()
    return frozen


def freeze_student_layers(student: nn.Module, indices: Iterable[int]) -> int:
    layers = get_layers(student)
    frozen = 0
    for idx in indices:
        if 0 <= idx < len(layers):
            layer = layers[idx]
            for param in layer.parameters():
                if param.requires_grad:
                    param.requires_grad_(False)
                    frozen += param.numel()
    return frozen


def count_layer_parameters(student: nn.Module, indices: Iterable[int]) -> Dict[str, int]:
    layers = get_layers(student)
    total = 0
    mlp = 0
    for idx in indices:
        if 0 <= idx < len(layers):
            layer = layers[idx]
            total += sum(p.numel() for p in layer.parameters())
            mlp_module = getattr(layer, "mlp", None)
            if mlp_module is not None:
                mlp += sum(p.numel() for p in mlp_module.parameters())
    return {
        "total": total,
        "mlp": mlp,
        "non_mlp": max(total - mlp, 0),
    }
