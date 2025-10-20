import copy
import os
import sys
from typing import Dict, Iterable, Optional

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
    student.backbone.embedding.weight.data.copy_(teacher.backbone.embed_tokens.weight.data)
    student.backbone.final_layernorm.weight.data.copy_(teacher.backbone.norm.weight.data)

    if hasattr(student.lm_head, "weight"):
        student.lm_head.weight.data.copy_(teacher.lm_head.weight.data)
    if (
        hasattr(student.lm_head, "bias")
        and hasattr(teacher.lm_head, "bias")
        and student.lm_head.bias is not None
        and teacher.lm_head.bias is not None
    ):
        student.lm_head.bias.data.copy_(teacher.lm_head.bias.data)

    total_layers = layer_count or min(len(student.backbone.layers), len(teacher.backbone.layers))
    for idx in range(total_layers):
        student_layer = student.backbone.layers[idx]
        teacher_layer = teacher.backbone.layers[idx]
        if hasattr(student_layer, "input_layernorm") and hasattr(teacher_layer, "input_layernorm"):
            student_layer.input_layernorm.weight.data.copy_(teacher_layer.input_layernorm.weight.data)
        if hasattr(student_layer, "post_attention_layernorm") and hasattr(teacher_layer, "post_attention_layernorm"):
            student_layer.post_attention_layernorm.weight.data.copy_(teacher_layer.post_attention_layernorm.weight.data)
        if hasattr(student_layer, "mlp") and hasattr(teacher_layer, "mlp"):
            copy_module_parameters(student_layer.mlp, teacher_layer.mlp)


def copy_module_parameters(target: nn.Module, source: nn.Module):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
