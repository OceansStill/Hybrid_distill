import logging
import os
from typing import Iterable, Optional


UNSUPPORTED_GEN_KW = {
    "stopping_criteria",
    "synced_gpus",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "attention_mask",
    "use_cache",
    "output_scores",
    "return_dict_in_generate",
    "output_hidden_states",
    "output_attentions",
    "num_return_sequences",
    "do_sample",
    "top_k",
    "top_p",
    "temperature",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
}


def configure_environment():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MAMBA_TRITON_AUTOTUNE", "0")
    os.environ.setdefault("MAMBA_TRITON_DISABLE_FALLBACK", "1")


def wrap_model_generate(model):
    if not hasattr(model, "generate"):
        return
    _orig_gen = model.generate

    def _patched_generate(*args, **kwargs):
        for key in list(kwargs.keys()):
            if key in UNSUPPORTED_GEN_KW:
                kwargs.pop(key)
        return _orig_gen(*args, **kwargs)

    model.generate = _patched_generate


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO, name: str = "finetune"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        directory = os.path.dirname(log_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def init_wandb(args, config):
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - helpful message
        raise ImportError(
            "检测到 --wandb 但未安装 wandb。请运行 `pip install wandb` 或移除该参数。"
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    )
    return run
