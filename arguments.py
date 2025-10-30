import argparse
from typing import Sequence, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用教师 Llama-3.1-8B-Instruct 对 Llamba-8B 学生模型进行蒸馏微调。"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--teacher_device",
        type=str,
        default=None,
        help="老师模型所在设备（例如 cuda:0）。为空则与 --device 相同。",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[10, 14, 17, 30],
        help="指定要用教师模型对应层替换的学生层索引。",
    )
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="按 epoch 训练的轮数（>0 时将忽略 --max_steps，按数据集完整遍历计算一次 epoch）。",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="CC-MAIN-2014-49",
        help="FineWeb 子集名称，默认只使用 CC-MAIN-2014-49。",
    )
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument(
        "--no_dataset_streaming",
        action="store_true",
        help="禁用数据流式加载，改为一次性读取数据集（仅当数据集较小时建议）。",
    )
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument(
        "--freeze_mlp",
        action="store_true",
        help="冻结学生模型中全部 MLP 参数，仅训练混合器及其余组件。",
    )
    parser.add_argument(
        "--freeze_replaced_layers",
        action="store_true",
        help="冻结通过 --layers 替换上来的教师层参数。",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="启用 Weights & Biases 训练监控。",
    )
    parser.add_argument("--wandb_project", type=str, default="llamba-distill")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases 运行模式（online/offline/disabled）。",
    )
    parser.add_argument("--max_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="KL(teacher || student) 的损失权重；设为 0 可跳过教师前向与通信。",
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="交叉熵（硬标签）损失权重；设为 0 可仅使用教师软目标。",
    )
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1000000)
    parser.add_argument("--output_dir", type=str, default="distilled_llamba")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument(
        "--second_dataset_name",
        type=str,
        default="teknium/OpenHermes-2.5",
        help="第二阶段微调使用的数据集名称，留空可跳过该阶段。",
    )
    parser.add_argument(
        "--skip_stage2",
        action="store_true",
        help="跳过阶段二（OpenHermes）蒸馏。",
    )
    parser.add_argument("--second_dataset_epochs", type=int, default=4)
    parser.add_argument("--second_dataset_tokens", type=int, default=200_000_000)
    parser.add_argument("--second_dataset_text_field", type=str, default="text")
    parser.add_argument("--second_dataset_split", type=str, default="train")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(args=args)
