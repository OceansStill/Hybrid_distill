#!/usr/bin/env python3
"""
快速查看 data.PackedTextDataset 在给定配置下的输出效果。

样例：
    python debug_packed_dataset.py \
        --dataset_name HuggingFaceFW/fineweb-edu \
        --dataset_subset CC-MAIN-2014-49 \
        --seq_length 128 \
        --samples 10 \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --local_files_only
"""

import argparse
import itertools
from typing import Optional

import torch
from transformers import AutoTokenizer

from data import PackedTextDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Inspect PackedTextDataset outputs (Hybrid_distill)")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace datasets 名称或本地路径")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_subset", type=str, default=None, help="可选子集名称，例如 CC-MAIN-2014-49")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=10, help="展示的样本数")
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--no_streaming", action="store_true", help="禁用 datasets streaming")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tokenizer_pad_eos", action="store_true", help="若需要，将 tokenizer.pad_token 设为 eos_token")
    return parser


def make_dataset(
    args: argparse.Namespace, tokenizer
) -> PackedTextDataset:
    dataset_kwargs: Optional[dict] = None
    if args.dataset_subset:
        dataset_kwargs = {"name": args.dataset_subset}

    dataset = PackedTextDataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        text_field=args.text_field,
        streaming=not args.no_streaming,
        shuffle_buffer=args.shuffle_buffer,
        revision=args.dataset_revision,
        local_files_only=args.local_files_only,
        world_size=1,
        rank=0,
        dataset_kwargs=dataset_kwargs,
    )
    return dataset


def decode_tokens(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
    )
    if args.tokenizer_pad_eos and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = make_dataset(args, tokenizer)
    iterator = iter(dataset)

    print(
        f"Inspecting dataset='{args.dataset_name}', split='{args.dataset_split}', "
        f"subset='{args.dataset_subset}', seq_length={args.seq_length}"
    )
    print("=" * 80)

    for idx, chunk in enumerate(itertools.islice(iterator, args.samples)):
        chunk = chunk.clone().detach()
        input_tokens = chunk[:-1]
        target_token = chunk[-1].unsqueeze(0)
        print(f"[Sample {idx:02d}]")
        print(f" - token ids: {chunk.tolist()}")
        print(f" - input ids ({len(input_tokens)}): {input_tokens.tolist()}")
        print(f" - target id: {target_token.item()}")
        print(f" - input text: {decode_tokens(tokenizer, input_tokens)}")
        print(f" - target token: {decode_tokens(tokenizer, target_token)}")
        print("-" * 80)


if __name__ == "__main__":
    main()
