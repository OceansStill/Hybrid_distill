import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="流式下载 HuggingFaceFW/fineweb-edu 的指定子集，并保存为 JSONL/GZ 分片。"
    )
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="CC-MAIN-2014-49")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=50_000,
        help="每个输出文件包含的样本数。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="限制总下载样本数（默认无限制）。",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "gzip"],
        default="gzip",
        help="输出文件压缩方式。",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="样本中存放文本的字段名称。",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="禁用 Hugging Face streaming，改为完整下载到缓存后再迭代。",
    )
    return parser.parse_args()


def open_output_file(base_path: Path, shard_idx: int, compression: str):
    filename = base_path / f"shard-{shard_idx:05d}.jsonl"
    if compression == "gzip":
        filename = filename.with_suffix(filename.suffix + ".gz")
        return gzip.open(filename, "wt", encoding="utf-8")
    return open(filename, "w", encoding="utf-8")


def iter_samples(dataset, max_samples: Optional[int] = None) -> Iterable[dict]:
    count = 0
    for sample in dataset:
        yield sample
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=not args.no_streaming,
    )

    shard_idx = 0
    samples_in_shard = 0
    writer = open_output_file(output_dir, shard_idx, args.compression)
    progress = tqdm(
        iter_samples(dataset, args.max_samples),
        desc="Downloading",
        unit="samples",
    )

    try:
        for sample in progress:
            text = sample.get(args.text_field, None)
            if text is None:
                continue
            record = {"text": text}
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples_in_shard += 1

            if samples_in_shard >= args.samples_per_file:
                writer.close()
                shard_idx += 1
                samples_in_shard = 0
                writer = open_output_file(output_dir, shard_idx, args.compression)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
