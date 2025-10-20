import argparse
import os
import re
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "下载整个 HuggingFace 数据集仓库(默认: HuggingFaceFW/fineweb-edu)到目标目录。"
            "使用 huggingface_hub.snapshot_download，支持断点续传与并发。"
        )
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Hugging Face 数据集仓库 ID (repo_type=dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/yijia/fineweb-edu",
        help="下载输出目录(将创建并填充为该仓库的快照)",
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default="/data/yijia/.cache/huggingface",
        help=(
            "Hugging Face 本地缓存目录(HF_HOME)。建议放在数据盘，以避免缓存占用系统盘。"
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="并发下载线程数(过大可能导致带宽或连接压力)",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="强制重新下载并覆盖本地缓存(默认断点续传)",
    )
    parser.add_argument(
        "--allow_patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "仅下载匹配的路径模式(如 *.parquet)。不指定则下载仓库内的全部文件。"
        ),
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        nargs="*",
        default=None,
        help="忽略匹配的路径模式(如 README* LICENSE*)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅统计文件与总大小，不实际下载。",
    )
    parser.add_argument(
        "--first_n_subsets",
        type=int,
        default=None,
        help="只下载前 N 个子集(按 data/<subset>/ 字典序)",
    )
    parser.add_argument(
        "--only_parquet",
        action="store_true",
        help="仅下载 parquet 分片(忽略非数据文件)",
    )
    parser.add_argument(
        "--list_subsets",
        action="store_true",
        help="仅列出可用子集并退出",
    )
    return parser.parse_args()


def bytes_to_readable(num: int) -> str:
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num < step:
            return f"{num:.2f} {unit}"
        num /= step
    return f"{num:.2f} EB"


def list_repo_files(repo_id: str) -> List[dict]:
    api = HfApi()
    # files_metadata=True 返回包含 size 的信息
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    # info.siblings 是文件列表，带 size (LFS) 等元数据
    files = []
    for f in info.siblings:
        meta = {"rfilename": f.rfilename}
        if hasattr(f, "size") and f.size is not None:
            meta["size"] = int(f.size)
        else:
            meta["size"] = 0
        files.append(meta)
    return files


def natural_key(s: str):
    # 使 "CC-MAIN-2014-10" 正确排序在 "CC-MAIN-2014-9" 之后
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_subsets_from_files(files: List[dict]) -> List[str]:
    subsets = set()
    for f in files:
        p = f.get("rfilename", "")
        if p.startswith("data/"):
            parts = p.split("/")
            if len(parts) >= 3:
                subsets.add(parts[1])
    return sorted(subsets, key=natural_key)


def main():
    args = parse_args()

    # 将 HF_HOME 指向数据盘，避免缓存写到系统盘
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预统计文件与体积，便于用户确认
    files = list_repo_files(args.repo_id)
    total_size = sum(f.get("size", 0) for f in files)
    total_files = len(files)
    print(
        f"[INFO] Repo: {args.repo_id} | Files: {total_files} | Size: {bytes_to_readable(total_size)}"
    )

    # 子集列表与选择
    subsets = list_subsets_from_files(files)
    print(f"[INFO] Found {len(subsets)} subsets under 'data/' (showing first 20):")
    print("       ", ", ".join(subsets[:20]) + (" ..." if len(subsets) > 20 else ""))

    allow_patterns = args.allow_patterns
    if args.first_n_subsets is not None:
        selected = subsets[: args.first_n_subsets]
        print(f"[INFO] Selecting first {args.first_n_subsets} subsets:")
        print("       ", ", ".join(selected))
        if args.only_parquet:
            computed = [f"data/{s}/*.parquet" for s in selected]
        else:
            computed = [f"data/{s}/*" for s in selected]
        if args.allow_patterns:
            print("[WARN] 已提供 --allow_patterns，与 --first_n_subsets 冲突；将按子集选择覆盖 allow_patterns")
        allow_patterns = computed

    if allow_patterns:
        print(f"[INFO] allow_patterns: {allow_patterns}")
    if args.ignore_patterns:
        print(f"[INFO] ignore_patterns: {args.ignore_patterns}")
    print(f"[INFO] HF_HOME: {os.environ.get('HF_HOME', '(default)')}")
    print(f"[INFO] Output dir: {str(out_dir)}")

    if args.dry_run:
        # 估算按 allow_patterns 选取的文件体积
        if allow_patterns:
            import fnmatch

            def matched(p: str) -> bool:
                return any(fnmatch.fnmatch(p, pat) for pat in allow_patterns)

            sel = [f for f in files if matched(f["rfilename"])]
            sel_size = sum(f.get("size", 0) for f in sel)
            print(f"[DRY-RUN] 预计下载 {len(sel)} 个文件, 总计 {bytes_to_readable(sel_size)}")
        print("[DRY-RUN] 仅统计信息，未执行下载。")
        return

    # 执行镜像下载
    # huggingface_hub 新流程直接写入 local_dir；HF_HOME 用于缓存。
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=args.ignore_patterns,
        force_download=args.force_download,
        max_workers=args.max_workers,
    )

    print(f"[DONE] Snapshot at: {snapshot_path}")


if __name__ == "__main__":
    main()
