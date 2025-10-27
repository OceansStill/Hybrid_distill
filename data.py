import os
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset


class PackedTextDataset(IterableDataset):
    """
    将文本数据打包成定长序列的 IterableDataset。

    设计思想
    --------
    - 适合大规模文本数据，尤其是 streaming 模式（无需一次性载入全部样本）。
    - 每次迭代读取一条文本，分词后 append 到缓存 buffer。
    - 当 buffer 长度 ≥ seq_length + 1 时，截断一段返回（含下一个 token 作为标签）。

    参数
    ----
    dataset_name: str
        HuggingFace datasets 名称或本地路径。
    split: str
        数据集切分，如 "train"。
    tokenizer: transformers.PreTrainedTokenizerBase
        分词器（必须提供 `__call__` 返回 input_ids）。
    seq_length: int
        序列长度（实际输出大小为 seq_length + 1，用于自回归预测）。
    text_field: str
        数据集中对应文本的字段名。
    streaming: bool
        是否启用流式读取。True 时可在大数据集上节省内存。
    shuffle_buffer: int
        流式 shuffle 时的缓冲区大小，越大随机性越好、内存越高。
    seed: int
        shuffle 随机种子，保证多进程间的可复现性。
    revision: Optional[str]
        指定数据集的 revision（适用于 datasets Hub）。
    world_size: int
        总进程数，用于多卡场景分片。
    rank: int
        当前进程编号（0-based）。
    dataset_kwargs: Optional[Dict]
        传递给 `load_dataset` 的额外参数，例如子集名称。
    local_files_only: bool
        是否仅使用本地缓存（无网络环境下需设置为 True）。
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        seq_length: int,
        text_field: str = "text",
        streaming: bool = True,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        revision: Optional[str] = None,
        world_size: int = 1,
        rank: int = 0,
        dataset_kwargs: Optional[Dict] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field
        dataset_kwargs = dataset_kwargs or {}

        if local_files_only:
            # datasets 库通过该环境变量强制离线模式
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            revision=revision,
            **dataset_kwargs,
        )

        # 多进程场景：通过 shard 保证每个 rank 读取 disjoint 切片
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)

        # Streaming 模式下可调用 shuffle，内部将维护一个缓冲区
        if streaming:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

        self.dataset = dataset
        self._column_names: Optional[List[str]] = None

    def __iter__(self) -> Iterable[torch.Tensor]:
        """
        返回一个迭代器，持续输出形状为 [seq_length + 1] 的 long tensor。
        该 tensor 包含输入 + 下一个 token，用于 causal LM 训练。
        """
        buffer: List[int] = []
        for sample in self.dataset:
            if self._column_names is None:
                self._column_names = list(sample.keys())

            text = sample.get(self.text_field, None)
            if text is None:
                raise KeyError(
                    f"字段 '{self.text_field}' 缺失，数据集可选列：{self._column_names}"
                )

            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)

            # 当缓存中可切出 seq_length+1 个 token 时，产出一次样本
            # 注意：每次切片后从 buffer 移除对应片段，确保顺序衔接
            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length + 1 :]
                yield torch.tensor(chunk, dtype=torch.long)


def build_dataloader(
    dataset_name: str,
    split: str,
    tokenizer,
    seq_length: int,
    text_field: str,
    batch_size: int,
    streaming: bool,
    shuffle_buffer: int,
    revision: Optional[str],
    local_files_only: bool,
    world_size: int,
    rank: int,
    dataset_kwargs: Optional[Dict] = None,
) -> DataLoader:
    """
    根据训练配置构建 DataLoader。

    返回 DataLoader 特性
    --------------------
    - dataset: PackedTextDataset（IterableDataset），无需 `sampler`。
    - batch_size: 按输入 batch_size，`collate_fn` 会 stack 成 [B, seq_length+1]。
    - pin_memory=True: 提升 GPU 传输性能。

    说明
    ----
    IterableDataset 与 DDP 配合时，需确保每个进程拿到互不重叠的数据。
    通过在 PackedTextDataset 初始化时传入 `world_size`、`rank` 控制。
    """
    dataset = PackedTextDataset(
        dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer,
        seq_length=seq_length,
        text_field=text_field,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        revision=revision,
        local_files_only=local_files_only,
        world_size=world_size,
        rank=rank,
        dataset_kwargs=dataset_kwargs,
    )

    def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
        """
        将若干 [seq_length+1] 的 tensor 堆叠为 [B, seq_length+1]。
        由于 PackedTextDataset 已保证每个样本长度一致，可直接 stack。
        """
        return torch.stack(batch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,   # 新增：确保各 rank batch 尺寸一致
    )
