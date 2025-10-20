import os
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset


class PackedTextDataset(IterableDataset):
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
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field
        dataset_kwargs = dataset_kwargs or {}
        if local_files_only:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            revision=revision,
            **dataset_kwargs,
        )
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        if streaming:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
        self.dataset = dataset
        self._column_names = None

    def __iter__(self):
        buffer: List[int] = []
        for sample in self.dataset:
            if self._column_names is None:
                self._column_names = list(sample.keys())
            text = sample.get(self.text_field, None)
            if text is None:
                raise KeyError(
                    f"Expected field '{self.text_field}' in dataset sample. Available columns: {self._column_names}"
                )
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)
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
):
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

    def collate_fn(batch: List[torch.Tensor]):
        return torch.stack(batch)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)
