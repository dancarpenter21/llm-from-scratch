import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader
from tiktoken import Encoding

class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
):
    tokenizer: Encoding = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # Keep the dataset on CPU. Move each batch to the GPU in the train loop with
    # move_batch_to_device(...). pin_memory can speed CPU-to-GPU copies.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader
