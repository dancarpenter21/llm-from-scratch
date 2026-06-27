import torch


def get_default_device() -> torch.device:
    """Return the accelerator device when PyTorch can see one, otherwise CPU.

    PyTorch uses the "cuda" device name for both NVIDIA CUDA and AMD ROCm/HIP
    builds, so this works for the Radeon setup documented in this project too.
    Reuse this in later modules instead of scattering `.cuda()` calls.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a dataloader batch to the selected device.

    Keep Dataset/DataLoader tensors on CPU, then call this immediately after
    reading each batch in a training or inference loop.
    """
    input_ids, target_ids = batch
    return input_ids.to(device, non_blocking=True), target_ids.to(device, non_blocking=True)
