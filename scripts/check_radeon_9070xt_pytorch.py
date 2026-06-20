#!/usr/bin/env python3
"""Check whether a Radeon RX 9070 XT is visible to PyTorch.

On AMD GPUs, ROCm/HIP-enabled PyTorch still exposes devices through the
``torch.cuda`` namespace. A working Radeon setup should normally show:

  * torch.version.hip is not None
  * torch.cuda.is_available() is True
  * a CUDA device whose name matches the Radeon RX 9070 XT
  * a small tensor operation succeeds on that device
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


DEFAULT_EXPECT_RE = r"9070\s*xt|9070xt|gfx1200|gfx1201"


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    total_memory_gib: float | None
    capability: tuple[int, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a Radeon RX 9070 XT is available through "
            "PyTorch's torch.cuda API."
        )
    )
    parser.add_argument(
        "--expect-regex",
        default=DEFAULT_EXPECT_RE,
        help=(
            "Case-insensitive regex used to identify the expected GPU. "
            f"Default: {DEFAULT_EXPECT_RE!r}"
        ),
    )
    parser.add_argument(
        "--any-gpu",
        action="store_true",
        help="Pass if any PyTorch CUDA/HIP device works, even if no 9070 XT is found.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def load_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        print("PyTorch is not installed in this Python environment.")
        print("With uv, install the project dependencies from pyproject.toml:")
        print("  uv sync")
        return None
    return torch


def collect_devices(torch) -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_memory_gib = None
        if getattr(props, "total_memory", None) is not None:
            total_memory_gib = props.total_memory / (1024**3)

        capability = None
        try:
            capability = torch.cuda.get_device_capability(index)
        except Exception:
            pass

        devices.append(
            DeviceInfo(
                index=index,
                name=torch.cuda.get_device_name(index),
                total_memory_gib=total_memory_gib,
                capability=capability,
            )
        )
    return devices


def run_smoke_test(torch, device_index: int) -> bool:
    device = torch.device(f"cuda:{device_index}")
    try:
        x = torch.randn((1024, 1024), device=device)
        y = x @ x.T
        torch.cuda.synchronize(device)
        checksum = float(y[0, 0].detach().cpu())
    except Exception as exc:
        print(f"Smoke test on cuda:{device_index} failed: {exc}")
        return False

    print(f"Smoke test on cuda:{device_index} passed; checksum={checksum:.6f}")
    return True


def main() -> int:
    args = parse_args()
    torch = load_torch()
    if torch is None:
        return 1

    print_header("PyTorch build")
    print(f"torch.__version__     : {torch.__version__}")
    print(f"torch.version.cuda    : {torch.version.cuda}")
    print(f"torch.version.hip     : {torch.version.hip}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.version.hip is None:
        print(
            "\nThis PyTorch build does not report HIP/ROCm support. "
            "A Radeon GPU will usually require a ROCm-enabled PyTorch wheel."
        )

    if not torch.cuda.is_available():
        print("\nNo PyTorch CUDA/HIP device is available.")
        return 2

    print_header("Detected devices")
    devices = collect_devices(torch)
    if not devices:
        print("torch.cuda.is_available() was true, but device_count() returned 0.")
        return 2

    for device in devices:
        memory = (
            f"{device.total_memory_gib:.2f} GiB"
            if device.total_memory_gib is not None
            else "unknown"
        )
        capability = device.capability if device.capability is not None else "unknown"
        print(f"cuda:{device.index}: {device.name}")
        print(f"  total memory: {memory}")
        print(f"  capability  : {capability}")

    expected = re.compile(args.expect_regex, flags=re.IGNORECASE)
    matching_devices = [device for device in devices if expected.search(device.name)]

    if matching_devices:
        test_device = matching_devices[0]
        print_header("Radeon RX 9070 XT match")
        print(f"Matched cuda:{test_device.index}: {test_device.name}")
    elif args.any_gpu:
        test_device = devices[0]
        print_header("GPU match")
        print(f"No 9070 XT name match; testing first available GPU: cuda:{test_device.index}")
    else:
        print_header("Radeon RX 9070 XT match")
        print(f"No device matched regex: {args.expect_regex!r}")
        print("Use --expect-regex to adjust the match, or --any-gpu to test any GPU.")
        return 3

    if not run_smoke_test(torch, test_device.index):
        return 4

    print("\nResult: PyTorch can use the selected device through torch.cuda.")
    if torch.version.hip is not None:
        print("Backend: ROCm/HIP, exposed through PyTorch's CUDA API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
