# LLM From Scratch by Me

Me learning LLM from scratch.

## Check Radeon RX 9070 XT with PyTorch

Install the UV-managed environment. The project config points `torch` at
PyTorch's nightly ROCm 7.2 wheel index, which matches AMD's current PyTorch
wheel guidance for ROCm 7.2:

```bash
uv sync
```

Then run:

```bash
uv run scripts/check_radeon_9070xt_pytorch.py
```

PyTorch exposes AMD ROCm/HIP devices through the `torch.cuda` API, so a working
Radeon setup should show `torch.version.hip`, `torch.cuda.is_available() == True`,
and a matching 9070 XT device.
