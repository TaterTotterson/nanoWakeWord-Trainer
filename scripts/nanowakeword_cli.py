#!/usr/bin/env python3
"""Run the upstream CLI with Apple Metal training support when available."""
from __future__ import annotations

import os

import torch


def requested_device() -> torch.device:
    requested = os.environ.get("NWW_TORCH_DEVICE", "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise SystemExit("NWW_TORCH_DEVICE requested CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("NWW_TORCH_DEVICE requested MPS, but Apple Metal is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported NWW_TORCH_DEVICE: {requested}")


def enable_training_device() -> torch.device:
    device = requested_device()
    from nanowakeword.modules.model import Model

    original_init = Model.__init__

    def device_aware_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.device = device

    Model.__init__ = device_aware_init
    return device


def main() -> int:
    device = enable_training_device()
    print(f"NanoWakeWord training device: {device}", flush=True)
    from nanowakeword.cli import main as upstream_main

    result = upstream_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
