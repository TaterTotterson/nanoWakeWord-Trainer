#!/usr/bin/env python3
"""Run the upstream CLI with predictable training and feature devices."""
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


def requested_feature_device() -> str:
    """Select the ONNX provider used to build training feature arrays.

    CPU remains the portable standalone default. NVIDIA trainer images select
    ``auto`` so CUDA is used when available, while Apple and CPU-only hosts can
    continue using the same command without a separate configuration.
    """
    requested = os.environ.get("NWW_FEATURE_DEVICE", "cpu").strip().lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        import onnxruntime as ort

        return "gpu" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if requested in {"cpu", "gpu"}:
        if requested == "gpu":
            import onnxruntime as ort

            if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
                raise SystemExit(
                    "NWW_FEATURE_DEVICE requested GPU, but CUDAExecutionProvider is unavailable"
                )
        return requested
    raise SystemExit(f"Unsupported NWW_FEATURE_DEVICE: {requested}")


def enable_training_device() -> torch.device:
    device = requested_device()
    from nanowakeword.modules.model import Model

    original_init = Model.__init__

    def device_aware_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.device = device

    Model.__init__ = device_aware_init
    return device


def enable_feature_device() -> str:
    """Force upstream AudioFeatures to use the selected ONNX provider."""
    device = requested_feature_device()
    from nanowakeword.data.AudioFeatures import AudioFeatures

    original_init = AudioFeatures.__init__

    def device_aware_init(self, *args, **kwargs):
        positional = list(args)
        if len(positional) >= 6:
            positional[5] = device
            kwargs.pop("device", None)
        else:
            kwargs["device"] = device
        original_init(self, *positional, **kwargs)

    AudioFeatures.__init__ = device_aware_init
    return device


def main() -> int:
    device = enable_training_device()
    print(f"NanoWakeWord training device: {device}", flush=True)
    feature_device = enable_feature_device()
    print(f"NanoWakeWord feature device: {feature_device}", flush=True)
    from nanowakeword.cli import main as upstream_main

    result = upstream_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
