from __future__ import annotations

from typing import Any

_ARCHITECTURE_ROWS: list[dict[str, Any]] = [
    {
        "id": "dnn",
        "label": "DNN",
        "use_case": "General use on resource-constrained devices, including MCUs.",
        "profile": "Fastest Training, Low Memory",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "rnn",
        "label": "RNN",
        "use_case": "Baseline experiments or educational runs.",
        "profile": "Better than DNN",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "cnn",
        "label": "CNN",
        "use_case": "Short, sharp, and explosive wake words.",
        "profile": "Efficient Feature Extraction",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "lstm",
        "label": "LSTM",
        "use_case": "Noisy environments or complex, multi-syllable phrases.",
        "profile": "Best-in-Class Noise Robustness",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "gru",
        "label": "GRU",
        "use_case": "A faster, lighter alternative to LSTM with similar high performance.",
        "profile": "Balanced: Speed & Robustness",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "crnn",
        "label": "CRNN",
        "use_case": "Challenging audio requiring both feature and context analysis.",
        "profile": "Hybrid Power: CNN + RNN",
        "status": "experimental",
        "status_label": "Upstream untested",
    },
    {
        "id": "tcn",
        "label": "TCN",
        "use_case": "Modern, high-speed sequential processing.",
        "profile": "Faster than RNN (Parallel)",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "bcresnet",
        "label": "BcResNet",
        "use_case": "Broadcasting-residual network.",
        "profile": "Accuracy Potential",
        "status": "experimental",
        "status_label": "Available, not production-listed",
    },
    {
        "id": "quartznet",
        "label": "QuartzNet",
        "use_case": "Top accuracy with a small footprint on edge devices.",
        "profile": "Parameter-Efficient & Accurate",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "transformer",
        "label": "Transformer",
        "use_case": "Deep contextual understanding via self-attention.",
        "profile": "SOTA Performance & Flexibility",
        "status": "production",
        "status_label": "Production-ready",
    },
    {
        "id": "conformer",
        "label": "Conformer",
        "use_case": "State-of-the-art hybrid for ultimate real-world performance.",
        "profile": "SOTA: Global + Local Features",
        "status": "experimental",
        "status_label": "Upstream untested",
    },
    {
        "id": "e_branchformer",
        "label": "E-Branchformer",
        "use_case": "Bleeding-edge research for potentially the highest accuracy.",
        "profile": "Accuracy Potential",
        "status": "experimental",
        "status_label": "Upstream untested",
    },
]

MODEL_ARCHITECTURES: dict[str, dict[str, Any]] = {row["id"]: dict(row) for row in _ARCHITECTURE_ROWS}

MODEL_ARCHITECTURE_ALIASES = {
    "e-branchformer": "e_branchformer",
    "ebranchformer": "e_branchformer",
    "e_branchformer": "e_branchformer",
    "bc-resnet": "bcresnet",
    "bc_resnet": "bcresnet",
    "quartz-net": "quartznet",
}


def architecture_list() -> list[dict[str, Any]]:
    return [dict(item) for item in MODEL_ARCHITECTURES.values()]


def architecture_metadata(model_type: str) -> dict[str, Any]:
    return dict(MODEL_ARCHITECTURES[normalize_model_type(model_type)])


def normalize_model_type(value: str) -> str:
    raw = (value or "").strip().lower()
    key = MODEL_ARCHITECTURE_ALIASES.get(raw, raw)
    if key not in MODEL_ARCHITECTURES:
        supported = ", ".join(MODEL_ARCHITECTURES)
        raise ValueError(f"Unsupported NanoWakeWord architecture '{value}'. Supported values: {supported}")
    return key
