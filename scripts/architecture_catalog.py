from __future__ import annotations

from typing import Any

_ARCHITECTURE_ROWS: list[dict[str, Any]] = [
    {
        "id": "cnn",
        "label": "CNN",
        "use_case": "Higher wake-word recall and fast training.",
        "profile": "Higher Recall",
        "layer_size": 32,
        "n_blocks": 1,
        "loss_bias": 0.70,
        "status": "production",
        "status_label": "Streaming validated",
    },
    {
        "id": "gru",
        "label": "GRU",
        "use_case": "Stronger rejection of similar-sounding phrases.",
        "profile": "Lower False Triggers",
        "layer_size": 128,
        "n_blocks": 1,
        "loss_bias": 0.75,
        "status": "production",
        "status_label": "Streaming validated",
    },
]

MODEL_ARCHITECTURES: dict[str, dict[str, Any]] = {row["id"]: dict(row) for row in _ARCHITECTURE_ROWS}

MODEL_ARCHITECTURE_ALIASES: dict[str, str] = {}


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
