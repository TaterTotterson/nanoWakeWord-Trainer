#!/usr/bin/env python3
"""Evaluate a NanoWakeWord ONNX model through the real streaming feature path.

Training-set metrics operate on already-computed feature arrays.  That can hide
audio placement and streaming/preprocessing problems, so this evaluator starts
from WAV files and feeds 80 ms PCM chunks exactly as a satellite does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

from nanowakeword.data.AudioFeatures import AudioFeatures

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
MODEL_FRAMES = 16
WARMUP_SAMPLES = SAMPLE_RATE * 2
TRAILING_SAMPLES = SAMPLE_RATE
CALIBRATION_THRESHOLDS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)


def consecutive_activation_score(scores: Iterable[float], patience: int) -> float:
    values = list(scores)
    patience = max(1, int(patience))
    if len(values) < patience:
        return 0.0
    return max(min(values[index : index + patience]) for index in range(len(values) - patience + 1))


def wav_paths(directories: Iterable[Path]) -> list[Path]:
    paths: dict[str, Path] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*.wav"):
            if path.is_file():
                paths[str(path.resolve())] = path.resolve()
    return sorted(paths.values(), key=lambda item: str(item))


def deterministic_sample(paths: list[Path], limit: int, partition: int = 0) -> list[Path]:
    """Choose a stable, well-distributed partition without retry-set leakage."""
    if limit <= 0 or len(paths) <= limit:
        return paths
    def stable_key(path: Path) -> bytes:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        # Do not let /data versus a host mount point change the holdout set.
        # File name and size are stable identifiers for generated corpora.
        return hashlib.sha256(f"{path.name}\0{size}".encode("utf-8")).digest()

    shuffled = sorted(paths, key=stable_key)
    start = (max(0, int(partition)) * limit) % len(shuffled)
    rotated = shuffled[start:] + shuffled[:start]
    return rotated[:limit]


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path} is not 16-bit PCM")
    audio = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        audio = audio.reshape(-1, channels).astype(np.int32).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio.astype(np.float32), SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.clip(np.rint(audio), -32768, 32767).astype(np.int16)


class StreamingScorer:
    def __init__(self, model_path: Path) -> None:
        self.features = AudioFeatures(device="cpu")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = list(model_input.shape)
        if len(shape) != 3 or shape[-2:] != [MODEL_FRAMES, 96]:
            raise ValueError(
                f"Expected an embedding model with input [batch,{MODEL_FRAMES},96]; got {shape}"
            )

    def scores(self, path: Path) -> list[float]:
        audio = read_pcm16(path)
        stream = np.concatenate(
            (
                np.zeros(WARMUP_SAMPLES, dtype=np.int16),
                audio,
                np.zeros(TRAILING_SAMPLES, dtype=np.int16),
            )
        )
        self.features.reset()
        scores: list[float] = []
        for offset in range(0, len(stream), CHUNK_SAMPLES):
            chunk = stream[offset : offset + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            self.features(chunk)
            if offset + CHUNK_SAMPLES <= WARMUP_SAMPLES:
                continue
            feature_window = self.features.get_features(MODEL_FRAMES)
            if feature_window.shape != (1, MODEL_FRAMES, 96):
                continue
            score = float(self.session.run(None, {self.input_name: feature_window})[0].reshape(-1)[0])
            scores.append(score)
        return scores

    def score(self, path: Path, patience: int = 1) -> float:
        """Return the strongest consecutive-hit score for the runtime policy.

        Tater requires ``patience`` consecutive frames above its threshold.  The
        minimum score in each consecutive window is therefore the meaningful
        threshold-independent value; taking the maximum of those windows lets
        the normal distribution/gate code reproduce that policy exactly.
        """
        scores = self.scores(path)
        return consecutive_activation_score(scores, patience)


def distribution(scores: list[float], threshold: float) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "at_or_above_threshold": 0, "rate": None}
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": len(scores),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
        "at_or_above_threshold": int(np.sum(values >= threshold)),
        "rate": float(np.mean(values >= threshold)),
    }


def _score_sequences(
    model_path: Path,
    positive_dirs: Iterable[Path],
    negative_dirs: Iterable[Path],
    max_clips_per_class: int,
    sample_partition: int,
) -> tuple[list[tuple[Path, list[float]]], list[tuple[Path, list[float]]]]:
    positive_paths = deterministic_sample(
        wav_paths(positive_dirs), max_clips_per_class, sample_partition
    )
    negative_paths = deterministic_sample(
        wav_paths(negative_dirs), max_clips_per_class, sample_partition
    )
    if not positive_paths:
        raise ValueError("No positive WAV files were available for streaming evaluation")
    if not negative_paths:
        raise ValueError("No negative WAV files were available for streaming evaluation")

    scorer = StreamingScorer(model_path)
    return (
        [(path, scorer.scores(path)) for path in positive_paths],
        [(path, scorer.scores(path)) for path in negative_paths],
    )


def _evaluate_sequences(
    model_path: Path,
    positive_sequences: list[tuple[Path, list[float]]],
    negative_sequences: list[tuple[Path, list[float]]],
    *,
    threshold: float,
    patience: int,
    sample_partition: int,
    minimum_recall: float,
    maximum_negative_rate: float,
) -> dict[str, Any]:
    positive_rows = [
        (path, consecutive_activation_score(scores, patience))
        for path, scores in positive_sequences
    ]
    negative_rows = [
        (path, consecutive_activation_score(scores, patience))
        for path, scores in negative_sequences
    ]
    positive_scores = [score for _, score in positive_rows]
    negative_scores = [score for _, score in negative_rows]
    positives = distribution(positive_scores, threshold)
    negatives = distribution(negative_scores, threshold)
    recall = float(positives["rate"])
    negative_rate = float(negatives["rate"])
    passed = recall >= minimum_recall and negative_rate <= maximum_negative_rate
    return {
        "model": str(model_path.resolve()),
        "streaming_contract": {
            "sample_rate": SAMPLE_RATE,
            "chunk_samples": CHUNK_SAMPLES,
            "feature_frames": MODEL_FRAMES,
            "threshold": threshold,
            "patience": patience,
        },
        "criteria": {
            "minimum_recall": minimum_recall,
            "maximum_negative_rate": maximum_negative_rate,
            "sample_partition": sample_partition,
        },
        "positive": positives,
        "negative": negatives,
        "examples": {
            "lowest_positive_scores": [
                {"path": str(path), "score": score}
                for path, score in sorted(positive_rows, key=lambda row: row[1])[:10]
            ],
            "highest_negative_scores": [
                {"path": str(path), "score": score}
                for path, score in sorted(negative_rows, key=lambda row: row[1], reverse=True)[:10]
            ],
        },
        "misclassified": {
            "missed_positive_paths": [
                str(path) for path, score in positive_rows if score < threshold
            ],
            "triggered_negative_paths": [
                str(path) for path, score in negative_rows if score >= threshold
            ],
        },
        "passed": passed,
        "failures": [
            message
            for condition, message in (
                (recall < minimum_recall, f"recall {recall:.3f} is below {minimum_recall:.3f}"),
                (
                    negative_rate > maximum_negative_rate,
                    f"negative trigger rate {negative_rate:.3f} is above {maximum_negative_rate:.3f}",
                ),
            )
            if condition
        ],
    }


def select_streaming_policy(
    candidates: list[dict[str, Any]],
    *,
    minimum_recall: float,
    maximum_negative_rate: float,
    target_recall: float = 0.95,
    target_negative_rate: float = 0.05,
) -> tuple[dict[str, Any], str]:
    """Select a stable low-latency policy without tuning against one metric alone."""
    valid = [item for item in candidates if item["passed"]]
    preferred_recall = max(minimum_recall, target_recall)
    preferred_negative_rate = min(maximum_negative_rate, target_negative_rate)
    preferred = [
        item
        for item in valid
        if item["recall"] >= preferred_recall
        and item["negative_rate"] <= preferred_negative_rate
    ]
    if preferred:
        # Prefer the least added latency, then the strictest threshold that
        # still keeps healthy recall and a false-trigger safety margin.
        return min(
            preferred,
            key=lambda item: (
                item["patience"],
                -item["threshold"],
                item["negative_rate"],
                -item["recall"],
            ),
        ), "preferred_targets"
    if valid:
        return max(
            valid,
            key=lambda item: (
                item["recall"] - item["negative_rate"] - 0.02 * item["patience"],
                -item["patience"],
                item["threshold"],
            ),
        ), "required_targets"

    def violation(item: dict[str, Any]) -> tuple[float, float, int, float]:
        amount = max(0.0, minimum_recall - item["recall"]) + max(
            0.0, item["negative_rate"] - maximum_negative_rate
        )
        return amount, -item["recall"], item["patience"], -item["threshold"]

    return min(candidates, key=violation), "closest_failed_policy"


def calibrate_model(
    model_path: Path,
    positive_dirs: Iterable[Path],
    negative_dirs: Iterable[Path],
    *,
    minimum_threshold: float = 0.90,
    minimum_patience: int = 2,
    maximum_patience: int = 5,
    max_clips_per_class: int = 200,
    sample_partition: int = 0,
    minimum_recall: float = 0.90,
    maximum_negative_rate: float = 0.05,
    target_recall: float = 0.95,
    target_negative_rate: float = 0.05,
) -> dict[str, Any]:
    """Calibrate and evaluate one deployment policy from the real audio stream."""
    positive_sequences, negative_sequences = _score_sequences(
        model_path,
        positive_dirs,
        negative_dirs,
        max_clips_per_class,
        sample_partition,
    )
    thresholds = sorted(
        {
            round(float(value), 6)
            for value in (*CALIBRATION_THRESHOLDS, minimum_threshold)
            if value >= minimum_threshold
        }
    )
    patience_values = range(max(1, minimum_patience), max(minimum_patience, maximum_patience) + 1)
    reports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for patience in patience_values:
        for threshold in thresholds:
            report = _evaluate_sequences(
                model_path,
                positive_sequences,
                negative_sequences,
                threshold=threshold,
                patience=patience,
                sample_partition=sample_partition,
                minimum_recall=minimum_recall,
                maximum_negative_rate=maximum_negative_rate,
            )
            reports.append(report)
            candidates.append(
                {
                    "threshold": threshold,
                    "patience": patience,
                    "recall": float(report["positive"]["rate"]),
                    "negative_rate": float(report["negative"]["rate"]),
                    "passed": bool(report["passed"]),
                }
            )

    selected, reason = select_streaming_policy(
        candidates,
        minimum_recall=minimum_recall,
        maximum_negative_rate=maximum_negative_rate,
        target_recall=target_recall,
        target_negative_rate=target_negative_rate,
    )
    selected_index = candidates.index(selected)
    report = reports[selected_index]
    report["calibration"] = {
        "selected_reason": reason,
        "target_recall": max(minimum_recall, target_recall),
        "target_negative_rate": min(maximum_negative_rate, target_negative_rate),
        "candidates": candidates,
    }
    return report


def evaluate_model(
    model_path: Path,
    positive_dirs: Iterable[Path],
    negative_dirs: Iterable[Path],
    *,
    threshold: float = 0.90,
    patience: int = 2,
    max_clips_per_class: int = 200,
    sample_partition: int = 0,
    minimum_recall: float = 0.90,
    maximum_negative_rate: float = 0.05,
) -> dict[str, Any]:
    positive_sequences, negative_sequences = _score_sequences(
        model_path,
        positive_dirs,
        negative_dirs,
        max_clips_per_class,
        sample_partition,
    )
    return _evaluate_sequences(
        model_path,
        positive_sequences,
        negative_sequences,
        threshold=threshold,
        patience=patience,
        sample_partition=sample_partition,
        minimum_recall=minimum_recall,
        maximum_negative_rate=maximum_negative_rate,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--positive-dir", action="append", type=Path, required=True)
    parser.add_argument("--negative-dir", action="append", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-clips-per-class", type=int, default=200)
    parser.add_argument("--sample-partition", type=int, default=0)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--maximum-negative-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_model(
        args.model,
        args.positive_dir,
        args.negative_dir,
        threshold=args.threshold,
        patience=args.patience,
        max_clips_per_class=args.max_clips_per_class,
        sample_partition=args.sample_partition,
        minimum_recall=args.minimum_recall,
        maximum_negative_rate=args.maximum_negative_rate,
    )
    payload = json.dumps(report, indent=2)
    print(payload, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
