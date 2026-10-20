#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from architecture_catalog import architecture_metadata, normalize_model_type
from evaluate_nanowakeword import calibrate_model, evaluate_model

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_SUFFIXES = {".onnx", ".pt", ".pth"}
DEFAULT_POSITIVE_SAMPLES = 2500
DEFAULT_NEGATIVE_SAMPLES = 5000
DEFAULT_VALIDATION_SAMPLES = 2000
DEFAULT_HARD_NEGATIVE_SAMPLES = 3000
DEFAULT_STEPS = 50000
DEFAULT_NUM_WORKERS = 0
DEFAULT_CLIP_LENGTH_SAMPLES = 32000
DEFAULT_QUALITY_MAX_CLIPS = 200
PERSONAL_POSITIVE_MAX_SAMPLES = 25600
PERSONAL_AUGMENTATION_ROUNDS = 50
NEGATIVE_GENERATION_RECIPE_VERSION = 2
FEATURE_GENERATION_RECIPE_VERSION = 2

FEATURE_BANKS = {
    "AE29H_float32.npy": ("ae", 100),
    "RACON_11h_v1.npy": ("b", 90),
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": ("oww", 1000),
}


def log(message: str) -> None:
    print(message, flush=True)


def safe_name(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "wakeword"


def existing_wavs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.glob("*.wav") if item.is_file())


def missing_generated_samples(path: Path, requested: int) -> int:
    """Return how many clips are needed without rebuilding an existing TTS set."""
    return max(0, int(requested) - len(existing_wavs(path)))


def prepare_generation_cache(
    directory: Path,
    fingerprint: dict[str, Any],
    *,
    adopt_legacy: bool,
) -> bool:
    """Validate a generated-audio cache and return whether it was invalidated.

    Fixed-phrase positive TTS is intentionally adopted when upgrading from an
    older trainer so an app update does not rebuild thousands of clips for the
    same wake word.  Negative recipes are different: silently reusing clips
    made by an obsolete recipe defeats the very false-positive fixes that the
    new recipe provides, so legacy/mismatched negative caches are rebuilt once.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / ".tater-generation.json"
    paths = existing_wavs(directory)
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None

    invalidated = False
    if paths and cached is None and not adopt_legacy:
        invalidated = True
    elif paths and cached is not None and cached != fingerprint:
        invalidated = True

    if invalidated:
        for path in paths:
            path.unlink()
        log(f"Rebuilding stale generated audio cache: {directory}")

    manifest_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    return invalidated


def invalidate_feature_file(model_dir: Path, filename: str, invalidated: bool) -> None:
    if not invalidated:
        return
    path = model_dir / "features" / filename
    if path.exists() or path.is_symlink():
        path.unlink()


def _write_pcm16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.asarray(audio, dtype="<i2").tobytes())


def speech_centered_crop(audio: np.ndarray, max_samples: int = PERSONAL_POSITIVE_MAX_SAMPLES) -> np.ndarray:
    """Return the most speech-dense window from a long positive recording.

    Upstream NanoWakeWord randomly crops clips longer than its two-second model
    window.  For a wake-word capture that can turn room tone into a positive
    example.  A 1.6-second foreground also leaves room for augmentation to vary
    its position inside the two-second training window.
    """
    audio = np.asarray(audio, dtype=np.int16)
    if audio.size <= max_samples:
        return audio

    frame_samples = 320  # 20 ms at 16 kHz
    frame_count = audio.size // frame_samples
    framed = audio[: frame_count * frame_samples].reshape(frame_count, frame_samples).astype(np.float64)
    rms = np.sqrt(np.mean(framed * framed, axis=1))
    window_frames = max(1, min(frame_count, max_samples // frame_samples))
    noise_floor = float(np.quantile(rms, 0.20))
    speech_level = float(np.quantile(rms, 0.95))
    threshold = noise_floor + 0.30 * max(0.0, speech_level - noise_floor)
    activity = np.maximum(rms - threshold, 0.0)
    if not np.any(activity):
        activity = rms
    scores = np.convolve(activity, np.ones(window_frames, dtype=np.float64), mode="valid")
    start_frame = int(np.argmax(scores))
    start_sample = start_frame * frame_samples
    return audio[start_sample : start_sample + max_samples]


def prepare_positive_training_dir(source_dir: Path, output_dir: Path) -> Path:
    """Stage long personal positives as deterministic speech-centered clips."""
    paths = existing_wavs(source_dir)
    if not paths:
        return source_dir

    # Short, purpose-recorded clips are already safe for upstream augmentation.
    needs_staging = False
    for path in paths:
        try:
            with wave.open(str(path), "rb") as handle:
                duration = handle.getnframes() / max(1, handle.getframerate())
            if duration > PERSONAL_POSITIVE_MAX_SAMPLES / 16000:
                needs_staging = True
                break
        except (OSError, wave.Error):
            # Let NanoWakeWord's normal verifier report unsupported/corrupt audio.
            return source_dir
    if not needs_staging:
        return source_dir

    staged_dir = output_dir / "_trainer_assets" / "positive_clips"
    manifest_path = staged_dir / "manifest.json"
    fingerprint = {
        "max_samples": PERSONAL_POSITIVE_MAX_SAMPLES,
        "sources": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ],
    }
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if cached == fingerprint and len(existing_wavs(staged_dir)) == len(paths):
        return staged_dir

    staged_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{index:04d}_{path.stem}.wav" for index, path in enumerate(paths)}
    for stale in existing_wavs(staged_dir):
        if stale.name not in expected_names:
            stale.unlink()

    # Importing from the evaluator keeps WAV decoding/resampling identical to
    # the streaming quality gate.
    from evaluate_nanowakeword import read_pcm16

    for index, path in enumerate(paths):
        audio = read_pcm16(path)
        cropped = speech_centered_crop(audio)
        destination = staged_dir / f"{index:04d}_{path.stem}.wav"
        _write_pcm16(destination, cropped)
        if audio.size > cropped.size:
            log(
                f"Speech-centered positive clip: {path.name} "
                f"({audio.size / 16000:.2f}s -> {cropped.size / 16000:.2f}s)"
            )
    manifest_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    return staged_dir


def feature_bank_dir(output_dir: Path) -> Path:
    return Path(os.environ.get("NWW_FEATURE_BANK_DIR", str(output_dir.parent / "feature_banks"))).resolve()


def ensure_alignment_background(output_dir: Path) -> Path:
    """Create quiet, deterministic room tone used to randomize phrase placement.

    NanoWakeWord 2.x/3.x places a short foreground clip at sample zero whenever
    no background file is configured.  That teaches a fixed position which does
    not match the rolling window used during inference.  A non-silent background
    makes upstream augmentation choose a new legal foreground offset each round.
    """
    directory = output_dir / "_trainer_assets" / "alignment_background"
    path = directory / "quiet_room_tone.wav"
    if path.exists():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)
    sample_count = 10 * 16000
    white = rng.normal(0.0, 1.0, sample_count + 63)
    filtered = np.convolve(white, np.ones(64, dtype=np.float64) / 8.0, mode="valid")[:sample_count]
    filtered /= max(float(np.std(filtered)), 1e-9)
    pcm = np.clip(np.rint(filtered * 190.0), -900, 900).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())
    return directory


def add_feature_bank_negatives(config: dict[str, Any], bank_dir: Path) -> None:
    for filename, (key, batch_weight) in FEATURE_BANKS.items():
        path = (bank_dir / filename).resolve()
        if not path.exists():
            continue
        config.setdefault("feature_manifest", {}).setdefault("negatives", {})[key] = str(path)
        config.setdefault("batch_composition", {})[key] = batch_weight


def nanowakeword_cmd() -> list[str]:
    wrapper = ROOT_DIR / "scripts" / "nanowakeword_cli.py"
    return [sys.executable, str(wrapper)]


def add_feature_task(
    config: dict[str, Any],
    *,
    key: str,
    label: str,
    source_dir: Path,
    feature_file: str,
    split: str,
    require_existing: bool = True,
    augmentation_rounds: int = 10,
    batch_weight: int | None = None,
    use_rir: bool = False,
) -> None:
    if require_existing and not existing_wavs(source_dir):
        return
    feature_path = Path(config["output_dir"]) / config["model_name"] / "features" / feature_file
    has_backgrounds = bool(config.get("background_paths"))
    has_rirs = bool(config.get("rir_paths")) and use_rir
    cache_path = feature_path.with_name(f"{feature_path.name}.tater.json")
    if require_existing:
        source_fingerprint: list[dict[str, Any]] = [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in existing_wavs(source_dir)
        ]
    else:
        # Generated TTS has its own cache fingerprint. Avoid hashing thousands
        # of WAVs here; make_config invalidates this feature whenever clips are
        # added or its generation recipe changes.
        source_fingerprint = [{"generated_directory": str(source_dir)}]
    supporting_audio: list[dict[str, Any]] = []
    for directory_value in [*(config.get("background_paths") or []), *(config.get("rir_paths") or [])]:
        supporting_audio.extend(
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in existing_wavs(Path(directory_value))
        )
    fingerprint = {
        "recipe_version": FEATURE_GENERATION_RECIPE_VERSION,
        "source": source_fingerprint,
        "supporting_audio": supporting_audio,
        "augmentation_rounds": augmentation_rounds,
        "use_background_noise": has_backgrounds,
        "use_rir": has_rirs,
        "audio_processing": config.get("audio_processing") or {},
        "augmentation_settings": config.get("augmentation_settings") or {},
    }
    try:
        cached_fingerprint = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached_fingerprint = None
    if (feature_path.exists() or feature_path.is_symlink()) and cached_fingerprint != fingerprint:
        feature_path.unlink()
        log(f"Rebuilding stale feature cache: {feature_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    config.setdefault("feature_generation_manifest", {})[label] = {
        "input_audio_dirs": [str(source_dir)],
        "output_filename": feature_file,
        "use_background_noise": has_backgrounds,
        "use_rir": has_rirs,
        "augmentation_rounds": augmentation_rounds,
    }
    manifest_keys = {
        "target": "targets",
        "negative": "negatives",
        "target_val": "targets_val",
        "negative_val": "negatives_val",
    }
    manifest_key = manifest_keys.get(split)
    if not manifest_key:
        raise ValueError(f"Unknown feature split: {split}")
    config.setdefault("feature_manifest", {}).setdefault(manifest_key, {})[key] = str(feature_path)
    if split in {"target", "negative"}:
        if batch_weight is None:
            batch_weight = 100
        if require_existing:
            available_rows = len(existing_wavs(source_dir)) * augmentation_rounds
            batch_weight = min(batch_weight, available_rows)
        if batch_weight <= 0:
            raise ValueError(f"No training rows are available for feature source {label}")
        config.setdefault("batch_composition", {})[key] = batch_weight


def negative_text_source(args: argparse.Namespace, samples: int) -> dict[str, Any]:
    negative_phrases = [item for item in args.custom_negative_phrase if item.strip()]
    if negative_phrases:
        return {
            "type": "from_list",
            "phrases": negative_phrases,
            "repeat_each": max(1, samples // max(1, len(negative_phrases))),
        }
    return {
        "type": "auto_adversarial",
        "base_phrase": args.phrase,
        # Upstream interprets these as probabilities, not booleans.  Passing
        # True made both values 1.0 and produced a narrow, repetitive negative
        # set.  Keep some partial/original-word near misses while retaining a
        # much broader set of fully substituted confusable phrases.
        "include_input_words": 0.35,
        "include_partial_phrase": 0.35,
        "multi_word_prob": 0.5,
        "max_multi_word_len": 3,
    }


def phoneme_negative_text_source(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "phoneme_adversarial",
        "base_phrase": args.phrase,
        "min_distance": 0.3,
    }


def make_config(args: argparse.Namespace, model_name: str, output_dir: Path) -> dict[str, Any]:
    architecture = architecture_metadata(args.model_type)
    layer_size = args.layer_size if args.layer_size > 0 else int(architecture.get("layer_size", 128))
    positive_source_dir = Path(args.positive_dir).resolve()
    positive_dir = prepare_positive_training_dir(positive_source_dir, output_dir)
    shared_positive_dir = (
        Path(args.positive_tts_dir).resolve() if args.positive_tts_dir else None
    )
    if shared_positive_dir is not None and not existing_wavs(shared_positive_dir):
        raise SystemExit(
            f"The shared positive TTS directory has no WAV files: {shared_positive_dir}"
        )
    negative_dir = Path(args.negative_dir).resolve()
    mined_negative_dir = Path(args.mined_negative_dir).resolve() if args.mined_negative_dir else None
    background_dir = Path(args.background_dir).resolve()
    rir_dir = Path(args.rir_dir).resolve()
    model_dir = output_dir / model_name
    generated_dir = model_dir / "generated"
    positive_generated = generated_dir / "positive"
    negative_generated = generated_dir / "negative"
    validation_generated = generated_dir / "validation_positive"
    validation_negative_generated = generated_dir / "validation_negative"
    if existing_wavs(background_dir):
        background_paths = [str(background_dir)]
        alignment_background = None
    else:
        alignment_background = ensure_alignment_background(output_dir)
        background_paths = [str(alignment_background)]
    rir_paths = [str(rir_dir)] if existing_wavs(rir_dir) else []
    hard_negative_generated = generated_dir / "negative_phoneme"
    hard_negative_samples = (
        min(DEFAULT_HARD_NEGATIVE_SAMPLES, max(0, int(args.negative_samples * 0.6)))
        if args.negative_samples > 0
        else 0
    )

    if args.positive_samples > 0 and shared_positive_dir is None:
        invalidated = prepare_generation_cache(
            positive_generated,
            {
                "kind": "fixed_phrase_positive",
                "phrase": args.phrase.strip().lower(),
            },
            adopt_legacy=True,
        )
        invalidate_feature_file(model_dir, "synthetic_positive_features.npy", invalidated)

    if args.validation_samples > 0:
        invalidated = prepare_generation_cache(
            validation_generated,
            {
                "kind": "fixed_phrase_validation_positive",
                "phrase": args.phrase.strip().lower(),
            },
            adopt_legacy=True,
        )
        invalidate_feature_file(model_dir, "synthetic_validation_positive_features.npy", invalidated)

        validation_negative_recipe = negative_text_source(args, args.validation_samples)
        invalidated = prepare_generation_cache(
            validation_negative_generated,
            {
                "kind": "validation_negative",
                "recipe_version": NEGATIVE_GENERATION_RECIPE_VERSION,
                "recipe": validation_negative_recipe,
            },
            adopt_legacy=False,
        )
        invalidate_feature_file(model_dir, "synthetic_validation_negative_features.npy", invalidated)

    if args.negative_samples > 0:
        negative_recipe = negative_text_source(args, args.negative_samples)
        invalidated = prepare_generation_cache(
            negative_generated,
            {
                "kind": "negative",
                "recipe_version": NEGATIVE_GENERATION_RECIPE_VERSION,
                "recipe": negative_recipe,
            },
            adopt_legacy=False,
        )
        invalidate_feature_file(model_dir, "synthetic_negative_features.npy", invalidated)

    if hard_negative_samples > 0:
        phoneme_recipe = phoneme_negative_text_source(args)
        invalidated = prepare_generation_cache(
            hard_negative_generated,
            {
                "kind": "phoneme_negative",
                "recipe_version": NEGATIVE_GENERATION_RECIPE_VERSION,
                "recipe": phoneme_recipe,
            },
            adopt_legacy=False,
        )
        invalidate_feature_file(model_dir, "synthetic_phoneme_hard_negative_features.npy", invalidated)

    config: dict[str, Any] = {
        "model_name": model_name,
        "target_phrase": args.phrase,
        "output_dir": str(output_dir),
        "positive_data_path": str(
            shared_positive_dir
            or (positive_generated if args.positive_samples > 0 else positive_dir)
        ),
        "negative_data_path": str(negative_generated if args.negative_samples > 0 else negative_dir),
        "model_type": args.model_type,
        "layer_size": layer_size,
        "n_blocks": int(architecture.get("n_blocks", 1)),
        "embedding_dim": 128,
        "dropout_prob": 0.3,
        "activation_function": "relu",
        "margin_pos": 2.0,
        "margin_neg": -2.0,
        "LOSS_BIAS": float(architecture.get("loss_bias", 0.75)),
        "logit_reg_weight": 0.0005,
        "logit_reg_margin": 4.0,
        "logit_min_margin": 1.5,
        "steps": args.steps,
        "stabilization_steps": max(1, min(20000, max(1000, args.steps // 2), max(1, args.steps - 1))),
        "num_workers": args.num_workers,
        "optimizer_type": "adamw",
        "learning_rate_max": 0.0008,
        "lr_scheduler_type": "onecycle",
        "weight_decay": 0.01,
        "momentum": 0.9,
        "target_accuracy": args.target_accuracy,
        "target_recall": args.target_recall,
        "target_false_positives_per_hour": args.target_fp_per_hour,
        "generate_clips": False,
        "transform_clips": True,
        "train_model": True,
        "distill": False,
        "convert_audio": True,
        "show_training_summary": True,
        "overwrite": bool(args.overwrite),
        "batch_composition": {},
        "feature_manifest": {"targets": {}, "negatives": {}},
        "data_generation_tasks": [],
        "feature_generation_manifest": {},
        "background_paths": background_paths,
        "rir_paths": rir_paths,
        "background_paths_duplication_rate": [1] if background_paths else [],
        "alignment_background": str(alignment_background) if alignment_background else None,
        "audio_processing": {"clip_length_samples": DEFAULT_CLIP_LENGTH_SAMPLES},
        "augmentation_batch_size": 16,
        "feature_gen_cpu_ratio": 1.0,
        "augmentation_settings": {
            "gain_prob": 1.0,
            "max_gain_in_db": 2.0,
            "max_pitch_semitones": 1.0,
            "max_snr_in_db": 35.0,
            "min_gain_in_db": -2.0,
            "min_pitch_semitones": -1.0,
            "min_snr_in_db": 15.0,
            "pitch_prob": 0.3,
            "rir_prob": 0.0,
        },
        "val_miss_weight": 4.0,
        # False wakes are just as damaging as misses for an always-listening
        # satellite.  The old 4:1 weighting selected overly eager checkpoints.
        "val_fp_weight": 4.0,
        "validation_batch_size": 256,
        "validation_smoothing_window": 3,
        "val_early_stopping_patience": 6000,
        "hardness_ema_alpha": 0.05,
        "hardness_floor": 0.05,
        "hardness_reset_interval": 5000,
        "hardness_reset_decay": 0.5,
        "checkpoint_averaging_top_k": 5,
        "checkpointing": {
            "enabled": True,
            "interval_steps": 1000,
            "limit": 2,
        },
        "early_stopping_patience": 0,
        "min_delta": 0.0001,
        "ema_alpha": 0.01,
        "onnx_opset_version": 17,
        "hard_negative_samples": hard_negative_samples,
        "quality_evaluation": {"positive_dirs": [], "negative_dirs": []},
        "training_audio_evaluation": {
            "positive_dirs": [str(positive_source_dir)] if existing_wavs(positive_source_dir) else [],
            "negative_dirs": [str(negative_dir)] if existing_wavs(negative_dir) else [],
        },
    }

    add_feature_bank_negatives(config, feature_bank_dir(output_dir))

    if shared_positive_dir is not None:
        add_feature_task(
            config,
            key="t",
            label="shared_tts_positive_features",
            source_dir=shared_positive_dir,
            feature_file="shared_tts_positive_features.npy",
            split="target",
            augmentation_rounds=10,
            batch_weight=100,
        )
    elif args.positive_samples > 0:
        missing = missing_generated_samples(positive_generated, args.positive_samples)
        invalidate_feature_file(model_dir, "synthetic_positive_features.npy", missing > 0)
        if missing:
            config["data_generation_tasks"].append(
                {
                    "name": "synthetic_positive",
                    "enabled": True,
                    "output_dir": str(positive_generated),
                    "num_samples": missing,
                    "file_prefix": "pos",
                    "text_source": {
                        "type": "fixed_phrase",
                        "phrase": args.phrase,
                    },
                }
            )
        add_feature_task(
            config,
            key="t",
            label="synthetic_positive_features",
            source_dir=positive_generated,
            feature_file="synthetic_positive_features.npy",
            split="target",
            require_existing=False,
            augmentation_rounds=10,
            batch_weight=100,
        )

    if args.validation_samples > 0:
        missing = missing_generated_samples(validation_generated, args.validation_samples)
        invalidate_feature_file(model_dir, "synthetic_validation_positive_features.npy", missing > 0)
        if missing:
            config["data_generation_tasks"].append(
                {
                    "name": "synthetic_validation_positive",
                    "enabled": True,
                    "output_dir": str(validation_generated),
                    "num_samples": missing,
                    "file_prefix": "val_pos",
                    "text_source": {
                        "type": "fixed_phrase",
                        "phrase": args.phrase,
                    },
                }
            )
        add_feature_task(
            config,
            key="tv",
            label="synthetic_validation_positive_features",
            source_dir=validation_generated,
            feature_file="synthetic_validation_positive_features.npy",
            split="target_val",
            require_existing=False,
            augmentation_rounds=10,
        )
        config["quality_evaluation"]["positive_dirs"].append(str(validation_generated))
        missing = missing_generated_samples(validation_negative_generated, args.validation_samples)
        invalidate_feature_file(model_dir, "synthetic_validation_negative_features.npy", missing > 0)
        if missing:
            config["data_generation_tasks"].append(
                {
                    "name": "synthetic_validation_negative",
                    "enabled": True,
                    "output_dir": str(validation_negative_generated),
                    "num_samples": missing,
                    "file_prefix": "val_neg",
                    "text_source": negative_text_source(args, missing),
                }
            )
        add_feature_task(
            config,
            key="nv",
            label="synthetic_validation_negative_features",
            source_dir=validation_negative_generated,
            feature_file="synthetic_validation_negative_features.npy",
            split="negative_val",
            require_existing=False,
            augmentation_rounds=10,
        )
        config["quality_evaluation"]["negative_dirs"].append(str(validation_negative_generated))

    validation_positive_dir = Path(args.validation_positive_dir).resolve() if args.validation_positive_dir else None
    if validation_positive_dir and existing_wavs(validation_positive_dir):
        add_feature_task(
            config,
            key="tvp",
            label="personal_validation_positive_features",
            source_dir=validation_positive_dir,
            feature_file="personal_validation_positive_features.npy",
            split="target_val",
            augmentation_rounds=10,
        )
        config["quality_evaluation"]["positive_dirs"].append(str(validation_positive_dir))

    validation_negative_dir = Path(args.validation_negative_dir).resolve() if args.validation_negative_dir else None
    if validation_negative_dir and existing_wavs(validation_negative_dir):
        add_feature_task(
            config,
            key="nvp",
            label="personal_validation_negative_features",
            source_dir=validation_negative_dir,
            feature_file="personal_validation_negative_features.npy",
            split="negative_val",
            augmentation_rounds=10,
        )
        config["quality_evaluation"]["negative_dirs"].append(str(validation_negative_dir))

    if existing_wavs(positive_dir):
        add_feature_task(
            config,
            key="tp",
            label="personal_positive_features",
            source_dir=positive_dir,
            feature_file="personal_positive_features.npy",
            split="target",
            augmentation_rounds=PERSONAL_AUGMENTATION_ROUNDS,
            batch_weight=200,
        )

    if args.negative_samples > 0:
        missing = missing_generated_samples(negative_generated, args.negative_samples)
        invalidate_feature_file(model_dir, "synthetic_negative_features.npy", missing > 0)
        if missing:
            config["data_generation_tasks"].append(
                {
                    "name": "synthetic_negative",
                    "enabled": True,
                    "output_dir": str(negative_generated),
                    "num_samples": missing,
                    "file_prefix": "neg",
                    "text_source": negative_text_source(args, missing),
                }
            )
        add_feature_task(
            config,
            key="n",
            label="synthetic_negative_features",
            source_dir=negative_generated,
            feature_file="synthetic_negative_features.npy",
            split="negative",
            require_existing=False,
            augmentation_rounds=10,
            batch_weight=100,
        )

    if hard_negative_samples > 0:
        missing = missing_generated_samples(hard_negative_generated, hard_negative_samples)
        invalidate_feature_file(model_dir, "synthetic_phoneme_hard_negative_features.npy", missing > 0)
        if missing:
            config["data_generation_tasks"].append(
                {
                    "name": "synthetic_phoneme_hard_negative",
                    "enabled": True,
                    "output_dir": str(hard_negative_generated),
                    "num_samples": missing,
                    "file_prefix": "neg_ph",
                    "text_source": phoneme_negative_text_source(args),
                }
            )
        add_feature_task(
            config,
            key="hn",
            label="synthetic_phoneme_hard_negative_features",
            source_dir=hard_negative_generated,
            feature_file="synthetic_phoneme_hard_negative_features.npy",
            split="negative",
            require_existing=False,
            augmentation_rounds=1,
            batch_weight=20,
        )

    if existing_wavs(negative_dir):
        add_feature_task(
            config,
            key="np",
            label="personal_negative_features",
            source_dir=negative_dir,
            feature_file="personal_negative_features.npy",
            split="negative",
            augmentation_rounds=PERSONAL_AUGMENTATION_ROUNDS,
            batch_weight=200,
        )

    if mined_negative_dir and existing_wavs(mined_negative_dir):
        add_feature_task(
            config,
            key="nm",
            label="mined_negative_features",
            source_dir=mined_negative_dir,
            feature_file="mined_negative_features.npy",
            split="negative",
            augmentation_rounds=10,
            batch_weight=100,
        )

    if not config["feature_manifest"]["targets"]:
        raise SystemExit("No positive data configured. Add positive clips or set --positive-samples above 0.")
    if not config["feature_manifest"]["negatives"]:
        log("WARNING: no negative data configured. Add negative clips or set --negative-samples above 0.")

    config["generate_clips"] = bool(config["data_generation_tasks"])

    return config


def sync_artifacts(output_dir: Path, export_dir: Path, model_name: str, metadata: dict[str, Any]) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
            continue
        dest = export_dir / path.name
        if dest.exists() and dest.resolve() == path.resolve():
            copied.append(dest)
            continue
        shutil.copy2(path, dest)
        copied.append(dest)

    metadata["artifacts"] = [path.name for path in copied]
    metadata["synced_at"] = datetime.now(timezone.utc).isoformat()
    (export_dir / f"{model_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if copied:
        log("Synced artifacts:")
        for path in copied:
            log(f"  {path}")
    else:
        log(f"WARNING: no NanoWakeWord model artifacts found in {output_dir}")
    return copied


def run_nanowakeword(config_path: Path, stage_flags: list[str]) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if bool(config.get("overwrite")):
        stage_flags.append("--overwrite")
    candidates = [
        [*nanowakeword_cmd(), "-c", str(config_path), *stage_flags],
        [*nanowakeword_cmd(), "--config", str(config_path), *stage_flags],
    ]
    last_code = 1
    for idx, cmd in enumerate(candidates, start=1):
        log("")
        log("$ " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=os.environ.copy())
        if result.returncode == 0:
            return
        last_code = result.returncode
        if idx < len(candidates):
            log(f"Command form exited {result.returncode}; trying the next NanoWakeWord CLI form.")
    raise SystemExit(last_code)


def write_stage_config(
    config: dict[str, Any],
    config_path: Path,
    suffix: str,
    *,
    generate_clips: bool = False,
    transform_clips: bool = False,
    train_model: bool = False,
    feature_generation_manifest: dict[str, Any] | None = None,
) -> Path:
    stage_config = copy.deepcopy(config)
    stage_config["generate_clips"] = generate_clips
    stage_config["transform_clips"] = transform_clips
    stage_config["train_model"] = train_model
    stage_config["distill"] = False
    stage_config.setdefault("distillation", {})["enabled"] = False
    if feature_generation_manifest is not None:
        stage_config["feature_generation_manifest"] = feature_generation_manifest
    path = config_path.with_name(f"{config_path.stem}.{suffix}{config_path.suffix}")
    path.write_text(yaml.safe_dump(stage_config, sort_keys=False), encoding="utf-8")
    return path


def run_training(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if config.get("generate_clips"):
        run_nanowakeword(
            write_stage_config(config, config_path, "generate", generate_clips=True),
            ["-G"],
        )

    feature_jobs = config.get("feature_generation_manifest") or {}
    if config.get("transform_clips") and feature_jobs:
        for job_name, recipe in feature_jobs.items():
            run_nanowakeword(
                write_stage_config(
                    config,
                    config_path,
                    f"features.{safe_name(str(job_name))}",
                    transform_clips=True,
                    feature_generation_manifest={job_name: recipe},
                ),
                ["-t"],
            )

    if config.get("train_model"):
        run_nanowakeword(
            write_stage_config(config, config_path, "train", train_model=True),
            ["-T"],
        )


def find_onnx_model(model_output_dir: Path, model_name: str) -> Path:
    preferred = model_output_dir / "model" / f"{model_name}.onnx"
    if preferred.exists():
        return preferred
    candidates = sorted(model_output_dir.rglob("*.onnx"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No ONNX model was produced under {model_output_dir}")
    return candidates[0]


def ensure_training_completed(model_output_dir: Path, model_name: str) -> None:
    """Reject NanoWakeWord's silent zero-step success mode.

    Upstream treats an exhausted batch sampler as a successful training run and
    still exports an untrained ONNX model.  Its journal records N/A metrics in
    that case, so require a real completed journal row before quality testing or
    artifact publication.
    """
    journal_path = model_output_dir.parent / "training_journal.md"
    if not journal_path.exists():
        raise SystemExit(f"NanoWakeWord did not write a training journal: {journal_path}")
    rows = [
        line
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if f"**{model_name}**" in line
    ]
    if not rows or "N/A" in rows[-1]:
        raise SystemExit(
            "NanoWakeWord exited before completing a training step. "
            "The untrained model was rejected and no published artifact was changed."
        )


def quality_misclassifications(report: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """Return unique missed positives and false-trigger negatives from a report."""
    positive: dict[str, Path] = {}
    negative: dict[str, Path] = {}
    sections = [report.get("validation") or {}]
    sections.extend(report.get("additional") or [])
    for section in sections:
        misclassified = section.get("misclassified") or {}
        for value in misclassified.get("missed_positive_paths") or []:
            path = Path(value).resolve()
            if path.is_file():
                positive[str(path)] = path
        for value in misclassified.get("triggered_negative_paths") or []:
            path = Path(value).resolve()
            if path.is_file():
                negative[str(path)] = path
    return list(positive.values()), list(negative.values())


def stage_retry_audio(paths: list[Path], directory: Path, *, positive: bool) -> Path:
    """Stage quality-gate misses as augmented retry inputs."""
    directory.mkdir(parents=True, exist_ok=True)
    for stale in existing_wavs(directory):
        stale.unlink()

    if positive:
        from evaluate_nanowakeword import read_pcm16

    for index, source in enumerate(sorted(paths, key=lambda item: str(item))):
        destination = directory / f"{index:04d}_{safe_name(source.stem)}.wav"
        if positive:
            _write_pcm16(destination, speech_centered_crop(read_pcm16(source)))
        else:
            shutil.copy2(source, destination)
    return directory


def make_quality_retry_config(
    config: dict[str, Any],
    model_output_dir: Path,
    missed_positive_paths: list[Path],
    triggered_negative_paths: list[Path],
) -> dict[str, Any]:
    """Create a second-pass config that emphasizes runtime gate failures."""
    retry = copy.deepcopy(config)
    retry["data_generation_tasks"] = []
    retry["generate_clips"] = False
    retry["feature_generation_manifest"] = {}
    retry["transform_clips"] = True
    retry["train_model"] = True

    if missed_positive_paths:
        source = stage_retry_audio(
            missed_positive_paths,
            model_output_dir / "_trainer_assets" / "quality_retry_positive",
            positive=True,
        )
        invalidate_feature_file(model_output_dir, "quality_retry_positive_features.npy", True)
        add_feature_task(
            retry,
            key="qrp",
            label="quality_retry_positive_features",
            source_dir=source,
            feature_file="quality_retry_positive_features.npy",
            split="target",
            augmentation_rounds=PERSONAL_AUGMENTATION_ROUNDS,
            batch_weight=200,
        )

    if triggered_negative_paths:
        source = stage_retry_audio(
            triggered_negative_paths,
            model_output_dir / "_trainer_assets" / "quality_retry_negative",
            positive=False,
        )
        invalidate_feature_file(model_output_dir, "quality_retry_negative_features.npy", True)
        add_feature_task(
            retry,
            key="qrn",
            label="quality_retry_negative_features",
            source_dir=source,
            feature_file="quality_retry_negative_features.npy",
            split="negative",
            augmentation_rounds=PERSONAL_AUGMENTATION_ROUNDS,
            batch_weight=200,
        )

    if not retry["feature_generation_manifest"]:
        raise ValueError("No quality-gate failures were available for a corrective retry")
    retry["quality_retry"] = {
        "missed_positive_paths": [str(path) for path in missed_positive_paths],
        "triggered_negative_paths": [str(path) for path in triggered_negative_paths],
    }
    return retry


def raise_quality_failure(report: dict[str, Any], report_path: Path) -> None:
    reasons = "; ".join(report.get("failures") or ["unknown quality failure"])
    raise SystemExit(
        f"NanoWakeWord model failed streaming quality validation: {reasons}. "
        f"The existing published model was left untouched. Report: {report_path}"
    )


def run_quality_gate(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_output_dir: Path,
    model_name: str,
    *,
    sample_partition: int = 0,
) -> dict[str, Any] | None:
    if args.skip_quality_gate:
        log("WARNING: streaming quality gate was explicitly skipped.")
        return None
    quality = config.get("quality_evaluation") or {}
    positive_dirs = [Path(item) for item in quality.get("positive_dirs") or []]
    negative_dirs = [Path(item) for item in quality.get("negative_dirs") or []]
    if not positive_dirs or not negative_dirs:
        raise SystemExit(
            "Streaming quality evaluation requires positive and negative validation WAVs. "
            "Keep --validation-samples above 0, provide validation directories, or explicitly "
            "use --skip-quality-gate."
        )

    model_path = find_onnx_model(model_output_dir, model_name)
    log("")
    log("Running streaming audio quality gate...")
    validation_report = calibrate_model(
        model_path,
        positive_dirs,
        negative_dirs,
        minimum_threshold=args.quality_threshold,
        minimum_patience=args.quality_patience,
        maximum_patience=getattr(args, "quality_maximum_patience", 4),
        max_clips_per_class=args.quality_max_clips,
        sample_partition=sample_partition,
        minimum_recall=args.quality_minimum_recall,
        maximum_negative_rate=args.quality_maximum_negative_rate,
        target_recall=getattr(args, "quality_target_recall", 0.95),
        target_negative_rate=getattr(args, "quality_target_negative_rate", 0.05),
    )
    selected_contract = validation_report["streaming_contract"]
    selected_threshold = float(selected_contract["threshold"])
    selected_patience = int(selected_contract["patience"])
    additional_positive = [Path(item).resolve() for item in args.quality_positive_dir]
    additional_negative = [Path(item).resolve() for item in args.quality_negative_dir]
    additional_reports: list[dict[str, Any]] = []
    training_audio = config.get("training_audio_evaluation") or {}
    training_positive = [Path(item) for item in training_audio.get("positive_dirs") or []]
    training_negative = [Path(item) for item in training_audio.get("negative_dirs") or []]
    if training_positive and training_negative:
        training_report = evaluate_model(
            model_path,
            training_positive,
            training_negative,
            threshold=selected_threshold,
            patience=selected_patience,
            max_clips_per_class=args.quality_max_clips,
            minimum_recall=args.quality_minimum_recall,
            maximum_negative_rate=args.quality_maximum_real_negative_rate,
        )
        additional_reports.append({"name": "training_audio", **training_report})
    if additional_positive or additional_negative:
        if not additional_positive or not additional_negative:
            raise SystemExit("Additional quality evaluation requires both positive and negative directories")
        if len(additional_positive) == len(additional_negative):
            quality_groups = zip(additional_positive, additional_negative)
        else:
            quality_groups = [(additional_positive, additional_negative)]
        for index, (positive_group, negative_group) in enumerate(quality_groups, start=1):
            positive_group_dirs = positive_group if isinstance(positive_group, list) else [positive_group]
            negative_group_dirs = negative_group if isinstance(negative_group, list) else [negative_group]
            group_report = evaluate_model(
                model_path,
                positive_group_dirs,
                negative_group_dirs,
                threshold=selected_threshold,
                patience=selected_patience,
                max_clips_per_class=args.quality_max_clips,
                minimum_recall=args.quality_minimum_recall,
                maximum_negative_rate=args.quality_maximum_real_negative_rate,
            )
            additional_reports.append({"name": f"additional_{index}", **group_report})
    report = {
        "passed": validation_report["passed"] and all(item["passed"] for item in additional_reports),
        "validation": validation_report,
        "additional": additional_reports,
    }
    report["failures"] = [
        f"validation: {message}" for message in validation_report["failures"]
    ] + [
        f"{item['name']}: {message}"
        for item in additional_reports
        for message in item["failures"]
    ]
    report_path = model_output_dir / f"{model_name}.quality.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(json.dumps(report, indent=2))
    if report["passed"]:
        log(f"Streaming quality gate passed. Report: {report_path}")
    else:
        log(f"Streaming quality gate failed. Report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NanoWakeWord wake word model")
    parser.add_argument("phrase")
    parser.add_argument("--output-root", default=os.environ.get("NWW_OUTPUT_ROOT", str(ROOT_DIR / "output")))
    parser.add_argument("--export-dir", default=os.environ.get("NWW_EXPORT_DIR", str(ROOT_DIR / "trained_wake_words")))
    parser.add_argument("--positive-dir", default=os.environ.get("NWW_PERSONAL_DIR", str(ROOT_DIR / "personal_samples")))
    parser.add_argument(
        "--positive-tts-dir",
        default=os.environ.get("NWW_POSITIVE_TTS_DIR", ""),
        help=(
            "Reuse an existing generated positive-TTS WAV corpus instead of "
            "generating NanoWakeWord positives. Personal --positive-dir clips "
            "are still included as a separate balanced source."
        ),
    )
    parser.add_argument("--negative-dir", default=os.environ.get("NWW_NEGATIVE_DIR", str(ROOT_DIR / "negative_samples")))
    parser.add_argument(
        "--mined-negative-dir",
        help="Optional separate hard-negative WAV directory with its own balanced sampler source.",
    )
    parser.add_argument("--background-dir", default=os.environ.get("NWW_BACKGROUND_DIR", str(ROOT_DIR / "background_samples")))
    parser.add_argument("--rir-dir", default=os.environ.get("NWW_RIR_DIR", str(ROOT_DIR / "rir_samples")))
    parser.add_argument("--positive-samples", type=int, default=DEFAULT_POSITIVE_SAMPLES)
    parser.add_argument("--negative-samples", type=int, default=DEFAULT_NEGATIVE_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--validation-positive-dir")
    parser.add_argument("--validation-negative-dir")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--model-type", type=model_type_arg, default="cnn")
    parser.add_argument(
        "--layer-size",
        type=int,
        default=0,
        help="Hidden layer size; 0 selects the architecture recommendation.",
    )
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--target-fp-per-hour", type=float, default=0.5)
    parser.add_argument("--custom-negative-phrase", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true", help="Regenerate feature files instead of reusing existing .npy artifacts.")
    parser.add_argument("--quality-positive-dir", action="append", default=[])
    parser.add_argument("--quality-negative-dir", action="append", default=[])
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--quality-patience", type=int, default=2)
    parser.add_argument(
        "--quality-maximum-patience",
        type=int,
        default=4,
        help="Maximum consecutive streaming hits considered during automatic calibration.",
    )
    parser.add_argument("--quality-max-clips", type=int, default=DEFAULT_QUALITY_MAX_CLIPS)
    parser.add_argument("--quality-minimum-recall", type=float, default=0.90)
    parser.add_argument(
        "--quality-target-recall",
        type=float,
        default=0.95,
        help="Preferred recall used to select a buffered deployment policy.",
    )
    parser.add_argument(
        "--quality-target-negative-rate",
        type=float,
        default=0.05,
        help="Preferred adversarial-negative rate used during automatic calibration.",
    )
    parser.add_argument(
        "--quality-maximum-negative-rate",
        type=float,
        default=0.10,
        help=(
            "Maximum trigger rate for deliberately confusable synthetic negatives "
            "(default: 0.10)."
        ),
    )
    parser.add_argument(
        "--quality-maximum-real-negative-rate",
        type=float,
        default=0.0,
        help="Maximum trigger rate for uploaded/real negative recordings (default: 0).",
    )
    parser.add_argument(
        "--quality-retries",
        type=int,
        default=1,
        help="Corrective retraining passes using runtime-gate misses (default: 1).",
    )
    parser.add_argument("--skip-quality-gate", action="store_true")
    return parser.parse_args()


def model_type_arg(value: str) -> str:
    try:
        return normalize_model_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    args = parse_args()
    model_name = safe_name(args.phrase)
    output_root = Path(args.output_root).resolve()
    export_dir = Path(args.export_dir).resolve()
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args, model_name, output_root)
    architecture = architecture_metadata(args.model_type)
    config_path = model_output_dir / f"{model_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metadata = {
        "name": model_name,
        "phrase": args.phrase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "model_type": args.model_type,
        "architecture": architecture,
        "steps": args.steps,
        "layer_size": config["layer_size"],
        "hard_negative_samples": config.get("hard_negative_samples", 0),
        "feature_bank_dir": str(feature_bank_dir(output_root)),
    }
    (model_output_dir / f"{model_name}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"NanoWakeWord config written to {config_path}")
    log(json.dumps(metadata, indent=2))
    run_training(config_path)
    ensure_training_completed(model_output_dir, model_name)
    quality = run_quality_gate(args, config, model_output_dir, model_name)
    retry_config = config
    retry_count = 0
    accumulated_positive: dict[str, Path] = {}
    accumulated_negative: dict[str, Path] = {}
    while quality is not None and not quality["passed"] and retry_count < max(0, args.quality_retries):
        missed_positive, triggered_negative = quality_misclassifications(quality)
        for path in missed_positive:
            accumulated_positive[str(path)] = path
        for path in triggered_negative:
            accumulated_negative[str(path)] = path
        if not accumulated_positive and not accumulated_negative:
            break

        retry_count += 1
        report_path = model_output_dir / f"{model_name}.quality.json"
        shutil.copy2(
            report_path,
            model_output_dir / f"{model_name}.quality.attempt-{retry_count}.json",
        )
        log(
            f"Quality retry {retry_count}/{args.quality_retries}: emphasizing "
            f"{len(accumulated_positive)} missed positives and "
            f"{len(accumulated_negative)} false-trigger negatives."
        )
        retry_config = make_quality_retry_config(
            retry_config,
            model_output_dir,
            list(accumulated_positive.values()),
            list(accumulated_negative.values()),
        )
        retry_path = model_output_dir / f"{model_name}.quality-retry-{retry_count}.yaml"
        retry_path.write_text(yaml.safe_dump(retry_config, sort_keys=False), encoding="utf-8")
        run_training(retry_path)
        ensure_training_completed(model_output_dir, model_name)
        # The corrective pass trains on the previous gate's misses. Evaluate
        # the next disjoint synthetic partition so a retry cannot pass merely
        # by memorizing the exact clips that caused it.
        quality = run_quality_gate(
            args,
            retry_config,
            model_output_dir,
            model_name,
            sample_partition=retry_count,
        )

    if quality is not None and not quality["passed"]:
        raise_quality_failure(quality, model_output_dir / f"{model_name}.quality.json")
    metadata["quality"] = quality
    metadata["quality_retries_used"] = retry_count
    if quality is not None:
        contract = quality["validation"]["streaming_contract"]
        metadata["recommended_threshold"] = float(contract["threshold"])
        metadata["recommended_patience"] = int(contract["patience"])
    sync_artifacts(model_output_dir, export_dir, model_name, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
