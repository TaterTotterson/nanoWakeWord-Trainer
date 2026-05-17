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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_SUFFIXES = {".onnx", ".pt", ".pth"}
DEFAULT_POSITIVE_SAMPLES = 2500
DEFAULT_NEGATIVE_SAMPLES = 5000
DEFAULT_VALIDATION_SAMPLES = 2000
DEFAULT_HARD_NEGATIVE_SAMPLES = 3000
DEFAULT_STEPS = 50000
DEFAULT_NUM_WORKERS = 0

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


def feature_bank_dir(output_dir: Path) -> Path:
    return Path(os.environ.get("NWW_FEATURE_BANK_DIR", str(output_dir.parent / "feature_banks"))).resolve()


def add_feature_bank_negatives(config: dict[str, Any], bank_dir: Path) -> None:
    for filename, (key, batch_weight) in FEATURE_BANKS.items():
        path = (bank_dir / filename).resolve()
        if not path.exists():
            continue
        config.setdefault("feature_manifest", {}).setdefault("negatives", {})[key] = str(path)
        config.setdefault("batch_composition", {})[key] = batch_weight


def nanowakeword_cmd() -> list[str]:
    exe = shutil.which("nanowakeword")
    if exe:
        return [exe]
    return [sys.executable, "-m", "nanowakeword"]


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
        "include_input_words": True,
        "include_partial_phrase": True,
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
    positive_dir = Path(args.positive_dir).resolve()
    negative_dir = Path(args.negative_dir).resolve()
    background_dir = Path(args.background_dir).resolve()
    rir_dir = Path(args.rir_dir).resolve()
    model_dir = output_dir / model_name
    generated_dir = model_dir / "generated"
    positive_generated = generated_dir / "positive"
    negative_generated = generated_dir / "negative"
    validation_generated = generated_dir / "validation_positive"
    validation_negative_generated = generated_dir / "validation_negative"
    background_paths = [str(background_dir)] if existing_wavs(background_dir) else []
    rir_paths = [str(rir_dir)] if existing_wavs(rir_dir) else []
    hard_negative_generated = generated_dir / "negative_phoneme"
    hard_negative_samples = (
        min(DEFAULT_HARD_NEGATIVE_SAMPLES, max(0, int(args.negative_samples * 0.6)))
        if args.negative_samples > 0
        else 0
    )

    config: dict[str, Any] = {
        "model_name": model_name,
        "target_phrase": args.phrase,
        "output_dir": str(output_dir),
        "positive_data_path": str(positive_generated if args.positive_samples > 0 else positive_dir),
        "negative_data_path": str(negative_generated if args.negative_samples > 0 else negative_dir),
        "model_type": args.model_type,
        "layer_size": args.layer_size,
        "n_blocks": 3,
        "embedding_dim": 128,
        "dropout_prob": 0.3,
        "activation_function": "relu",
        "margin_pos": 2.0,
        "margin_neg": -2.0,
        "LOSS_BIAS": 0.65,
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
        "generate_clips": (
            args.positive_samples > 0
            or args.negative_samples > 0
            or args.validation_samples > 0
            or hard_negative_samples > 0
        ),
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
        "val_fp_weight": 1.0,
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
    }

    add_feature_bank_negatives(config, feature_bank_dir(output_dir))

    if args.positive_samples > 0:
        config["data_generation_tasks"].append(
            {
                "name": "synthetic_positive",
                "enabled": True,
                "output_dir": str(positive_generated),
                "num_samples": args.positive_samples,
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
        config["data_generation_tasks"].append(
            {
                "name": "synthetic_validation_positive",
                "enabled": True,
                "output_dir": str(validation_generated),
                "num_samples": args.validation_samples,
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
        config["data_generation_tasks"].append(
            {
                "name": "synthetic_validation_negative",
                "enabled": True,
                "output_dir": str(validation_negative_generated),
                "num_samples": args.validation_samples,
                "file_prefix": "val_neg",
                "text_source": negative_text_source(args, args.validation_samples),
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

    if existing_wavs(positive_dir):
        add_feature_task(
            config,
            key="tp",
            label="personal_positive_features",
            source_dir=positive_dir,
            feature_file="personal_positive_features.npy",
            split="target",
            augmentation_rounds=10,
            batch_weight=100,
        )

    if args.negative_samples > 0:
        config["data_generation_tasks"].append(
            {
                "name": "synthetic_negative",
                "enabled": True,
                "output_dir": str(negative_generated),
                "num_samples": args.negative_samples,
                "file_prefix": "neg",
                "text_source": negative_text_source(args, args.negative_samples),
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
        config["data_generation_tasks"].append(
            {
                "name": "synthetic_phoneme_hard_negative",
                "enabled": True,
                "output_dir": str(hard_negative_generated),
                "num_samples": hard_negative_samples,
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
            augmentation_rounds=10,
            batch_weight=100,
        )

    if not config["feature_manifest"]["targets"]:
        raise SystemExit("No positive data configured. Add positive clips or set --positive-samples above 0.")
    if not config["feature_manifest"]["negatives"]:
        log("WARNING: no negative data configured. Add negative clips or set --negative-samples above 0.")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NanoWakeWord wake word model")
    parser.add_argument("phrase")
    parser.add_argument("--output-root", default=os.environ.get("NWW_OUTPUT_ROOT", str(ROOT_DIR / "output")))
    parser.add_argument("--export-dir", default=os.environ.get("NWW_EXPORT_DIR", str(ROOT_DIR / "trained_wake_words")))
    parser.add_argument("--positive-dir", default=os.environ.get("NWW_PERSONAL_DIR", str(ROOT_DIR / "personal_samples")))
    parser.add_argument("--negative-dir", default=os.environ.get("NWW_NEGATIVE_DIR", str(ROOT_DIR / "negative_samples")))
    parser.add_argument("--background-dir", default=os.environ.get("NWW_BACKGROUND_DIR", str(ROOT_DIR / "background_samples")))
    parser.add_argument("--rir-dir", default=os.environ.get("NWW_RIR_DIR", str(ROOT_DIR / "rir_samples")))
    parser.add_argument("--positive-samples", type=int, default=DEFAULT_POSITIVE_SAMPLES)
    parser.add_argument("--negative-samples", type=int, default=DEFAULT_NEGATIVE_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--model-type", default="dnn")
    parser.add_argument("--layer-size", type=int, default=32)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--target-fp-per-hour", type=float, default=0.5)
    parser.add_argument("--custom-negative-phrase", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true", help="Regenerate feature files instead of reusing existing .npy artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = safe_name(args.phrase)
    output_root = Path(args.output_root).resolve()
    export_dir = Path(args.export_dir).resolve()
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args, model_name, output_root)
    config_path = model_output_dir / f"{model_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metadata = {
        "name": model_name,
        "phrase": args.phrase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "model_type": args.model_type,
        "steps": args.steps,
        "hard_negative_samples": config.get("hard_negative_samples", 0),
        "feature_bank_dir": str(feature_bank_dir(output_root)),
    }
    (model_output_dir / f"{model_name}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"NanoWakeWord config written to {config_path}")
    log(json.dumps(metadata, indent=2))
    run_training(config_path)
    sync_artifacts(model_output_dir, export_dir, model_name, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
