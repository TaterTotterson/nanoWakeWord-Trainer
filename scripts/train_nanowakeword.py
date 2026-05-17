#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
) -> None:
    if require_existing and not existing_wavs(source_dir):
        return
    feature_path = Path(config["output_dir"]) / config["model_name"] / "features" / feature_file
    has_backgrounds = bool(config.get("background_paths"))
    has_rirs = bool(config.get("rir_paths"))
    config.setdefault("feature_generation_manifest", {})[label] = {
        "input_audio_dirs": [str(source_dir)],
        "output_filename": feature_file,
        "use_background_noise": has_backgrounds,
        "use_rir": has_rirs,
        "augmentation_rounds": 1,
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
        config.setdefault("batch_composition", {})[key] = 32 if split == "target" else 64


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
        "include_partial_phrase": True,
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

    config: dict[str, Any] = {
        "model_name": model_name,
        "target_phrase": args.phrase,
        "output_dir": str(output_dir),
        "positive_data_path": str(positive_generated if args.positive_samples > 0 else positive_dir),
        "negative_data_path": str(negative_generated if args.negative_samples > 0 else negative_dir),
        "model_type": args.model_type,
        "layer_size": args.layer_size,
        "steps": args.steps,
        "num_workers": args.num_workers,
        "target_accuracy": args.target_accuracy,
        "target_recall": args.target_recall,
        "target_false_positives_per_hour": args.target_fp_per_hour,
        "generate_clips": args.positive_samples > 0 or args.negative_samples > 0 or args.validation_samples > 0,
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
    }

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
        )

    if existing_wavs(positive_dir):
        add_feature_task(
            config,
            key="tp",
            label="personal_positive_features",
            source_dir=positive_dir,
            feature_file="personal_positive_features.npy",
            split="target",
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
        )

    if existing_wavs(negative_dir):
        add_feature_task(
            config,
            key="np",
            label="personal_negative_features",
            source_dir=negative_dir,
            feature_file="personal_negative_features.npy",
            split="negative",
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


def run_training(config_path: Path) -> None:
    stage_flags = ["-G", "-t", "-T"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NanoWakeWord wake word model")
    parser.add_argument("phrase")
    parser.add_argument("--output-root", default=os.environ.get("NWW_OUTPUT_ROOT", str(ROOT_DIR / "output")))
    parser.add_argument("--export-dir", default=os.environ.get("NWW_EXPORT_DIR", str(ROOT_DIR / "trained_wake_words")))
    parser.add_argument("--positive-dir", default=os.environ.get("NWW_PERSONAL_DIR", str(ROOT_DIR / "personal_samples")))
    parser.add_argument("--negative-dir", default=os.environ.get("NWW_NEGATIVE_DIR", str(ROOT_DIR / "negative_samples")))
    parser.add_argument("--background-dir", default=os.environ.get("NWW_BACKGROUND_DIR", str(ROOT_DIR / "background_samples")))
    parser.add_argument("--rir-dir", default=os.environ.get("NWW_RIR_DIR", str(ROOT_DIR / "rir_samples")))
    parser.add_argument("--positive-samples", type=int, default=2000)
    parser.add_argument("--negative-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=400)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=4)
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
    }
    (model_output_dir / f"{model_name}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"NanoWakeWord config written to {config_path}")
    log(json.dumps(metadata, indent=2))
    run_training(config_path)
    sync_artifacts(model_output_dir, export_dir, model_name, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
