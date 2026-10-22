from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import train_nanowakeword as trainer  # noqa: E402
from architecture_catalog import (  # noqa: E402
    architecture_list,
    architecture_metadata,
    normalize_model_type,
)
from evaluate_nanowakeword import (  # noqa: E402
    consecutive_activation_score,
    deterministic_sample,
    select_streaming_policy,
)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(audio, dtype="<i2").tobytes())


class TrainerDataTests(unittest.TestCase):
    def test_supported_architecture_defaults(self) -> None:
        self.assertEqual([item["id"] for item in architecture_list()], ["cnn", "gru"])
        self.assertEqual(architecture_metadata("cnn")["layer_size"], 32)
        self.assertEqual(architecture_metadata("gru")["layer_size"], 128)
        self.assertEqual(architecture_metadata("cnn")["loss_bias"], 0.70)
        self.assertEqual(architecture_metadata("gru")["loss_bias"], 0.75)
        self.assertEqual(architecture_metadata("gru")["n_blocks"], 1)
        with self.assertRaises(ValueError):
            normalize_model_type("dnn")

    def test_cli_defaults_to_cnn_and_strict_recall_gate(self) -> None:
        with mock.patch.object(sys, "argv", ["train_nanowakeword.py", "hey tater"]):
            args = trainer.parse_args()
        self.assertEqual(args.model_type, "cnn")
        self.assertEqual(args.quality_minimum_recall, 0.90)
        self.assertEqual(args.quality_target_recall, 0.95)
        self.assertEqual(args.quality_target_negative_rate, 0.05)
        self.assertEqual(args.quality_maximum_patience, 5)
        self.assertEqual(args.quality_max_clips, 500)
        self.assertEqual(args.quality_retries, 2)
        self.assertEqual(args.positive_tts_dir, "")

    def test_shared_tts_replaces_nano_positive_generation_and_keeps_personal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "generated_samples"
            personal = root / "personal_samples"
            negative = root / "negative_samples"
            background = root / "background"
            rir = root / "rir"
            write_wav(shared / "tts.wav", np.zeros(16000, dtype=np.int16))
            write_wav(personal / "mine.wav", np.zeros(16000, dtype=np.int16))
            write_wav(negative / "not_it.wav", np.zeros(16000, dtype=np.int16))
            argv = [
                "train_nanowakeword.py",
                "hey tater",
                "--output-root", str(root / "output"),
                "--positive-dir", str(personal),
                "--positive-tts-dir", str(shared),
                "--negative-dir", str(negative),
                "--background-dir", str(background),
                "--rir-dir", str(rir),
                "--positive-samples", "10",
                "--negative-samples", "0",
                "--validation-samples", "0",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = trainer.parse_args()
            config = trainer.make_config(args, "hey_tater", root / "output")

            self.assertFalse(any(row["name"] == "synthetic_positive" for row in config["data_generation_tasks"]))
            self.assertIn("t", config["feature_manifest"]["targets"])
            self.assertIn("tp", config["feature_manifest"]["targets"])
            self.assertEqual(
                config["feature_generation_manifest"]["shared_tts_positive_features"]["input_audio_dirs"],
                [str(shared.resolve())],
            )

    def test_runtime_patience_requires_consecutive_hits(self) -> None:
        scores = [0.10, 0.95, 0.20, 0.93, 0.91, 0.30]
        self.assertEqual(consecutive_activation_score(scores, 1), 0.95)
        self.assertEqual(consecutive_activation_score(scores, 2), 0.91)
        self.assertEqual(consecutive_activation_score(scores, 3), 0.30)

    def test_quality_retry_uses_a_disjoint_deterministic_partition(self) -> None:
        paths = [Path(f"sample-{index}.wav") for index in range(100)]
        first = deterministic_sample(paths, 20, partition=0)
        second = deterministic_sample(paths, 20, partition=1)
        self.assertEqual(first, deterministic_sample(paths, 20, partition=0))
        self.assertTrue(set(first).isdisjoint(second))

    def test_streaming_calibration_prefers_buffered_three_hit_policy(self) -> None:
        candidates = [
            {"threshold": 0.99, "patience": 2, "recall": 0.97, "negative_rate": 0.07, "passed": True},
            {"threshold": 0.98, "patience": 3, "recall": 0.98, "negative_rate": 0.05, "passed": True},
            {"threshold": 0.99, "patience": 3, "recall": 0.96, "negative_rate": 0.05, "passed": True},
            {"threshold": 0.99, "patience": 4, "recall": 0.95, "negative_rate": 0.02, "passed": True},
        ]
        selected, reason = select_streaming_policy(
            candidates,
            minimum_recall=0.90,
            maximum_negative_rate=0.10,
        )
        self.assertEqual(reason, "preferred_targets")
        self.assertEqual((selected["threshold"], selected["patience"]), (0.99, 3))

    def test_real_negatives_use_a_stricter_gate_than_synthetic_decoys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "wake"
            model_path = model_dir / "model" / "wake.onnx"
            model_path.parent.mkdir(parents=True)
            model_path.touch()
            args = argparse.Namespace(
                skip_quality_gate=False,
                quality_threshold=0.90,
                quality_patience=2,
                quality_max_clips=200,
                quality_minimum_recall=0.80,
                quality_maximum_negative_rate=0.10,
                quality_maximum_real_negative_rate=0.0,
                quality_positive_dir=[],
                quality_negative_dir=[],
            )
            config = {
                "quality_evaluation": {
                    "positive_dirs": ["synthetic-positive"],
                    "negative_dirs": ["synthetic-negative"],
                },
                "training_audio_evaluation": {
                    "positive_dirs": ["real-positive"],
                    "negative_dirs": ["real-negative"],
                },
            }
            calls: list[float] = []

            def fake_evaluate(*_args, **kwargs):
                calls.append(kwargs["maximum_negative_rate"])
                return {
                    "passed": True,
                    "failures": [],
                    "streaming_contract": {"threshold": 0.99, "patience": 3},
                    "misclassified": {
                        "missed_positive_paths": [],
                        "triggered_negative_paths": [],
                    },
                }

            with (
                mock.patch.object(trainer, "calibrate_model", side_effect=fake_evaluate),
                mock.patch.object(trainer, "evaluate_model", side_effect=fake_evaluate),
            ):
                trainer.run_quality_gate(args, config, model_dir, "wake")

            self.assertEqual(calls, [0.10, 0.0])

    def test_missing_generated_samples_reuses_existing_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(3):
                write_wav(directory / f"sample-{index}.wav", np.zeros(160, dtype=np.int16))
            self.assertEqual(trainer.missing_generated_samples(directory, 5), 2)
            self.assertEqual(trainer.missing_generated_samples(directory, 2), 0)

    def test_legacy_positive_generation_cache_is_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "positive"
            clip = directory / "sample.wav"
            write_wav(clip, np.zeros(160, dtype=np.int16))
            fingerprint = {"kind": "fixed_phrase_positive", "phrase": "hey tater"}

            self.assertFalse(
                trainer.prepare_generation_cache(directory, fingerprint, adopt_legacy=True)
            )
            self.assertTrue(clip.exists())
            self.assertEqual(
                json.loads((directory / ".tater-generation.json").read_text(encoding="utf-8")),
                fingerprint,
            )

    def test_legacy_negative_generation_cache_is_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "negative"
            clip = directory / "sample.wav"
            write_wav(clip, np.zeros(160, dtype=np.int16))
            fingerprint = {"kind": "negative", "recipe_version": 2}

            self.assertTrue(
                trainer.prepare_generation_cache(directory, fingerprint, adopt_legacy=False)
            )
            self.assertFalse(clip.exists())
            self.assertEqual(trainer.missing_generated_samples(directory, 1), 1)

    def test_speech_centered_crop_selects_late_activity(self) -> None:
        rng = np.random.default_rng(7)
        audio = rng.normal(0, 20, 48000).astype(np.int16)
        audio[30000:41000] += (np.sin(np.arange(11000) / 7) * 5000).astype(np.int16)
        cropped = trainer.speech_centered_crop(audio)
        self.assertEqual(cropped.size, trainer.PERSONAL_POSITIVE_MAX_SAMPLES)
        self.assertGreater(float(np.sqrt(np.mean(cropped.astype(np.float64) ** 2))), 1000)

    def test_long_positive_is_staged_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "positive"
            output = root / "output"
            audio = np.zeros(48000, dtype=np.int16)
            audio[24000:36000] = 4000
            original = source / "wake.wav"
            write_wav(original, audio)

            staged = trainer.prepare_positive_training_dir(source, output)
            staged_wavs = trainer.existing_wavs(staged)
            self.assertNotEqual(staged, source)
            self.assertEqual(len(staged_wavs), 1)
            with wave.open(str(staged_wavs[0]), "rb") as handle:
                self.assertEqual(handle.getnframes(), trainer.PERSONAL_POSITIVE_MAX_SAMPLES)
            first_mtime = staged_wavs[0].stat().st_mtime_ns

            self.assertEqual(trainer.prepare_positive_training_dir(source, output), staged)
            self.assertEqual(staged_wavs[0].stat().st_mtime_ns, first_mtime)
            self.assertEqual(original.stat().st_size, 48000 * 2 + 44)
            manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["max_samples"], trainer.PERSONAL_POSITIVE_MAX_SAMPLES)

    def test_alignment_background_is_non_silent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = trainer.ensure_alignment_background(Path(temporary))
            path = directory / "quiet_room_tone.wav"
            with wave.open(str(path), "rb") as handle:
                audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            self.assertEqual(audio.size, 160000)
            self.assertGreater(int(np.max(np.abs(audio))), 4)

    def test_small_personal_source_caps_batch_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_wav(source / "one.wav", np.zeros(1600, dtype=np.int16))
            config = {
                "output_dir": str(root / "output"),
                "model_name": "wake",
                "background_paths": [],
                "rir_paths": [],
            }
            trainer.add_feature_task(
                config,
                key="np",
                label="personal_negative_features",
                source_dir=source,
                feature_file="personal.npy",
                split="negative",
                augmentation_rounds=10,
                batch_weight=100,
            )
            self.assertEqual(config["batch_composition"]["np"], 10)

    def test_legacy_feature_array_is_invalidated_without_losing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            clip = source / "one.wav"
            write_wav(clip, np.zeros(1600, dtype=np.int16))
            feature = root / "output" / "wake" / "features" / "personal.npy"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b"stale")
            config = {
                "output_dir": str(root / "output"),
                "model_name": "wake",
                "background_paths": [],
                "rir_paths": [],
            }

            trainer.add_feature_task(
                config,
                key="tp",
                label="personal_features",
                source_dir=source,
                feature_file="personal.npy",
                split="target",
            )
            self.assertFalse(feature.exists())
            self.assertTrue(clip.exists())
            self.assertTrue(feature.with_name("personal.npy.tater.json").exists())

    def test_unlabelled_cuda_feature_cache_is_rebuilt_for_cpu_runtime_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_wav(source / "one.wav", np.zeros(1600, dtype=np.int16))
            feature = root / "output" / "wake" / "features" / "personal.npy"
            config = {
                "output_dir": str(root / "output"),
                "model_name": "wake",
                "background_paths": [],
                "rir_paths": [],
            }

            trainer.add_feature_task(
                config,
                key="tp",
                label="personal_features",
                source_dir=source,
                feature_file="personal.npy",
                split="target",
            )
            cache = feature.with_name("personal.npy.tater.json")
            fingerprint = json.loads(cache.read_text(encoding="utf-8"))
            fingerprint.pop("feature_device")
            cache.write_text(json.dumps(fingerprint), encoding="utf-8")
            feature.write_bytes(b"unlabelled-cuda-features")

            with mock.patch.dict(
                os.environ,
                {"NWW_FEATURE_DEVICE": "cpu", "NWW_TORCH_CUDA": "cu128"},
                clear=False,
            ):
                trainer.add_feature_task(
                    config,
                    key="tp",
                    label="personal_features",
                    source_dir=source,
                    feature_file="personal.npy",
                    split="target",
                )

            self.assertFalse(feature.exists())
            self.assertEqual(
                json.loads(cache.read_text(encoding="utf-8"))["feature_device"],
                "cpu",
            )

    def test_adversarial_probabilities_are_not_booleans(self) -> None:
        args = argparse.Namespace(custom_negative_phrase=[], phrase="hey tater")
        source = trainer.negative_text_source(args, 100)
        self.assertEqual(source["include_input_words"], 0.35)
        self.assertEqual(source["include_partial_phrase"], 0.35)
        self.assertIs(type(source["include_input_words"]), float)

    def test_custom_negative_phrases_are_normalized_and_never_include_target(self) -> None:
        args = argparse.Namespace(
            custom_negative_phrase=[" hey later ", "HEY TATER", "hey later", "okay tater"],
            phrase="hey tater",
        )
        source = trainer.negative_text_source(args, 100)
        self.assertEqual(source["type"], "from_list")
        self.assertEqual(source["phrases"], ["hey later", "okay tater"])

    def test_custom_negatives_augment_instead_of_replace_auto_adversarial(self) -> None:
        args = argparse.Namespace(
            custom_negative_phrase=["hey later", "okay tater"], phrase="hey tater"
        )
        sources = trainer.negative_text_sources(args, 100)
        self.assertEqual([(name, count) for name, count, _recipe in sources], [("auto", 70), ("custom", 30)])
        self.assertEqual(sources[0][2]["type"], "auto_adversarial")
        self.assertEqual(sources[1][2]["type"], "from_list")

    def test_uploaded_device_audio_gets_a_stable_unseen_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for index in range(10):
                write_wav(source / f"clip-{index}.wav", np.zeros(160, dtype=np.int16))

            training, holdout = trainer.partition_uploaded_audio(source, root / "output", "positive")

            self.assertIsNotNone(holdout)
            self.assertEqual(len(trainer.existing_wavs(training)), 8)
            self.assertEqual(len(trainer.existing_wavs(holdout)), 2)
            self.assertTrue(set(path.name for path in trainer.existing_wavs(training)).isdisjoint(
                path.name for path in trainer.existing_wavs(holdout)
            ))

    def test_zero_step_training_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "wake"
            model_dir.mkdir()
            (model_dir.parent / "training_journal.md").write_text(
                "| **wake** | N/A | N/A | N/A | 0.0 |\n", encoding="utf-8"
            )
            with self.assertRaises(SystemExit):
                trainer.ensure_training_completed(model_dir, "wake")

    def test_quality_failures_become_balanced_retry_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive = root / "missed.wav"
            negative = root / "triggered.wav"
            write_wav(positive, np.zeros(48000, dtype=np.int16))
            write_wav(negative, np.zeros(16000, dtype=np.int16))
            report = {
                "validation": {
                    "misclassified": {
                        "missed_positive_paths": [str(positive)],
                        "triggered_negative_paths": [str(negative)],
                    }
                },
                "additional": [],
            }
            missed, triggered = trainer.quality_misclassifications(report)
            self.assertEqual(missed, [positive.resolve()])
            self.assertEqual(triggered, [negative.resolve()])

            output = root / "output" / "wake"
            config = {
                "model_name": "wake",
                "output_dir": str(root / "output"),
                "batch_composition": {},
                "feature_manifest": {"targets": {}, "negatives": {}},
                "feature_generation_manifest": {},
            }
            retry = trainer.make_quality_retry_config(config, output, missed, triggered)
            self.assertEqual(
                retry["batch_composition"]["qrp"], trainer.PERSONAL_AUGMENTATION_ROUNDS
            )
            self.assertEqual(
                retry["batch_composition"]["qrn"], trainer.PERSONAL_AUGMENTATION_ROUNDS
            )
            self.assertIn("quality_retry_positive_features", retry["feature_generation_manifest"])
            self.assertIn("quality_retry_negative_features", retry["feature_generation_manifest"])

    def test_cnn_false_trigger_retry_raises_negative_loss_bias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            negative = root / "triggered.wav"
            write_wav(negative, np.zeros(16000, dtype=np.int16))
            config = {
                "model_name": "wake", "model_type": "cnn", "LOSS_BIAS": 0.70,
                "output_dir": str(root / "output"), "batch_composition": {},
                "feature_manifest": {"targets": {}, "negatives": {}},
                "feature_generation_manifest": {},
            }
            retry = trainer.make_quality_retry_config(
                config, root / "output" / "wake", [], [negative]
            )
            self.assertEqual(retry["LOSS_BIAS"], 0.75)

    def test_validate_existing_candidate_skips_training_and_exports_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            model_dir = output_root / "wake"
            model_path = model_dir / "model" / "wake.onnx"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"onnx")
            (model_dir / "wake.yaml").write_text(
                "model_name: wake\nmodel_type: cnn\nlayer_size: 32\nLOSS_BIAS: 0.7\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                phrase="wake", output_root=str(output_root), export_dir=str(root / "export"),
                model_type="cnn", seed=42, steps=50000, validate_existing_candidate=True,
                skip_quality_gate=False, quality_retries=2,
            )
            report = {
                "passed": True,
                "validation": {
                    "passed": True, "failures": [],
                    "streaming_contract": {"threshold": 0.99, "patience": 5},
                    "criteria": {"sample_partition": 1},
                    "positive": {"rate": 0.98, "count": 200},
                    "negative": {"rate": 0.015, "count": 200},
                },
                "additional": [], "failures": [],
            }
            with (
                mock.patch.object(trainer, "parse_args", return_value=args),
                mock.patch.object(trainer, "provision_feature_bank"),
                mock.patch.object(trainer, "run_training") as run_training,
                mock.patch.object(trainer, "run_quality_gate", return_value=report),
            ):
                self.assertEqual(trainer.main(), 0)
            run_training.assert_not_called()
            metadata = json.loads((root / "export" / "wake.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["recommended_patience"], 5)
            self.assertTrue(metadata["validated_existing_candidate"])
            self.assertTrue(metadata["quality_evaluated"])
            self.assertTrue(metadata["quality_passed"])
            self.assertEqual(metadata["quality_status"], "passed")

    def test_validate_existing_candidate_exports_with_quality_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            model_dir = output_root / "wake"
            model_path = model_dir / "model" / "wake.onnx"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"onnx")
            (model_dir / "wake.yaml").write_text(
                "model_name: wake\nmodel_type: cnn\nlayer_size: 32\nLOSS_BIAS: 0.7\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                phrase="wake", output_root=str(output_root), export_dir=str(root / "export"),
                model_type="cnn", seed=42, steps=50000, validate_existing_candidate=True,
                skip_quality_gate=False, quality_retries=2,
            )
            report = {
                "passed": False,
                "validation": {
                    "passed": False, "failures": ["recall below target"],
                    "streaming_contract": {"threshold": 0.90, "patience": 5},
                    "criteria": {"sample_partition": 1},
                    "positive": {"rate": 0.84, "count": 200},
                    "negative": {"rate": 0.01, "count": 200},
                },
                "additional": [], "failures": ["recall below target"],
            }
            with (
                mock.patch.object(trainer, "parse_args", return_value=args),
                mock.patch.object(trainer, "provision_feature_bank"),
                mock.patch.object(trainer, "run_training") as run_training,
                mock.patch.object(trainer, "run_quality_gate", return_value=report),
            ):
                self.assertEqual(trainer.main(), 0)
            run_training.assert_not_called()
            metadata = json.loads((root / "export" / "wake.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["quality_evaluated"])
            self.assertFalse(metadata["quality_passed"])
            self.assertEqual(metadata["quality_status"], "low_quality")
            self.assertEqual(metadata["quality"]["failures"], ["recall below target"])


if __name__ == "__main__":
    unittest.main()
