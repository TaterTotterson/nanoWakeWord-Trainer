from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nanowakeword_cli  # noqa: E402


class TrainingDependencyTests(unittest.TestCase):
    def test_cuda_install_matches_nanowakeword_torch_requirement(self) -> None:
        launcher = (ROOT / "train_nanowakeword.sh").read_text(encoding="utf-8")

        self.assertIn('NWW_TORCH_VERSION:-2.8.0', launcher)
        self.assertIn('requirements-train.torch-constraints.txt', launcher)
        self.assertIn('pip install -c "$TORCH_CONSTRAINTS"', launcher)
        self.assertIn('NWW_ONNX_RUNTIME_VERSION:-1.22.0', launcher)

    def test_feature_provider_defaults_to_runtime_compatible_cpu(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(nanowakeword_cli.requested_feature_device(), "cpu")

    def test_feature_provider_can_be_overridden(self) -> None:
        with mock.patch.dict(os.environ, {"NWW_FEATURE_DEVICE": "auto"}, clear=True):
            expected = "gpu" if nanowakeword_cli.torch.cuda.is_available() else "cpu"
            self.assertEqual(nanowakeword_cli.requested_feature_device(), expected)

        with mock.patch.dict(os.environ, {"NWW_FEATURE_DEVICE": "invalid"}, clear=True):
            with self.assertRaisesRegex(SystemExit, "Unsupported NWW_FEATURE_DEVICE"):
                nanowakeword_cli.requested_feature_device()


if __name__ == "__main__":
    unittest.main()
