#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./train_nanowakeword.sh "hey tater" [options]

Options are passed through to scripts/train_nanowakeword.py:
  --steps 20000
  --positive-samples 2000
  --negative-samples 2000
  --validation-samples 400
  --model-type dnn|lstm|tc-resnet
  --layer-size 32
  --num-workers 4
  --custom-negative-phrase "phrase"

Environment:
  NWW_DATA_DIR=/data                  persistent Docker data root
  NWW_TORCH_CUDA=cu124               install CUDA PyTorch wheels before training
  NWW_FORCE_CPU=1                    disable CUDA PyTorch wheel install
  NWW_PYTHON_BIN=/path/to/python3.11  override Python
EOF
  exit 1
fi

DATA_ROOT="${NWW_DATA_DIR:-$ROOT_DIR}"
OUTPUT_ROOT="${NWW_OUTPUT_ROOT:-$DATA_ROOT/output}"
EXPORT_DIR="${NWW_EXPORT_DIR:-${NWW_TRAINED_DIR:-$DATA_ROOT/trained_wake_words}}"
POSITIVE_DIR="${NWW_PERSONAL_DIR:-$DATA_ROOT/personal_samples}"
NEGATIVE_DIR="${NWW_NEGATIVE_DIR:-$DATA_ROOT/negative_samples}"
BACKGROUND_DIR="${NWW_BACKGROUND_DIR:-$DATA_ROOT/background_samples}"
RIR_DIR="${NWW_RIR_DIR:-$DATA_ROOT/rir_samples}"
VENV_DIR="${NWW_VENV_DIR:-$DATA_ROOT/.venv}"

if [[ -n "${NWW_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$NWW_PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.10)"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

PY="$VENV_DIR/bin/python"
TRAIN_DEPS_KEY="cpu"
if [[ -n "${NWW_TORCH_CUDA:-}" && "${NWW_FORCE_CPU:-0}" != "1" ]]; then
  TRAIN_DEPS_KEY="${NWW_TORCH_VERSION:-2.6.0}+${NWW_TORCH_CUDA}"
fi

mkdir -p "$OUTPUT_ROOT" "$EXPORT_DIR" "$POSITIVE_DIR" "$NEGATIVE_DIR" "$BACKGROUND_DIR" "$RIR_DIR" "$DATA_ROOT/logs"

if [[ ! -x "$PY" ]]; then
  echo "Creating training venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

install_cuda_torch() {
  local torch_version torch_cuda
  torch_version="${NWW_TORCH_VERSION:-2.6.0}"
  torch_cuda="${NWW_TORCH_CUDA:-}"
  if [[ -z "$torch_cuda" || "${NWW_FORCE_CPU:-0}" == "1" ]]; then
    return
  fi
  echo "Installing CUDA PyTorch wheels: torch ${torch_version}+${torch_cuda}"
  "$PY" -m pip install --index-url "https://download.pytorch.org/whl/${torch_cuda}" \
    "torch==${torch_version}+${torch_cuda}" \
    "torchaudio==${torch_version}+${torch_cuda}"
}

write_filtered_train_requirements() {
  local filtered="$VENV_DIR/requirements-train.no-torch.txt"
  "$PY" - "$ROOT_DIR/requirements-train.txt" "$filtered" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
skip = {"torch", "torchaudio"}
lines = []
for line in source.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
    if match and match.group(1).lower() in skip:
        continue
    lines.append(line)
dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(dest)
PY
}

if [[ ! -f "$VENV_DIR/.train_deps_installed" || "$(cat "$VENV_DIR/.train_deps_key" 2>/dev/null || true)" != "$TRAIN_DEPS_KEY" ]]; then
  "$PY" -m pip install -U pip setuptools wheel
  if [[ -n "${NWW_TORCH_CUDA:-}" && "${NWW_FORCE_CPU:-0}" != "1" ]]; then
    install_cuda_torch
    FILTERED_REQUIREMENTS="$(write_filtered_train_requirements)"
    "$PY" -m pip install -r "$FILTERED_REQUIREMENTS"
  else
    "$PY" -m pip install -r requirements-train.txt
  fi
  touch "$VENV_DIR/.train_deps_installed"
  printf "%s\n" "$TRAIN_DEPS_KEY" > "$VENV_DIR/.train_deps_key"
else
  echo "Reusing existing training venv"
fi

export NWW_OUTPUT_ROOT="$OUTPUT_ROOT"
export NWW_EXPORT_DIR="$EXPORT_DIR"
export NWW_PERSONAL_DIR="$POSITIVE_DIR"
export NWW_NEGATIVE_DIR="$NEGATIVE_DIR"
export NWW_BACKGROUND_DIR="$BACKGROUND_DIR"
export NWW_RIR_DIR="$RIR_DIR"

"$PY" scripts/train_nanowakeword.py "$@"
