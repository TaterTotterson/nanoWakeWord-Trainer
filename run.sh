#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${NWW_DATA_DIR:-$ROOT_DIR}"
VENV_DIR="${REC_VENV_DIR:-$DATA_DIR/.recorder-venv}"
PYTHON_BIN="${REC_PYTHON_BIN:-python3}"
HOST="${REC_HOST:-0.0.0.0}"
PORT="${REC_PORT:-8792}"
TRAIN_VENV_DIR="${NWW_VENV_DIR:-$DATA_DIR/.venv}"
OUTPUT_ROOT="${NWW_OUTPUT_ROOT:-$DATA_DIR/output}"
EXPORT_DIR="${NWW_EXPORT_DIR:-$DATA_DIR/trained_wake_words}"

PY="$VENV_DIR/bin/python"
PIN_FILE="$VENV_DIR/.ui_deps_installed"

echo "NanoWakeWord Trainer UI"
echo "ROOT: $ROOT_DIR"
echo "DATA: $DATA_DIR"
echo "VENV: $VENV_DIR"

mkdir -p \
  "$DATA_DIR" \
  "$DATA_DIR/personal_samples" \
  "$DATA_DIR/negative_samples" \
  "$DATA_DIR/background_samples" \
  "$DATA_DIR/rir_samples" \
  "$DATA_DIR/feature_banks" \
  "$DATA_DIR/captured_audio" \
  "$DATA_DIR/logs" \
  "$EXPORT_DIR" \
  "$OUTPUT_ROOT"

if [[ ! -x "$PY" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

install_ui_deps() {
  "$PY" -m pip install -r requirements-ui.txt
}

if [[ ! -f "$PIN_FILE" ]]; then
  "$PY" -m pip install -U pip setuptools wheel
  install_ui_deps
  touch "$PIN_FILE"
else
  echo "Reusing existing UI venv"
fi

UVICORN="$VENV_DIR/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  install_ui_deps
fi

export NWW_DATA_DIR="$DATA_DIR"
export NWW_OUTPUT_ROOT="$OUTPUT_ROOT"
export NWW_EXPORT_DIR="$EXPORT_DIR"
export NWW_VENV_DIR="$TRAIN_VENV_DIR"
export NWW_PERSONAL_DIR="${NWW_PERSONAL_DIR:-$DATA_DIR/personal_samples}"
export NWW_NEGATIVE_DIR="${NWW_NEGATIVE_DIR:-$DATA_DIR/negative_samples}"
export NWW_BACKGROUND_DIR="${NWW_BACKGROUND_DIR:-$DATA_DIR/background_samples}"
export NWW_RIR_DIR="${NWW_RIR_DIR:-$DATA_DIR/rir_samples}"
export NWW_FEATURE_BANK_DIR="${NWW_FEATURE_BANK_DIR:-$DATA_DIR/feature_banks}"
export NWW_CAPTURED_DIR="${NWW_CAPTURED_DIR:-$DATA_DIR/captured_audio}"
export NWW_LOG_DIR="${NWW_LOG_DIR:-$DATA_DIR/logs}"
export NWW_TRAINED_DIR="${NWW_TRAINED_DIR:-$EXPORT_DIR}"
export STATIC_DIR="${STATIC_DIR:-$ROOT_DIR/static}"

echo "Launching http://127.0.0.1:$PORT"
exec "$UVICORN" trainer_server:app --host "$HOST" --port "$PORT"
