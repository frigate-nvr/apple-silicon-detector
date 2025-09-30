#!/bin/zsh

set -euo pipefail

# Project root is one level up from macos/
PROJECT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="python3.11"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3.11 not found. Attempting to use system python3..."
  PYTHON_BIN="python3"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PIP_BIN="$VENV_DIR/bin/pip3"
PY_BIN="$VENV_DIR/bin/python3"

echo "Upgrading pip and installing requirements..."
"$PIP_BIN" install --upgrade pip
"$PIP_BIN" install -r "$PROJECT_DIR/requirements.txt"

ENDPOINT_DEFAULT="ipc:///tmp/cache/zmq_detector"
PROVIDERS_DEFAULT="CoreMLExecutionProvider CPUExecutionProvider"

echo "Launching detector with AUTO model..."
"$PY_BIN" "$PROJECT_DIR/detector/zmq_onnx_client.py" \
  --model AUTO \
  --endpoint "${ENDPOINT:-$ENDPOINT_DEFAULT}" \
  --providers "${PROVIDERS:-$PROVIDERS_DEFAULT}" \
  "$@"


