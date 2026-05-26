#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [ ! -d "api/.venv" ]; then
  python3 -m venv api/.venv
  api/.venv/bin/pip install -r api/requirements.txt
fi

exec api/.venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
