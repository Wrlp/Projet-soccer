#!/usr/bin/env bash
# Lance l'API (8000) et le front (5173) en une commande
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

cleanup() {
  trap - INT TERM EXIT
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "$FRONT_PID" ] && kill "$FRONT_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM EXIT

# API
if [ ! -d "api/.venv" ]; then
  echo "→ Création venv API..."
  python3 -m venv api/.venv
fi
echo "→ Dépendances API..."
api/.venv/bin/pip install -q -r api/requirements.txt
echo "→ Modèles (Git LFS)..."
bash scripts/materialize-models.sh
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "→ API http://localhost:8000"
api/.venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Front
if [ ! -d "front/node_modules" ]; then
  echo "→ npm install (front)..."
  (cd front && npm install)
fi
echo "→ Front http://localhost:5173"
(cd front && npm run dev) &
FRONT_PID=$!

echo ""
echo "SportInsight — Ctrl+C pour arrêter"
echo "  Front : http://localhost:5173"
echo "  API   : http://localhost:8000/docs"
wait
