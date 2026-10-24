#!/usr/bin/env bash
# Start anomality: creates the virtual environment on first run, then serves http://localhost:8000
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
PORT="${PORT:-8000}"
echo "anomality running at http://localhost:$PORT"
( sleep 1.5 && open "http://localhost:$PORT" 2>/dev/null || true ) &
exec .venv/bin/python -W ignore -m uvicorn backend.app:app --port "$PORT"
