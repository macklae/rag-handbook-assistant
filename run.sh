#!/usr/bin/env bash
set -e
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add your OPENAI_API_KEY, then run this again."
  exit 1
fi
uvicorn app.main:app --reload --port 8000
