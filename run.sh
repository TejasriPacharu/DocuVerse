#!/usr/bin/env bash
set -e

trap 'kill 0' EXIT

cd "$(dirname "$0")"

echo "Starting backend..."
(cd server && uv run uvicorn main:app --reload) > backend.log 2>&1 &

echo "Starting frontend..."
(cd client && uv run streamlit run app.py) > frontend.log 2>&1 &

echo "Backend logs: backend.log"
echo "Frontend logs: frontend.log"
echo "Press Ctrl+C to stop both."

wait
