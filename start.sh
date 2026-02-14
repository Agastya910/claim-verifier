#!/bin/bash

# Start FastAPI backend in background on port 8000
echo "Starting FastAPI backend..."
uv run uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 &

# Wait a moment for backend to start
sleep 30

# Start Streamlit UI on the Render port (usually 10000 or $PORT)
echo "Starting Streamlit UI..."
uv run streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
