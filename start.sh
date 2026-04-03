#!/bin/bash
# DeceptiVision — Start both servers
echo "Starting DeceptiVision..."
echo "FastAPI backend  → http://localhost:8000"
echo "React frontend   → http://localhost:5173"
echo ""

# Start FastAPI
cd "$(dirname "$0")"
python api.py &
BACKEND_PID=$!

# Start React
cd frontend && npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Open http://localhost:5173 in your browser."
echo "Press Ctrl+C to stop both servers."

wait
