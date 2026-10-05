#!/bin/bash

echo "🚀 Starting MCP-Powered CRS..."

# Start backend
echo "▶ Starting backend..."
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "⏳ Waiting for backend..."
until curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
  sleep 1
done
echo "✅ Backend is ready!"

# Start frontend
echo "▶ Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
echo "⏳ Waiting for frontend..."
until curl -s http://localhost:3000 > /dev/null 2>&1; do
  sleep 1
done
echo "✅ Frontend is ready!"

# Open browser
echo "🌐 Opening browser..."
open http://localhost:3000

echo ""
echo "✅ MCP-Powered CRS is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://127.0.0.1:8000"
echo ""
echo "Press Ctrl+C to stop everything."

# Wait and cleanup on exit
trap "echo '🛑 Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait