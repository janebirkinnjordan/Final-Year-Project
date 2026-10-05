@echo off
echo Starting MCP-Powered CRS...

:: Start backend
echo Starting backend...
cd backend
start /B cmd /C "call .venv\Scripts\activate && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
cd ..

:: Wait for backend
echo Waiting for backend...
:WAIT_BACKEND
curl -s http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
  %SystemRoot%\System32\timeout.exe /t 1 /nobreak >nul
  goto WAIT_BACKEND
)
echo Backend is ready!

:: Start frontend
echo Starting frontend...
cd frontend
start /B cmd /C "npm run dev"
cd ..

:: Wait for frontend
echo Waiting for frontend...
:WAIT_FRONTEND
curl -s http://localhost:3000 >nul 2>&1
if errorlevel 1 (
  %SystemRoot%\System32\timeout.exe /t 1 /nobreak >nul
  goto WAIT_FRONTEND
)
echo Frontend is ready!

:: Open browser
echo Opening browser...
start http://localhost:3000

echo.
echo MCP-Powered CRS is running!
echo    Frontend: http://localhost:3000
echo    Backend:  http://127.0.0.1:8000
echo.
echo Press Ctrl+C to stop everything.
pause
