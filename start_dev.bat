@echo off
echo Starting Research Agent Development Environment...

REM Start FastAPI Backend in a new window
start "FastAPI Backend" cmd /c "python -m uvicorn research_agent.web:app --host 127.0.0.1 --port 8000 --reload & pause"

REM Start Vite React Frontend in a new window
start "Vite Frontend" cmd /c "cd frontend && npm run dev & pause"

echo Both services are starting in separate windows.
echo - Backend available at: http://127.0.0.1:8000
echo - Frontend available at: http://localhost:5173
echo.
echo Press any key to exit this launcher...
pause >nul
