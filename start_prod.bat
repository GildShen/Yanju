@echo off
echo Building frontend and starting Research Agent Production Environment...

REM Build the Vite frontend
echo [1/2] Building frontend...
cd frontend
call npm run build
cd ..

REM Start FastAPI Backend (which now serves the built frontend)
echo [2/2] Starting backend service...
echo The application will be available at: http://127.0.0.1:8000
python -m uvicorn research_agent.web:app --host 127.0.0.1 --port 8000

pause
