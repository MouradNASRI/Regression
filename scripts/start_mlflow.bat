@echo off

REM Move to project root (parent directory of this script)
cd /d "%~dp0\.."

echo ==============================
echo Starting MLflow Server...
echo ==============================

mlflow server ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root ./mlruns ^
  --host 127.0.0.1 ^
  --port 5000

pause
