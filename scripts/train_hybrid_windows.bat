@echo off
REM Windows wrapper for train_hybrid.py
REM This handles path separators correctly on Windows

echo ======================================================================
echo AURORA Hybrid Training - Windows Wrapper
echo ======================================================================
echo.

REM Use forward slashes (work on Windows too!)
python scripts\train_hybrid.py --datasets-dir data/open_datasets --synthetic 1000 --output models/neural_oracle_v1.pkl --metadata-file models/neural_oracle_v1.json

echo.
echo ======================================================================
echo Training complete! Check models/ directory for output.
echo ======================================================================
pause
