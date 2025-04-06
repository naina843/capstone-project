@echo off
cd /d "%~dp0"
echo Launching Streamlit App using Python module...
python -m streamlit run "Stock Forecast App.py"
pause
