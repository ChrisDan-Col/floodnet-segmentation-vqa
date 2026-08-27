@echo off
REM ==========================================================================
REM  FloodNet app - INSTALLER (creates a venv environment, NO Anaconda)
REM  Requires Python 3.9+ on the PATH. Double-click to install and open the app.
REM ==========================================================================
cd /d "%~dp0"

echo Creating virtual environment .venv_app ...
python -m venv .venv_app
if errorlevel 1 (
  echo.
  echo [ERROR] Python was not found on the PATH. Install Python 3.9+ and retry.
  pause
  exit /b 1
)

call ".venv_app\Scripts\activate.bat"
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CPU)...
REM  If you have an NVIDIA GPU, replace the line below with the CUDA build, e.g.:
REM  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing app dependencies...
REM  PySide6 is pinned to 6.8.3 (lowest that supports Python 3.13; built against an older Visual C++
REM  runtime than 6.11, which failed with "DLL load failed ... specified procedure could not be found").
REM  If PySide6 still fails to load, install the Microsoft Visual C++ Redistributable (vc_redist.x64.exe).
pip install -r requirements_app.txt
python -m pip install -U scikit-image
pip install peft

echo.
echo Installation complete. Opening the app...
python app.py
pause
