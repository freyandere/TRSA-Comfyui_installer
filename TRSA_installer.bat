@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: TRSA ComfyUI Installer v2.6.0
:: Auto-downloads and executes the latest installer from GitHub

title TRSA ComfyUI SageAttention Installer v2.6.0

echo ====================================================
echo  TRSA ComfyUI SageAttention Installer v2.6.0
echo ====================================================
echo.
echo Downloading latest installer from GitHub...
echo.

:: GitHub repository details
set "REPO_URL=https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
set "CORE_FILE=installer_core.py"
set "LANG_FILE=installer_core_lang.py"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in system PATH!
    echo Please ensure you are running this script in ComfyUI's python_embeded folder.
    echo.
    pause
    exit /b 1
)

:: Download installer files
echo [1/3] Downloading %CORE_FILE%...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%CORE_FILE%', '%CORE_FILE%')" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to download %CORE_FILE%
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo [2/3] Downloading %LANG_FILE%...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%LANG_FILE%', '%LANG_FILE%')" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to download %LANG_FILE%
    echo Please check your internet connection and try again.
    del "%CORE_FILE%" >nul 2>&1
    pause
    exit /b 1
)

echo [3/3] Launching installer...
echo.

:: Execute the installer
python "%CORE_FILE%"

:: Cleanup downloaded files
echo.
echo Cleaning up temporary files...
timeout /t 2 /nobreak >nul
del "%CORE_FILE%" >nul 2>&1
del "%LANG_FILE%" >nul 2>&1

echo.
echo ====================================================
echo  Installation process completed
echo ====================================================
echo.
pause

