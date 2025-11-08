@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================================================
:: TRSA ComfyUI SageAttention Installer v2.6.0
:: Repository: https://github.com/freyandere/TRSA-Comfyui_installer
:: ============================================================================

title TRSA ComfyUI SageAttention Installer v2.6.0

echo.
echo ============================================================
echo   TRSA ComfyUI SageAttention Installer v2.6.0
echo ============================================================
echo.
echo Downloading latest installer from GitHub...
echo.

:: GitHub repository configuration
set "REPO_URL=https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
set "SCRIPT_FOLDER=script_files"
set "CORE_FILE=installer_core.py"
set "LANG_FILE=installer_core_lang.py"

:: Verify Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python executable not found!
    echo.
    echo Please ensure you are running this script from:
    echo   ComfyUI\python_embeded\
    echo.
    echo The portable ComfyUI installation includes Python in this folder.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python detected successfully
echo.

:: Download installer core files
echo [1/3] Downloading %CORE_FILE%...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_FOLDER%/%CORE_FILE%', '%CORE_FILE%')" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to download %CORE_FILE%
    echo.
    echo Possible reasons:
    echo   - No internet connection
    echo   - GitHub is unavailable
    echo   - Firewall blocking the request
    echo.
    echo Please check your connection and try again.
    echo.
    pause
    exit /b 1
)
echo [SUCCESS] %CORE_FILE% downloaded

echo [2/3] Downloading %LANG_FILE%...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_FOLDER%/%LANG_FILE%', '%LANG_FILE%')" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to download %LANG_FILE%
    echo.
    echo Cleaning up partial downloads...
    del "%CORE_FILE%" >nul 2>&1
    pause
    exit /b 1
)
echo [SUCCESS] %LANG_FILE% downloaded

echo [3/3] Launching installer...
echo.
echo ============================================================
echo.

:: Execute the Python installer
python "%CORE_FILE%"
set "INSTALLER_EXIT_CODE=%ERRORLEVEL%"

:: Cleanup temporary files
echo.
echo ============================================================
echo   Cleaning up temporary files...
echo ============================================================
timeout /t 2 /nobreak >nul
del "%CORE_FILE%" >nul 2>&1
del "%LANG_FILE%" >nul 2>&1

echo.
echo ============================================================
echo   Installation process completed
echo ============================================================
echo.

:: Exit with installer's exit code
exit /b %INSTALLER_EXIT_CODE%
