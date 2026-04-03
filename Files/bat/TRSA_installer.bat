@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================================================
:: TRSA ComfyUI Installer v2.7.0
:: Repository: https://github.com/freyandere/TRSA-Comfyui_installer
:: ============================================================================

title TRSA ComfyUI Installer v2.7.0

:: Handle --restore flag
if "%1"=="--restore" goto :restore

echo.
echo ============================================================
echo   TRSA ComfyUI Installer v2.7.0
echo ============================================================
echo.
echo Downloading latest installer from GitHub...
echo.

:: GitHub repository configuration
set "REPO_URL=https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
set "SCRIPT_FOLDER=script_files"
set "CORE_FILE=installer_core.py"
set "LANG_FILE=installer_core_lang.py"

:: Cache buster: GitHub raw CDN caches aggressively; this forces fresh downloads
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set "dt=%%a"
set "CACHE_BUSTER=%dt:~0,8%"
set "DOWNLOAD_SUFFIX=?cb=%CACHE_BUSTER%"

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

:: Clean up any leftover stale files from a previous crashed run
del "%CORE_FILE%" >nul 2>&1
del "%LANG_FILE%" >nul 2>&1

:: Download installer core files (cache-busted to defeat CDN caching)
echo [1/3] Downloading %CORE_FILE%...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_FOLDER%/%CORE_FILE%%DOWNLOAD_SUFFIX%', '%CORE_FILE%')" 2>nul
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
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_FOLDER%/%LANG_FILE%%DOWNLOAD_SUFFIX%', '%LANG_FILE%')" 2>nul
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
echo   To restore a previous state, run: TRSA_installer.bat --restore
echo.

:: Exit with installer's exit code
exit /b %INSTALLER_EXIT_CODE%

:: ============================================================================
:: Restore Mode
:: ============================================================================

:restore
echo.
echo ============================================================
echo   TRSA ComfyUI Restore
echo ============================================================
echo.
echo Restoring your previous configuration...
echo.

:: Verify Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python executable not found!
    echo.
    echo Please ensure you are running this script from:
    echo   ComfyUI\python_embeded\
    echo.
    pause
    exit /b 1
)

echo [1/3] Downloading restore script...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_FOLDER%/%CORE_FILE%%DOWNLOAD_SUFFIX%', '%CORE_FILE%')" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to download script. Cannot restore.
    pause
    exit /b 1
)
echo [SUCCESS] %CORE_FILE% downloaded

echo [2/3] Downloading language file...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_FOLDER%/%LANG_FILE%%DOWNLOAD_SUFFIX%', '%LANG_FILE%')" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to download language file. Cannot restore.
    del "%CORE_FILE%" >nul 2>&1
    pause
    exit /b 1
)
echo [SUCCESS] %LANG_FILE% downloaded

echo [3/3] Running restore...
python "%CORE_FILE%" --restore

:: Cleanup
del "%CORE_FILE%" >nul 2>&1
del "%LANG_FILE%" >nul 2>&1

echo.
echo Restore process completed.
echo.
pause
