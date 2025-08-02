@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ========================================
:: ComfyUI Accelerator v3.0 - Launcher
:: Modern portable architecture
:: ========================================

set "APP_NAME=ComfyUI Accelerator"
set "APP_VERSION=v3.0"
set "REPO_URL=https://raw.githubusercontent.com/your-username/comfyui-accelerator/main"

title %APP_NAME% %APP_VERSION%

:: Check python.exe
if not exist "python.exe" (
    echo.
    echo ❌ ERROR: python.exe not found!
    echo.
    echo 📍 Place launcher.bat in folder with python.exe
    echo    Usually: ComfyUI_windows_portable\python_embedded\
    echo.
    pause
    exit /b 1
)

:: Test python
.\python.exe --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python not working!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   🚀 %APP_NAME% %APP_VERSION%
echo   Loading modules from repository...
echo ================================================================

:: Create bootstrap loader
set "BOOTSTRAP=temp_bootstrap_%RANDOM%.py"

(
echo # -*- coding: utf-8 -*-
echo """ComfyUI Accelerator v3.0 Bootstrap Loader"""
echo.
echo import sys
echo import os
echo import urllib.request
echo import tempfile
echo import atexit
echo.
echo class BootstrapLoader:
echo     def __init__^(self^):
echo         self.temp_files = []
echo         atexit.register^(self.cleanup^)
echo.
echo     def load_and_run^(self^):
echo         try:
echo             print^("🌐 Loading core application..."^)
echo             
echo             # Create config for main app
echo             os.environ['COMFYUI_ACC_REPO'] = '%REPO_URL%'
echo             
echo             # Download and execute core_app.py
echo             app_url = '%REPO_URL%/core_app.py'
echo             temp_app = 'temp_core_app.py'
echo             
echo             urllib.request.urlretrieve^(app_url, temp_app^)
echo             self.temp_files.append^(temp_app^)
echo             
echo             print^("✅ Core application loaded"^)
echo             print^(^)
echo             
echo             # Execute main application
echo             exec^(open^(temp_app^).read^(^)^)
echo             
echo         except Exception as e:
echo             print^(f"❌ Bootstrap failed: {e}"^)
echo             input^("Press Enter to exit..."^)
echo             sys.exit^(1^)
echo.
echo     def cleanup^(self^):
echo         for f in self.temp_files:
echo             try:
echo                 os.remove^(f^)
echo             except:
echo                 pass
echo.
echo if __name__ == '__main__':
echo     loader = BootstrapLoader^(^)
echo     loader.load_and_run^(^)
) > %BOOTSTRAP%

:: Run bootstrap
.\python.exe %BOOTSTRAP%

:: Cleanup
if exist "%BOOTSTRAP%" del "%BOOTSTRAP%" >nul 2>&1

:: Final cleanup of any remaining temp files
for %%f in (temp_*.py) do del "%%f" >nul 2>&1

echo.
echo ✅ %APP_NAME% session completed
if not "%1"=="silent" pause
exit /b 0
