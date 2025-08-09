@echo off
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 >nul

REM ===== Config =====
set "REPO_URL=https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
set "SCRIPT_NAME=installer_core.py"
set "TEMP_FILE=temp_installer_core.py"

REM Optional environment knobs (uncomment to use):
REM set "ACC_LANG=en"            REM en or ru
REM set "COMFYUI_ACC_AUTO_TORCH_FIX=y"  REM auto-approve torch reinstall

title ComfyUI Accelerator (Core)

REM ===== Pre-flight: find python.exe next to this .bat =====
if not exist "python.exe" (
    echo ❌ ERROR: python.exe not found! Please place this .bat into: ComfyUI_windows_portable\python_embeded
    echo ❌ Ошибка: python.exe не найден! Поместите .bat в папку: ComfyUI_windows_portable\python_embeded
    pause
    exit /b 1
)

echo 🚀 ComfyUI Accelerator (Core)
echo 🌐 Downloading %SCRIPT_NAME% from main...

REM ===== Try PowerShell download first (safe args), fallback to python urllib =====
set "DL_OK="
for %%P in (pwsh.exe powershell.exe) do (
    where %%P >nul 2>&1
    if not errorlevel 1 (
        %%P -NoProfile -NonInteractive -Command ^
            "try { Invoke-WebRequest -Uri '%REPO_URL%/%SCRIPT_NAME%' -OutFile '%TEMP_FILE%' -ErrorAction Stop } catch { exit 1 }"
        if exist "%TEMP_FILE%" (
            set "DL_OK=1"
            goto :RUN_SCRIPT
        )
    )
)

echo ℹ️ PowerShell not available or download failed. Trying Python urllib...
.\python.exe -c "import urllib.request; urllib.request.urlretrieve('%REPO_URL%/%SCRIPT_NAME%', r'%TEMP_FILE%')" 2>nul
if exist "%TEMP_FILE%" (
    set "DL_OK=1"
) else (
    echo ❌ Failed to download %SCRIPT_NAME%
    echo ❌ Не удалось скачать %SCRIPT_NAME%
    pause
    exit /b 2
)

:RUN_SCRIPT
echo ▶ Running installer...

REM Propagate optional env if set (ACC_LANG, COMFYUI_ACC_AUTO_TORCH_FIX)
REM Execute the downloaded script; keep console output visible to the user.
.\python.exe "%TEMP_FILE%"

REM Cleanup
del "%TEMP_FILE%" >nul 2>&1

echo.
echo ✅ Finished. Press any key to exit.
pause >nul

endlocal
exit /b 0
