@echo off
echo 🧹 Emergency cleanup for ComfyUI Accelerator...

:: Kill any running Python processes (осторожно!)
:: taskkill /f /im python.exe 2>nul

:: Clean temp files
del temp_*.py 2>nul
del *_temp.whl 2>nul
del *.whl 2>nul
del cleanup_*.tmp 2>nul

:: Clean scheduled tasks
schtasks /query /tn "ComfyUI_Cleanup_*" >nul 2>&1 && (
    echo Removing scheduled cleanup tasks...
    schtasks /delete /tn "ComfyUI_Cleanup_*" /f >nul 2>&1
)

echo ✅ Cleanup completed!