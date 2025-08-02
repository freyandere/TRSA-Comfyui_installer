@echo off
chcp 65001 >nul

set "REPO_URL=https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/dev"

title ComfyUI Accelerator v3.0

if not exist "python.exe" (
    echo ❌ ERROR: python.exe not found! Please place the .bat file to path: ComfyUI_windows_portable\python_embeded
    echo ❌ Ошибка: python.exe не найден! Пожалуйста поместите данный .bat файл по этому пути: ComfyUI_windows_portable\python_embeded
    pause
    exit /b 1
)

echo 🚀 ComfyUI Accelerator v3.0
echo Loading from repository...

.\python.exe -c "import urllib.request,os; os.environ['COMFYUI_ACC_REPO']='%REPO_URL%'; urllib.request.urlretrieve('%REPO_URL%/core_app.py', 'temp_app.py'); exec(open('temp_app.py', encoding='utf-8').read())"

del temp_app.py >nul 2>&1

pause
