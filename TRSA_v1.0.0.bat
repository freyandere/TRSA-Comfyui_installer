chcp 65001 >nul

@echo off
setlocal enabledelayedexpansion
title Frey's Triton and Sage Attention 2++ Installer, additional libs auto-install

echo ========================================
echo Frey's Triton and Sage Attention 2++ Installer,
echo additional libs auto-install
echo ========================================
echo.

:: Check if we're in the python_embeded directory
if not exist "python.exe" (
    echo ERROR: python.exe not found in current directory!
    echo Please place this script in the python_embeded folder inside your ComfyUI portable directory.
    echo Expected location: ComfyUI_windows_portable\python_embeded\
    echo.
    pause
    exit /b 1
)

:: Verify we can run python
.\python.exe --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Cannot execute python.exe!
    echo Please verify your ComfyUI portable installation.
    echo.
    pause
    exit /b 1
)

:: Check Python version
echo Detecting Python version...
for /f "tokens=2" %%i in ('.\python.exe --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

:MENU
cls
echo ========================================
echo Frey's Triton and Sage Attention 2++ Installer,
echo additional libs auto-install
echo ========================================
echo.
echo Current location: %CD%
echo Python version: %PYTHON_VERSION%
echo.
echo 1. Check System Compatibility
echo 2. Upgrade pip
echo 3. Install Triton (Standard)
echo 4. Install Triton (Pre-release 3.3)
echo 5. Install Sage Attention 2++
echo 6. Auto-download and setup include/libs folders
echo 7. Manual setup include/libs folders
echo 8. Install Teacache (Optional)
echo 9. Force Reinstall All Components
echo 0. Verify Installation
echo E. Exit
echo.
set /p choice="Enter your choice: "

if "%choice%"=="1" goto CHECK_SYSTEM
if "%choice%"=="2" goto UPGRADE_PIP
if "%choice%"=="3" goto INSTALL_TRITON
if "%choice%"=="4" goto INSTALL_TRITON_PRE
if "%choice%"=="5" goto INSTALL_SAGE
if "%choice%"=="6" goto AUTO_SETUP_FOLDERS
if "%choice%"=="7" goto MANUAL_SETUP_FOLDERS
if "%choice%"=="8" goto INSTALL_TEACACHE
if "%choice%"=="9" goto FORCE_REINSTALL
if "%choice%"=="0" goto VERIFY
if /i "%choice%"=="E" goto EXIT
goto MENU

:CHECK_SYSTEM
cls
echo ========================================
echo  System Compatibility Check
echo  CUDA 12.8 + PyTorch 2.7.1 Required
echo ========================================
echo.

echo Checking Python version...
.\python -c "import platform; print('Python:', platform.python_version()); print('Architecture:', platform.architecture()[0])"
echo.

echo Checking CUDA version via nvidia-smi...
nvidia-smi --version 2>nul
if %errorlevel% neq 0 (
    echo ⚠ nvidia-smi not available - trying alternative method
) else (
    echo CUDA Driver Information:
    nvidia-smi | findstr "CUDA Version"
)
echo.

echo Checking PyTorch installation and versions...

:: Create a comprehensive version check script
echo import sys > temp_version_check.py
echo try: >> temp_version_check.py
echo     import torch >> temp_version_check.py
echo     print('✓ PyTorch installed') >> temp_version_check.py
echo     print('PyTorch version:', torch.__version__) >> temp_version_check.py
echo     print('PyTorch CUDA version:', torch.version.cuda) >> temp_version_check.py
echo     print('CUDA available:', torch.cuda.is_available()) >> temp_version_check.py
echo     if torch.cuda.is_available(): >> temp_version_check.py
echo         print('GPU:', torch.cuda.get_device_name(0)) >> temp_version_check.py
echo         print('GPU Compute Capability:', torch.cuda.get_device_capability(0)) >> temp_version_check.py
echo     else: >> temp_version_check.py
echo         print('⚠ CUDA not available in PyTorch') >> temp_version_check.py
echo     # Check for specific version requirements >> temp_version_check.py
echo     torch_version = torch.__version__ >> temp_version_check.py
echo     cuda_version = torch.version.cuda >> temp_version_check.py
echo     print() >> temp_version_check.py
echo     print('=== SageAttention Compatibility Check ===') >> temp_version_check.py
echo     if torch_version.startswith('2.7.'): >> temp_version_check.py
echo         print('✓ PyTorch 2.7.x detected - Compatible with SageAttention') >> temp_version_check.py
echo     else: >> temp_version_check.py
echo         print('⚠ PyTorch version', torch_version, '- SageAttention requires 2.7.1+') >> temp_version_check.py
echo     if cuda_version and '12.8' in cuda_version: >> temp_version_check.py
echo         print('✓ CUDA 12.8 detected - Compatible with SageAttention') >> temp_version_check.py
echo     else: >> temp_version_check.py
echo         print('⚠ CUDA version', cuda_version, '- SageAttention requires 12.8') >> temp_version_check.py
echo except ImportError: >> temp_version_check.py
echo     print('✗ PyTorch not installed') >> temp_version_check.py
echo     print('SageAttention requires PyTorch 2.7.1+ with CUDA 12.8') >> temp_version_check.py
echo except Exception as e: >> temp_version_check.py
echo     print('Error checking PyTorch:', str(e)) >> temp_version_check.py

.\python temp_version_check.py
del temp_version_check.py >nul 2>&1

echo.
echo ========================================
echo  GPU Architecture Compatibility
echo ========================================
echo **RTX 50xx (Blackwell)**: Requires CUDA 12.8 + PyTorch 2.7.1+
echo **RTX 40xx (Ada)**: Fully supported with CUDA 12.8 + PyTorch 2.7.1+
echo **RTX 30xx (Ampere)**: Fully supported with CUDA 12.8 + PyTorch 2.7.1+
echo **RTX 20xx and older**: Limited compatibility
echo.
echo ========================================
echo  Version Requirements Summary
echo ========================================
echo **Required for SageAttention wheel:**
echo - CUDA: 12.8 (cu128)
echo - PyTorch: 2.7.1+
echo - Python: 3.10+ recommended
echo.
echo **Installation Commands for Required Versions:**
echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo pip install --pre torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
echo.
echo **Note**: PyTorch 2.7+ with CUDA 12.8 adds Blackwell GPU support
echo **Your current Python version**: %PYTHON_VERSION%
echo.
pause
goto MENU



:UPGRADE_PIP
cls
echo Upgrading pip...
.\python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo.
    echo WARNING: pip upgrade failed. Continuing anyway...
) else (
    echo.
    echo pip upgraded successfully!
)
echo.
pause
goto MENU

:INSTALL_TRITON
cls
echo Installing Triton (Standard)...
echo.
.\python -m pip install -U triton-windows
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Triton installation failed!
    echo.
    echo This might be due to:
    echo - Package availability issues on PyPI
    echo - Network connectivity problems
    echo.
    echo **Alternative:** Try option 4 for pre-release version
    echo **Manual:** Download triton wheel from official source
) else (
    echo.
    echo Triton installed successfully!
)
echo.
pause
goto MENU

:INSTALL_TRITON_PRE
cls
echo Installing Triton (Pre-release 3.3)...
echo.
.\python -m pip install -U --pre triton-windows
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Triton pre-release installation failed!
    echo Check your internet connection and try again.
) else (
    echo.
    echo Triton pre-release installed successfully!
)
echo.
pause
goto MENU

:INSTALL_SAGE
cls
echo ========================================
echo  Installing Sage Attention 2.2.0
echo  CUDA 12.8 + PyTorch 2.7.1 Build
echo ========================================
echo.
echo **Downloading SageAttention 2.2.0 wheel file...**
echo Repository: https://github.com/freyandere/TRSA-Comfyui_installer
echo File: sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl
echo.

:: Download the specific wheel file
echo Downloading SageAttention wheel file...
powershell -Command "try { Invoke-WebRequest -Uri 'https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0%%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl' -OutFile 'sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl' -ErrorAction Stop; Write-Host 'Download completed successfully' } catch { Write-Host 'Download failed: ' $_.Exception.Message; exit 1 }"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Download failed!
    echo.
    echo **Fallback options:**
    echo 1. Check your internet connection
    echo 2. Try manual download from: https://github.com/freyandere/TRSA-Comfyui_installer
    echo 3. Download the .whl file and place it in this directory
    echo 4. Run this option again
    echo.
    goto INSTALL_SAGE_ERROR
)

:: Verify the file was downloaded
if not exist "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl" (
    echo ERROR: Downloaded wheel file not found!
    goto INSTALL_SAGE_ERROR
)

echo.
echo **Installing SageAttention 2.2.0 from wheel file...**
.\python.exe -m pip install sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl

if %errorlevel% neq 0 (
    echo.
    echo **Installation failed. Trying with force reinstall...**
    .\python.exe -m pip install --force-reinstall sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl
    
    if !errorlevel! neq 0 (
        echo.
        echo ERROR: SageAttention installation failed!
        echo.
        echo **Troubleshooting steps:**
        echo 1. Make sure Triton is installed first (options 3 or 4)
        echo 2. Verify PyTorch 2.7.1 + CUDA 12.8 is installed (option 3)
        echo 3. Check that include/libs folders are present (option 6)
        echo 4. Restart as administrator if permission errors occur
        echo.
        goto INSTALL_SAGE_CLEANUP
    ) else (
        echo.
        echo ✓ SageAttention 2.2.0 installed successfully (force reinstall)!
        echo **Build**: CUDA 12.8 + PyTorch 2.7.1
        echo **Architecture**: cp39-abi3 (Python 3.9+ compatible)
    )
) else (
    echo.
    echo ✓ SageAttention 2.2.0 installed successfully!
    echo **Build**: CUDA 12.8 + PyTorch 2.7.1  
    echo **Architecture**: cp39-abi3 (Python 3.9+ compatible)
)

echo.
echo **Verifying installation...**
.\python -c "try: import sageattention; print('✓ SageAttention import successful'); print('Version:', sageattention.__version__ if hasattr(sageattention, '__version__') else 'Version info not available'); except Exception as e: print('✗ Import failed:', str(e))"

goto INSTALL_SAGE_CLEANUP

:INSTALL_SAGE_ERROR
echo.
echo **Manual Installation Instructions:**
echo 1. Download manually: https://github.com/freyandere/TRSA-Comfyui_installer/blob/main/sageattention-2.2.0%%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl
echo 2. Place the .whl file in this directory: %CD%
echo 3. Run this option again
echo.
pause
goto MENU

:INSTALL_SAGE_CLEANUP
echo.
echo **Cleaning up downloaded wheel file...**
del "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl" 2>nul

echo.
echo **Important Notes:**
echo - This build requires CUDA 12.8 and PyTorch 2.7.1+
echo - Compatible with RTX 50xx (Blackwell) architecture  
echo - Optimized for video generation workflows (WAN2.1, Hunyuan, etc.)
echo - Remember to restart ComfyUI to use SageAttention
echo.
pause
goto MENU


:AUTO_SETUP_FOLDERS
cls
echo ========================================
echo  Auto-downloading include/libs folders
echo  from TRSA-Comfyui_installer repository
echo ========================================
echo.
echo Repository: https://github.com/freyandere/TRSA-Comfyui_installer
echo File: python_3.12.7_include_libs.zip
echo.
echo **Warning**: This will download and extract files to current directory
echo Current directory: %CD%
echo.
set /p confirm="Continue with automatic download? (y/n): "
if /i not "%confirm%"=="y" goto MENU

echo.
echo ========================================
echo  Method 1: PowerShell Download (Primary)
echo ========================================

:: Create a PowerShell script file instead of inline command
echo Write-Host "Starting download..." > download_script.ps1
echo try { >> download_script.ps1
echo     $ProgressPreference = 'SilentlyContinue' >> download_script.ps1
echo     Invoke-WebRequest -Uri 'https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip' -OutFile 'python_3.12.7_include_libs.zip' -UseBasicParsing >> download_script.ps1
echo     Write-Host "Download completed successfully" >> download_script.ps1
echo     exit 0 >> download_script.ps1
echo } catch { >> download_script.ps1
echo     Write-Host "PowerShell download failed: $_" >> download_script.ps1
echo     exit 1 >> download_script.ps1
echo } >> download_script.ps1

:: Execute the PowerShell script
powershell -ExecutionPolicy Bypass -File download_script.ps1
set DOWNLOAD_RESULT=%errorlevel%

:: Clean up the script file
del download_script.ps1 2>nul

:: Check if download was successful
if %DOWNLOAD_RESULT% neq 0 goto DOWNLOAD_METHOD2

if not exist "python_3.12.7_include_libs.zip" goto DOWNLOAD_METHOD2

:: Verify file size
for %%A in ("python_3.12.7_include_libs.zip") do set filesize=%%~zA
if %filesize% LSS 1000 goto DOWNLOAD_METHOD2

echo ✓ PowerShell download successful (size: %filesize% bytes)
goto EXTRACT_FILES

:DOWNLOAD_METHOD2
echo.
echo ========================================
echo  Method 2: Alternative Download (Fallback)
echo ========================================
echo PowerShell method failed, trying Python method...

.\python -c "import urllib.request; print('Downloading...'); urllib.request.urlretrieve('https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip', 'python_3.12.7_include_libs.zip'); print('Download complete')" 2>nul

if exist "python_3.12.7_include_libs.zip" (
    for %%A in ("python_3.12.7_include_libs.zip") do set filesize=%%~zA
    if !filesize! GTR 1000 (
        echo ✓ Python download successful (size: !filesize! bytes)
        goto EXTRACT_FILES
    )
)

echo.
echo ❌ All download methods failed!
goto AUTO_SETUP_ERROR

:EXTRACT_FILES
echo.
echo ========================================
echo  Extracting Files
echo ========================================

:: Fixed PowerShell extraction script with correct syntax
echo Write-Host "Starting extraction..." > extract_script.ps1
echo try { >> extract_script.ps1
echo     Add-Type -AssemblyName System.IO.Compression.FileSystem >> extract_script.ps1
echo     [System.IO.Compression.ZipFile]::ExtractToDirectory('python_3.12.7_include_libs.zip', '.') >> extract_script.ps1
echo     Write-Host "Extraction completed successfully" >> extract_script.ps1
echo     exit 0 >> extract_script.ps1
echo } catch { >> extract_script.ps1
echo     Write-Host "PowerShell extraction failed: $_" >> extract_script.ps1
echo     exit 1 >> extract_script.ps1
echo } >> extract_script.ps1

:: Execute extraction
powershell -ExecutionPolicy Bypass -File extract_script.ps1
set EXTRACT_RESULT=%errorlevel%

:: Clean up script
del extract_script.ps1 2>nul

:: If PowerShell extraction failed, try Python
if %EXTRACT_RESULT% neq 0 (
    echo PowerShell extraction failed, trying Python method...
    .\python -c "import zipfile; zipfile.ZipFile('python_3.12.7_include_libs.zip').extractall('.'); print('Python extraction completed')" 2>nul
    set EXTRACT_RESULT=!errorlevel!
)

:: Wait for file system to catch up
timeout /t 2 /nobreak >nul

:: Verify extraction results
if not exist "include" (
    if not exist "libs" (
        echo.
        echo ❌ Extraction failed - no folders found!
        goto AUTO_SETUP_ERROR
    )
)

echo.
echo ========================================
echo  Verification and Cleanup
echo ========================================

:: Clean up downloaded file
echo Cleaning up downloaded zip file...
del "python_3.12.7_include_libs.zip" 2>nul

:: Initialize success flag
set SETUP_SUCCESS=1

:: Detailed verification
if exist "include" (
    echo ✓ include folder extracted successfully
    dir "include" /a /b 2>nul | find /c /v "" > temp_include_count.txt 2>nul
    if exist temp_include_count.txt (
        set /p include_count=<temp_include_count.txt
        del temp_include_count.txt
        echo   - Contains !include_count! items
    )
) else (
    echo ❌ include folder missing after extraction
    set SETUP_SUCCESS=0
)

if exist "libs" (
    echo ✓ libs folder extracted successfully
    dir "libs" /a /b 2>nul | find /c /v "" > temp_libs_count.txt 2>nul
    if exist temp_libs_count.txt (
        set /p libs_count=<temp_libs_count.txt
        del temp_libs_count.txt
        echo   - Contains !libs_count! items
    )
) else (
    echo ❌ libs folder missing after extraction
    set SETUP_SUCCESS=0
)

:: Final success determination
echo.
if %SETUP_SUCCESS% equ 1 (
    echo ========================================
    echo  ✅ SUCCESS: Auto-setup completed!
    echo ========================================
    echo The include and libs folders have been extracted to:
    echo %CD%
    echo.
    echo **Folder Structure Verified:**
    echo ├── include\ ^(%include_count% items^) - C++ headers for Triton
    echo └── libs\ ^(%libs_count% items^) - Library files for linking
    echo.
    echo **Next steps:**
    echo 1. Install Triton ^(options 3 or 4^)
    echo 2. Install Sage Attention ^(option 5^)
    echo 3. Verify installation ^(option 0^)
) else (
    echo ========================================
    echo  ⚠️ PARTIAL SUCCESS: Setup incomplete
    echo ========================================
    echo Some folders may be missing. Please verify manually.
    echo Try manual installation ^(option 7^) if needed.
)

echo.
echo Press any key to continue...
pause >nul
goto MENU

:AUTO_SETUP_ERROR
echo.
echo ========================================
echo  ❌ Automatic Installation Failed
echo ========================================
echo.
echo **Manual Installation Instructions:**
echo.
echo 1. Visit: https://github.com/freyandere/TRSA-Comfyui_installer
echo 2. Download: python_3.12.7_include_libs.zip
echo 3. Extract to this folder: %CD%
echo.
echo **Expected Result:**
echo ├── include\ ^(C++ headers^)
echo └── libs\ ^(library files^)
echo.
echo Press any key to return to menu...
pause >nul
goto MENU


:MANUAL_SETUP_FOLDERS
cls
echo ========================================
echo  Manual Setup: include and libs folders
echo ========================================
echo.
echo **Manual Download Instructions:**
echo.
echo 1. Visit: https://github.com/freyandere/TRSA-Comfyui_installer
echo 2. Download: python_3.12.7_include_libs.zip
echo 3. Extract the contents to this directory: %CD%
echo.
echo **Expected structure after extraction:**
echo   %CD%\include\
echo   %CD%\libs\
echo.
echo **Alternative sources for different Python versions:**
echo - Check the original guide for compatible files
echo - Look for HuggingFace repositories with include/libs folders
echo.
echo Checking current folder structure...
if exist "include" (
    echo ✓ include folder found
) else (
    echo ✗ include folder missing
)
if exist "libs" (
    echo ✓ libs folder found
) else (
    echo ✗ libs folder missing
)
echo.
echo **Note:** These folders are crucial for Triton compilation
echo If Triton fails to work, this is usually the missing piece.
echo.
pause
goto MENU

:INSTALL_TEACACHE
cls
echo ========================================
echo  Installing ComfyUI-TeaCache
echo  Official Repository Integration
echo ========================================
echo.
echo Repository: https://github.com/welltop-cn/ComfyUI-TeaCache
echo Installation Location: ComfyUI\custom_nodes\ComfyUI-TeaCache
echo.
echo **TeaCache Features:**
echo - 1.5x to 3x speed improvement for diffusion models
echo - Supports FLUX, HunyuanVideo, LTX-Video, CogVideoX, Wan2.1
echo - Training-free acceleration technique
echo - Compatible with LoRA and ControlNet
echo.

:: Check if we can navigate to the custom_nodes directory
echo Checking ComfyUI directory structure...
if not exist "..\ComfyUI\custom_nodes" (
    echo.
    echo ERROR: ComfyUI custom_nodes folder not found!
    echo Expected path: ..\ComfyUI\custom_nodes
    echo.
    echo **Please verify your ComfyUI portable structure:**
    echo ComfyUI_windows_portable\
    echo ├── ComfyUI\
    echo │   ├── custom_nodes\  ^(Target folder^)
    echo │   └── models\
    echo └── python_embeded\  ^(Current location^)
    echo.
    goto TEACACHE_ERROR
)

:: Check if Git is available
echo Checking Git availability...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Git is not installed or not available in PATH!
    echo.
    echo **Git Installation Required:**
    echo 1. Download Git from: https://git-scm.com/download/win
    echo 2. Install with default settings
    echo 3. Restart this script after installation
    echo.
    goto TEACACHE_ERROR
)

echo ✓ Git is available
echo.

:: Check if TeaCache is already installed
if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache" (
    echo **ComfyUI-TeaCache is already installed!**
    echo.
    set /p update_choice="Do you want to update to the latest version? (y/n): "
    if /i not "%update_choice%"=="y" goto MENU
    
    echo.
    echo Updating existing installation...
    cd "..\ComfyUI\custom_nodes\ComfyUI-TeaCache"
    git pull origin main
    if %errorlevel% neq 0 (
        echo.
        echo WARNING: Git pull failed. Continuing anyway...
    ) else (
        echo ✓ TeaCache updated successfully!
    )
    cd "..\..\..\python_embeded"
    goto TEACACHE_VERIFY
)

:: Proceed with fresh installation
set /p confirm="Continue with TeaCache installation? (y/n): "
if /i not "%confirm%"=="y" goto MENU

echo.
echo ========================================
echo  Cloning ComfyUI-TeaCache Repository
echo ========================================

:: Navigate to custom_nodes directory and clone
echo Navigating to custom_nodes directory...
cd "..\ComfyUI\custom_nodes"

echo Cloning ComfyUI-TeaCache repository...
git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Git clone failed!
    echo.
    echo **Possible causes:**
    echo - Network connectivity issues
    echo - Repository temporarily unavailable
    echo - Insufficient disk space
    echo.
    cd "..\..\python_embeded"
    goto TEACACHE_ERROR
)

echo ✓ Repository cloned successfully!

:: Navigate into the cloned directory
cd ComfyUI-TeaCache

:: Check for requirements.txt and install dependencies
echo.
echo ========================================
echo  Installing Dependencies
echo ========================================

if exist "requirements.txt" (
    echo Installing Python dependencies...
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo WARNING: Some dependencies may have failed to install.
        echo TeaCache might still work, but check for errors when using it.
    ) else (
        echo ✓ Dependencies installed successfully!
    )
) else (
    echo No requirements.txt found - TeaCache may not need additional dependencies.
)

:: Return to python_embeded directory
cd "..\..\..\python_embeded"

:TEACACHE_VERIFY
echo.
echo ========================================
echo  Installation Verification
echo ========================================

:: Verify installation
if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache" (
    echo ✓ ComfyUI-TeaCache folder exists
) else (
    echo ✗ Installation folder not found
    goto TEACACHE_ERROR
)

if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache\__init__.py" (
    echo ✓ TeaCache node module found
) else (
    echo ✗ TeaCache module files missing
)

if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache\nodes.py" (
    echo ✓ Node definitions found
) else (
    echo ⚠ Some TeaCache files may be missing
)

echo.
echo ========================================
echo  ✅ TeaCache Installation Complete!
echo ========================================
echo.
echo **Installation Summary:**
echo Repository: https://github.com/welltop-cn/ComfyUI-TeaCache
echo Location: ComfyUI\custom_nodes\ComfyUI-TeaCache
echo.
echo **Recommended Settings for Different Models:**
echo.
echo **FLUX Models:**
echo - rel_l1_thresh: 0.4
echo - Speedup: ~2x
echo.
echo **HunyuanVideo:**
echo - rel_l1_thresh: 0.15  
echo - Speedup: ~1.9x
echo.
echo **Wan2.1 Models:**
echo - rel_l1_thresh: 0.08-0.26 (depending on model)
echo - Speedup: ~1.6-2.3x
echo.
echo **IMPORTANT NEXT STEPS:**
echo 1. **Restart ComfyUI** completely (close and reopen)
echo 2. Look for **TeaCache** node in the node menu
echo 3. Connect TeaCache node after "Load Diffusion Model" 
echo 4. Set appropriate rel_l1_thresh value for your model
echo 5. Enjoy 2-3x faster generation speeds!
echo.
echo **Note:** TeaCache provides a good trade-off between speed and quality.
echo Start with recommended settings and adjust based on your needs.
echo.
pause
goto MENU

:TEACACHE_ERROR
echo.
echo ========================================
echo  ❌ TeaCache Installation Failed
echo ========================================
echo.
echo **Alternative Installation Methods:**
echo.
echo **Method 1 - ComfyUI Manager (Recommended):**
echo 1. Open ComfyUI
echo 2. Click "Manager" button
echo 3. Go to "Custom Nodes Manager"
echo 4. Search for "ComfyUI-TeaCache"
echo 5. Click "Install"
echo.
echo **Method 2 - Manual Git Commands:**
echo 1. Open Command Prompt as Administrator
echo 2. Navigate to: %CD%\..\ComfyUI\custom_nodes
echo 3. Run: git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git
echo 4. Restart ComfyUI
echo.
echo **Method 3 - Download ZIP:**
echo 1. Visit: https://github.com/welltop-cn/ComfyUI-TeaCache
echo 2. Click "Code" ^> "Download ZIP"
echo 3. Extract to: ComfyUI\custom_nodes\ComfyUI-TeaCache
echo 4. Restart ComfyUI
echo.
pause
goto MENU


:FORCE_REINSTALL
cls
echo Force reinstalling all components...
echo.
echo Upgrading pip first...
.\python -m pip install --upgrade pip
echo.
echo Reinstalling Triton...
.\python -m pip install --force-reinstall triton-windows
echo.
echo Reinstalling Sage Attention...
.\python -m pip install --force-reinstall sageattention
echo.
echo Force reinstallation completed!
echo **Note:** Make sure include/libs folders are in place (option 6 or 7)
echo.
pause
goto MENU

:VERIFY
cls
echo ========================================
echo  Installation Verification
echo ========================================
echo.
echo Checking Python version...
.\python --version
echo.
echo Checking installed packages...
.\python -m pip list | findstr /i "triton sageattention"
echo.
echo Testing imports...

:: Create a proper Python script for import testing
echo # Import testing script > test_imports.py
echo try: >> test_imports.py
echo     import triton >> test_imports.py
echo     print('✓ Triton import successful') >> test_imports.py
echo except Exception as e: >> test_imports.py
echo     print('✗ Triton import failed:', str(e)) >> test_imports.py
echo. >> test_imports.py
echo try: >> test_imports.py
echo     import sageattention >> test_imports.py
echo     print('✓ Sage Attention import successful') >> test_imports.py
echo     try: >> test_imports.py
echo         print('  Version:', sageattention.__version__) >> test_imports.py
echo     except: >> test_imports.py
echo         print('  Version info not available') >> test_imports.py
echo except Exception as e: >> test_imports.py
echo     print('✗ Sage Attention import failed:', str(e)) >> test_imports.py

:: Run the import test script
.\python test_imports.py

:: Clean up the test script
del test_imports.py 2>nul

echo.
echo Checking folder structure...
if exist "include" (
    echo ✓ include folder present
) else (
    echo ✗ include folder missing - may cause Triton compilation issues
)
if exist "libs" (
    echo ✓ libs folder present
) else (
    echo ✗ libs folder missing - may cause Triton compilation issues
)

echo.
echo Checking TeaCache installation (Custom Node)...
if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache" (
    echo ✓ TeaCache custom node folder found
    if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache\__init__.py" (
        echo   - Node initialization file present
    ) else (
        echo   - Warning: Node initialization file missing
    )
    if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache\nodes.py" (
        echo   - Node definitions file present
    ) else (
        echo   - Warning: Node definitions file missing
    )
) else (
    echo ✗ TeaCache not installed (Custom Node not found)
    echo   Location checked: ..\ComfyUI\custom_nodes\ComfyUI-TeaCache
    echo   Install via: Option 8 or ComfyUI Manager
)

echo.
echo ========================================
echo  Installation Summary
echo ========================================
echo Repository used: https://github.com/freyandere/TRSA-Comfyui_installer
echo Python version compatibility: Designed for 3.12.7 (current: %PYTHON_VERSION%)
echo.
echo **Component Status:**
.\python -c "import sys; packages = ['triton', 'sageattention']; [print(f'✓ {pkg} - Successfully installed') if __import__(pkg) else None for pkg in packages]" 2>nul || echo Some packages may need verification

echo.
echo **GPU Compatibility Check:**
.\python -c "try: import torch; print('✓ PyTorch available:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available()); print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU') except: print('⚠ PyTorch not available for GPU check')" 2>nul

echo.
echo ========================================
echo  Next Steps
echo ========================================
echo **Your installation status:**

:: Final component check
set INSTALL_COMPLETE=1

.\python -c "import triton" 2>nul || set INSTALL_COMPLETE=0
.\python -c "import sageattention" 2>nul || set INSTALL_COMPLETE=0

if %INSTALL_COMPLETE% equ 1 (
    echo ✅ **Core installation complete!** Triton + SageAttention ready
) else (
    echo ⚠️ **Some components missing** - check error messages above
)

if exist "..\ComfyUI\custom_nodes\ComfyUI-TeaCache" (
    echo ✅ **TeaCache installed** - Speed optimization available
) else (
    echo ⚠️ **TeaCache not installed** - Optional speed boost missing
)

echo.
echo **To use your installation:**
echo 1. **Restart ComfyUI** completely (close and reopen)
echo 2. **Load a SageAttention workflow** (2-3x speed improvement)
echo 3. **Look for TeaCache nodes** in ComfyUI node menu (if installed)
echo 4. **Test with video generation models** (WAN2.1, Hunyuan, etc.)
echo.
echo **Troubleshooting:**
echo - If models fail to load: Try force reinstall (option 9)
echo - If CUDA errors occur: Verify PyTorch 2.7.1 + CUDA 12.8
echo - If TeaCache missing: Install via option 8 or ComfyUI Manager
echo.
pause
goto MENU



:EXIT
echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo **Important Reminders:**
echo - Restart ComfyUI if it was running during installation
echo - Repository: https://github.com/freyandere/TRSA-Comfyui_installer
echo - The include/libs folders are crucial for Triton functionality
echo - Test with a Sage Attention workflow to verify everything works
echo.
echo **Troubleshooting:**
echo - If Triton fails: Ensure include/libs folders are present
echo - If Sage Attention fails: Reinstall Triton first
echo - Check ComfyUI logs for detailed error messages
echo.
pause
exit /b 0
