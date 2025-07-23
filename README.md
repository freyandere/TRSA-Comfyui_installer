Overview
The TRSA ComfyUI Installer transforms a simple batch script into a professional-grade installer for ComfyUI Portable on Windows. It automates the setup of Triton, SageAttention and TeaCache with color-coded status feedback, robust error handling and multiple fallback methods. Users gain 2–3× faster video generation alongside simplified dependency management.

Features
Robust error handling with PowerShell, Curl and Python fallbacks

Color-coded console output for instant status recognition (Red/Green/Yellow/Cyan/Magenta/White)

Automated downloads from your GitHub repository

Smart archive extraction via native .NET API or Python zipfile

Comprehensive verification of Triton, SageAttention and include/libs folders

SageAttention 2.2.0 wheel installer optimized for RTX 50xx (CUDA 12.8 + PyTorch 2.7.1)

TeaCache custom-node integration with git cloning into ComfyUI/custom_nodes

Menu-driven interface with clear guidance and retry options

Installation
Download or clone this repository into your ComfyUI Portable directory.

Place the .bat installer inside python_embeded.

Run the installer by double-clicking the batch file or via Command Prompt:

text
cd path\to\ComfyUI_windows_portable\python_embeded
installer.bat
Follow the interactive menu to install components in the recommended order (check system, upgrade pip, install Triton, install PyTorch, install SageAttention, auto-setup include/libs, install TeaCache, verify).

Usage
Check System: Validates Python version, CUDA driver and GPU support.

Upgrade pip: Ensures latest pip for dependency installs.

Install Triton: Supports stable and pre-release builds.

Install PyTorch: Automatically installs PyTorch 2.7.1 with CUDA 12.8.

Install SageAttention: Downloads and installs the pre-built wheel for RTX 50xx.

Auto-setup include/libs: Fetches and extracts required folders for Triton compilation.

Install TeaCache: Clones and configures the TeaCache custom node.

Verify: Runs import tests and folder checks to confirm a successful setup.

Troubleshooting
If a download or extraction fails, retry with alternate methods provided in the menu.

Missing include/libs folders may cause Triton build errors—use the manual setup option or extract via Python.

TeaCache installation issues often stem from missing Git or incorrect directory structure—refer to the manual instructions.

For GPU or CUDA mismatches, confirm driver version with nvidia-smi and verify PyTorch CUDA support with the system check menu option.

Contributing
Fork the repository and create a feature branch.

Follow the existing batch-script style and color conventions.

Submit a pull request describing your changes and test results on Windows 10/11.
