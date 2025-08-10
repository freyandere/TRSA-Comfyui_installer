# ComfyUI Accelerator — v2.5

Speeds up ComfyUI by 2-3 times on Windows using Triton and SageAttention.

**Languages**: [English](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main) | [Русский](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main-ru)

## Table of Contents

- [What's new in v2.5](#-whats-new-in-v25)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [System Requirements](#-system-requirements)
- [Performance](#-performance)
- [Features](#%EF%B8%8F-features)
- [Diagnostics](#-diagnostics)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Support the Project](#-support-the-project)
- [Links](#-links)
- [Related Projects](#-related-projects)


## What's new in v2.5

- Single-file core installer (installer_core.py) with secure HTTPS downloads and safe ZIP extraction.
- Step-by-step TUI without progress bars: clear statuses and a final summary.
- Strict compatibility checks:
    - Torch “2.7.1+cu128” is correctly identified as 2.7.1.
    - Strictly CUDA 12.8 for the target configuration.
    - SageAttention wheel is validated against win_amd64, cpXY/abi3, and Torch 2.7.1.
- Strict pinning: triton-windows<3.4.
- Final report for each component (torch/include+libs/triton/sageattention).


## Quick Start

1) Place the .bat file in:
ComfyUI_windows_portable/
└─ python_embeded/
├─ python.exe
└─ TRSA_installer.bat

text

2) Run the .bat file (double-click) — it will download installer_core.py and execute it with the embedded Python.
3) Follow the prompts:

- If there's a Torch/CUDA mismatch, it will offer to reinstall Torch 2.7.1 (CUDA 12.8) (~2.5GB).
- Step order: include/libs → Triton (<3.4) → SageAttention wheel.
- A summary report is provided at the end.


## Installation

Installs:

- Triton for Windows: `triton-windows<3.4`
- SageAttention 2.2.x (CUDA 12.8 + Torch 2.7.1)
- include/ and libs/ for the portable Python
- Utility dependencies (minimum required)


## System Requirements

- Windows 10/11 x64
- Embedded Python 3.11/3.12 (ComfyUI portable)
- NVIDIA GPU (CUDA)
- Up to ~2.5GB of traffic and disk space if reinstalling Torch


## Performance

- Typical speed-up: 2–3× with correct Torch/CUDA versions and Triton/SageAttention installed.
- For detailed benchmarks, see the SageAttention documentation.


## Features

- Auto-detection of language (EN/RU) with an option for manual override via environment variables.
- Step-by-step TUI installation without progress bars.
- Detailed final report on all components.
- Strict compatibility checks and secure downloads.


## Diagnostics

- Quick check for the presence of include/libs.
- Validation of Torch/CUDA versions and SageAttention wheel compatibility.
- Verification of successful Triton installation and module imports.


## Troubleshooting

- Torch/CUDA mismatch: confirm the reinstallation of Torch to 2.7.1 (CUDA 12.8).
- “not a supported wheel”: check the cp-tag (cp311/cp312) and platform (win_amd64).
- Network/SSL errors: check your proxy/AV; try again (HTTPS + certificate verification is used).
- No PowerShell: urllib is used; PowerShell is only a fallback.


### **Manual Installation**

If the automatic installation fails:

```bash
# 1. Install Triton manually
python -m pip install -U "triton-windows<3.4"

# 2. Download the SageAttention wheel
# Visit: https://github.com/freyandere/TRSA-Comfyui_installer/releases

# 3. Install from the wheel
python -m pip install sageattention-2.2.0*.whl

# 4. Verify the installation
python -c "import triton, sageattention; print('Success!')"
```


### **Getting Help**

- **Issues**: [GitHub Issues](https://github.com/freyandere/TRSA-Comfyui_installer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/freyandere/TRSA-Comfyui_installer/discussions)
- **Documentation**: [Wiki](https://github.com/freyandere/TRSA-Comfyui_installer/wiki)


### **Contribution Guidelines**

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Test thoroughly** on a clean ComfyUI installation
4. **Document** your changes in the code and README
5. **Submit** a pull request with a detailed description

## License

Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the developers of the projects that made this possible:

- **[Triton Windows](https://github.com/woct0rdho/triton-windows)** - Triton port for Windows by [@woct0rdho](https://github.com/woct0rdho) and the community
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - Quantized attention from researchers at Tsinghua University
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - A powerful node-based interface by [@comfyanonymous](https://github.com/comfyanonymous)
- **[Telegram channel - Psy Eyes](https://t.me/Psy_Eyes)** - for highlighting the repository and community support.
- **[Telegram channel - FRALID | НАСМОТРЕННОСТЬ](https://t.me/fralid95)** - for support and guidance.

Without their incredible work, this project would not be possible!

## Support the Project

If ComfyUI Accelerator has helped speed up your workflows:

- **Star** the repository
- **Report** any issues you encounter
- **Suggest** new features
- **Contribute** improvements
- **Share** with the community


## Links

- **Main Repository**: [English Branch](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main)
- **Russian Version**: [Russian Branch](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main-ru)
- **Issues**: [Report Issues](https://github.com/freyandere/TRSA-Comfyui_installer/issues)
- **Discussions**: [Community Forum](https://github.com/freyandere/TRSA-Comfyui_installer/discussions)
- **Wiki**: [Documentation](https://github.com/freyandere/TRSA-Comfyui_installer/wiki)


### **Related Projects**

- **[Triton Windows](https://github.com/woct0rdho/triton-windows)** - GPU compiler for acceleration
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - Quantized attention mechanisms
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - Node-based interface for AI

*Made with love for the ComfyUI community*
