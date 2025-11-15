<div align="center">

# ⚡ TRSA ComfyUI Installer

Accelerate your ComfyUI workflows on Windows by integrating Triton and SageAttention.

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/releases)
[![GitHub stars](https://img.shields.io/github/stars/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Windows-brightgreen.svg)](https://github.com/freyandere/TRSA-Comfyui_installer)
[![Python](https://img.shields.io/badge/Python-3.9%20%E2%80%93%203.13-blue)](https://python.org)

[![starline](https://starlines.qoo.monster/assets/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/stargazers)

Achieve up to 30% performance boost on Windows systems.

[Quick Start](#quick-start) • [Features](#features) • [Installation Details](#installation-details) • [Performance](#performance-requirements) • [Troubleshooting](#troubleshooting) • [Support](#support-the-project)

</div>

---

## What is TRSA?

TRSA (Triton + SageAttention) is a one‑click installer that supercharges ComfyUI performance on Windows by integrating optimized attention kernels (SageAttention) and Triton for GPU‑accelerated kernels.  
Designed for AI artists, developers, and power users running image generation workflows on ComfyUI portable builds.

### Key Benefits

- Up to 30%  speed improvement on supported GPUs.  
- One‑click setup via a single batch file.  
- Safety features: environment checks, version validation, rollback.  
- Multilingual interface (English & Russian).  
- Portable‑ready: works with embedded Python inside ComfyUI portable.

---

## Quick Start

> Requirements:  
> - Windows 10/11 x64  
> - NVIDIA GPU (RTX 30/40 series or similar, CUDA‑capable)  
> - ComfyUI portable installation with embedded Python (3.9–3.13)  

### Method 1: Automated Installation (Recommended)

1. Place the installer in your ComfyUI portable folder.

   ```
   ComfyUI_windows_portable/
   └─ python_embeded/
      ├─ python.exe
      └─ TRSA_installer.bat   ← here
   ```

2. Download and run.

   - Download `TRSA_installer.bat` from the Releases page.  
   - Double‑click `TRSA_installer.bat`.  
   - Follow the interactive prompts:
     - Select language (English / Русский).
     - The installer will check Python, PyTorch, CUDA and existing SageAttention.
     - If needed, it will offer to upgrade PyTorch to a supported version.

3. Restart ComfyUI.

   - Launch your usual `ComfyUI\run.bat` or custom start script.  
   - In the console log you should see:
     - `pytorch version: 2.9.0+cu130` (or another supported combo)  
     - `Using sage attention`  

If these lines appear and ComfyUI starts normally, TRSA is installed and active.

---

## Features

### Technical Features

- SageAttention 2.2.x integration:
  - Optimized attention kernels for speed and lower memory usage.
  - Pre‑built wheels for:
    - Python 3.9 + CUDA 12.4–13.0 (several Torch versions).
    - Python 3.13 + CUDA 13.0 (Torch 2.9.0 and 2.10.0).  
- Triton support on Windows:
  - For Python 3.13: install via `triton-windows` from PyPI.  
  - For Python 3.9–3.12: optional install from pre‑built wheels (where available).  
- Version‑aware installer:
  - Checks Python, PyTorch, CUDA and SageAttention versions.
  - Only installs compatible combinations from a built‑in compatibility table.

### Installation Features

- Smart detection:
  - Automatically detects system language (EN/RU).
  - Reads current Python, PyTorch and CUDA versions from the embedded environment.
- Compatibility checks:
  - Validates that your setup is supported before making changes.
  - Offers PyTorch upgrades when your version is too old or sub‑optimal.
- Rollback support:
  - If SageAttention was installed before, the installer can restore the previous version if something goes wrong.
- Cleanup:
  - Temporary files (downloaded wheels, Triton wheel) are removed after the run.
- Logging:
  - Detailed log file is created in the working directory, named like:
    - `TRSA_install_HH.MM-DD.MM.YYYY.log`

### User Experience

- Interactive TUI in console:
  - Clear step‑by‑step prompts.
  - English and Russian texts controlled by a localization module.
- Minimal dependencies:
  - Runs using the embedded Python inside ComfyUI portable.
  - No need to install a separate system‑wide Python.
- Transparent summary:
  - Final summary shows:
    - Previous and new SageAttention versions.
    - Python, PyTorch, CUDA versions.
    - Any errors encountered.
    - Path to the log file.

---

## Installation Details

### What the installer does

Depending on your environment, the installer may:

- Check current environment:
  - Python version from `sys.version_info`.
  - PyTorch version and CUDA (via `torch.__version__` and `nvcc --version` where available).
  - Installed SageAttention version (if any).
- Optionally upgrade PyTorch:
  - For example, to `torch==2.9.0+cu130` on CUDA 13.0, when running on modern RTX GPUs.
- Install Triton:
  - For Python 3.13: `pip install -U "triton-windows<3.6"`.  
  - For older supported Python versions: downloads a matching wheel from the Triton‑Windows GitHub releases and installs it.
- Install SageAttention:
  - Selects the best matching wheel based on your PyTorch + CUDA + Python combination.
  - Downloads the wheel from the `wheels/` folder in this repository.
  - Installs (or reinstalls) SageAttention using `pip`.

### Supported Python / PyTorch / CUDA combinations

The installer currently targets:

- Python:
  - 3.9 (portable ComfyUI builds)  
  - 3.13 (latest ComfyUI portable with experimental Python 3.13)  
- PyTorch:
  - 2.5.1+ on CUDA 12.4  
  - 2.6.0+ on CUDA 12.6  
  - 2.7.1 / 2.8.0 on CUDA 12.8  
  - 2.9.0 / 2.10.0 on CUDA 13.0  
- CUDA:
  - 12.4–13.0 on NVIDIA RTX GPUs.

If your current setup is outside these ranges, the installer will either propose an upgrade (for PyTorch) or refuse to install incompatible wheels.

---

## Performance Requirements

<details>
<summary><strong>GPU Compatibility (practical)</strong></summary>

| GPU | Status | Notes |
|-----|--------|-------|
| RTX 4090 | ✅ Fully Supported (Optimal Performance) | Best results, high VRAM |
| RTX 4080 | ✅ High performance | Up to 30% average speedup vs baseline |
| RTX 3090 | ✅ Full SageAttention support | Good for heavy models |
| RTX 3080 | ✅ Good performance | Recommended for SDXL |
| RTX 3070 | ✅ Supported | 8 GB VRAM is tight but usable |
| RTX 3060 (12 GB) | ✅ Supported | Good mid‑range option |
| RTX 20xx / GTX 10xx | ⚠️ Limited Support | May run, but gains are smaller |
| Non‑NVIDIA / iGPU | ❌ Not Supported | Requires CUDA and Tensor Cores |

</details>

---

## Troubleshooting

<details>
<summary><strong>"Torch/CUDA version mismatch"</strong></summary>

The installer detected that your current PyTorch/CUDA combination is not supported by the available SageAttention wheels.

**What to do:**

- Allow the installer to upgrade PyTorch to a recommended version (for example, `2.9.0+cu130` on CUDA 13.0).  
- If you prefer manual control, you can run:
  ```
  python -m pip install "torch==2.9.0+cu130" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  ```
  then re-run `TRSA_installer.bat`.
</details>

<details>
<summary><strong>"Not a supported wheel"</strong></summary>

Usually indicates a mismatch between:

- Python version (e.g. not cp39/abi3 or cp313), or  
- Windows architecture (must be 64‑bit), or  
- Attempting to install a wheel for a different CUDA/PyTorch combo.

Make sure you are using:

- The portable ComfyUI build with embedded Python 3.9–3.13.  
- A PyTorch version that matches one of the supported combos described above, or allow the installer to upgrade it.
</details>

<details>
<summary><strong>"Network/SSL errors"</strong></summary>

**Check:**

- Internet connection.  
- Firewall / antivirus rules for Python and `curl`/`wget`.  
- Try running the installer as Administrator if your environment is heavily locked down.  

If the automated installer fails repeatedly, you can fall back to manual steps:

1. Download the appropriate SageAttention wheel from the `wheels/` folder in this repository.  
2. Run `python -m pip install <wheel_file>.whl` inside the `python_embeded` folder.
</details>

<details>
<summary><strong>"Failed to find cuobjdump.exe / nvdisasm.exe"</strong></summary>

These warnings come from Triton and mean that CUDA Toolkit’s debug/profiling tools are not present in PATH.  
They do not prevent SageAttention from working, and can usually be ignored for normal ComfyUI usage.
</details>

---

## Getting Help

- Bug reports: open an issue on GitHub.  
- Feature requests: use GitHub Discussions or Issues.  
- For logs: attach the latest `TRSA_install_*.log` and the relevant ComfyUI console output.

---

## Contributing

1. Fork this repository.  
2. Create a feature branch:

   ```
   git checkout -b feature/my-improvement
   ```

3. Test on a clean ComfyUI portable installation.  
4. Document your changes in the code and, if needed, in `README.md` and `CHANGELOG.md`.  
5. Open a pull request with a clear description and reproduction steps.

---

## Acknowledgments

This project depends on and builds upon:

- Triton Windows – Windows port of Triton by @woct0rdho.  
- SageAttention – Quantized attention kernels by Tsinghua University.  
- ComfyUI – Node‑based UI by @comfyanonymous and the community.

---

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

---

## Support the Project

If TRSA has accelerated your ComfyUI workflows:

- Star this repository.  
- Report any issues you encounter.  
- Suggest new features.  
- Share it with the community.  
- Contribute improvements via pull requests.

```
