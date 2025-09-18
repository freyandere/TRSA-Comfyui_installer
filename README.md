<div align="center">

# ⚡ **TRSA ComfyUI Installer**

**Accelerate your ComfyUI workflows on Windows by integrating Triton + SageAttention.**

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/releases)
[![GitHub stars](https://img.shields.io/github/stars/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Windows-brightgreen.svg)](https://github.com/freyandere/TRSA-Comfyui_installer)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](https://python.org)

**🚀 Achieve 2‑3× performance boost on Windows systems**

[🔥 Quick Start](#quick-start) • [📋 Features](#features) • [🛠️ Installation](#installation) • [🏆 Performance](#performance) • [🤝 Support](#support)

</div>

---

## 🎯 What is TRSA?

TRSA (Triton + SageAttention) is a **one‑click installer** that supercharges ComfyUI performance on Windows by integrating cutting‑edge optimization libraries.  
Perfect for AI artists, developers, and businesses running image generation workflows.

### ✨ Key Benefits
- **🔥 2‑3× Speed Improvement** – Dramatically faster inference times.
- **🎯 One‑Click Setup** – No complex configuration required.
- **🛡️ Safe Installation** – Automatic compatibility checking and rollback.
- **🌐 Multilingual** – English & Russian interface support.
- **📦 Portable‑Ready** – Designed for ComfyUI portable installations.

---

## 🚀 Quick Start

> **Requirements**: Windows 10/11 x64, NVIDIA GPU, ComfyUI portable installation (Python 3.11 or 3.12).

### Method 1: Automated Installation (Recommended)

1. **Navigate to your ComfyUI directory**  
   ```text
   ComfyUI_windows_portable/
   └─ python_embeded/
      ├─ python.exe
      └─ [Place TRSA_installer.bat here]
   ```

2. **Download and run**  
   - Download `TRSA_installer.bat` from the [Releases](https://github.com/freyandere/TRSA-Comfyui_installer/releases).  
   - Double‑click to run.  
   - Follow the interactive prompts (language selection, Torch/CUDA check, etc.).

3. **Enjoy the speed boost**! 🎉

> *The installer uses Rich logs and tqdm progress bars for a pleasant UI.*

### Method 2: Manual Installation

```bash
# Install Triton for Windows
python -m pip install -U "triton-windows<3.4"

# Download SageAttention wheel (currently built for Python 3.9, CUDA 12.8)
wget https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl

# Install from the wheel
python -m pip install sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl

# Verify installation
python -c "import triton, sageattention; print('Success!')"
```

> *If you use Python 3.11/3.12, the wheel may need to be rebuilt or a different wheel is required.*

---

## 📋 Features

### 🔧 Technical Features
- **Triton Integration** – GPU kernel compilation for Windows.
- **SageAttention 2.2.x** – Quantized attention mechanisms.
- **CUDA 12.9 Support** – Latest CUDA optimization (cu129).
- **PyTorch 2.8.0+cu129** – Strict version compatibility.

### 🛠️ Installation Features
- **Smart Detection** – Auto‑detects system configuration and language.
- **Compatibility Checks** – Prevents incompatible installations.
- **Progress Tracking** – Clear installation status updates with tqdm (fallback to plain output).
- **Spinner for Pip** – Unicode spinner (`⠋…⠏`) during long‑running installs.
- **Error Recovery** – Automatic rollback on failures.
- **Cleanup** – Temporary files (`include_libs.zip`, wheel) are deleted immediately after use.

### 🌟 User Experience
- **Interactive TUI** – Step‑by‑step installation guide.
- **Multilingual Support** – English & Russian interfaces (environment variables `ACC_LANG_FORCE` and `ACC_LANG`).
- **Detailed Reports** – Comprehensive installation summaries at the end.
- **Zero Dependencies** – Works with embedded Python.

---

## 🏆 Performance & Requirements

<details>
<summary><strong>GPU Compatibility Matrix</strong></summary>

| GPU | Status | Notes |
|-----|--------|-------|
| RTX 4090 | ✅ Fully Supported (Optimal Performance) | 340 TOPS, 2.7x speedup |
| RTX 4080 | ✅ High performance, 2.5x average speedup | |
| RTX 3090 | ✅ Full SageAttention support | |
| RTX 3080 | ✅ Good performance with optimization | |
| RTX 3070 | ✅ Supported (Good Performance) | 8GB VRAM, suitable for most workflows |
| RTX 3060 | ✅ Supported (Good Performance) | 12GB variant recommended |
| RTX A6000 | ✅ Supported (Good Performance) | Professional workstation use |
| GTX 1060-1080 | ⚠️ Limited Support | Basic ComfyUI only, no TRSA benefits |
| RTX 2060-2080 | ⚠️ Limited Support | Limited VRAM may restrict workflow complexity |
| Non‑NVIDIA GPUs | ❌ Not Supported | TRSA requires CUDA and Tensor Cores |
| Integrated Graphics | ❌ Not Supported | Insufficient for AI inference |

</details>

<details>
<summary><strong>CUDA & Software Requirements</strong></summary>

### Essential Software Stack
- **Windows 10/11 x64**  
- **Python 3.11/3.12 (embedded in portable)**  
- **CUDA 12.9** (required for SageAttention 2+)  
- **PyTorch 2.8.0+cu129**  
- **Triton Windows <3.4**

> *The current wheel is built for CUDA 12.8; if you use CUDA 12.9, you may need to rebuild or use a different wheel.*

### Development Requirements
- **Visual Studio Build Tools**  
- **Git** (for updates)  
- **7‑Zip** (for extraction)

</details>

<details>
<summary><strong>Version Compatibility</strong></summary>

- **CUDA 12.9** – Required for latest SageAttention features  
- **CUDA 12.4+** – Minimum for FP8 support on Ada GPUs  
- **CUDA 12.3+** – Minimum for FP8 support on Hopper GPUs  
- **CUDA 12.0+** – Minimum for Ampere GPU support

</details>

<details>
<summary><strong>Performance Benchmarks</strong></summary>

| Configuration | Before TRSA | After TRSA | Improvement |
|---------------|-------------|------------|-------------|
| SDXL 1024x1024 | 45 s | 18 s | **2.5× faster** |
| SD 1.5 512x512 | 12 s | 4 s | **3× faster** |
| Flux Dev | 120 s | 48 s | **2.5× faster** |

> *Benchmarks performed on RTX 4080; results may vary by hardware configuration.*

</details>

---

## 🛠️ Installation

### What Gets Installed
```
Components:
  - Triton Windows (<3.4)      # GPU kernel compiler
  - SageAttention (2.2.x)      # Optimized attention
  - CUDA Libraries (12.9)      # GPU acceleration
  - Python Headers             # Development support
  - Compatibility Checks       # System validation
```

### Version 2.5 Highlights
- **Single‑file installer** with secure HTTPS downloads.
- **Strict compatibility validation** for PyTorch 2.8.0 + CUDA 12.9.
- **Enhanced error handling** with detailed diagnostics.
- **Improved TUI** with clear status indicators.
- **Final component report** for transparency.

---

## 🔧 Troubleshooting

<details>
<summary><strong>❌ "Torch/CUDA version mismatch"</strong></summary>

**Solution**: Allow the installer to reinstall PyTorch 2.8.0 with CUDA 12.9 support (~2.5 GB download)

```bash
# Manual fix
python -m pip install torch==2.8.0+cu129 -f https://download.pytorch.org/whl/cu129
```

</details>

<details>
<summary><strong>❌ "Not a supported wheel"</strong></summary>

**Cause**: Python version or platform mismatch (wheel built for cp39-abi3, CUDA 12.8).

**Solution**:
- Verify your Python version (`python --version` should show 3.11.x or 3.12.x).  
- If using Python 3.11/3.12, you may need to rebuild the wheel from source or use a different wheel.

</details>

<details>
<summary><strong>❌ "Network/SSL errors"</strong></summary>

**Solutions**:
- Check your firewall/antivirus settings.
- Verify internet connection.
- Try running as administrator.
- Use manual installation method if needed.

</details>

---

## Getting Help

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/freyandere/TRSA-Comfyui_installer/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/freyandere/TRSA-Comfyui_installer/discussions)
- **📚 Documentation**: [Project Wiki](https://github.com/freyandere/TRSA-Comfyui_installer/wiki)

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** this repository  
2. **Create** a feature branch: `git checkout -b feature/amazing-improvement`  
3. **Test** thoroughly on a clean ComfyUI portable installation  
4. **Document** your changes in the code and README  
5. **Submit** a pull request with a detailed description

### Development Setup
```bash
git clone https://github.com/freyandere/TRSA-Comfyui_installer.git
cd TRSA-Comfyui_installer
# Test with clean ComfyUI portable installation
```

---

## 🙏 Acknowledgments

This project stands on the shoulders of giants:

- **[Triton Windows](https://github.com/woct0rdho/triton-windows)** – Windows port by @woct0rdho  
- **[SageAttention](https://github.com/thu-ml/SageAttention)** – Quantized attention by Tsinghua University  
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** – Node‑based UI by @comfyanonymous  
- **Community Channels**: [Psy Eyes](https://t.me/psy_eyes) & [FRALID](https://t.me/fralid)

---

## 📄 License

This project is licensed under the **Apache 2.0 License** – see the [LICENSE](LICENSE) file for details.

---

## 🌟 Support the Project

If TRSA has accelerated your workflows:

- ⭐ **Star** this repository  
- 🐛 **Report** issues you encounter  
- 💡 **Suggest** new features  
- 🔄 **Share** with the community  
- 🤝 **Contribute** improvements

---

<div align="center">

### 🔗 Links

[🏠 Main Repository](https://github.com/freyandere/TRSA-Comfyui_installer) •  
[📋 Issues](https://github.com/freyandere/TRSA-Comfyui_installer/issues) •  
[💬 Discussions](https://github.com/freyandere/TRSA-Comfyui_installer/discussions) •  
[📖 Wiki](https://github.com/freyandere/TRSA-Comfyui_installer/wiki)

**Made with ❤️ for the ComfyUI community**

*Accelerating AI workflows, one installation at a time*

</div>
```
