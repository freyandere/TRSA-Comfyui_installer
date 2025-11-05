#!/usr/bin/env python3
# installer_core.py
# Flexible CUDA/Torch version installer with user choice

import logging, os, re, sys, shutil, subprocess, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import threading
import time

# Optional Rich imports for enhanced UI
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.prompt import Prompt
    rich_available = True
except ImportError:
    rich_available = False
    Console = None
    RichHandler = None
    Table = None
    Prompt = None

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

console = Console() if rich_available else None

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
LOG = logging.getLogger("installer_core")

def init_logging() -> None:
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    handler = RichHandler() if RichHandler is not None else logging.StreamHandler()
    handler.setFormatter(formatter)
    LOG.handlers.clear()
    LOG.addHandler(handler)

init_logging()

# ----------------------------------------------------------------------
# Language selection
# ----------------------------------------------------------------------
def _prompt_language_choice() -> str:
    """Detect user language preference."""
    if os.environ.get("ACC_LANG_FORCE", "").lower() in ("1", "true", "yes", "y"):
        lang = os.environ.get("ACC_LANG", "").strip().lower()
        return "ru" if lang == "ru" else "en"

    existing = os.environ.get("ACC_LANG", "").strip().lower()
    if existing in ("ru", "en"):
        return existing

    print("Select language / Выберите язык:\n  1) RU (Русский)\n  2) EN (English)")
    choice = input("Choice (1/2, Enter=Auto): ").strip()
    if choice == "1":
        return "ru"
    if choice == "2":
        return "en"

    # Try PowerShell culture detection
    try:
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            r = subprocess.run(
                [ps, "-NoProfile", "-NonInteractive", "-Command", "(Get-Culture).Name"],
                capture_output=True, text=True, timeout=3, encoding="utf-8", errors="replace"
            )
            culture = (r.stdout or "").strip().lower()
            if culture.startswith("ru"):
                return "ru"
            if culture.startswith("en"):
                return "en"
    except Exception:
        pass

    loc = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").lower()
    return "ru" if loc.startswith("ru") else "en"

_selected_lang = _prompt_language_choice()
os.environ["ACC_LANG"] = _selected_lang
L = _selected_lang

# ----------------------------------------------------------------------
# Localization strings
# ----------------------------------------------------------------------
T: dict[str, str] = {
    "ru": {
        "intro": "🚀 Установщик ускорителя ComfyUI",
        "detected_versions": "Обнаружено: torch={torch_ver}, CUDA={cuda_ver}",
        "select_version": "Выберите версию для установки:",
        "version_option": "  {idx}) Torch {torch} + CUDA {cuda} {tag}",
        "checking_torch": "Шаг 1/4: Проверка текущей установки PyTorch...",
        "confirm_install": "Установить выбранную версию? (y/N): ",
        "disclaimer_header": "ВНИМАНИЕ: переустановка PyTorch может повлиять на другие пайплайны.",
        "disclaimer_common": "Загрузка ~2.8ГБ; это может занять время.",
        "cancelled": "Операция отменена пользователем.",
        "installing_pytorch": "Установка PyTorch {ver} + CUDA {cuda}...",
        "pytorch_done": "✅ PyTorch установлен успешно.",
        "pytorch_fail": "❌ Ошибка установки PyTorch: {err}",
        "setup_inc_libs": "Шаг 2/4: Распаковка include/libs...",
        "setup_ok": "✅ include/libs готовы.",
        "setup_fail": "❌ Ошибка распаковки include/libs: {err}",
        "install_triton": "Шаг 3/4: Установка Triton...",
        "triton_ok": "✅ Triton установлен.",
        "triton_fail": "❌ Ошибка установки Triton: {err}",
        "install_sage": "Шаг 4/4: Установка SageAttention...",
        "sage_ok": "✅ SageAttention установлен.",
        "sage_fail": "❌ Ошибка установки SageAttention: {err}",
        "download_fail": "Ошибка загрузки: {err}",
        "report_title": "\n📋 Итоговый отчёт:",
        "goodbye": "✅ Установка завершена!",
    },
    "en": {
        "intro": "🚀 ComfyUI Accelerator Installer",
        "detected_versions": "Detected: torch={torch_ver}, CUDA={cuda_ver}",
        "select_version": "Select version to install:",
        "version_option": "  {idx}) Torch {torch} + CUDA {cuda} {tag}",
        "checking_torch": "Step 1/4: Checking current PyTorch installation...",
        "confirm_install": "Install selected version? (y/N): ",
        "disclaimer_header": "WARNING: reinstalling PyTorch may affect other pipelines.",
        "disclaimer_common": "Download size ~2.8GB; this may take a while.",
        "cancelled": "Operation cancelled by user.",
        "installing_pytorch": "Installing PyTorch {ver} + CUDA {cuda}...",
        "pytorch_done": "✅ PyTorch installed successfully.",
        "pytorch_fail": "❌ PyTorch installation failed: {err}",
        "setup_inc_libs": "Step 2/4: Unpacking include/libs...",
        "setup_ok": "✅ include/libs are ready.",
        "setup_fail": "❌ include/libs extraction error: {err}",
        "install_triton": "Step 3/4: Installing Triton...",
        "triton_ok": "✅ Triton installed.",
        "triton_fail": "❌ Triton installation failed: {err}",
        "install_sage": "Step 4/4: Installing SageAttention...",
        "sage_ok": "✅ SageAttention installed.",
        "sage_fail": "❌ SageAttention installation failed: {err}",
        "download_fail": "Download error: {err}",
        "report_title": "\n📋 Final Report:",
        "goodbye": "✅ Installation completed!",
    },
}[L]

# ----------------------------------------------------------------------
# Version configurations
# ----------------------------------------------------------------------
@dataclass
class VersionConfig:
    """Configuration for a specific CUDA/Torch version."""
    name: str
    torch_version: str
    cuda_version: str  # e.g., "12.9" or "12.10"
    pytorch_index_url: str
    torchvision_version: str
    torchaudio_version: str
    sage_wheel_urlenc: str
    sage_wheel_local: str
    triton_pin: str
    recommended: bool = False

# Available version configurations
VERSIONS = [
    VersionConfig(
        name="CUDA 12.9 + Torch 2.8.0",
        torch_version="2.8.0",
        cuda_version="12.9",
        pytorch_index_url="https://download.pytorch.org/whl/cu129",
        torchvision_version="0.23.0",
        torchaudio_version="2.8.0",
        sage_wheel_urlenc="sageattention-2.2.0%2Bcu128torch2.8.0.post2-cp39-abi3-win_amd64.whl",
        sage_wheel_local="sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl",
        triton_pin="triton-windows<3.4",
        recommended=False  # Stable but not latest
    ),
    VersionConfig(
        name="CUDA 12.10 + Torch 2.9.0",
        torch_version="2.9.0",
        cuda_version="12.10",
        pytorch_index_url="https://download.pytorch.org/whl/cu1210",
        torchvision_version="0.24.0",
        torchaudio_version="2.9.0",
        sage_wheel_urlenc="sageattention-2.2.0%2Bcu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
        sage_wheel_local="sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
        triton_pin="triton-windows<3.5",
        recommended=True  # Latest
    ),
]

@dataclass
class InstallConfig:
    repo_base: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
    include_zip: str = "python_3.12.7_include_libs.zip"
    max_total_uncompressed: int = 600 * 1024 * 1024

# ----------------------------------------------------------------------
# Installer core
# ----------------------------------------------------------------------
class InstallerCore:
    def __init__(self, python_exe: str | None = None, cfg: InstallConfig | None = None):
        self.cfg = cfg or InstallConfig()
        self.python = python_exe or sys.executable
        self.selected_version: Optional[VersionConfig] = None
        
        if not shutil.which(self.python):
            raise FileNotFoundError(f"Python executable not found at: {self.python}")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _run(self, args: list[str], timeout: int = 3600) -> Tuple[bool, str]:
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace"
            )
            return r.returncode == 0, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)

    def _pip(self, *args: str, timeout: int = 3600) -> Tuple[bool, str]:
        return self._run([self.python, "-m", "pip", *args], timeout=timeout)

    def _current_torch_cuda(self) -> Tuple[str, str]:
        """Get current torch and CUDA versions."""
        code = """\
import torch
print(torch.__version__)
cuda_ver = getattr(getattr(torch, 'version', None), 'cuda', '')
print(cuda_ver if cuda_ver else '')
"""
        ok, out = self._run([self.python, "-c", code], timeout=30)
        lines = [s.strip() for s in (out or "").splitlines()] if ok else []
        return (lines[0] if len(lines) > 0 else "", lines[1] if len(lines) > 1 else "")

    @staticmethod
    def _spinner(msg: str = "") -> threading.Event:
        """Create a spinner for long-running operations."""
        stop_event = threading.Event()
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        
        def spin():
            idx = 0
            while not stop_event.is_set():
                print(f"\r{msg} {frames[idx % len(frames)]}", end="", flush=True)
                idx += 1
                time.sleep(0.1)
        
        t = threading.Thread(target=spin, daemon=True)
        t.start()
        return stop_event

    # ------------------------------------------------------------------
    # Version selection
    # ------------------------------------------------------------------
    def select_version(self) -> VersionConfig:
        """Prompt user to select a version."""
        LOG.info(T["checking_torch"])
        
        # Show current installation
        cur_torch, cur_cuda = self._current_torch_cuda()
        shown_torch = cur_torch or ("not installed" if L == "en" else "не установлен")
        shown_cuda = cur_cuda or ("unknown" if L == "en" else "неизвестно")
        LOG.info(T["detected_versions"].format(torch_ver=shown_torch, cuda_ver=shown_cuda))
        
        print("\n" + "="*60)
        print(T["select_version"])
        
        # Display available versions
        for idx, ver in enumerate(VERSIONS, 1):
            tag = "⭐ Recommended" if ver.recommended else ""
            if L == "ru" and tag:
                tag = "⭐ Рекомендуется"
            
            print(T["version_option"].format(
                idx=idx,
                torch=ver.torch_version,
                cuda=ver.cuda_version,
                tag=tag
            ))
        
        print("="*60)
        
        # Get user choice
        while True:
            try:
                if rich_available and Prompt:
                    choice = Prompt.ask("Choice", default="2")
                else:
                    choice = input(f"Choice (1-{len(VERSIONS)}, default=2): ").strip() or "2"
                
                idx = int(choice) - 1
                if 0 <= idx < len(VERSIONS):
                    self.selected_version = VERSIONS[idx]
                    LOG.info(f"Selected: {self.selected_version.name}")
                    return self.selected_version
                else:
                    print(f"Please enter a number between 1 and {len(VERSIONS)}")
            except (ValueError, KeyboardInterrupt):
                print(f"Invalid input. Using recommended version (2).")
                self.selected_version = VERSIONS[1]  # Default to latest
                return self.selected_version

    # ------------------------------------------------------------------
    # Installation steps
    # ------------------------------------------------------------------
    def install_pytorch(self, version: VersionConfig) -> Tuple[bool, str]:
        """Install PyTorch with specified version."""
        try:
            LOG.info("\n" + "="*60)
            LOG.warning(T["disclaimer_header"])
            LOG.warning(T["disclaimer_common"])
            LOG.info("="*60)
            
            ans = input(T["confirm_install"]).strip().lower()
            if ans not in ("y", "yes", "д", "да"):
                LOG.info(T["cancelled"])
                return False, "User cancelled"
            
            LOG.info(T["installing_pytorch"].format(ver=version.torch_version, cuda=version.cuda_version))
            
            cuda_tag = version.cuda_version.replace(".", "")
            packages = [
                f"torch=={version.torch_version}",
                f"torchvision=={version.torchvision_version}",
                f"torchaudio=={version.torchaudio_version}",
                "--force-reinstall",
                "--index-url", version.pytorch_index_url
            ]
            
            stop_spinner = self._spinner("Installing PyTorch")
            ok, out = self._pip(*packages)
            stop_spinner.set()
            print("\r" + " " * 80 + "\r", end="")
            
            if ok:
                LOG.info(T["pytorch_done"])
                return True, ""
            else:
                LOG.error(T["pytorch_fail"].format(err=out[-500:]))
                return False, out
                
        except Exception as e:
            LOG.error(T["pytorch_fail"].format(err=str(e)))
            return False, str(e)

    def download_and_extract_include_libs(self) -> Tuple[bool, str]:
        """Download and extract include/libs."""
        url = f"{self.cfg.repo_base}/{self.cfg.include_zip}"
        dest_path = Path.cwd() / self.cfg.include_zip
        LOG.info(T["setup_inc_libs"])

        try:
            if tqdm:
                with urllib.request.urlopen(url) as resp:
                    total = int(resp.getheader("Content-Length") or 0)
                    with open(dest_path, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True, desc="Downloading include/libs"
                    ) as pbar:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                urllib.request.urlretrieve(url, dest_path)

            with zipfile.ZipFile(dest_path) as zf:
                members = zf.infolist()
                if tqdm:
                    for member in tqdm(members, desc="Extracting include/libs"):
                        zf.extract(member)
                else:
                    zf.extractall()

            LOG.info(T["setup_ok"])
            return True, ""
            
        except Exception as e:
            LOG.error(T["setup_fail"].format(err=str(e)))
            return False, str(e)
        finally:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)

    def install_triton(self, version: VersionConfig) -> Tuple[bool, str]:
        """Install Triton."""
        LOG.info(T["install_triton"])
        
        stop_spinner = self._spinner("Installing Triton")
        ok, out = self._pip("install", version.triton_pin, "--force-reinstall")
        stop_spinner.set()
        print("\r" + " " * 80 + "\r", end="")

        if ok:
            LOG.info(T["triton_ok"])
            return True, ""
        else:
            LOG.error(T["triton_fail"].format(err=out[-500:]))
            return False, out

    def install_sage_attention(self, version: VersionConfig) -> Tuple[bool, str]:
        """Install SageAttention."""
        LOG.info(T["install_sage"])
        url = f"{self.cfg.repo_base}/{version.sage_wheel_urlenc}"
        dest_path = Path.cwd() / version.sage_wheel_local
        
        try:
            if tqdm:
                with urllib.request.urlopen(url) as resp:
                    total = int(resp.getheader("Content-Length") or 0)
                    with open(dest_path, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True, desc="Downloading SageAttention"
                    ) as pbar:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                urllib.request.urlretrieve(url, dest_path)

            stop_spinner = self._spinner("Installing SageAttention")
            ok, out = self._pip("install", str(dest_path), "--force-reinstall")
            stop_spinner.set()
            print("\r" + " " * 80 + "\r", end="")

            if ok:
                LOG.info(T["sage_ok"])
                return True, ""
            else:
                LOG.error(T["sage_fail"].format(err=out[-500:]))
                return False, out
                
        except Exception as e:
            LOG.error(T["download_fail"].format(err=str(e)))
            return False, str(e)
        finally:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Execute the full installation flow."""
        try:
            # Step 0: Version selection
            version = self.select_version()
            
            # Step 1: PyTorch
            ok_torch, _ = self.install_pytorch(version)
            if not ok_torch:
                return 1
            print()
            
            # Step 2: Include/libs
            ok_extract, _ = self.download_and_extract_include_libs()
            if not ok_extract:
                return 1
            print()
            
            # Step 3: Triton
            ok_triton, _ = self.install_triton(version)
            if not ok_triton:
                return 1
            print()
            
            # Step 4: SageAttention
            ok_sage, _ = self.install_sage_attention(version)
            if not ok_sage:
                return 1
            
            # Summary
            steps = [
                (f"PyTorch {version.torch_version}", ok_torch),
                ("Include/libs", ok_extract),
                ("Triton", ok_triton),
                ("SageAttention", ok_sage),
            ]
            
            if rich_available and console:
                table = Table(title="Installation Summary")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="bold")
                
                for name, status in steps:
                    color = "green" if status else "red"
                    status_text = "✅ Success" if status else "❌ Failed"
                    table.add_row(name, f"[{color}]{status_text}[/{color}]")
                
                console.print("\n")
                console.print(table)
            else:
                LOG.info(T["report_title"])
                for name, status in steps:
                    status_text = "✅" if status else "❌"
                    LOG.info(f"  {status_text} {name}")
            
            LOG.info(T["goodbye"])
            return 0
            
        except KeyboardInterrupt:
            LOG.info(f"\n{T['cancelled']}")
            return 1
        except Exception as e:
            LOG.error(f"Fatal error: {e}")
            return 1

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if console:
        console.print(f"[bold magenta]{T['intro']}[/bold magenta]\n")
    else:
        LOG.info(T["intro"])

    if not os.path.exists(sys.executable):
        LOG.error("Could not determine valid Python executable.")
        sys.exit(1)

    exit_code = InstallerCore().run()
    sys.exit(exit_code)
