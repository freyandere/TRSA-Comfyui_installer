#!/usr/bin/env python3
# installer_core.py
# Smart version recommendation based on current installation

import logging, os, re, sys, shutil, subprocess, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import threading
import time

# Optional Rich imports for enhanced UI
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.markdown import Markdown
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
        "version": "v1.2.0",
        "detected_versions": "Обнаружено: torch={torch_ver}, CUDA={cuda_ver}",
        "select_version": "Выберите версию для установки:",
        "version_option": "  {idx}) Torch {torch} + CUDA {cuda} {tags}",
        "tag_recommended": "⭐ Рекомендуется",
        "tag_installed": "✓ Установлено",
        "tag_latest": "🆕 Новейшая",
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
        "system_diagnostic": "📊 Диагностика системы",
        "update_available": "💡 Доступно обновление:",
        "sage_detected": "SAGE уже установлен: v{version}",
        "no_sage_installed": "SAGE не установлен",
        "welcome_msg": "Добро пожаловать в установщик ускорителя ComfyUI",
    },
    "en": {
        "intro": "🚀 ComfyUI Accelerator Installer",
        "version": "v1.2.0",
        "detected_versions": "Detected: torch={torch_ver}, CUDA={cuda_ver}",
        "select_version": "Select version to install:",
        "version_option": "  {idx}) Torch {torch} + CUDA {cuda} {tags}",
        "tag_recommended": "⭐ Recommended",
        "tag_installed": "✓ Installed",
        "tag_latest": "🆕 Latest",
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
        "system_diagnostic": "📊 System Diagnostic",
        "update_available": "💡 Update Available:",
        "sage_detected": "SAGE already installed: v{version}",
        "no_sage_installed": "SAGE not installed",
        "welcome_msg": "Welcome to ComfyUI Accelerator Installer",
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
    is_latest: bool = False  # Is this the newest version?

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
        is_latest=False
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
        is_latest=True
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
        self.current_torch: str = ""
        self.current_cuda: str = ""
        
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

    @staticmethod
    def _strip_local(v: str) -> str:
        """Strip local version identifier (e.g., '2.8.0+cu129' -> '2.8.0')"""
        return v.split("+", 1)[0] if v else v

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

    def _current_sage_version(self) -> str:
        """Get current SageAttention version."""
        code = """\
try:
    import sageattention
    print(sageattention.__version__)
except ImportError:
    print("not installed")
"""
        ok, out = self._run([self.python, "-c", code], timeout=30)
        return (out or "").strip()

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

    def _find_matching_version(self, torch_ver: str, cuda_ver: str) -> Optional[int]:
        """Find index of version that matches current installation."""
        if not torch_ver or not cuda_ver:
            return None
        
        clean_torch = self._strip_local(torch_ver)
        
        for idx, ver in enumerate(VERSIONS):
            if ver.torch_version == clean_torch and ver.cuda_version == cuda_ver:
                return idx
        
        return None

    # ------------------------------------------------------------------
    # Welcome screen
    # ------------------------------------------------------------------
    def welcome_screen(self) -> None:
        """Display the welcome screen with version info."""
        print("\n" + "="*60)
        
        if rich_available:
            console.print(f"[bold magenta][center]{T['intro']}[/center][/bold magenta]")
            console.print(f"[bold blue][center]Version: {T['version']}[/center][/bold blue]")
        else:
            print(T["intro"])
            print("Version:", T["version"])
            
        print("="*60)
        
        # Language selection
        if L == "ru":
            print("\nЯзык интерфейса установлен на русский")
        else:
            print("\nInterface language set to English")
        
        # Check for Sage installation
        sage_version = self._current_sage_version()
        if sage_version != "not installed":
            LOG.info(T["sage_detected"].format(version=sage_version))
        else:
            LOG.info(T["no_sage_installed"])
            
        # Show current torch/cuda versions
        self.current_torch, self.current_cuda = self._current_torch_cuda()
        
        if not self.current_torch or not self.current_cuda:
            LOG.warning("Could not detect installed PyTorch/CUDA version")
        else:
            LOG.info(T["detected_versions"].format(torch_ver=self.current_torch, cuda_ver=self.current_cuda))
            
        print("\n" + "="*60)
        input("Нажмите Enter для продолжения...")

    # ------------------------------------------------------------------
    # System diagnostic
    # ------------------------------------------------------------------
    def system_diagnostic(self) -> Tuple[str, str]:
        """Show current installation status."""
        LOG.info("=" * 60)
        LOG.info(T["system_diagnostic"])
        LOG.info("=" * 60)

        # Show Sage version if installed
        sage_version = self._current_sage_version()
        if sage_version != "not installed":
            print(f"✅ {T['sage_detected'].format(version=sage_version)}")
        else:
            print(f"❌ {T['no_sage_installed']}")
            
        # Show current torch/cuda versions
        self.current_torch, self.current_cuda = self._current_torch_cuda()
        
        if not self.current_torch or not self.current_cuda:
            LOG.warning("Could not detect installed PyTorch/CUDA version")
        else:
            print(f"🔧 {T['detected_versions'].format(torch_ver=self.current_torch, cuda_ver=self.current_cuda)}")
            
        # Check for latest version
        if self.current_torch == "2.9.0" and self.current_cuda == "12.10":
            LOG.info("✅ Latest version already installed - no PyTorch reinstallation needed")
            return self.current_torch, self.current_cuda
            
        print("=" * 60)
        
        # Wait for user confirmation
        input("Press Enter to continue...")

    # ------------------------------------------------------------------
    # Version selection with smart defaults
    # ------------------------------------------------------------------
    def select_version(self) -> VersionConfig:
        """Prompt user to select a version with smart recommendations."""
        LOG.info(T["select_version"])
        
        # Check if we should skip PyTorch installation (if latest already installed)
        if self.current_torch == "2.9.0" and self.current_cuda == "12.10":
            print("✅ Latest PyTorch version is already installed - skipping reinstallation")
            return VERSIONS[-1]  # Return the latest version for Sage installation
        
        # Display available versions
        for idx, ver in enumerate(VERSIONS):
            tags = []
            
            # Check if this is the currently installed version
            current_version_idx = self._find_matching_version(self.current_torch, self.current_cuda)
            if current_version_idx == idx:
                tags.append(T["tag_installed"])
            
            # Mark latest version
            if ver.is_latest:
                tags.append(T["tag_latest"])
                
            tag_str = " ".join(tags) if tags else ""
            print(T["version_option"].format(
                idx=idx + 1,
                torch=ver.torch_version,
                cuda=ver.cuda_version,
                tags=tag_str
            ))
        
        # Set default choice - smart logic based on current installation
        if self.current_torch == "2.9.0" and self.current_cuda == "12.10":
            # Already latest version, choose last (latest) for Sage only
            default_choice = len(VERSIONS)
        elif current_version_idx is not None:
            # There's an existing installation, keep it unless outdated
            default_choice = current_version_idx + 1
        else:
            # No installation at all - recommend latest for new users
            default_choice = len(VERSIONS)  # Latest is last item
        
        print("="*60)
        
        # Get user choice with default set to latest when appropriate
        while True:
            try:
                if rich_available and Prompt:
                    choice = Prompt.ask("Choice", default=str(default_choice))
                else:
                    choice = input(f"Choice (1-{len(VERSIONS)}, default={default_choice}): ").strip() or str(default_choice)
                
                idx = int(choice) - 1
                if 0 <= idx < len(VERSIONS):
                    self.selected_version = VERSIONS[idx]
                    LOG.info(f"Selected: {self.selected_version.name}")
                    return self.selected_version
                else:
                    print(f"Please enter a number between 1 and {len(VERSIONS)}")
            except (ValueError, KeyboardInterrupt):
                print(f"Invalid input. Using default choice ({default_choice}).")
                self.selected_version = VERSIONS[default_choice - 1]
                return self.selected_version

    # ------------------------------------------------------------------
    # Installation steps
    # ------------------------------------------------------------------
    def install_pytorch(self, version: VersionConfig) -> Tuple[bool, str]:
        """Install PyTorch with specified version."""
        try:
            # Check if we already have latest installed - skip installation if so
            if self.current_torch == "2.9.0" and self.current_cuda == "12.10":
                LOG.info("✅ Latest PyTorch is already installed, skipping reinstallation")
                return True, ""
            
            LOG.info("\n" + "="*60)
            LOG.warning(T["disclaimer_header"])
            LOG.warning(T["disclaimer_common"])
            LOG.info("="*60)
            
            ans = input(T["confirm_install"]).strip().lower()
            if ans not in ("y", "yes", "д", "да"):
                LOG.info(T["cancelled"])
                return False, "User cancelled"
            
            LOG.info(T["installing_pytorch"].format(ver=version.torch_version, cuda=version.cuda_version))
            
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
        
        # Check if Sage is already installed and matches target version
        current_sage = self._current_sage_version()
        if current_sage != "not installed":
            print(f"✅ SAGE already installed (v{current_sage}) - skipping")
            return True, ""
            
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
            # Step 0: Welcome screen
            self.welcome_screen()
            
            # Step 1: System diagnostic
            self.system_diagnostic()
            
            # Step 2: Version selection (smart defaults)
            version = self.select_version()
            
            # Step 3: PyTorch installation (if needed)
            if not (self.current_torch == "2.9.0" and self.current_cuda == "12.10"):
                ok_torch, _ = self.install_pytorch(version)
                if not ok_torch:
                    return 1
                print()
            
            # Step 4: Include/libs
            ok_extract, _ = self.download_and_extract_include_libs()
            if not ok_extract:
                return 1
            print()
            
            # Step 5: Triton
            ok_triton, _ = self.install_triton(version)
            if not ok_triton:
                return 1
            print()
            
            # Step 6: SageAttention (install only if needed)
            ok_sage, _ = self.install_sage_attention(version)
            if not ok_sage:
                return 1
            
            # Summary
            steps = [
                ("PyTorch", "not installed" if self.current_torch == "2.9.0" and self.current_cuda == "12.10" else f"{version.torch_version}"),
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
