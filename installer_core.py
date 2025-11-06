#!/usr/bin/env python3
# installer_core.py
# ComfyUI Accelerator Installer v3.0
# Bias: Latest stable (PyTorch 2.9.0 + CUDA 12.10)

import logging
import os
import platform
import re
import sys
import shutil
import subprocess
import urllib.request
import zipfile
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

# Optional Rich imports for enhanced UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Optional tqdm for fallback progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════
# Constants and Configuration
# ══════════════════════════════════════════════════════════════════════

VERSION = "3.0"
REPO_BASE = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"

@dataclass
class VersionConfig:
    """Configuration for a specific CUDA/Torch version."""
    name: str
    torch_version: str
    cuda_version: str
    pytorch_index_url: str
    torchvision_version: str
    torchaudio_version: str
    sage_wheel_urlenc: str
    sage_wheel_local: str
    triton_pin: str
    is_latest: bool = False
    benefits: List[str] = None

# Available configurations (ordered: stable → latest)
VERSIONS = [
    VersionConfig(
        name="Torch 2.8.0 + CUDA 12.9",
        torch_version="2.8.0",
        cuda_version="12.9",
        pytorch_index_url="https://download.pytorch.org/whl/cu129",
        torchvision_version="0.23.0",
        torchaudio_version="2.8.0",
        sage_wheel_urlenc="sageattention-2.2.0%2Bcu128torch2.8.0.post2-cp39-abi3-win_amd64.whl",
        sage_wheel_local="sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl",
        triton_pin="triton-windows<3.4",
        is_latest=False,
        benefits=["Stable", "Well-tested"]
    ),
    VersionConfig(
        name="Torch 2.9.0 + CUDA 12.10",
        torch_version="2.9.0",
        cuda_version="12.10",
        pytorch_index_url="https://download.pytorch.org/whl/cu1210",
        torchvision_version="0.24.0",
        torchaudio_version="2.9.0",
        sage_wheel_urlenc="sageattention-2.2.0%2Bcu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
        sage_wheel_local="sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
        triton_pin="triton-windows<3.5",
        is_latest=True,
        benefits=["10-15% faster", "Latest features", "Long-term support"]
    ),
]

INCLUDE_ZIP = "python_3.12.7_include_libs.zip"

# ══════════════════════════════════════════════════════════════════════
# Logging Setup
# ══════════════════════════════════════════════════════════════════════

LOG_FILE = "installation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
LOG = logging.getLogger("installer")

# ══════════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════════

def print_banner(text: str, width: int = 70) -> None:
    """Print a centered banner."""
    if RICH_AVAILABLE:
        console.print(Panel(f"[bold magenta]{text}[/bold magenta]", expand=False, border_style="magenta"))
    else:
        border = "═" * width
        padding = (width - len(text) - 2) // 2
        print(f"\n{border}")
        print(f"║{' ' * padding}{text}{' ' * (width - len(text) - padding - 2)}║")
        print(f"{border}\n")

def print_section(title: str) -> None:
    """Print a section divider."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")
    else:
        print(f"\n{'═' * 70}")
        print(f"{title}")
        print(f"{'═' * 70}\n")

def run_command(cmd: List[str], timeout: int = 3600) -> Tuple[bool, str]:
    """Execute a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def strip_local_version(version: str) -> str:
    """Strip local version identifier (e.g., '2.8.0+cu129' -> '2.8.0')"""
    return version.split("+")[0] if version else version

# ══════════════════════════════════════════════════════════════════════
# Installer Core Class
# ══════════════════════════════════════════════════════════════════════

class ComfyUIAcceleratorInstaller:
    """Main installer class implementing the full workflow."""
    
    def __init__(self):
        self.python_exe = sys.executable
        self.current_torch = ""
        self.current_cuda = ""
        self.current_triton = ""
        self.current_sage = ""
        self.selected_version: Optional[VersionConfig] = None
        self.installation_results = {}
        
    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Splash Screen
    # ──────────────────────────────────────────────────────────────────
    
    def show_splash_screen(self) -> None:
        """Display welcome banner and version info."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if RICH_AVAILABLE:
            splash = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        🚀 ComfyUI Accelerator Installer                 ║
    ║                   Version 3.0                            ║
    ║                                                          ║
    ║        Optimize your ComfyUI with SageAttention         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
            """
            console.print(splash, style="bold magenta")
        else:
            print("\n" + "="*70)
            print("        🚀 ComfyUI Accelerator Installer")
            print("                   Version 3.0")
            print("="*70 + "\n")
        
        time.sleep(1)
    
    # ──────────────────────────────────────────────────────────────────
    # Stage 2: System Diagnostic
    # ──────────────────────────────────────────────────────────────────
    
    def detect_current_installation(self) -> None:
        """Detect current PyTorch, CUDA, and related packages."""
        LOG.info("Detecting current installation...")
        
        # Detect PyTorch and CUDA
        code = """
import torch
print(torch.__version__)
print(torch.version.cuda if hasattr(torch.version, 'cuda') else '')
"""
        ok, output = run_command([self.python_exe, "-c", code], timeout=30)
        if ok:
            lines = [l.strip() for l in output.splitlines() if l.strip()]
            self.current_torch = lines[0] if len(lines) > 0 else ""
            self.current_cuda = lines[1] if len(lines) > 1 else ""
        
        # Detect Triton
        code_triton = "import triton; print(triton.__version__)"
        ok, output = run_command([self.python_exe, "-c", code_triton], timeout=10)
        if ok:
            self.current_triton = output.strip().splitlines()[0] if output.strip() else ""
        
        # Detect SageAttention
        code_sage = "import sageattention; print(sageattention.__version__)"
        ok, output = run_command([self.python_exe, "-c", code_sage], timeout=10)
        if ok:
            self.current_sage = output.strip().splitlines()[0] if output.strip() else ""
    
    def show_diagnostic_screen(self) -> None:
        """Display system diagnostic information."""
        print_section("📊 System Diagnostic")
        
        # Environment info
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        os_info = f"{platform.system()} {platform.release()}"
        
        if RICH_AVAILABLE:
            table = Table(title="Current Environment", show_header=True, header_style="bold cyan")
            table.add_column("Component", style="cyan", width=20)
            table.add_column("Version/Status", style="green")
            
            table.add_row("Python", py_version)
            table.add_row("Platform", os_info)
            table.add_row("", "")
            table.add_row("PyTorch", self.current_torch or "[red]Not installed[/red]")
            table.add_row("CUDA", self.current_cuda or "[yellow]Unknown[/yellow]")
            table.add_row("Triton", self.current_triton or "[red]Not installed[/red]")
            table.add_row("SageAttention", self.current_sage or "[red]Not installed[/red]")
            
            console.print(table)
        else:
            print(f"Environment:")
            print(f"  Python:        {py_version}")
            print(f"  Platform:      {os_info}")
            print(f"\nCurrent Installation:")
            print(f"  PyTorch:       {self.current_torch or 'Not installed'}")
            print(f"  CUDA:          {self.current_cuda or 'Unknown'}")
            print(f"  Triton:        {self.current_triton or 'Not installed'}")
            print(f"  SageAttention: {self.current_sage or 'Not installed'}")
        
        # Check if update available
        latest_version = VERSIONS[-1]
        current_torch_clean = strip_local_version(self.current_torch)
        
        if self.current_torch and current_torch_clean != latest_version.torch_version:
            print(f"\n💡 [bold yellow]Update Available:[/bold yellow]" if RICH_AVAILABLE else "\n💡 Update Available:")
            print(f"   → PyTorch {latest_version.torch_version} + CUDA {latest_version.cuda_version} (Latest)")
            print(f"   → Improved performance and stability")
            print(f"   → Recommended for new features")
        
        print("\n" + "="*70)
        input("Press Enter to continue...")
    
    # ──────────────────────────────────────────────────────────────────
    # Stage 3: Version Selection
    # ──────────────────────────────────────────────────────────────────
    
    def select_target_version(self) -> VersionConfig:
        """Let user select target configuration with smart defaults."""
        print_section("🔧 Select Target Configuration")
        
        # Determine smart default
        current_torch_clean = strip_local_version(self.current_torch)
        latest_version = VERSIONS[-1]
        
        # Find matching version index
        current_match_idx = None
        for idx, ver in enumerate(VERSIONS):
            if ver.torch_version == current_torch_clean and ver.cuda_version == self.current_cuda:
                current_match_idx = idx
                break
        
        # Default logic: prefer latest unless already on latest
        if current_match_idx is not None and current_match_idx == len(VERSIONS) - 1:
            # Already on latest
            default_idx = current_match_idx
            print("✨ You're already on the latest version!")
        else:
            # Recommend upgrade to latest
            default_idx = len(VERSIONS) - 1
            if self.current_torch:
                print("We recommend upgrading to the latest version for:")
                print("  • Better performance and stability")
                print("  • Support for newest ComfyUI features")
                print("  • Longer-term compatibility")
            else:
                print("Starting fresh? Get the best experience with the latest version!")
        
        print(f"\n{'='*70}")
        print("Available configurations:\n")
        
        # Display options
        for idx, ver in enumerate(VERSIONS):
            tags = []
            
            # Check if currently installed
            if current_match_idx == idx:
                tags.append("✓ Currently Installed")
            
            # Mark latest
            if ver.is_latest:
                tags.append("🆕 Latest")
            
            # Mark recommended (default)
            if idx == default_idx:
                tags.append("⭐ RECOMMENDED")
            
            tag_str = "  " + "  ".join(tags) if tags else ""
            
            print(f"  {idx + 1}) {ver.name}{tag_str}")
            
            # Show benefits for latest
            if ver.is_latest and ver.benefits:
                for benefit in ver.benefits:
                    print(f"     • {benefit}")
            
            print(f"     ├─ SageAttention: {ver.sage_wheel_local.split('-')[1]}")
            print(f"     ├─ Triton:        {ver.triton_pin}")
            print(f"     └─ Download:      ~2.8-2.9 GB\n")
        
        print(f"{'='*70}\n")
        
        # Get user choice
        while True:
            try:
                if RICH_AVAILABLE:
                    choice = Prompt.ask("Choice", default=str(default_idx + 1))
                else:
                    choice = input(f"Choice (1-{len(VERSIONS)}, default={default_idx + 1}): ").strip()
                    choice = choice or str(default_idx + 1)
                
                idx = int(choice) - 1
                if 0 <= idx < len(VERSIONS):
                    self.selected_version = VERSIONS[idx]
                    LOG.info(f"Selected: {self.selected_version.name}")
                    return self.selected_version
                else:
                    print(f"Please enter a number between 1 and {len(VERSIONS)}")
            except (ValueError, KeyboardInterrupt):
                print(f"Invalid input. Using default ({default_idx + 1}).")
                self.selected_version = VERSIONS[default_idx]
                return self.selected_version
    
    # ──────────────────────────────────────────────────────────────────
    # Stage 4: Confirmation
    # ──────────────────────────────────────────────────────────────────
    
    def confirm_installation(self, version: VersionConfig) -> bool:
        """Show upgrade plan and get user confirmation."""
        print_section("📦 Installation Confirmation")
        
        # Show what will be installed
        print("You are about to install:")
        print(f"  • PyTorch {version.torch_version} + CUDA {version.cuda_version}")
        print(f"  • torchvision {version.torchvision_version}")
        print(f"  • torchaudio {version.torchaudio_version}")
        print(f"  • Triton {version.triton_pin}")
        print(f"  • SageAttention (optimized wheel)")
        print(f"  • include/libs for compilation")
        
        print(f"\nDownload size: ~2.8-2.9 GB")
        print(f"Estimated time: 5-15 minutes (depends on connection)")
        
        # Show upgrade path if applicable
        if self.current_torch and strip_local_version(self.current_torch) != version.torch_version:
            print(f"\n📈 Upgrade Path:")
            print(f"  {self.current_torch} → {version.torch_version}+cu{version.cuda_version.replace('.', '')}")
        
        print(f"\n⚠️  WARNING:")
        print(f"  • This will REINSTALL PyTorch packages")
        print(f"  • Other Python environments may be affected")
        print(f"  • Ensure stable internet connection")
        
        print(f"\n{'='*70}\n")
        
        # Get confirmation
        if RICH_AVAILABLE:
            return Confirm.ask("Continue with installation?", default=False)
        else:
            ans = input("Continue with installation? (y/N): ").strip().lower()
            return ans in ('y', 'yes')
    
    # ──────────────────────────────────────────────────────────────────
    # Stage 5: Installation Process
    # ──────────────────────────────────────────────────────────────────
    
    def install_pytorch(self, version: VersionConfig) -> bool:
        """Install PyTorch packages."""
        LOG.info(f"Installing PyTorch {version.torch_version}...")
        
        packages = [
            f"torch=={version.torch_version}",
            f"torchvision=={version.torchvision_version}",
            f"torchaudio=={version.torchaudio_version}",
            "--force-reinstall",
            "--index-url",
            version.pytorch_index_url
        ]
        
        cmd = [self.python_exe, "-m", "pip", "install"] + packages
        ok, output = run_command(cmd, timeout=1800)
        
        if ok:
            LOG.info("✅ PyTorch installed successfully")
        else:
            LOG.error(f"❌ PyTorch installation failed: {output[-500:]}")
        
        return ok
    
    def download_and_extract_include_libs(self) -> bool:
        """Download and extract include/libs."""
        LOG.info("Downloading include/libs...")
        
        url = f"{REPO_BASE}/{INCLUDE_ZIP}"
        dest = Path(INCLUDE_ZIP)
        
        try:
            # Download
            urllib.request.urlretrieve(url, dest)
            
            # Extract
            LOG.info("Extracting include/libs...")
            with zipfile.ZipFile(dest, 'r') as zf:
                zf.extractall()
            
            LOG.info("✅ include/libs ready")
            return True
            
        except Exception as e:
            LOG.error(f"❌ include/libs extraction failed: {e}")
            return False
        finally:
            dest.unlink(missing_ok=True)
    
    def install_triton(self, version: VersionConfig) -> bool:
        """Install Triton."""
        LOG.info(f"Installing Triton {version.triton_pin}...")
        
        cmd = [self.python_exe, "-m", "pip", "install", "-U", version.triton_pin, "--force-reinstall"]
        ok, output = run_command(cmd, timeout=600)
        
        if ok:
            LOG.info("✅ Triton installed")
        else:
            LOG.error(f"❌ Triton installation failed: {output[-500:]}")
        
        return ok
    
    def install_sageattention(self, version: VersionConfig) -> bool:
        """Install SageAttention wheel."""
        LOG.info("Installing SageAttention...")
        
        url = f"{REPO_BASE}/{version.sage_wheel_urlenc}"
        dest = Path(version.sage_wheel_local)
        
        try:
            # Download wheel
            urllib.request.urlretrieve(url, dest)
            
            # Install
            cmd = [self.python_exe, "-m", "pip", "install", str
