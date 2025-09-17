#!/usr/bin/env python3
# installer_core.py
#
import logging, os, re, sys, shutil, subprocess, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import threading
import time

# Optional Rich imports for enhanced UI
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    rich_available = True
except ImportError:  # pragma: no cover
    rich_available = False
    Console = None
    RichHandler = None
    Table = None

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
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
                [ps, "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-Culture).Name"],
                capture_output=True,
                text=True,
                timeout=3,
                encoding="utf-8",
                errors="replace"
            )
            culture = (r.stdout or "").strip().lower()
            if culture.startswith("ru"):
                return "ru"
            if culture.startswith("en"):
                return "en"
    except Exception:  # pragma: no cover
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
        "intro": "Старт установки ускорителя ComfyUI (ядро).",
        "checking_torch": "Шаг 1/4: Проверка версий torch, torchvision, torchaudio...",
        "detected_versions": "Обнаружено: torch={torch_ver}, CUDA={cuda_ver}",
        "target_versions": "Цель: torch={target_torch} (CUDA {target_cuda})",
        "mismatch": "Несовместимость версий: требуется соответствие целевым версиям.",
        "single_choice": "Требуется переустановка до целевой версии. Выполнить ПЕРЕУСТАНОВКУ сейчас? (y/N): ",
        "disclaimer_header": "ВНИМАНИЕ: переустановка PyTorch может повлиять на другие пайплайны.",
        "disclaimer_common": "Загрузка ~2.8ГБ; это может занять значительное время.",
        "cancelled": "Операция отменена пользователем. Установка остановлена.",
        "pip_index": "Используется индекс PyTorch: {url}",
        "torch_fix_start": "Переустановка torch, torchvision, torchaudio до {ver} (CUDA {cuda})...",
        "torch_fix_done": "Переустановка пакетов PyTorch завершена успешно.",
        "torch_fix_fail": "Не удалось переустановить пакеты PyTorch: {err}",
        "setup_inc_libs": "Шаг 2/4: Распаковка include/libs...",
        "setup_ok": "Папки include/libs готовы.",
        "setup_fail": "Ошибка распаковки include/libs: {err}",
        "install_triton": "Шаг 3/4: Установка Triton (triton-windows<3.4)...",
        "install_triton_ok": "Triton установлен.",
        "install_triton_fail": "Ошибка установки Tritон: {err}",
        "install_sage": "Шаг 4/4: Установка SageAttention из wheel...",
        "sage_ok": "SageAttention установлен.",
        "sage_fail": "Ошибка установки SageAttention: {err}",
        "wheel_bad": "Wheel несовместим (ожидалось torch {expected}, фактически {actual}). Причина: {why}",
        "download_fail": "Ошибка загрузки: {err}",
        "stop_due_to_torch": "Установка остановлена из‑за несовместимости torch/CUDA.",
        "report_title": "\nИтоговый отчёт:",
        "report_line": "- {name}: {status}",
        "status_ok": "успешно",
        "status_fail": "ошибка",
        "goodbye": "Готово.",
    },
    "en": {
        "intro": "Starting ComfyUI accelerator install (core).",
        "checking_torch": "Step 1/4: Checking torch, torchvision, torchaudio versions...",
        "detected_versions": "Detected: torch={torch_ver}, CUDA={cuda_ver}",
        "target_versions": "Target: torch={target_torch} (CUDA {target_cuda})",
        "mismatch": "Version mismatch: exact match to target versions is required.",
        "single_choice": "Reinstall to target version now? (y/N): ",
        "disclaimer_header": "WARNING: reinstalling PyTorch may affect other pipelines.",
        "disclaimer_common": "Download size is ~2.8GB; this may take a while.",
        "cancelled": "Operation cancelled by user. Installation stopped.",
        "pip_index": "Using PyTorch index: {url}",
        "torch_fix_start": "Reinstalling torch, torchvision, torchaudio to {ver} (CUDA {cuda})...",
        "torch_fix_done": "PyTorch packages reinstall completed successfully.",
        "torch_fix_fail": "Failed to reinstall PyTorch packages: {err}",
        "setup_inc_libs": "Step 2/4: Unpacking include/libs...",
        "setup_ok": "include/libs are ready.",
        "setup_fail": "include/libs extraction error: {err}",
        "install_triton": "Step 3/4: Installing Triton (triton-windows<3.4)...",
        "install_triton_ok": "Triton installed.",
        "install_triton_fail": "Triton installation failed: {err}",
        "install_sage": "Step 4/4: Installing SageAttention from wheel...",
        "sage_ok": "SageAttention installed.",
        "sage_fail": "SageAttention installation failed: {err}",
        "wheel_bad": "Wheel incompatible (expected torch {expected}, actual {actual}). Reason: {why}",
        "download_fail": "Download error: {err}",
        "stop_due_to_torch": "Installation stopped due to torch/CUDA incompatibility.",
        "report_title": "\nFinal report:",
        "report_line": "- {name}: {status}",
        "status_ok": "success",
        "status_fail": "failure",
        "goodbye": "Done.",
    },
}[L]

# ----------------------------------------------------------------------
# Target versions
# ----------------------------------------------------------------------
PYTORCH_MIN = (2, 8, 0)
PYTORCH_MAX = (2, 8, 0)

TARGET_TORCH_VERSION = "2.8.0"
TARGET_CUDA_MAJOR = "12"
TARGET_CUDA_MINOR = "9"
TARGET_CUDA_FULL = f"{TARGET_CUDA_MAJOR}.{TARGET_CUDA_MINOR}"
PYTORCH_INDEX_URL = f"https://download.pytorch.org/whl/cu{TARGET_CUDA_MAJOR}{TARGET_CUDA_MINOR}"

# Sub‑package versions that match the target torch
TORCHVISION_VERSION = "0.15.1"
TORCHAUDIO_VERSION = "2.8.0"

# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------
@dataclass
class InstallConfig:
    triton_pin: str = "triton-windows<3.4"
    repo_base: str = (
        "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
    )
    include_zip: str = "python_3.12.7_include_libs.zip"
    sage_wheel_name_urlenc: str = (
        "sageattention-2.2.0%2Bcu128torch2.8.0.post2-cp39-abi3-win_amd64.whl"
    )
    sage_wheel_local: str = (
        "sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl"
    )
    max_total_uncompressed: int = 600 * 1024 * 1024

# ----------------------------------------------------------------------
# Installer core
# ----------------------------------------------------------------------
class InstallerCore:
    def __init__(self, python_exe: str | None = None, cfg: InstallConfig | None = None):
        self.cfg = cfg or InstallConfig()
        self.python = python_exe or sys.executable
        if not shutil.which(self.python):
            raise FileNotFoundError(f"Python executable not found at: {self.python}")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _run(self, args: list[str], timeout: int = 3600) -> Tuple[bool, str]:
        try:
            r = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
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
        return v.split("+", 1)[0] if v else v

    @classmethod
    def _parse_semver(cls, v: str) -> Tuple[int, int, int]:
        core = cls._strip_local(v or "")
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", core)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)

    def _current_torch_cuda(self) -> Tuple[str, str]:
        code = """\
import sys
try:
    import torch
    print(getattr(torch, '__version__', ''))
    cuda_ver = getattr(getattr(torch, 'version', None), 'cuda', '')
    print(cuda_ver if cuda_ver else '')
except (ImportError, AttributeError):
    print('')
    print('')
"""
        ok, out = self._run([self.python, "-c", code], timeout=30)
        lines = [s.strip() for s in (out or "").splitlines()] if ok else []
        cur_t = lines[0] if len(lines) > 0 else ""
        cur_c = lines[1] if len(lines) > 1 else ""
        return cur_t, cur_c

    @staticmethod
    def _spinner(msg: str = "") -> threading.Event:
        stop_event = threading.Event()
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        def spin():
            while not stop_event.is_set():
                for ch in frames:
                    print(f"\r{msg} {ch}", end="", flush=True)
                    time.sleep(0.1)
        t = threading.Thread(target=spin, daemon=True)
        t.start()
        return stop_event

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    def ensure_torch_compatible_interactive(self) -> Tuple[bool, str]:
        LOG.info(T["checking_torch"])
        cur_t, cur_c = self._current_torch_cuda()
        shown_t = cur_t or ("not installed" if L == "en" else "не установлен")
        shown_c = cur_c or ("unknown" if L == "en" else "неизвестно")
        LOG.info(T["detected_versions"].format(torch_ver=shown_t, cuda_ver=shown_c))
        LOG.info(
            T["target_versions"]
            .format(target_torch=TARGET_TORCH_VERSION, target_cuda=TARGET_CUDA_FULL)
        )

        cur_tuple = self._parse_semver(cur_t)
        need_fix = (
            not cur_t
            or cur_tuple < PYTORCH_MIN
            or cur_tuple > PYTORCH_MAX
            or not cur_c
            or cur_c != TARGET_CUDA_FULL
        )
        if not need_fix:
            LOG.info("Torch is already compatible.")
            return True, "OK"

        LOG.warning(T["mismatch"])
        LOG.warning("\n" + "=" * 60)
        LOG.warning(T["disclaimer_header"])
        LOG.warning(T["disclaimer_common"])
        LOG.warning("=" * 60)

        ans = os.environ.get("COMFYUI_ACC_AUTO_TORCH_FIX", "") or input(
            T["single_choice"]
        ).strip().lower()
        if ans not in ("y", "yes", "д", "да", "1", "true"):
            LOG.info(T["cancelled"])
            return False, "User declined"

        LOG.info(T["pip_index"].format(url=PYTORCH_INDEX_URL))
        LOG.info(
            T["torch_fix_start"]
            .format(ver=TARGET_TORCH_VERSION, cuda=TARGET_CUDA_FULL)
        )

        torch_pkg = f"torch=={TARGET_TORCH_VERSION}+cu{TARGET_CUDA_MAJOR}{TARGET_CUDA_MINOR}"
        torchvision_pkg = (
            f"torchvision=={TORCHVISION_VERSION}+cu{TARGET_CUDA_MAJOR}{TARGET_CUDA_MINOR}"
        )
        torchaudio_pkg = (
            f"torchaudio=={TORCHAUDIO_VERSION}+cu{TARGET_CUDA_MAJOR}{TARGET_CUDA_MINOR}"
        )

        stop_spinner = self._spinner("Installing PyTorch")
        ok, out = self._pip(
            "install",
            torch_pkg,
            torchvision_pkg,
            torchaudio_pkg,
            "--force-reinstall",
            "--index-url",
            PYTORCH_INDEX_URL,
        )
        stop_spinner.set()
        print("\r" + " " * 80 + "\r", end="")  # clear spinner line

        if not ok:
            LOG.error(T["torch_fix_fail"].format(err=out))
            return False, out
        LOG.info(T["torch_fix_done"])
        return True, "Reinstalled"

    def download_and_extract_include_libs(self) -> Tuple[bool, str]:
        url = f"{self.cfg.repo_base}/{self.cfg.include_zip}"
        dest_path = Path.cwd() / self.cfg.include_zip
        LOG.info(T["setup_inc_libs"])

        try:
            # Download with progress bar if available
            if tqdm:
                with urllib.request.urlopen(url) as resp:
                    total = int(resp.getheader("Content-Length") or 0)
                    with open(dest_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading include/libs",
                    ) as pbar:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                urllib.request.urlretrieve(url, dest_path)

            # Extract zip with progress bar if available
            with zipfile.ZipFile(dest_path) as zf:
                members = zf.infolist()
                if tqdm:
                    for member in tqdm(members, desc="Extracting include/libs"):
                        zf.extract(member)
                else:
                    zf.extractall()

            LOG.info(T["setup_ok"])
            return True, ""
        except Exception as e:  # pragma: no cover
            LOG.error(T["setup_fail"].format(err=str(e)))
            return False, str(e)
        finally:
            if dest_path.exists():
                try:
                    dest_path.unlink()
                    LOG.debug(f"Removed temporary file {dest_path}")
                except Exception:  # pragma: no cover
                    pass

    def install_triton(self) -> Tuple[bool, str]:
        LOG.info(T["install_triton"])
        stop_spinner = self._spinner("Installing Triton")
        ok, out = self._pip(
            "install",
            self.cfg.triton_pin,
            "--force-reinstall",
        )
        stop_spinner.set()
        print("\r" + " " * 80 + "\r", end="")  # clear spinner line

        if not ok:
            LOG.error(T["install_triton_fail"].format(err=out))
            return False, out
        LOG.info(T["install_triton_ok"])
        return True, ""

    def install_sage_attention(self) -> Tuple[bool, str]:
        LOG.info(T["install_sage"])
        url = f"{self.cfg.repo_base}/{self.cfg.sage_wheel_name_urlenc}"
        dest_path = Path.cwd() / self.cfg.sage_wheel_local
        try:
            # Download wheel with progress bar if available
            if tqdm:
                with urllib.request.urlopen(url) as resp:
                    total = int(resp.getheader("Content-Length") or 0)
                    with open(dest_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading SageAttention",
                    ) as pbar:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                urllib.request.urlretrieve(url, dest_path)

            # Install wheel
            stop_spinner = self._spinner("Installing SageAttention")
            ok, out = self._pip(
                "install",
                str(dest_path),
                "--force-reinstall",
            )
            stop_spinner.set()
            print("\r" + " " * 80 + "\r", end="")  # clear spinner line

            if not ok:
                LOG.error(T["sage_fail"].format(err=out))
                return False, out
            LOG.info(T["sage_ok"])
            return True, ""
        except Exception as e:  # pragma: no cover
            LOG.error(T["download_fail"].format(err=str(e)))
            return False, str(e)
        finally:
            if dest_path.exists():
                try:
                    dest_path.unlink()
                    LOG.debug(f"Removed temporary file {dest_path}")
                except Exception:  # pragma: no cover
                    pass

    def run(self) -> int:
        """Execute the full installation flow."""
        # Step 1 – Torch compatibility
        ok_torch, _ = self.ensure_torch_compatible_interactive()
        if not ok_torch:
            LOG.error("Torch compatibility check failed.")
            return 1

        if console: console.print("")
        else: print()

        # Step 2 – Include/libs extraction
        ok_extract, _ = self.download_and_extract_include_libs()
        if not ok_extract:
            LOG.error("Include/libs extraction failed.")
            return 1

        if console: console.print("")
        else: print()

        # Step 3 – Triton installation
        ok_triton, _ = self.install_triton()
        if not ok_triton:
            LOG.error("Triton installation failed.")
            return 1

        if console: console.print("")
        else: print()

        # Step 4 – SageAttention installation
        ok_sage, _ = self.install_sage_attention()
        if not ok_sage:
            LOG.error("SageAttention installation failed.")
            return 1

        # Summary table
        steps = [
            ("Torch compatibility", ok_torch),
            ("Include/libs extraction", ok_extract),
            ("Triton installation", ok_triton),
            ("SageAttention installation", ok_sage),
        ]

        if rich_available and console is not None:
            table = Table(title="Installation Summary")
            table.add_column("Step", style="cyan", no_wrap=True)
            for name, status in steps:
                status_str = T["status_ok"] if status else T["status_fail"]
                color = "green" if status else "red"
                table.add_row(name, f"[{color}]{status_str}[/{color}]")
            console.print(table)
        else:
            LOG.info(T["report_title"])
            for name, status in steps:
                status_str = T["status_ok"] if status else T["status_fail"]
                LOG.info(f"  - {name}: {status_str}")

        LOG.info(T["goodbye"])
        return 0

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Print intro message with optional styling
    if console:
        console.print(f"[bold magenta]{T['intro']}[/bold magenta]")
    else:
        LOG.info(T["intro"])

    # Add a check for being run directly without a valid executable path
    if not os.path.exists(sys.executable):
        LOG.error(
            "Could not determine a valid Python executable path. Please run with a specific python interpreter."
        )
        sys.exit(1)

    exit_code = InstallerCore().run()
    sys.exit(exit_code)
TORCHVISION_VERSION = "0.23.0" 
