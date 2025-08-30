# installer_core.py (Python 3.11+)
# Minimal addition: interactive language selector (RU/EN) before localization table is built.
# Updated for torch 2.8.0 + SageAttention 2.2.0+cu129torch2.8.0.post2

from __future__ import annotations
import logging, os, re, sys, shutil, ssl, subprocess, urllib.parse, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

LOG = logging.getLogger("installer_core")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------- Language selection (new) --------
def _prompt_language_choice() -> str:
    """
    Interactive language choice.
    Returns 'ru' or 'en'.
    Priority:
      1) ACC_LANG env if already set (respected, no prompt if ACC_LANG_FORCE=1)
      2) Prompt user: 1=RU, 2=EN, Enter = auto-detect
      3) Auto-detect fallback (Windows culture or POSIX LANG/LC_ALL)
    """
    # If caller wants to force env (for automation), skip prompting
    if os.environ.get("ACC_LANG_FORCE", "") in ("1", "true", "TRUE", "yes", "y"):
        lang = os.environ.get("ACC_LANG", "").strip().lower()
        return "ru" if lang == "ru" else "en"

    existing = os.environ.get("ACC_LANG", "").strip().lower()
    if existing in ("ru", "en"):
        return existing

    # Prompt
    print("Select language / Выберите язык:")
    print("  1) RU (Русский)")
    print("  2) EN (English)")
    choice = input("Choice (1/2, Enter=Auto): ").strip()

    if choice == "1":
        return "ru"
    if choice == "2":
        return "en"

    # Auto-detect
    try:
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            r = subprocess.run(
                [ps, "-NoProfile", "-NonInteractive", "-Command", "(Get-Culture).Name"],
                capture_output=True, text=True, timeout=3
            )
            culture = (r.stdout or "").strip().lower()
            if culture.startswith("ru"):
                return "ru"
            if culture.startswith("en"):
                return "en"
    except Exception:
        pass
    loc = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").lower()
    if loc.startswith("ru"):
        return "ru"
    if loc.startswith("en"):
        return "en"
    return "en"

def _detect_lang() -> str:
    # Called after we possibly set ACC_LANG via prompt
    env = os.environ.get("ACC_LANG", "").strip().lower()
    if env in ("ru", "en"):
        return env
    # Fallback to auto
    try:
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            r = subprocess.run(
                [ps, "-NoProfile", "-NonInteractive", "-Command", "(Get-Culture).Name"],
                capture_output=True, text=True, timeout=3
            )
            culture = (r.stdout or "").strip().lower()
            if culture.startswith("ru"):
                return "ru"
            if culture.startswith("en"):
                return "en"
    except Exception:
        pass
    loc = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").lower()
    if loc.startswith("ru"):
        return "ru"
    if loc.startswith("en"):
        return "en"
    return "en"

# Prompt the user once and export ACC_LANG for the session
_selected_lang = _prompt_language_choice()
os.environ["ACC_LANG"] = _selected_lang

# -------- Localization --------
L = _detect_lang()
T = {
    "ru": {
        "intro": "Старт установки ускорителя ComfyUI (ядро).",
        "checking_torch": "Шаг 1/4: Проверка версий torch/CUDA...",
        "detected_versions": "Обнаружено: torch={torch_ver}, CUDA={cuda_ver}",
        "target_versions": "Цель: torch=2.8.0 (CUDA 12.9)",
        "mismatch": "Несовместимость версий: требуется точное соответствие целевым версиям.",
        "single_choice": "Требуется переустановка до целевой версии. Выполнить ПЕРЕУСТАНОВКУ сейчас? (y/N): ",
        "disclaimer_header": "ВНИМАНИЕ: переустановка PyTorch может повлиять на другие пайплайны.",
        "disclaimer_common": "Загрузка ~2.5ГБ; это может занять значительное время. Возможны конфликты с другими окружениями.",
        "cancelled": "Операция отменена пользователем. Установка остановлена.",
        "pip_index": "Используется индекс PyTorch: {url}",
        "torch_fix_start": "Переустановка torch до {ver} (CUDA {cuda})...",
        "torch_fix_done": "Переустановка torch завершена успешно.",
        "torch_fix_fail": "Не удалось переустановить torch: {err}",
        "setup_inc_libs": "Шаг 2/4: Распаковка include/libs...",
        "setup_ok": "Папки include/libs готовы.",
        "setup_fail": "Ошибка распаковки include/libs: {err}",
        "install_triton": "Шаг 3/4: Установка Triton (triton-windows<3.4)...",
        "install_triton_ok": "Triton установлен.",
        "install_triton_fail": "Ошибка установки Triton: {err}",
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
        "checking_torch": "Step 1/4: Checking torch/CUDA versions...",
        "detected_versions": "Detected: torch={torch_ver}, CUDA={cuda_ver}",
        "target_versions": "Target: torch=2.8.0 (CUDA 12.9)",
        "mismatch": "Version mismatch: exact match to target versions is required.",
        "single_choice": "Reinstall to target version now? (y/N): ",
        "disclaimer_header": "WARNING: reinstalling PyTorch may affect other pipelines.",
        "disclaimer_common": "Download size is ~2.5GB; this may take a while. May conflict with other environments.",
        "cancelled": "Operation cancelled by user. Installation stopped.",
        "pip_index": "Using PyTorch index: {url}",
        "torch_fix_start": "Reinstalling torch to {ver} (CUDA {cuda})...",
        "torch_fix_done": "Torch reinstall completed successfully.",
        "torch_fix_fail": "Failed to reinstall torch: {err}",
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

# ---------------- Pins ----------------
PYTORCH_MIN = (2, 8, 0)
PYTORCH_MAX = (2, 8, 0)
TARGET_TORCH = "2.8.0"
TARGET_CUDA = "12.9"
PYTORCH_INDEX_CU129 = "https://download.pytorch.org/whl/cu129"

@dataclass
class InstallConfig:
    triton_pin: str = "triton-windows<3.4"
    repo_base: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
    include_zip: str = "python_3.12.7_include_libs.zip"
    sage_wheel_name_urlenc: str = "sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl"
    sage_wheel_local: str = "sageattention-2.2.0+cu129torch2.8.0.post2-cp39-abi3-win_amd64.whl"
    max_total_uncompressed: int = 600 * 1024 * 1024  # 600MB safety cap for ZIP bombs

# ---------------- Safe I/O helpers ----------------
def _https_url_or_raise(url: str, allowed_hosts: tuple[str, ...] | None = None) -> str:
    """
    Validate HTTPS URL and optional host allow-list.
    Raises ValueError on invalid inputs.
    """
    parts = urllib.parse.urlparse(url)
    if parts.scheme.lower() != "https":
        raise ValueError(f"Only HTTPS is allowed: {url}")
    host = parts.netloc.lower()
    if allowed_hosts and not any(host.endswith(h) for h in allowed_hosts):
        raise ValueError(f"Host not allowed: {host}")
    return url

def _ssl_context_with_certifi() -> ssl.SSLContext:
    """
    Build default SSL context and load certifi CA bundle if available.
    """
    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    try:
        # Try stdlib's bundled pip vendor certifi first to avoid extra dependency.
        from pip._vendor import certifi as pip_certifi  # type: ignore
        ctx.load_verify_locations(cafile=pip_certifi.where())
        return ctx
    except Exception:
        pass
    try:
        import certifi  # optional dependency
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        # Fallback to system CAs already loaded by create_default_context
        pass
    return ctx

def safe_download_https(url: str, dest: Path, allowed_hosts: tuple[str, ...] = ()) -> Tuple[bool, str]:
    """
    Download over HTTPS using urllib with verified SSL; fallback to PowerShell with safe args.
    Returns (ok, message). Never disables certificate verification.
    """
    try:
        _https_url_or_raise(url, allowed_hosts=allowed_hosts or ("github.com", "raw.githubusercontent.com", "download.pytorch.org"))
    except ValueError as e:
        return False, str(e)

    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context_with_certifi()
    req = urllib.request.Request(url, headers={"User-Agent": "python-installer-core/1.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=300) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        if dest.exists() and dest.stat().st_size > 0:
            return True, "OK"
        return False, "Downloaded file missing or empty"
    except Exception as e:
        # Fallback to PowerShell 7+ Invoke-WebRequest with safe parameter passing.
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if not ps:
            return False, f"{T['download_fail'].format(err=e)}"
        try:
            # Use array args, no string concat; set -UseBasicParsing is deprecated, not needed in PS7.
            # Note: -OutFile path is quoted by PowerShell parsing; we pass as separate args.
            cmd = [ps, "-NoProfile", "-NonInteractive", "-Command",
                   "Invoke-WebRequest", "-Uri", url, "-OutFile", str(dest), "-ErrorAction", "Stop"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                return True, "OK"
            return False, f"PowerShell download failed: {r.stderr.strip() or r.stdout.strip()}"
        except Exception as pe:
            return False, f"{T['download_fail'].format(err=pe)}"

def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        base_resolved = base.resolve(strict=False)
        target_resolved = target.resolve(strict=False)
        return str(target_resolved).startswith(str(base_resolved) + os.sep) or (target_resolved == base_resolved)
    except Exception:
        # Fallback: naive check
        return str(target).startswith(str(base) + os.sep)

def safe_extract_zip(zip_path: Path, dest_dir: Path, max_total_uncompressed: int) -> None:
    """
    Safely extract a ZIP archive into dest_dir:
    - Prevent path traversal (canonical path check).
    - Enforce total uncompressed size cap to mitigate zip bombs.
    Raises ValueError on validation failure.
    """
    if not zip_path.exists():
        raise ValueError(f"ZIP not found: {zip_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        total_uncompressed = sum(i.file_size for i in infos)
        if total_uncompressed > max_total_uncompressed:
            raise ValueError(f"Uncompressed size too large: {total_uncompressed} bytes (> {max_total_uncompressed})")

        for info in infos:
            # Normalize path: reject absolute, drive letters, parent traversal
            name = info.filename
            if not name or name.replace("\\", "/").startswith(("/", "\\")):
                raise ValueError(f"Unsafe absolute path in ZIP: {name}")
            if ".." in Path(name).parts:
                raise ValueError(f"Path traversal attempt in ZIP: {name}")

            target_path = dest_dir / name
            if not _is_within_directory(dest_dir, target_path):
                raise ValueError(f"Extraction outside target dir: {name}")

            if name.endswith("/") or name.endswith("\\"):
                (dest_dir / name).mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

# ---------------- Core installer ----------------
class InstallerCore:
    """
    Core installer:
    - Ensure torch core 2.8.0 (accept '2.8.0+cu129') and CUDA 12.9; single 'reinstall' prompt on mismatch.
    - Setup include/libs (safe ZIP extraction).
    - Install Triton 'triton-windows<3.4'.
    - Install SageAttention wheel with platform/ABI checks.
    """
    def __init__(self, python_exe: Optional[str] = None, cfg: Optional[InstallConfig] = None):
        self.cfg = cfg or InstallConfig()
        self.python = python_exe or ("python.exe" if os.path.exists("python.exe") else "python")
        if not shutil.which(self.python) and not os.path.exists(self.python):
            raise FileNotFoundError("python executable not found")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # ----- utils -----
    def _run(self, args: list[str], timeout: int = 3600) -> Tuple[bool, str]:
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)

    def _pip(self, *args: str, timeout: int = 3600) -> Tuple[bool, str]:
        return self._run([self.python, "-m", "pip", *args], timeout=timeout)

    # ----- version helpers -----
    @staticmethod
    def _strip_local(v: str) -> str:
        return v.split("+", 1)[0] if v else v

    @classmethod
    def _parse_semver(cls, v: str) -> Tuple[int, int, int]:
        core = cls._strip_local(v or "")
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", core)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)

    # ----- torch/cuda detection and enforcement -----
    def _current_torch_cuda(self) -> Tuple[str, str]:
        code = (
            "import sys\n"
            "try:\n"
            " import torch\n"
            " print(getattr(torch,'__version__',''))\n"
            " print(getattr(getattr(torch,'version',None),'cuda',''))\n"
            "except Exception:\n"
            " print('')\n"
            " print('')\n"
        )
        ok, out = self._run([self.python, "-c", code], timeout=30)
        lines = [s.strip() for s in (out or "").splitlines()] if ok else []
        cur_t = lines[0] if len(lines) >= 1 else ""
        cur_c = lines[1] if len(lines) >= 2 else ""
        return cur_t, cur_c

    def ensure_torch_compatible_interactive(self) -> Tuple[bool, str]:
        print(T["checking_torch"])
        cur_t, cur_c = self._current_torch_cuda()
        shown_t = cur_t if cur_t else ("not installed" if L == "en" else "не установлен")
        shown_c = cur_c if cur_c else ("unknown" if L == "en" else "неизвестно")
        print(T["detected_versions"].format(torch_ver=shown_t, cuda_ver=shown_c))
        print(T["target_versions"])

        cur_tuple = self._parse_semver(cur_t)
        need_fix = (not cur_t) or cur_tuple < PYTORCH_MIN or cur_tuple > PYTORCH_MAX or (cur_c and cur_c != TARGET_CUDA)
        if not need_fix:
            return True, "OK"

        print(T["mismatch"])
        print("\n" + "=" * 60)
        print(T["disclaimer_header"])
        print(T["disclaimer_common"])
        print("=" * 60)
        ans = os.environ.get("COMFYUI_ACC_AUTO_TORCH_FIX", "")
        if not ans:
            ans = input(T["single_choice"]).strip().lower()
        if ans not in ("y", "yes", "д", "да", "1", "true"):
            print(T["cancelled"])
            return False, "User declined"

        print(T["pip_index"].format(url=PYTORCH_INDEX_CU129))
        print(T["torch_fix_start"].format(ver=TARGET_TORCH, cuda=TARGET_CUDA))
        ok, out = self._pip("install", f"torch=={TARGET_TORCH}", "--force-reinstall", "--index-url", PYTORCH_INDEX_CU129)
        if not ok:
            return False, T["torch_fix_fail"].format(err=out[-800:])
        print(T["torch_fix_done"])
        return True, "Reinstalled"

    # ----- include/libs (safe) -----
    def setup_include_libs(self) -> Tuple[bool, str]:
        print(T["setup_inc_libs"])
        if Path("include").is_dir() and Path("libs").is_dir():
            return True, T["setup_ok"]
        url = f"{self.cfg.repo_base}/{self.cfg.include_zip}"
        tmp = Path("temp_include_libs.zip")
        ok, msg = safe_download_https(url, tmp)
        if not ok:
            return False, msg
        try:
            safe_extract_zip(tmp, Path("."), self.cfg.max_total_uncompressed)
        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False, T["setup_fail"].format(err=e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if Path("include").is_dir() and Path("libs").is_dir():
            return True, T["setup_ok"]
        return False, "Extraction finished but include/libs not found"

    # ----- Triton -----
    def install_triton(self) -> Tuple[bool, str]:
        print(T["install_triton"])
        ok, out = self._pip("install", "-U", self.cfg.triton_pin)
        return (True, T["install_triton_ok"]) if ok else (False, T["install_triton_fail"].format(err=out[-800:]))

    # ----- SageAttention -----
    def _validate_wheel(self, wheel_path: Path) -> Tuple[bool, str]:
        """
        Validate platform and torch compatibility for the wheel.
        Accept torch reported as '2.8.0+cu129' by stripping '+...' for comparison.
        """
        name = wheel_path.name.lower()
        py = sys.version_info
        cp_tag = f"cp{py.major}{py.minor}"
        if "win_amd64" not in name:
            return False, "not win_amd64"
        if cp_tag not in name and "abi3" not in name:
            return False, f"cp tag mismatch (need {cp_tag} or abi3)"
        cur_t, _ = self._current_torch_cuda()
        core = self._strip_local(cur_t)
        if "cu129" in name and core != TARGET_TORCH:
            return False, f"expected core {TARGET_TORCH}, got {cur_t}"
        return True, "OK"

    def install_sageattention(self) -> Tuple[bool, str]:
        print(T["install_sage"])
        wheel_url = f"{self.cfg.repo_base}/{self.cfg.sage_wheel_name_urlenc}"
        local = Path(self.cfg.sage_wheel_local)
        ok, msg = safe_download_https(wheel_url, local)
        if not ok:
            return False, msg
        if not local.exists():
            return False, T["download_fail"].format(err="wheel not found after download")
        ok, why = self._validate_wheel(local)
        if not ok:
            try:
                local.unlink(missing_ok=True)
            except Exception:
                pass
            return False, T["wheel_bad"].format(expected=TARGET_TORCH, actual=(self._current_torch_cuda()[0] or "n/a"), why=why)
        ok, out = self._pip("install", str(local))
        if not ok:
            ok, out = self._pip("install", "--force-reinstall", str(local))
        try:
            local.unlink(missing_ok=True)
        except Exception:
            pass
        return (True, T["sage_ok"]) if ok else (False, T["sage_fail"].format(err=out[-800:]))

# ---------------- Orchestration ----------------
def _print_report(results: Dict[str, bool]) -> None:
    print(T["report_title"])
    names = {
        "torch": "PyTorch (2.8.0+cu129)",
        "include_libs": "include/libs",
        "triton": "Triton (<3.4)",
        "sageattention": "SageAttention wheel",
    }
    for key in ["torch", "include_libs", "triton", "sageattention"]:
        status = T["status_ok"] if results.get(key, False) else T["status_fail"]
        print(T["report_line"].format(name=names[key], status=status))
    print(T["goodbye"])

def smart_core_flow() -> int:
    print(T["intro"])
    results: Dict[str, bool] = {"torch": False, "include_libs": False, "triton": False, "sageattention": False}
    inst = InstallerCore()

    ok_t, _ = inst.ensure_torch_compatible_interactive()
    results["torch"] = ok_t
    if not ok_t:
        print(T["stop_due_to_torch"])
        _print_report(results)
        return 1

    ok_inc, _ = inst.setup_include_libs()
    results["include_libs"] = ok_inc
    if not ok_inc:
        _print_report(results)
        return 1

    ok_tr, _ = inst.install_triton()
    results["triton"] = ok_tr
    if not ok_tr:
        _print_report(results)
        return 1

    ok_sg, _ = inst.install_sageattention()
    results["sageattention"] = ok_sg

    _print_report(results)
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    raise SystemExit(smart_core_flow())
