#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Accelerator v3.0 - Installation Module
"""

import subprocess
import urllib.request
import os
import logging
from typing import Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class InstallConfig:
    triton_command: str = "triton-windows<3.4"  # всегда
    repo_base: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
    sage_wheel: str = "sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    sage_wheel_local: str = "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    include_zip: str = "python_3.12.7_include_libs.zip"

class ComponentInstaller:
    def __init__(self, config: InstallConfig | None = None) -> None:
        self.config = config or InstallConfig()
        self.logger = logging.getLogger(__name__)
        self.python_exe = self._find_python()

    def _find_python(self) -> str:
        if os.path.exists("python.exe"): return "python.exe"
        if os.path.exists("python"): return "python"
        raise FileNotFoundError("Python executable not found")

    def _run_pip(self, args: List[str], timeout: int = 300) -> Tuple[bool, str]:
        try:
            r = subprocess.run([self.python_exe, "-m", "pip", *args], capture_output=True, text=True, timeout=timeout)
            return (r.returncode == 0, (r.stdout + r.stderr))
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)

    def _run_powershell(self, cmd: str, timeout: int = 120) -> Tuple[bool, str]:
        try:
            r = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, timeout=timeout)
            return (r.returncode == 0, (r.stdout + r.stderr))
        except Exception as e:
            return False, str(e)

    def upgrade_pip(self) -> Tuple[bool, str]:
        ok, out = self._run_pip(["install", "--upgrade", "pip"])
        return ok, ("pip upgraded successfully" if ok else f"pip upgrade failed: {out}")

    def install_triton(self) -> Tuple[bool, str]:
        ok, out = self._run_pip(["install", "-U", self.config.triton_command])
        return (True, "Triton Windows installed successfully") if ok else (False, f"Triton installation failed: {out}")

    def _download_file(self, url: str, dst: str) -> Tuple[bool, str]:
        ps = f"try {{ Invoke-WebRequest -Uri '{url}' -OutFile '{dst}' -ErrorAction Stop }} catch {{ exit 1 }}"
        ok, out = self._run_powershell(ps)
        if ok and os.path.exists(dst):
            return True, "OK"
        # fallback urllib
        try:
            urllib.request.urlretrieve(url, dst)
            return (True, "OK") if os.path.exists(dst) else (False, "Downloaded file missing")
        except Exception as e:
            return False, f"Download failed: {e}"

    def _validate_wheel_tag(self, wheel_path: str) -> Tuple[bool, str]:
        """Сверка cp-тега и win_amd64 с текущим Python/платформой; torch/cuda совместимость."""
        py_ver = sys.version_info
        cp_tag = f"cp{py_ver.major}{py_ver.minor}"
        name = os.path.basename(wheel_path).lower()
        if "win_amd64" not in name:
            return False, "Wheel is not win_amd64"
        if cp_tag not in name and "abi3" not in name:
            return False, f"Wheel tag mismatch (expected {cp_tag} or abi3)"
        # Проверка торча/куды
        ok, out = self._run_pip(["show", "torch"])
        torch_ver = ""
        if ok:
            m = re.search(r"Version:\s*([^\s]+)", out)
            if m: torch_ver = m.group(1)
        # Если указан cu128 в имени — требуем cuda 12.8 + torch 2.7.x
        if "cu128" in name:
            if not torch_ver.startswith("2.7."):
                return False, f"Incompatible torch {torch_ver}; require 2.7.x for cu128 wheel"
        return True, "OK"

    def _ensure_torch_compatible(self) -> Tuple[bool, str]:
        """Проверить torch/cuda; при несовместимости спросить и выполнить ап/даунгрейд в диапазоне 2.7.1..2.7.9 (cu128)."""
        # выясним текущий torch и cuda
        code = "import torch,sys;print(torch.__version__);print(getattr(torch.version,'cuda', ''))"
        ok, out = self._run_pip(["run", "-q"])  # заглушка для pip; ниже вызов через python -c
        try:
            r = subprocess.run([self.python_exe, "-c", code], capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                return True, "Torch not installed; will proceed"
            lines = [s.strip() for s in (r.stdout or "").splitlines() if s.strip()]
            cur_ver = lines[0] if lines else ""
            cur_cuda = lines[1] if len(lines) > 1 else ""
        except Exception:
            cur_ver, cur_cuda = "", ""
        # требования
        target_cuda = "12.8"
        min_v, max_v = (2,7,1), (2,7,9)
        def parse(v: str) -> Tuple[int,int,int]:
            m = re.match(r"(\d+)\.(\d+)\.(\d+)", v or "")
            return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0,0,0)
        need_fix = (not cur_ver) or parse(cur_ver) < min_v or parse(cur_ver) > max_v or (cur_cuda and cur_cuda != "12.8")
        if not need_fix:
            return True, "Torch/CUDA compatible"
        # запрос подтверждения в non-UI среде — по env-флагу; по умолчанию спросим в консоли
        answer = os.environ.get("COMFYUI_ACC_AUTO_TORCH_FIX", "").lower()
        if not answer:
            try:
                print("Detected incompatible torch/cuda.")
                ans = input("Reinstall torch to 2.7.x with CUDA 12.8 now? (y/N): ").strip().lower()
                answer = "y" if ans in ("y","yes","д","да") else "n"
            except KeyboardInterrupt:
                answer = "n"
        if answer != "y":
            return False, "User declined torch/cuda fix"
        target_version = "2.7.9"  # выбираем максимально допустимую в диапазоне
        index_url = "https://download.pytorch.org/whl/cu128"
        ok, out = self._run_pip(["install", f"torch=={target_version}", "--force-reinstall", "--index-url", index_url], timeout=1200)
        return (True, f"torch=={target_version} installed") if ok else (False, f"torch reinstall failed: {out}")

    def install_sageattention(self) -> Tuple[bool, str]:
        self.logger.info("Installing SageAttention 2.2.0...")
        wheel_url = f"{self.config.repo_base}/{self.config.sage_wheel}"
        local = self.config.sage_wheel_local
        ok, msg = self._download_file(wheel_url, local)
        if not ok:
            return False, f"SageAttention download failed: {msg}"
        if not os.path.exists(local):
            return False, "Downloaded wheel file not found"
        ok, why = self._validate_wheel_tag(local)
        if not ok:
            try: os.remove(local)
            except Exception: pass
            # Попытка согласовать torch при несовместимости
            fix_ok, fix_msg = self._ensure_torch_compatible()
            return (False, f"Wheel incompatible: {why}; {fix_msg}") if not fix_ok else (False, f"Wheel incompatible: {why}. Re-run step after torch fix.")
        ok, out = self._run_pip(["install", local])
        if not ok:
            ok, out = self._run_pip(["install", "--force-reinstall", local])
        try: os.remove(local)
        except Exception: pass
        return (True, "SageAttention 2.2.0 installed successfully") if ok else (False, f"SageAttention installation failed: {out}")

    def setup_include_libs(self) -> Tuple[bool, str]:
        if os.path.exists("include") and os.path.exists("libs"):
            return True, "include/libs folders already exist"
        url = f"{self.config.repo_base}/{self.config.include_zip}"
        tmp = "temp_include_libs.zip"
        ok, msg = self._download_file(url, tmp)
        if not ok:
            return False, f"include/libs download failed: {msg}"
        try:
            with zipfile.ZipFile(tmp, "r") as zf:
                names = zf.namelist()
                if not any(n.rstrip("/").endswith("include") for n in names) or not any(n.rstrip("/").endswith("libs") for n in names):
                    return False, "Archive does not contain include/libs"
                zf.extractall(".")
        except Exception as e:
            return False, f"Extraction failed: {e}"
        finally:
            try: os.remove(tmp)
            except Exception: pass
        if os.path.exists("include") and os.path.exists("libs"):
            return True, "include/libs folders created successfully"
        return False, "Extraction successful but folders not found"

    def verify_installation(self) -> Dict[str, bool]:
        res: Dict[str, bool] = {}
        try:
            r = subprocess.run([self.python_exe, "-c", "import triton"], capture_output=True, timeout=10)
            res['triton'] = (r.returncode == 0)
        except Exception:
            res['triton'] = False
        try:
            r = subprocess.run([self.python_exe, "-c", "import sageattention"], capture_output=True, timeout=10)
            res['sageattention'] = (r.returncode == 0)
        except Exception:
            res['sageattention'] = False
        res['include'] = os.path.exists("include")
        res['libs'] = os.path.exists("libs")
        return res

    def smart_install(self) -> Tuple[bool, Dict[str, Any]]:
        steps = [
            ("pip_upgrade", self.upgrade_pip),
            ("include_libs", self.setup_include_libs),
            ("triton", self.install_triton),
            ("sageattention", self.install_sageattention),
        ]
        results: Dict[str, Any] = {}
        ok_all = True
        for name, fn in steps:
            ok, msg = fn()
            results[name] = {'success': ok, 'message': msg}
            ok_all &= ok
            (self.logger.info if ok else self.logger.error)("%s: %s", name, msg)
        results['verification'] = self.verify_installation()
        return ok_all, results

    def clean_install(self) -> Tuple[bool, str]:
        self._run_pip(["uninstall", "-y", "triton", "sageattention"])
        for folder in ["include", "libs"]:
            if os.path.exists(folder):
                try: shutil.rmtree(folder)
                except Exception: pass
        for f in [self.config.sage_wheel_local, "temp_include_libs.zip"]:
            if os.path.exists(f):
                try: os.remove(f)
                except Exception: pass
        ok, _ = self.smart_install()
        return ok, ("Clean installation completed" if ok else "Clean installation failed")


