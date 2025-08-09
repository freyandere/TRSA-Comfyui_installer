#!/usr/bin/env python3
"""
ComfyUI Accelerator v3.0 - System Checker Module
Performs comprehensive system diagnostics
"""

import subprocess
import sys
import os
import platform
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SystemInfo:
    python_version: str = ""
    pip_version: str = ""
    platform_info: str = ""
    gpu_name: str = ""
    gpu_driver: str = ""
    cuda_available: bool = False
    pytorch_version: str = ""
    cuda_version: str = ""

@dataclass
class ComponentStatus:
    triton: bool = False
    triton_version: str = ""
    sageattention: bool = False
    include_folder: bool = False
    libs_folder: bool = False
    pytorch: bool = False

class SystemChecker:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.python_exe = self._find_python()

    def _find_python(self) -> str:
        if os.path.exists("python.exe"):
            return "python.exe"
        if os.path.exists("python"):
            return "python"
        raise FileNotFoundError("Python executable not found")

    def _run_python_code(self, code: str, timeout: int = 10) -> Tuple[bool, str]:
        try:
            r = subprocess.run([self.python_exe, "-c", code], capture_output=True, text=True, timeout=timeout)
            return (r.returncode == 0, (r.stdout or r.stderr).strip())
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)

    def _run_system_command(self, cmd: str, timeout: int = 10) -> Tuple[bool, str]:
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return (r.returncode == 0, (r.stdout or r.stderr).strip())
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)

    def get_system_info(self) -> SystemInfo:
        info = SystemInfo()
        try:
            info.python_version = sys.version.split()[0]
        except Exception:
            ok, out = self._run_python_code("import sys; print(sys.version.split()[0])")
            info.python_version = out if ok else "Unknown"
        ok, out = self._run_python_code("import pip; print(pip.__version__)")
        if not ok:
            ok, out = self._run_system_command(f"{self.python_exe} -m pip --version")
            info.pip_version = (out.split()[1] if ok and len(out.split()) > 1 else out)
        else:
            info.pip_version = out
        info.platform_info = f"{__import__('platform').system()} {__import__('platform').release()} {__import__('platform').machine()}"
        pytorch_code = r"""
import sys
try:
    import torch
    print(f"pytorch:{torch.__version__}")
    print(f"cuda_available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_version:{torch.version.cuda}")
        try:
            print(f"gpu_name:{torch.cuda.get_device_name(0)}")
        except Exception:
            print("gpu_name:Unknown")
    else:
        print("cuda_version:Not available")
        print("gpu_name:No CUDA GPU")
except ImportError:
    print("pytorch:Not installed")
    print("cuda_available:False")
    print("cuda_version:Not available")
    print("gpu_name:PyTorch not available")
"""
        ok, out = self._run_python_code(pytorch_code, timeout=20)
        if ok:
            for line in out.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k == "pytorch":
                        info.pytorch_version = v
                    elif k == "cuda_available":
                        info.cuda_available = v.strip().lower() == "true"
                    elif k == "cuda_version":
                        info.cuda_version = v
                    elif k == "gpu_name":
                        info.gpu_name = v
        ok, out = self._run_system_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits", timeout=5)
        info.gpu_driver = (out.splitlines()[0] if ok and out else "nvidia-smi not available")
        return info

    def check_components(self) -> ComponentStatus:
        status = ComponentStatus()
        ok, out = self._run_python_code("import triton, sys; print(getattr(triton, '__version__', ''))")
        if ok:
            status.triton = True
            status.triton_version = out
        ok, _ = self._run_python_code("import sageattention")
        status.sageattention = ok
        cur = os.getcwd()
        status.include_folder = os.path.exists(os.path.join(cur, "include"))
        status.libs_folder = os.path.exists(os.path.join(cur, "libs"))
        ok, _ = self._run_python_code("import torch")
        status.pytorch = ok
        return status

    def run_quick_check(self) -> Tuple[bool, Dict[str, bool]]:
        c = self.check_components()
        quick = {
            "triton": c.triton,
            "sageattention": c.sageattention,
            "include_folder": c.include_folder,
            "libs_folder": c.libs_folder,
            "pytorch": c.pytorch,
        }
        return all(quick.values()), quick

    def run_gpu_benchmark(self) -> Dict[str, Any]:
        code = r"""
import torch, time
if not torch.cuda.is_available():
    print("benchmark_status:CUDA not available")
else:
    dev = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    # адаптивный размер
    size = 1024 if total < 6 * 1024**3 else 2048
    print(f"benchmark_device:{dev}")
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    torch.cuda.synchronize()
    for _ in range(2):
        c = a @ b
        torch.cuda.synchronize()
    start = time.time()
    iters = 5
    for _ in range(iters):
        c = a @ b
        torch.cuda.synchronize()
    avg = (time.time() - start) / iters
    gflops = (2 * (size**3)) / (avg * 1e9)
    print(f"benchmark_time:{avg:.4f}")
    print(f"benchmark_gflops:{gflops:.2f}")
    print("benchmark_status:SUCCESS")
"""
        res: Dict[str, Any] = {'status': 'UNKNOWN', 'device': 'Unknown', 'time_ms': 0.0, 'gflops': 0.0, 'error': None}
        ok, out = self._run_python_code(code, timeout=60)
        if ok:
            for line in out.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k == "benchmark_status": res['status'] = v
                    elif k == "benchmark_device": res['device'] = v
                    elif k == "benchmark_time": res['time_ms'] = float(v) * 1000
                    elif k == "benchmark_gflops": res['gflops'] = float(v)
        else:
            res['status'] = 'ERROR'
            res['error'] = out
        return res

    def check_comfyui_compatibility(self) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}
        indicators = ["ComfyUI", "main.py", "nodes.py", "model_management.py"]
        checks['comfyui_directory'] = any(os.path.exists(p) for p in indicators)
        checks['python_embeded'] = os.path.exists("python.exe") or os.path.exists("Scripts")
        pkgs = ["torch", "torchvision", "transformers", "diffusers", "safetensors", "accelerate", "xformers"]
        checks['packages'] = {}
        for pkg in pkgs:
            ok, ver = self._run_python_code(f"import {pkg}; print(getattr({pkg}, '__version__', ''))", timeout=10)
            checks['packages'][pkg] = {'installed': ok, 'version': (ver if ok else None)}
        return checks

    def _calculate_health_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        max_score = 0
        issues: List[str] = []
        comp = report['components']
        if comp['triton']: score += 15
        else: issues.append("Triton not installed")
        if comp['sageattention']: score += 15
        else: issues.append("SageAttention not installed")
        if comp['include_folder'] and comp['libs_folder']: score += 10
        else: issues.append("include/libs folders missing")
        max_score += 40
        sysi = report['system_info']
        if sysi['cuda_available']: score += 20
        else: issues.append("CUDA not available")
        if sysi['pytorch_version']: score += 10
        else: issues.append("PyTorch not installed")
        max_score += 30
        bench = report['gpu_benchmark']
        if bench['status'] == 'SUCCESS':
            g = bench['gflops']
            score += 30 if g > 1000 else 20 if g > 500 else 10 if g > 100 else 5
        else:
            issues.append("GPU benchmark failed")
        max_score += 30
        pct = (score / max_score * 100) if max_score else 0.0
        return {'score': score, 'max_score': max_score, 'percentage': round(pct, 1), 'grade': self._get_grade(pct), 'issues': issues}

    def _get_grade(self, pct: float) -> str:
        return "A+ (Excellent)" if pct >= 90 else "A (Very Good)" if pct >= 80 else "B (Good)" if pct >= 70 else "C (Fair)" if pct >= 60 else "D (Poor)" if pct >= 50 else "F (Critical Issues)"

    def generate_detailed_report(self) -> Dict[str, Any]:
        rep = {
            'timestamp': str(__import__('datetime').datetime.now()),
            'system_info': asdict(self.get_system_info()),
            'components': asdict(self.check_components()),
            'gpu_benchmark': self.run_gpu_benchmark(),
            'comfyui_compatibility': self.check_comfyui_compatibility(),
        }
        rep['health_score'] = self._calculate_health_score(rep)
        return rep


