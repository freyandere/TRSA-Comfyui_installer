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
    """System information container"""
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
    """Component installation status"""
    triton: bool = False
    triton_version: str = ""
    sageattention: bool = False
    include_folder: bool = False
    libs_folder: bool = False
    pytorch: bool = False

class SystemChecker:
    """Comprehensive system diagnostics and health check"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.python_exe = self._find_python()
        
    def _find_python(self) -> str:
        """Find python executable"""
        if os.path.exists("python.exe"):
            return "python.exe"
        if os.path.exists("python"):
            return "python"
        return "python"  # Fallback
    
    def _run_python_code(self, code: str, timeout: int = 10) -> Tuple[bool, str]:
        """Execute Python code and return result"""
        try:
            result = subprocess.run([
                self.python_exe, "-c", code
            ], capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)
    
    def _run_system_command(self, cmd: str, timeout: int = 10) -> Tuple[bool, str]:
        """Execute system command"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip()
        except Exception as e:
            return False, str(e)
    
    def get_system_info(self) -> SystemInfo:
        """Collect comprehensive system information"""
        info = SystemInfo()
        
        # Python version
        try:
            info.python_version = sys.version.split()[0]
        except:
            success, output = self._run_python_code("import sys; print(sys.version.split()[0])")
            info.python_version = output if success else "Unknown"
        
        # pip version
        success, output = self._run_python_code("import pip; print(pip.__version__)")
        if not success:
            success, output = self._run_system_command(f"{self.python_exe} -m pip --version")
            if success:
                info.pip_version = output.split()[1] if len(output.split()) > 1 else "Unknown"
        else:
            info.pip_version = output
        
        # Platform info
        info.platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"
        
        # PyTorch and CUDA info
        pytorch_code = '''
try:
    import torch
    print(f"pytorch:{torch.__version__}")
    print(f"cuda_available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_version:{torch.version.cuda}")
        try:
            print(f"gpu_name:{torch.cuda.get_device_name(0)}")
        except:
            print("gpu_name:Unknown")
    else:
        print("cuda_version:Not available")
        print("gpu_name:No CUDA GPU")
except ImportError:
    print("pytorch:Not installed")
    print("cuda_available:False")
    print("cuda_version:Not available")
    print("gpu_name:PyTorch not available")
'''
        
        success, output = self._run_python_code(pytorch_code)
        if success:
            for line in output.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key == "pytorch":
                        info.pytorch_version = value
                    elif key == "cuda_available":
                        info.cuda_available = value.lower() == "true"
                    elif key == "cuda_version":
                        info.cuda_version = value
                    elif key == "gpu_name":
                        info.gpu_name = value
        
        # GPU driver version via nvidia-smi
        success, output = self._run_system_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
        if success:
            info.gpu_driver = output.split('\n')[0] if output else "Unknown"
        else:
            info.gpu_driver = "nvidia-smi not available"
        
        return info
    
    def check_components(self) -> ComponentStatus:
        """Check installation status of all components"""
        status = ComponentStatus()
        
        # Check Triton
        success, output = self._run_python_code("import triton; print(triton.__version__)")
        if success:
            status.triton = True
            status.triton_version = output
        
        # Check SageAttention
        success, _ = self._run_python_code("import sageattention")
        status.sageattention = success
        
        # Check folders
        status.include_folder = os.path.exists("include")
        status.libs_folder = os.path.exists("libs")
        
        # Check PyTorch
        success, _ = self._run_python_code("import torch")
        status.pytorch = success
        
        return status
    
    def run_gpu_benchmark(self) -> Dict[str, Any]:
        """Run simple GPU performance test"""
        benchmark_code = '''
try:
    import torch
    import time
    
    if not torch.cuda.is_available():
        print("benchmark_status:CUDA not available")
    else:
        device = torch.cuda.get_device_name(0)
        print(f"benchmark_device:{device}")
        
        # Simple matrix multiplication benchmark
        size = 2048
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        
        # Warmup
        torch.cuda.synchronize()
        for _ in range(3):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        gflops = (2 * size**3) / (avg_time * 1e9)
        
        print(f"benchmark_time:{avg_time:.4f}")
        print(f"benchmark_gflops:{gflops:.2f}")
        print("benchmark_status:SUCCESS")
        
except Exception as e:
    print(f"benchmark_status:ERROR - {str(e)}")
'''
        
        result = {
            'status': 'UNKNOWN',
            'device': 'Unknown',
            'time_ms': 0.0,
            'gflops': 0.0,
            'error': None
        }
        
        success, output = self._run_python_code(benchmark_code, timeout=30)
        if success:
            for line in output.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key == "benchmark_status":
                        result['status'] = value
                    elif key == "benchmark_device":
                        result['device'] = value
                    elif key == "benchmark_time":
                        result['time_ms'] = float(value) * 1000
                    elif key == "benchmark_gflops":
                        result['gflops'] = float(value)
        else:
            result['status'] = 'ERROR'
            result['error'] = output
        
        return result
    
    def check_comfyui_compatibility(self) -> Dict[str, Any]:
        """Check ComfyUI specific compatibility"""
        checks = {}
        
        # Check if we're in ComfyUI directory structure
        comfyui_indicators = [
            "ComfyUI",
            "main.py",
            "nodes.py",
            "model_management.py"
        ]
        
        checks['comfyui_directory'] = any(
            os.path.exists(indicator) for indicator in comfyui_indicators
        )
        
        # Check Python embeded structure
        checks['python_embeded'] = os.path.exists("python.exe") or os.path.exists("Scripts")
        
        # Check common ComfyUI packages
        packages_to_check = [
            "torch", "torchvision", "transformers", "diffusers", 
            "safetensors", "accelerate", "xformers"
        ]
        
        checks['packages'] = {}
        for package in packages_to_check:
            success, version = self._run_python_code(f"import {package}; print({package}.__version__)")
            checks['packages'][package] = {
                'installed': success,
                'version': version if success else None
            }
        
        return checks
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        self.logger.info("Generating detailed system report...")
        
        report = {
            'timestamp': str(__import__('datetime').datetime.now()),
            'system_info': asdict(self.get_system_info()),
            'components': asdict(self.check_components()),
            'gpu_benchmark': self.run_gpu_benchmark(),
            'comfyui_compatibility': self.check_comfyui_compatibility()
        }
        
        # Add health score
        score = self._calculate_health_score(report)
        report['health_score'] = score
        
        return report
    
    def _calculate_health_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system health score"""
        score = 0
        max_score = 0
        issues = []
        
        # Component checks (40 points max)
        components = report['components']
        if components['triton']:
            score += 15
        else:
            issues.append("Triton not installed")
        
        if components['sageattention']:
            score += 15
        else:
            issues.append("SageAttention not installed")
        
        if components['include_folder'] and components['libs_folder']:
            score += 10
        else:
            issues.append("include/libs folders missing")
        
        max_score += 40
        
        # System checks (30 points max)
        system = report['system_info']
        if system['cuda_available']:
            score += 20
        else:
            issues.append("CUDA not available")
        
        if system['pytorch_version']:
            score += 10
        else:
            issues.append("PyTorch not installed")
        
        max_score += 30
        
        # Performance check (30 points max)
        benchmark = report['gpu_benchmark']
        if benchmark['status'] == 'SUCCESS':
            if benchmark['gflops'] > 1000:
                score += 30
            elif benchmark['gflops'] > 500:
                score += 20
            elif benchmark['gflops'] > 100:
                score += 10
            else:
                score += 5
        else:
            issues.append("GPU benchmark failed")
        
        max_score += 30
        
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': round(percentage, 1),
            'grade': self._get_grade(percentage),
            'issues': issues
        }
    
    def _get_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return "A+ (Excellent)"
        elif percentage >= 80:
            return "A (Very Good)"
        elif percentage >= 70:
            return "B (Good)"
        elif percentage >= 60:
            return "C (Fair)"
        elif percentage >= 50:
            return "D (Poor)"
        else:
            return "F (Critical Issues)"
    
    def run_quick_check(self) -> Tuple[bool, Dict[str, bool]]:
        """Quick health check - returns overall status and component status"""
        components = self.check_components()
        
        quick_status = {
            'triton': components.triton,
            'sageattention': components.sageattention,
            'folders': components.include_folder and components.libs_folder,
            'pytorch': components.pytorch
        }
        
        overall_healthy = all(quick_status.values())
        
        return overall_healthy, quick_status

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    checker = SystemChecker()
    
    print("🔍 Running system check...")
    healthy, status = checker.run_quick_check()
    
    print(f"\n🎯 Overall Status: {'✅ Healthy' if healthy else '❌ Issues Found'}")
    for component, ok in status.items():
        print(f"   {component}: {'✅' if ok else '❌'}")
    
    if not healthy:
        print("\n📊 Generating detailed report...")
        report = checker.generate_detailed_report()
        print(f"Health Score: {report['health_score']['percentage']}% ({report['health_score']['grade']})")
