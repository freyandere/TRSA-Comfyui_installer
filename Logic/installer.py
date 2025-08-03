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
    """Installation configuration"""
    triton_command: str = "triton-windows<3.4"
    repo_base: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
    sage_wheel: str = "sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    sage_wheel_local: str = "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    include_zip: str = "python_3.12.7_include_libs.zip"

class ComponentInstaller:
    """Handles installation of ComfyUI acceleration components"""
    
    def __init__(self, config: InstallConfig = None):
        self.config = config or InstallConfig()
        self.logger = logging.getLogger(__name__)
        self.python_exe = self._find_python()
        
    def _find_python(self) -> str:
        """Find python executable"""
        if os.path.exists("python.exe"):
            return "python.exe"
        if os.path.exists("python"):
            return "python"
        raise FileNotFoundError("Python executable not found")
    
    def _run_pip_command(self, args: list) -> Tuple[bool, str]:
        """Execute pip command with error handling"""
        cmd = [self.python_exe, "-m", "pip"] + args
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)
    
    def _run_powershell(self, command: str) -> Tuple[bool, str]:
        """Execute PowerShell command"""
        try:
            result = subprocess.run([
                "powershell", "-Command", command
            ], capture_output=True, text=True, timeout=120)
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def upgrade_pip(self) -> Tuple[bool, str]:
        """Upgrade pip to latest version"""
        success, output = self._run_pip_command(["install", "--upgrade", "pip"])
        return success, "pip upgraded successfully" if success else f"pip upgrade failed: {output}"
    
    def install_triton(self) -> Tuple[bool, str]:
        """Install Triton with Windows-specific command"""
        self.logger.info("Installing Triton Windows...")
        
        success, output = self._run_pip_command([
            "install", "-U", self.config.triton_command
        ])
        
        if success:
            return True, "Triton Windows installed successfully"
        return False, f"Triton installation failed: {output}"
    
    def install_sageattention(self) -> Tuple[bool, str]:
        """Install SageAttention using exact bat file logic"""
        self.logger.info("Installing SageAttention 2.2.0...")
        
        wheel_url = f"{self.config.repo_base}/{self.config.sage_wheel}"
        local_wheel = self.config.sage_wheel_local
        
        # Download wheel file using PowerShell
        download_cmd = f"""
        try {{ 
            Invoke-WebRequest -Uri '{wheel_url}' -OutFile '{local_wheel}' -ErrorAction Stop
            Write-Host 'Download completed successfully' 
        }} catch {{ 
            Write-Host 'Download failed: ' $_.Exception.Message
            exit 1 
        }}
        """
        
        success, output = self._run_powershell(download_cmd)
        if not success:
            return False, f"SageAttention download failed: {output}"
        
        # Verify file exists
        if not os.path.exists(local_wheel):
            return False, "Downloaded wheel file not found"
        
        # Install from wheel
        success, output = self._run_pip_command([
            "install", local_wheel
        ])
        
        if not success:
            # Try force reinstall
            success, output = self._run_pip_command([
                "install", "--force-reinstall", local_wheel
            ])
        
        # Cleanup wheel file
        try:
            os.remove(local_wheel)
        except:
            pass
        
        if success:
            return True, "SageAttention 2.2.0 installed successfully"
        return False, f"SageAttention installation failed: {output}"
    
    def setup_include_libs(self) -> Tuple[bool, str]:
        """Setup include and libs folders"""
        if os.path.exists("include") and os.path.exists("libs"):
            return True, "include/libs folders already exist"
        
        zip_url = f"{self.config.repo_base}/{self.config.include_zip}"
        temp_zip = "temp_include_libs.zip"
        
        # Download using PowerShell
        download_cmd = f"""
        try {{ 
            Invoke-WebRequest -Uri '{zip_url}' -OutFile '{temp_zip}' -ErrorAction Stop
            Write-Host 'Download completed'
        }} catch {{ 
            Write-Host 'Download failed: ' $_.Exception.Message
            exit 1 
        }}
        """
        
        success, _ = self._run_powershell(download_cmd)
        if not success:
            return False, "include/libs download failed"
        
        # Extract zip
        try:
            import zipfile
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            os.remove(temp_zip)
            
            if os.path.exists("include") and os.path.exists("libs"):
                return True, "include/libs folders created successfully"
            return False, "Extraction successful but folders not found"
            
        except Exception as e:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            return False, f"Extraction failed: {str(e)}"
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify all components are installed correctly"""
        results = {}
        
        # Check Triton
        try:
            result = subprocess.run([
                self.python_exe, "-c", "import triton"
            ], capture_output=True, timeout=10)
            results['triton'] = result.returncode == 0
        except:
            results['triton'] = False
        
        # Check SageAttention with import test
        try:
            result = subprocess.run([
                self.python_exe, "-c", 
                "import sageattention; print('✓ SageAttention import successful')"
            ], capture_output=True, timeout=10)
            results['sageattention'] = result.returncode == 0
        except:
            results['sageattention'] = False
        
        # Check folders
        results['include'] = os.path.exists("include")
        results['libs'] = os.path.exists("libs")
        
        return results
    
    def smart_install(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute full installation with progress tracking"""
        steps = [
            ("pip_upgrade", self.upgrade_pip),
            ("include_libs", self.setup_include_libs),
            ("triton", self.install_triton),
            ("sageattention", self.install_sageattention),
        ]
        
        results = {}
        total_success = True
        
        for step_name, step_func in steps:
            success, message = step_func()
            results[step_name] = {
                'success': success,
                'message': message
            }
            
            if not success:
                total_success = False
                self.logger.error(f"Step {step_name} failed: {message}")
            else:
                self.logger.info(f"Step {step_name} completed: {message}")
        
        # Final verification
        verification = self.verify_installation()
        results['verification'] = verification
        
        return total_success, results
    
    def clean_install(self) -> Tuple[bool, str]:
        """Clean installation - remove and reinstall everything"""
        # Uninstall packages
        self._run_pip_command(["uninstall", "-y", "triton", "sageattention"])
        
        # Remove folders
        import shutil
        for folder in ["include", "libs"]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except:
                    pass
        
        # Clean temporary files
        for temp_file in [self.config.sage_wheel_local, "temp_include_libs.zip"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Reinstall
        success, results = self.smart_install()
        return success, "Clean installation completed" if success else "Clean installation failed"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    installer = ComponentInstaller()
    success, results = installer.smart_install()
    print(f"Installation {'successful' if success else 'failed'}: {results}")
