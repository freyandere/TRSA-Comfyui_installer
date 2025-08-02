#!/usr/bin/env python3
"""
ComfyUI Accelerator v3.0 - Installation Module
Handles Triton, SageAttention, and include/libs installation
"""

import subprocess
import urllib.request
import zipfile
import tempfile
import os
import logging
from typing import Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class InstallConfig:
    """Installation configuration"""
    triton_version: str = "3.3.0"
    repo_base: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
    sage_wheel: str = "sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
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
    
    def _run_pip_command(self, args: list, check_output: bool = False) -> Tuple[bool, str]:
        """Execute pip command with error handling"""
        cmd = [self.python_exe, "-m", "pip"] + args
        
        try:
            if check_output:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                return result.returncode == 0, result.stdout + result.stderr
            else:
                result = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300
                )
                return result.returncode == 0, ""
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)
    
    def _download_file(self, url: str, filename: str) -> Tuple[bool, str]:
        """Download file with fallback methods"""
        try:
            # Method 1: PowerShell (Windows)
            if os.name == 'nt':
                ps_cmd = f'powershell -Command "Invoke-WebRequest -Uri \'{url}\' -OutFile \'{filename}\' -UseBasicParsing"'
                result = subprocess.run(ps_cmd, shell=True, capture_output=True)
                if result.returncode == 0 and os.path.exists(filename):
                    return True, f"Downloaded {filename} via PowerShell"
            
            # Method 2: Python urllib
            urllib.request.urlretrieve(url, filename)
            if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                return True, f"Downloaded {filename} via Python"
            
            return False, "Download failed - file too small or missing"
            
        except Exception as e:
            return False, f"Download error: {str(e)}"
    
    def upgrade_pip(self) -> Tuple[bool, str]:
        """Upgrade pip to latest version"""
        self.logger.info("Upgrading pip...")
        success, output = self._run_pip_command(["install", "--upgrade", "pip"])
        return success, "pip upgraded successfully" if success else f"pip upgrade failed: {output}"
    
    def install_triton(self) -> Tuple[bool, str]:
        """Install Triton with specified version"""
        self.logger.info(f"Installing Triton {self.config.triton_version}...")
        
        success, output = self._run_pip_command([
            "install", f"triton=={self.config.triton_version}"
        ])
        
        if success:
            return True, f"Triton {self.config.triton_version} installed successfully"
        return False, f"Triton installation failed: {output}"
    
    def install_sageattention(self) -> Tuple[bool, str]:
        """Install SageAttention from wheel file"""
        self.logger.info("Installing SageAttention...")
        
        wheel_url = f"{self.config.repo_base}/{self.config.sage_wheel}"
        temp_wheel = "sage_temp.whl"
        
        # Download wheel file
        success, msg = self._download_file(wheel_url, temp_wheel)
        if not success:
            return False, f"SageAttention download failed: {msg}"
        
        # Install from wheel
        success, output = self._run_pip_command([
            "install", temp_wheel, "--no-deps"
        ])
        
        # Cleanup
        if os.path.exists(temp_wheel):
            os.remove(temp_wheel)
        
        if success:
            return True, "SageAttention installed successfully"
        return False, f"SageAttention installation failed: {output}"
    
    def setup_include_libs(self) -> Tuple[bool, str]:
        """Setup include and libs folders"""
        self.logger.info("Setting up include/libs folders...")
        
        if os.path.exists("include") and os.path.exists("libs"):
            return True, "include/libs folders already exist"
        
        zip_url = f"{self.config.repo_base}/{self.config.include_zip}"
        temp_zip = "temp_include_libs.zip"
        
        # Download zip file
        success, msg = self._download_file(zip_url, temp_zip)
        if not success:
            return False, f"include/libs download failed: {msg}"
        
        # Extract zip
        try:
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            # Cleanup
            os.remove(temp_zip)
            
            if os.path.exists("include") and os.path.exists("libs"):
                return True, "include/libs folders created successfully"
            return False, "Extraction successful but folders not found"
            
        except zipfile.BadZipFile:
            return False, "Downloaded file is not a valid zip"
        except Exception as e:
            return False, f"Extraction failed: {str(e)}"
        finally:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify all components are installed correctly"""
        self.logger.info("Verifying installation...")
        
        results = {}
        
        # Check Triton
        try:
            result = subprocess.run([
                self.python_exe, "-c", "import triton"
            ], capture_output=True, timeout=10)
            results['triton'] = result.returncode == 0
        except:
            results['triton'] = False
        
        # Check SageAttention
        try:
            result = subprocess.run([
                self.python_exe, "-c", "import sageattention"
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
        self.logger.info("Starting smart installation...")
        
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
        self.logger.info("Starting clean installation...")
        
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
        for temp_file in ["sage_temp.whl", "temp_include_libs.zip"]:
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
