#!/usr/bin/env python3
"""
ComfyUI Accelerator v3.0 - Core Application
Main application orchestrator with dynamic module loading
"""

import sys
import os
import tempfile
import urllib.request
import importlib.util
import atexit
import logging
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration"""
    repo_base: str = "https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/dev"
    app_version: str = "3.0"
    modules: list = None
    
    def __post_init__(self):
        if self.modules is None:
            self.modules = ["installer", "checker", "ui_manager"]

class ComfyUIAccelerator:
    """Main application class with dynamic module loading"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.temp_dir = tempfile.mkdtemp(prefix="comfyui_acc_")
        self.loaded_modules = {}
        self.ui = None
        self.installer = None
        self.checker = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        # Check python.exe
        if not self._check_python():
            print("❌ ERROR: python.exe not found!")
            print("📍 Place this script in folder with python.exe")
            print("   Usually: ComfyUI_windows_portable\\python_embedded\\")
            input("Press Enter to exit...")
            sys.exit(1)
    
    def _check_python(self) -> bool:
        """Check if python.exe exists"""
        return os.path.exists("python.exe") or os.path.exists("python")
    
    def _download_module(self, module_name: str) -> Optional[str]:
        """Download module from repository"""
        url = f"{self.config.repo_base}/{module_name}.py"
        local_path = os.path.join(self.temp_dir, f"{module_name}.py")
        
        try:
            self.logger.info(f"Downloading {module_name} from {url}")
            urllib.request.urlretrieve(url, local_path)
            
            # Verify file was downloaded and has content
            if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                self.logger.info(f"Successfully downloaded {module_name}")
                return local_path
            else:
                self.logger.error(f"Downloaded {module_name} is too small or empty")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to download {module_name}: {str(e)}")
            return None
    
    def _load_module(self, module_name: str, local_path: str):
        """Import module from file"""
        try:
            spec = importlib.util.spec_from_file_location(module_name, local_path)
            if spec is None:
                raise ImportError(f"Could not load spec for {module_name}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.loaded_modules[module_name] = module
            self.logger.info(f"Successfully loaded module {module_name}")
            return module
            
        except Exception as e:
            self.logger.error(f"Failed to load module {module_name}: {str(e)}")
            return None
    
    def get_module(self, module_name: str):
        """Get loaded module or download and load it"""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        # Download and load module
        local_path = self._download_module(module_name)
        if local_path:
            return self._load_module(module_name, local_path)
        
        return None
    
    def initialize_modules(self) -> bool:
        """Initialize core modules"""
        print("🌐 Loading ComfyUI Accelerator modules...")
        
        # Load UI Manager first
        ui_module = self.get_module("ui_manager")
        if not ui_module:
            print("❌ Failed to load UI manager")
            return False
        
        try:
            # Initialize UI Manager
            self.ui = ui_module.UIManager()
            
            # Load other modules
            installer_module = self.get_module("installer")
            if installer_module:
                self.installer = installer_module.ComponentInstaller()
            
            checker_module = self.get_module("checker")
            if checker_module:
                self.checker = checker_module.SystemChecker()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize modules: {str(e)}")
            return False
    
    def show_main_menu(self) -> bool:
        """Display main application menu"""
        if not self.ui:
            print("❌ UI not initialized")
            return False
        
        menu_items = [
            (
                self.ui.t('menu_smart_install'),
                self.ui.t('menu_smart_install_desc'),
                self.smart_install
            ),
            (
                self.ui.t('menu_check_install'),
                self.ui.t('menu_check_install_desc'),
                self.check_installation
            ),
            (
                self.ui.t('menu_reinstall'),
                self.ui.t('menu_reinstall_desc'),
                self.force_reinstall
            ),
            (
                self.ui.t('menu_detailed_report'),
                self.ui.t('menu_detailed_report_desc'),
                self.detailed_report
            ),
            (
                self.ui.t('menu_exit'),
                "",
                self.exit_app
            )
        ]
        
        return self.ui.show_menu(menu_items)
    
    def smart_install(self) -> bool:
        """Execute smart installation"""
        if not self.installer or not self.ui:
            self.ui.show_error("Installer not available")
            return True
        
        self.ui.print_header("🚀 Smart Installation")
        
        steps = [
            (self.ui.t('install_upgrading_pip'), 'upgrade_pip'),
            (self.ui.t('install_setup_folders'), 'setup_include_libs'),
            (self.ui.t('install_triton'), 'install_triton'),
            (self.ui.t('install_sageattention'), 'install_sageattention'),
            (self.ui.t('install_verifying'), 'verify_installation')
        ]
        
        total_steps = len(steps)
        
        for i, (description, method_name) in enumerate(steps, 1):
            self.ui.show_step_progress(i-1, total_steps, description)
            
            method = getattr(self.installer, method_name)
            success, message = method()
            
            self.ui.show_step_progress(i, total_steps, description)
            
            if success:
                self.ui.show_success(f"{description} - {message}")
            else:
                self.ui.show_error(f"{description} - {message}")
                if not self.ui.confirm("Continue with remaining steps? "):
                    break
        
        print()
        self.ui.show_success(self.ui.t('install_complete'))
        self.ui.show_info(self.ui.t('install_restart_note'))
        
        self.ui.wait_for_input()
        return True
    
    def check_installation(self) -> bool:
        """Check installation status"""
        if not self.checker or not self.ui:
            self.ui.show_error("Checker not available")
            return True
        
        self.ui.print_header("🔍 Installation Check")
        
        self.ui.show_info("Checking system components...")
        
        # Get full components status
        components = self.checker.check_components()
        
        # Calculate healthy based on all components
        healthy = all([
            components.triton,
            components.sageattention,
            components.pytorch,
            components.include_folder,
            components.libs_folder
        ])
        
        print()
        self.ui.show_component_status(components.__dict__)
        
        # System info
        print()
        self.ui.print_divider("-", 40)
        system_info = self.checker.get_system_info()
        self.ui.show_system_info(system_info.__dict__)
        
        print()
        if healthy:
            self.ui.show_success("🎉 All systems operational!")
        else:
            self.ui.show_warning("⚠️ Some issues found. Consider reinstalling.")
        
        self.ui.wait_for_input()
        return True
    
    def force_reinstall(self) -> bool:
        """Force reinstallation of all components"""
        if not self.installer or not self.ui:
            self.ui.show_error("Installer not available")
            return True
        
        self.ui.print_header("🔄 Force Reinstall")
        
        if not self.ui.confirm(self.ui.t('confirm_reinstall')):
            return True
        
        self.ui.show_info("Cleaning previous installation...")
        success, message = self.installer.clean_install()
        
        if success:
            self.ui.show_success(message)
        else:
            self.ui.show_error(message)
        
        self.ui.wait_for_input()
        return True
    
    def detailed_report(self) -> bool:
        """Generate and display detailed system report"""
        if not self.checker or not self.ui:
            self.ui.show_error("Checker not available")
            return True
        
        self.ui.print_header("📊 Detailed System Report")
        
        self.ui.show_info("Collecting system information...")
        self.ui.show_spinner("Running diagnostics...", 2.0)
        
        report = self.checker.generate_detailed_report()
        
        # Display report sections
        print()
        self.ui.print_divider("=", 50)
        print(f"  📋 SYSTEM REPORT - {report['timestamp'][:19]}")
        self.ui.print_divider("=", 50)
        
        # System Information
        print("\n🖥️  SYSTEM INFORMATION")
        self.ui.print_divider("-", 30)
        self.ui.show_system_info(report['system_info'])
        
        # Component Status
        print("\n📦 COMPONENT STATUS")
        self.ui.print_divider("-", 30)
        self.ui.show_component_status(report['components'])
        
        # GPU Benchmark
        print("\n🚀 GPU PERFORMANCE")
        self.ui.print_divider("-", 30)
        benchmark = report['gpu_benchmark']
        if benchmark['status'] == 'SUCCESS':
            print(f"🎮 Device: {benchmark['device']}")
            print(f"⏱️  Average time: {benchmark['time_ms']:.2f}ms")
            print(f"🔥 Performance: {benchmark['gflops']:.1f} GFLOPS")
        else:
            self.ui.show_error(f"Benchmark failed: {benchmark.get('error', 'Unknown')}")
        
        # Health Score
        print("\n🎯 HEALTH SCORE")
        self.ui.print_divider("-", 30)
        self.ui.show_health_score(report['health_score'])
        
        print()
        self.ui.print_divider("=", 50)
        
        self.ui.wait_for_input()
        return True
    
    def exit_app(self) -> bool:
        """Exit application"""
        if self.ui:
            print(self.ui.t('goodbye'))
        else:
            print("👋 Goodbye!")
        return False
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
    
    def run(self):
        """Main application loop"""
        try:
            # Initialize modules
            if not self.initialize_modules():
                print("❌ Failed to initialize application")
                input("Press Enter to exit...")
                return
            
            # Language selection if UI supports it
            if hasattr(self.ui, 'language_selection_menu'):
                selected_lang = self.ui.language_selection_menu()
                self.ui.set_language(selected_lang)
            
            # Main application loop
            while True:
                try:
                    if not self.show_main_menu():
                        break
                except KeyboardInterrupt:
                    print(f"\n{self.ui.t('goodbye') if self.ui else '👋 Goodbye!'}")
                    break
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    if self.ui:
                        self.ui.show_error(error_msg)
                        self.ui.wait_for_input()
                    else:
                        print(f"❌ {error_msg}")
                        input("Press Enter to continue...")
        
        except Exception as e:
            print(f"❌ Critical error: {str(e)}")
            input("Press Enter to exit...")

if __name__ == "__main__":
    # Allow configuration via environment or command line
    repo_url = os.environ.get(
        'COMFYUI_ACC_REPO', 
        "https://raw.githubusercontent.com/your-username/comfyui-accelerator/main"
    )
    
    config = AppConfig(repo_base=repo_url)
    app = ComfyUIAccelerator(config)
    app.run()


