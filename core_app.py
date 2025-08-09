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
    """Application configuration."""
    # Всегда main по умолчанию; можно переопределить COMFYUI_ACC_REPO
    repo_base: str = "https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
    app_version: str = "3.0"
    modules: List[str] = None
    def __post_init__(self) -> None:
        if self.modules is None:
            self.modules = ["installer", "checker", "ui_manager"]

class ComfyUIAccelerator:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.temp_dir = __import__("tempfile").mkdtemp(prefix="comfyui_acc_")
        self.loaded_modules: Dict[str, Any] = {}
        self.ui = self.installer = self.checker = None
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        __import__("atexit").register(self.cleanup)
        if not self._check_python():
            print("❌ ERROR: python.exe not found!\nPlace this script into folder with python.exe")
            input("Press Enter to exit...")
            sys.exit(1)

    def _check_python(self) -> bool:
        return os.path.exists("python.exe") or os.path.exists("python")

    def _download_module(self, module_name: str) -> Optional[str]:
        url = f"{self.config.repo_base}/{module_name}.py"
        local_path = os.path.join(self.temp_dir, f"{module_name}.py")
        try:
            urllib.request.urlretrieve(url, local_path)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                return local_path
            self.logger.error("Downloaded %s is empty", module_name)
            return None
        except Exception as e:
            self.logger.error("Failed to download %s: %s", module_name, e)
            return None

    def _load_module(self, module_name: str, local_path: str):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, local_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {module_name}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.loaded_modules[module_name] = module
            return module
        except Exception as e:
            self.logger.error("Failed to load module %s: %s", module_name, e)
            return None

    def get_module(self, module_name: str):
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        lp = self._download_module(module_name)
        return self._load_module(module_name, lp) if lp else None

    def initialize_modules(self) -> bool:
        print("🌐 Loading ComfyUI Accelerator modules...")
        ui_module = self.get_module("ui_manager")
        if not ui_module:
            print("❌ Failed to load UI manager")
            return False
        try:
            self.ui = ui_module.UIManager()
            installer_module = self.get_module("installer")
            if installer_module:
                self.installer = installer_module.ComponentInstaller()
            checker_module = self.get_module("checker")
            if checker_module:
                self.checker = checker_module.SystemChecker()
            return True
        except Exception as e:
            print(f"❌ Failed to initialize modules: {e}")
            return False

    def show_main_menu(self) -> bool:
        if not self.ui:
            print("❌ UI not initialized")
            return False
        # Пункт 2 удалён. Пункт 4 переименован.
        menu_items = [
            (self.ui.t('menu_smart_install'), self.ui.t('menu_smart_install_desc'), self.smart_install),
            (self.ui.t('menu_reinstall'), self.ui.t('menu_reinstall_desc'), self.force_reinstall),
            (self.ui.t('menu_check_install_and_report'), self.ui.t('menu_check_install_and_report_desc'), self.detailed_report),
            (self.ui.t('menu_exit'), "", self.exit_app),
        ]
        return self.ui.show_menu(menu_items)

    def smart_install(self) -> bool:
        if not self.installer or not self.ui:
            self.ui.show_error("Installer not available")
            return True
        self.ui.print_header("🚀 Smart Installation")
        steps = [
            (self.ui.t('install_upgrading_pip'), 'upgrade_pip'),
            (self.ui.t('install_setup_folders'), 'setup_include_libs'),
            (self.ui.t('install_triton'), 'install_triton'),
            (self.ui.t('install_sageattention'), 'install_sageattention'),
            (self.ui.t('install_verifying'), 'verify_installation'),
        ]
        total = len(steps)
        for i, (desc, method_name) in enumerate(steps, 1):
            self.ui.show_step_progress(i-1, total, desc)
            method = getattr(self.installer, method_name)
            ok, msg = method()
            self.ui.show_step_progress(i, total, desc)
            (self.ui.show_success if ok else self.ui.show_error)(f"{desc} - {msg}")
            if not ok and not self.ui.confirm("Continue with remaining steps?"):
                break
        self.ui.show_success(self.ui.t('install_complete'))
        self.ui.show_info(self.ui.t('install_restart_note'))
        self.ui.wait_for_input()
        return True

    def check_installation(self) -> bool:
        if not self.checker or not self.ui:
            self.ui.show_error("Checker not available")
            return True
        self.ui.print_header("🔍 Installation Check")
        components = self.checker.check_components()
        healthy = all([components.triton, components.sageattention, components.pytorch,
                       components.include_folder, components.libs_folder])
        self.ui.show_component_status(components.__dict__)
        system_info = self.checker.get_system_info()
        print()
        self.ui.print_divider("-", 40)
        self.ui.show_system_info(system_info.__dict__)
        print()
        (self.ui.show_success if healthy else self.ui.show_warning)(
            "🎉 All systems operational!" if healthy else "⚠️ Some issues found. Consider reinstalling."
        )
        self.ui.wait_for_input()
        return True

    def detailed_report(self) -> bool:
        if not self.checker or not self.ui:
            self.ui.show_error("Checker not available")
            return True
        self.ui.print_header("📊 System check and detailed report")
        self.ui.show_spinner("Running diagnostics...", 2.0)
        report = self.checker.generate_detailed_report()
        print()
        self.ui.print_divider("=", 50)
        print(f" 📋 SYSTEM REPORT - {report['timestamp'][:19]}")
        self.ui.print_divider("=", 50)
        print("\n🖥️ SYSTEM INFORMATION")
        self.ui.print_divider("-", 30)
        self.ui.show_system_info(report['system_info'])
        print("\n📦 COMPONENT STATUS")
        self.ui.print_divider("-", 30)
        self.ui.show_component_status(report['components'])
        print("\n🚀 GPU PERFORMANCE")
        self.ui.print_divider("-", 30)
        bench = report['gpu_benchmark']
        if bench['status'] == 'SUCCESS':
            print(f"🎮 Device: {bench['device']}")
            print(f"⏱️ Average time: {bench['time_ms']:.2f}ms")
            print(f"🔥 Performance: {bench['gflops']:.1f} GFLOPS")
        else:
            self.ui.show_error(f"Benchmark failed: {bench.get('error', 'Unknown')}")
        print("\n🎯 HEALTH SCORE")
        self.ui.print_divider("-", 30)
        self.ui.show_health_score(report['health_score'])
        print()
        self.ui.print_divider("=", 50)
        self.ui.wait_for_input()
        return True

    def exit_app(self) -> bool:
        print(self.ui.t('goodbye') if self.ui else "👋 Goodbye!")
        return False

    def cleanup(self) -> None:
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logging.getLogger(__name__).error("Cleanup failed: %s", e)

def main() -> None:
    repo_url = os.environ.get(
        "COMFYUI_ACC_REPO",
        "https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main",
    )
    app = ComfyUIAccelerator(AppConfig(repo_base=repo_url))
    app.run = ComfyUIAccelerator.run  # keep existing run
    app.run()
