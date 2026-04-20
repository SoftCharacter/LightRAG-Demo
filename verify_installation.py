#!/usr/bin/env python3
"""
LightRAG Installation Verification Script
Checks if all required components are properly installed
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)"


def check_required_packages() -> List[Tuple[str, bool, str]]:
    """Check if required packages are installed"""
    packages = [
        ("yaml", "pyyaml"),
        ("gradio", "gradio"),
        ("rich", "rich"),
        ("lightrag", "lightrag"),
        ("openai", "openai"),
        ("pypdf", "pypdf"),
        ("docx", "python-docx"),
        ("bs4", "beautifulsoup4"),
        ("pyvis", "pyvis"),
        ("networkx", "networkx"),
        ("numpy", "numpy"),
    ]
    
    results = []
    for module_name, package_name in packages:
        try:
            importlib.import_module(module_name)
            results.append((package_name, True, f"✓ {package_name}"))
        except ImportError:
            results.append((package_name, False, f"✗ {package_name} - run: pip install {package_name}"))
    
    return results


def check_project_structure() -> List[Tuple[str, bool, str]]:
    """Check if required files and directories exist"""
    required_items = [
        ("config/config.yaml", "file"),
        ("config/prompts.yaml", "file"),
        ("src/core/config_loader.py", "file"),
        ("src/core/rag_engine.py", "file"),
        ("src/factories/llm_factory.py", "file"),
        ("src/factories/embedding_factory.py", "file"),
        ("src/cli/cli.py", "file"),
        ("src/webui/webui.py", "file"),
        ("main.py", "file"),
        ("requirements.txt", "file"),
        ("README.md", "file"),
        ("documents", "dir"),
        ("logs", "dir"),
        ("rag_storage", "dir"),
    ]
    
    results = []
    for item_path, item_type in required_items:
        path = Path(item_path)
        
        if item_type == "file":
            exists = path.is_file()
        else:
            exists = path.is_dir()
        
        status = "✓" if exists else "✗"
        results.append((item_path, exists, f"{status} {item_path}"))
    
    return results


def check_environment_variables() -> List[Tuple[str, bool, str]]:
    """Check environment variables"""
    import os
    
    vars_to_check = [
        ("OPENAI_API_KEY", False),  # Optional
        ("NEO4J_PASSWORD", False),  # Optional
        ("ANTHROPIC_API_KEY", False),  # Optional
    ]
    
    results = []
    for var_name, required in vars_to_check:
        is_set = bool(os.getenv(var_name))
        
        if required:
            status = "✓" if is_set else "✗"
            msg = f"{status} {var_name} {'SET' if is_set else 'NOT SET (REQUIRED)'}"
        else:
            status = "ℹ"
            msg = f"{status} {var_name} {'SET' if is_set else 'not set (optional)'}"
        
        results.append((var_name, is_set or not required, msg))
    
    return results


def main():
    """Run all checks"""
    print("=" * 60)
    print("LightRAG Installation Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    print("1. Python Version:")
    python_ok, python_msg = check_python_version()
    print(f"   {python_msg}")
    print()
    
    # Check packages
    print("2. Required Packages:")
    package_results = check_required_packages()
    all_packages_ok = all(ok for _, ok, _ in package_results)
    
    for package, ok, msg in package_results:
        print(f"   {msg}")
    print()
    
    # Check project structure
    print("3. Project Structure:")
    structure_results = check_project_structure()
    all_structure_ok = all(ok for _, ok, _ in structure_results)
    
    for path, ok, msg in structure_results:
        print(f"   {msg}")
    print()
    
    # Check environment variables
    print("4. Environment Variables:")
    env_results = check_environment_variables()
    all_env_ok = all(ok for _, ok, _ in env_results)
    
    for var, ok, msg in env_results:
        print(f"   {msg}")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    checks = [
        ("Python Version", python_ok),
        ("Required Packages", all_packages_ok),
        ("Project Structure", all_structure_ok),
        ("Environment Variables", all_env_ok),
    ]
    
    all_ok = all(ok for _, ok in checks)
    
    for check_name, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print()
    
    if all_ok:
        print("🎉 All checks passed! Your installation is ready.")
        print()
        print("Next steps:")
        print("1. Set your API keys in .env file (copy from .env.example)")
        print("2. Launch Web UI: python main.py --mode webui")
        print("3. Or use CLI: python main.py --mode cli --help")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("To install missing packages:")
        print("  pip install -r requirements.txt")
    
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

