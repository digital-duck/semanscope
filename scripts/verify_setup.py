#!/usr/bin/env python3
"""
Semanscope Setup Verification Script

Checks that the installation is complete and working properly.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path
import importlib


def check_mark(status):
    """Return check mark or X based on status."""
    return "✓" if status else "✗"


def test_import(module_name):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 70)
    print("Semanscope Setup Verification")
    print("=" * 70)

    all_passed = True

    # Check Python version
    print("\n[1/7] Python Version")
    py_version = sys.version_info
    version_ok = py_version.major == 3 and py_version.minor >= 9
    print(f"  {check_mark(version_ok)} Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not version_ok:
        print("  ⚠ Warning: Python 3.9+ recommended")
        all_passed = False

    # Check core dependencies
    print("\n[2/7] Core Dependencies")
    core_deps = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "torch": "PyTorch",
        "transformers": "Transformers",
        "sentence_transformers": "Sentence Transformers",
        "sklearn": "Scikit-learn",
        "scipy": "SciPy",
    }

    for module, name in core_deps.items():
        status = test_import(module)
        print(f"  {check_mark(status)} {name}")
        if not status:
            all_passed = False

    # Check dimensionality reduction libraries
    print("\n[3/7] Dimensionality Reduction")
    dimred_deps = {
        "umap": "UMAP",
        "phate": "PHATE",
        "pacmap": "PaCMAP",
        "trimap": "TriMap",
    }

    for module, name in dimred_deps.items():
        status = test_import(module)
        print(f"  {check_mark(status)} {name}")
        if not status:
            print(f"      Note: {name} is optional but recommended")

    # Check UI dependencies
    print("\n[4/7] UI Dependencies (optional)")
    ui_deps = {
        "streamlit": "Streamlit",
        "plotly": "Plotly",
        "streamlit_echarts": "Streamlit ECharts",
    }

    for module, name in ui_deps.items():
        status = test_import(module)
        print(f"  {check_mark(status)} {name}")
        if not status:
            print(f"      Note: Install with 'pip install -e \".[ui]\"'")

    # Check semanscope package
    print("\n[5/7] Semanscope Package")
    semanscope_modules = [
        "semanscope",
        "semanscope.config",
        "semanscope.components.semantic_affinity",
        "semanscope.components.embedding_viz",
        "semanscope.models.model_manager",
        "semanscope.utils.embedding_cache",
    ]

    for module in semanscope_modules:
        status = test_import(module)
        module_name = module.split(".")[-1]
        print(f"  {check_mark(status)} {module}")
        if not status:
            all_passed = False

    # Check GPU availability
    print("\n[6/7] GPU Support")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  {check_mark(cuda_available)} CUDA available")
        if cuda_available:
            print(f"      Device: {torch.cuda.get_device_name(0)}")
            print(f"      CUDA Version: {torch.version.cuda}")
        else:
            print("      Note: Running on CPU (GPU recommended for large models)")
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")

    # Check data directory
    print("\n[7/7] Data Directory")
    package_root = Path(__file__).parent.parent
    data_dir = package_root / "data" / "input"
    datasets_exist = data_dir.exists() and len(list(data_dir.glob("*"))) > 0

    print(f"  {check_mark(data_dir.exists())} Data directory exists: {data_dir}")
    if datasets_exist:
        dataset_count = len(list(data_dir.glob("*.txt"))) + len(list(data_dir.glob("*.csv")))
        print(f"  {check_mark(dataset_count > 0)} Datasets found: {dataset_count} files")
    else:
        print("  ⚠ Warning: No datasets found. Run scripts/download_datasets.py")

    # Summary
    print("\n" + "=" * 70)
    if all_passed and datasets_exist:
        print("✓ All checks passed! Semanscope is ready to use.")
        print("\nQuick start:")
        print("  • Launch UI:         python run_app.py")
        print("  • Run demo:          python demo/basic_visualization.py")
        print("  • List models:       python -c 'from semanscope.config import MODEL_INFO; print(list(MODEL_INFO.keys()))'")
        print("  • Batch benchmark:   semanscope-benchmark-sa --help")
    else:
        print("⚠ Some checks failed. Please install missing dependencies.")
        print("\nInstallation:")
        print("  • Core package:      pip install -e .")
        print("  • With UI:           pip install -e \".[ui]\"")
        print("  • Full install:      pip install -e \".[all]\"")
        print("  • Development:       pip install -e \".[dev]\"")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
