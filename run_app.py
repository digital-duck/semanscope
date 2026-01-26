#!/usr/bin/env python3
"""
Semanscope UI Launcher

This script launches the Semanscope Streamlit UI.

Usage:
    python run_app.py

Or after installation:
    semanscope-ui
"""
import subprocess
import sys
from pathlib import Path


def launch_ui():
    """Launch the Semanscope Streamlit UI."""
    # Add the package directory to PYTHONPATH so imports work
    package_dir = str(Path(__file__).parent.absolute())

    import os
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{package_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = package_dir

    ui_path = Path(__file__).parent / "ui" / "Welcome.py"

    if not ui_path.exists():
        print(f"Error: UI file not found at {ui_path}")
        sys.exit(1)

    # Launch Streamlit with updated PYTHONPATH
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ui_path)],
            env=env,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down Semanscope UI...")
        sys.exit(0)


if __name__ == "__main__":
    launch_ui()
