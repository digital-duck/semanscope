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
    ui_path = Path(__file__).parent / "ui" / "Welcome.py"

    if not ui_path.exists():
        print(f"Error: UI file not found at {ui_path}")
        sys.exit(1)

    # Launch Streamlit
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ui_path)],
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
