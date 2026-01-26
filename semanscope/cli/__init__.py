"""
Semanscope CLI Module

This module provides command-line interfaces for batch benchmarking
and the Streamlit UI launcher.
"""
import subprocess
import sys
from pathlib import Path


def launch_ui():
    """
    Launch the Semanscope Streamlit UI.

    This function is used as an entry point for the semanscope-ui command.
    """
    # Get the package root directory
    package_root = Path(__file__).parent.parent.parent
    ui_path = package_root / "ui" / "Welcome.py"

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


__all__ = ['launch_ui']
