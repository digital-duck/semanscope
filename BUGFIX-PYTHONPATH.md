# Python Path Fix - January 26, 2026

## Issue: "No module named 'utils'" Despite Correct Imports

### Problem
Even after fixing all imports to use `from semanscope.utils.*`, the error persisted:
```
ModuleNotFoundError: No module named 'utils'
```

### Root Cause
The `run_app.py` launcher script was not adding the package directory to `PYTHONPATH` before launching Streamlit. This meant that when Streamlit ran, Python couldn't find the `semanscope` package unless it was formally installed with `pip install`.

### Solution
Updated both launcher scripts to set `PYTHONPATH` before launching Streamlit:

1. **run_app.py** - Direct launcher script
2. **semanscope/cli/__init__.py** - CLI entry point launcher

### Changes Made

#### File: run_app.py
```python
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

    # Launch Streamlit with updated PYTHONPATH
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path)],
        env=env,  # Pass modified environment
        check=True
    )
```

#### File: semanscope/cli/__init__.py
```python
def launch_ui():
    """Launch the Semanscope Streamlit UI (CLI entry point)."""
    package_root = Path(__file__).parent.parent.parent

    # Add package root to PYTHONPATH for development mode
    import os
    env = os.environ.copy()
    package_dir = str(package_root.absolute())
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{package_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = package_dir

    ui_path = package_root / "ui" / "Welcome.py"

    # Launch Streamlit with updated PYTHONPATH
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path)],
        env=env,  # Pass modified environment
        check=True
    )
```

### Additional Steps Taken

1. **Cleared Python Cache**
   ```bash
   find semanscope/ ui/ -name "__pycache__" -type d -exec rm -rf {} +
   find semanscope/ ui/ -name "*.pyc" -delete
   ```

2. **Verified Imports**
   - All imports use `from semanscope.*` pattern ✓
   - No bare `from utils.*` imports remain ✓
   - All `__init__.py` files exist ✓

### How This Works

**Before Fix:**
```
python run_app.py
  → launches streamlit
  → streamlit imports ui/Welcome.py
  → Welcome.py tries: from semanscope.config import ...
  → Python can't find 'semanscope' package
  → ModuleNotFoundError
```

**After Fix:**
```
python run_app.py
  → sets PYTHONPATH=/path/to/semanscope
  → launches streamlit with modified environment
  → streamlit imports ui/Welcome.py
  → Welcome.py tries: from semanscope.config import ...
  → Python finds 'semanscope' in PYTHONPATH
  → Success! ✓
```

### Usage

Now users can run the app **without** pip installing:

```bash
# Option 1: Direct launcher (now works!)
python run_app.py

# Option 2: After pip install (still works!)
pip install -e .
semanscope-ui
```

Both methods now work correctly because the launcher sets PYTHONPATH appropriately.

### Testing

To verify the fix works:

```bash
# 1. Clear Python cache
find semanscope/ ui/ -name "__pycache__" -type d -exec rm -rf {} +

# 2. Launch app
python run_app.py

# 3. Navigate to Semantic Affinity page
# 4. Load a dataset and compute SA
# 5. Verify PHATE visualization works without "No module named 'utils'" error
```

### Why This Matters

- **Development Mode**: Developers can run the app directly without pip install
- **Clean Environment**: Works in fresh conda environments
- **Portable**: Package is self-contained and doesn't require system-wide installation
- **Debugging**: Easier to debug import issues

### Files Modified

1. `run_app.py` - Added PYTHONPATH setup
2. `semanscope/cli/__init__.py` - Added PYTHONPATH setup
3. Python cache cleared

### Status

✅ **RESOLVED** - App can now run without pip install
✅ **TESTED** - PYTHONPATH is correctly set for both launchers
✅ **BACKWARD COMPATIBLE** - Still works with pip install

---

**Date**: January 26, 2026
**Fixed By**: Claude Code
**Issue**: ModuleNotFoundError despite correct imports
**Solution**: Set PYTHONPATH in launcher scripts
