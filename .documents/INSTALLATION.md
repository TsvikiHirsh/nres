# Installation Guide

## Quick Install (Recommended)

The simplest way to install nres:

```bash
pip install git+https://github.com/TsvikiHirsh/nres
```

This will automatically handle all build dependencies including pybind11 and compile the C++ extensions.

## Common Installation Issues

### Issue: "ModuleNotFoundError: No module named 'pybind11'"

**Problem**: This happens when using `--no-build-isolation` flag, which prevents pip from installing build dependencies.

**Solution**: Remove the `--no-build-isolation` flag:

```bash
# ❌ Don't do this:
pip install git+https://github.com/TsvikiHirsh/nres --no-build-isolation

# ✅ Do this instead:
pip install git+https://github.com/TsvikiHirsh/nres
```

**Why**: The `--no-build-isolation` flag tells pip to skip installing build dependencies (like pybind11) defined in `pyproject.toml`. This causes the setup.py to fail when it tries to import pybind11.

### Issue: Behind a corporate proxy/firewall

If you're behind a firewall or need to trust specific hosts:

```bash
# Use --trusted-host flags but WITHOUT --no-build-isolation
pip install git+https://github.com/TsvikiHirsh/nres \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org
```

### Issue: Need to install build dependencies manually

If you absolutely must use `--no-build-isolation` (not recommended), install pybind11 first:

```bash
pip install pybind11
pip install git+https://github.com/TsvikiHirsh/nres --no-build-isolation
```

## Installation from Source (Development)

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/TsvikiHirsh/nres.git
cd nres

# Install in editable mode with development dependencies
pip install -e .[dev]
```

## Requirements

- **Python**: 3.8 or higher
- **C++ Compiler**: Required for building the pybind11 extensions
  - Linux: GCC with C++17 support
  - macOS: Clang (comes with Xcode Command Line Tools)
  - Windows: MSVC 2017 or later

### Installing C++ Compiler

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install build-essential
```

**macOS**:
```bash
xcode-select --install
```

**Windows (WSL)**:
```bash
sudo apt update
sudo apt install build-essential g++
```

## Verifying Installation

After installation, verify everything works:

```python
import nres
print(f"nres version: {nres.__version__}")

# Test C++ extension
import nres._integrate_xs
print("✓ C++ extension module loaded successfully")

# Test basic functionality
Si = nres.CrossSection('Silicon')
print("✓ Cross-section loaded successfully")
```

## Troubleshooting

### C++ Compilation Errors

If you get compilation errors, make sure you have:
1. A C++17 compatible compiler installed
2. Python development headers: `sudo apt install python3-dev` (Linux)

### Import Errors

If `import nres` fails:
1. Make sure you're using Python 3.8 or higher: `python --version`
2. Try reinstalling: `pip uninstall nres && pip install git+https://github.com/TsvikiHirsh/nres`

### WSL-Specific Issues

On Windows Subsystem for Linux (WSL):
1. Make sure you're using a Linux-based pip (not Windows pip)
2. Install build-essential: `sudo apt install build-essential`
3. Don't use `--no-build-isolation` flag

## Getting Help

If you continue to have issues:
1. Check the [GitHub Issues](https://github.com/TsvikiHirsh/nres/issues) page
2. Create a new issue with:
   - Your OS and version
   - Python version (`python --version`)
   - Complete error message
   - Installation command you used
