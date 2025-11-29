#!/bin/bash
# Run Python test script with proper environment setup
# This script will attempt to use a virtual environment if available,
# or fall back to system/user installation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
TEST_SCRIPT="$SCRIPT_DIR/test_semantic_python.py"

echo "=" | head -c 60 && echo ""
echo "Python MiniLM Test Runner"
echo "=" | head -c 60 && echo ""

# Try to use virtual environment if it exists
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "✓ Using virtual environment"
elif command -v conda > /dev/null 2>&1; then
    echo "Conda detected. You may want to create a conda environment:"
    echo "  conda create -n minilm python=3.10"
    echo "  conda activate minilm"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "Continue with current environment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "No virtual environment found."
    echo ""
    echo "To set up a virtual environment, run:"
    echo "  bash scripts/setup_venv.sh"
    echo ""
    echo "Or install python3-venv first:"
    echo "  sudo apt install python3-venv"
    echo ""
    read -p "Continue with system Python? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if required packages are installed
echo ""
echo "Checking for required packages..."
python3 -c "import torch; import transformers; import sklearn; print('✓ All packages available')" 2>&1 || {
    echo "✗ Missing required packages"
    echo ""
    echo "Please install them with one of:"
    echo "  1. Using pip (if available):"
    echo "     pip install -r requirements.txt"
    echo ""
    echo "  2. Using pip with --user flag:"
    echo "     pip install --user -r requirements.txt"
    echo ""
    echo "  3. Install python3-venv and use virtual environment:"
    echo "     sudo apt install python3-venv"
    echo "     bash scripts/setup_venv.sh"
    exit 1
}

echo ""
echo "Running Python test..."
echo "=" | head -c 60 && echo ""
python3 "$TEST_SCRIPT"

