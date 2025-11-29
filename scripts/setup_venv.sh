#!/bin/bash
# Setup script for Python virtual environment
# This script helps set up a virtual environment for running the Python tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

echo "Setting up Python virtual environment..."
echo "Project directory: $PROJECT_DIR"

# Check if venv module is available
if python3 -m venv --help > /dev/null 2>&1; then
    echo "Using python3 -m venv..."
    python3 -m venv "$VENV_DIR"
elif command -v virtualenv > /dev/null 2>&1; then
    echo "Using virtualenv..."
    virtualenv -p python3 "$VENV_DIR"
else
    echo "Error: Neither 'python3 -m venv' nor 'virtualenv' is available."
    echo ""
    echo "On Debian/Ubuntu, install with:"
    echo "  sudo apt install python3-venv"
    echo ""
    echo "Or install virtualenv:"
    echo "  sudo apt install python3-virtualenv"
    echo ""
    echo "Alternatively, you can install packages system-wide (not recommended):"
    echo "  sudo apt install python3-pip"
    echo "  pip3 install -r requirements.txt"
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "✓ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the Python test:"
echo "  python3 scripts/test_semantic_python.py"

