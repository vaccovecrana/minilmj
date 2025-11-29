#!/bin/bash
# Complete setup and run script for Python test
# Run this script after installing python3-venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

echo "=" | head -c 60 && echo ""
echo "Python MiniLM Test - Complete Setup"
echo "=" | head -c 60 && echo ""

# Check if python3-venv is available
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "✗ python3-venv is not available"
    echo ""
    echo "Please install it first with:"
    echo "  sudo apt install python3-venv"
    echo ""
    echo "Then run this script again:"
    echo "  bash scripts/install_and_run.sh"
    exit 1
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing requirements..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

echo ""
echo "✓ Setup complete!"
echo ""

# Run the test
echo "Running Python test..."
echo "=" | head -c 60 && echo ""
python3 "$SCRIPT_DIR/test_semantic_python.py"

