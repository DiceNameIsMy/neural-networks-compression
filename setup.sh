#!/bin/bash

VENV_DIR="venv"
COMPUTE_PLATFORM_URL="${COMPUTE_PLATFORM_URL:-https://download.pytorch.org/whl/cpu}"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    # Create the virtual environment
    python3 -m venv $VENV_DIR
    echo "Virtual environment created in $VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR"
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install the required packages
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url $COMPUTE_PLATFORM_URL
pip install -r colab_requirements.txt  # These requirements are needed in general + in colab notebooks.
pip install -e .
echo "Required packages were installed"
