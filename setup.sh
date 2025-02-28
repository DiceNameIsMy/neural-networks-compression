#!/bin/bash

VENV_DIR="venv"

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
pip install -r pytorch_requirements.txt --index-url https://download.pytorch.org/whl/cpu
pip install -r colab_requirements.txt
pip install -e .
echo "Required packages installed from requirements.txt"
