#!/bin/bash

VENV_DIR="venv"
# Check if the virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
    # Remove the existing virtual environment
    rm -rf $VENV_DIR
    echo "Existing virtual environment removed from $VENV_DIR"
fi

# Create the virtual environment
python3 -m venv $VENV_DIR
echo "Virtual environment created in $VENV_DIR"

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install the required packages
pip install -r requirements.txt
echo "Required packages installed from requirements.txt"
