#!/bin/bash
# Quick start script for Email Assistant (Linux/Mac)

echo "========================================"
echo "Email Assistant - Quick Start"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo

# Run the assistant
echo "Starting Email Assistant..."
echo
python main.py

# Deactivate virtual environment
deactivate






