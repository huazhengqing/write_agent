#!/bin/bash

echo "================================================="
echo " Story Agent Project Environment Setup"
echo "================================================="


# Check if the virtual environment directory exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating it now..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please ensure Python 3 is installed."
        exit 1
    fi
fi


echo "Activating virtual environment..."
source venv/bin/activate


echo "Upgrading pip..."
# pip install --upgrade pip
echo


echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt


# HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-small-zh --local-dir ./models/bge-small-zh 
# HF_ENDPOINT=https://hf-mirror.com hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2 


playwright install


# echo "Installing main package in development mode..."
# pip install -v -e .
# echo


