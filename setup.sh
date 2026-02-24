#!/bin/bash
set -e

echo "Setting up ESCI Search Relevance Engine..."

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p results

echo "Setup complete. Activate venv: source venv/bin/activate"
