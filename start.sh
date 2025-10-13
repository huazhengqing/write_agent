#!/bin/bash


docker-compose up -d

source venv/bin/activate

streamlit run ui/ui.py > main.log 2>&1
