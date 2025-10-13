#!/bin/bash


docker-compose up -d

source venv/bin/activate

streamlit run ui/home.py > main.log 2>&1
