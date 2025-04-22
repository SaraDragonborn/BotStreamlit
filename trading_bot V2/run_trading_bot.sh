#!/bin/bash

# Simple script to run the trading bot

echo "Starting Trading Bot..."

# Change to the streamlit app directory
cd streamlit_app

# Run the Streamlit app
streamlit run Home.py

# If the app fails to start, try the fallback method
if [ $? -ne 0 ]; then
    echo "Error starting the app. Trying fallback method..."
    
    # Check if python3 command is available
    if command -v python3 &> /dev/null; then
        python3 -m streamlit run Home.py
    else
        # Try with python command if python3 is not available
        python -m streamlit run Home.py
    fi
fi