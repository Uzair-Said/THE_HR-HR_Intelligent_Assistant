#!/bin/bash

# THE HR - HR Intelligent Assistant Startup Script

echo "======================================"
echo "   THE HR - HR Intelligent Assistant  "
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip is not installed. Installing pip..."
    python3 -m ensurepip --default-pip
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || venv\Scripts\activate

# Install or update requirements
echo "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Clear any previous Streamlit cache
echo "Clearing cache..."
rm -rf .streamlit/cache 2>/dev/null

# Launch the application
echo ""
echo "Starting THE HR Assistant..."
echo "================================"
echo "Admin Login Credentials:"
echo "Username: hradmin"
echo "Password: hrpass123"
echo "================================"
echo ""
echo "Opening in your default browser..."
echo "If browser doesn't open, navigate to: http://localhost:8501"
echo ""

# Run Streamlit
streamlit run hr_assistant.py --server.port=8501 --server.headless=true
