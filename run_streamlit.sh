#!/bin/bash
# Run Streamlit app in virtual environment

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if streamlit is installed
if ! python -m pip show streamlit &>/dev/null; then
    echo "📦 Installing streamlit..."
    pip install streamlit
fi

# Run Streamlit app
echo "🚀 Starting Eldercare Communication Analyzer..."
echo "📍 Access at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run with specific settings for better performance
streamlit run src/app/streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#2196F3" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --theme.textColor "#262730"