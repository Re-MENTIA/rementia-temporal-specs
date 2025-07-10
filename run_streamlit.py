"""
Run Streamlit app with virtual environment
"""
import subprocess
import sys
import os
from pathlib import Path

def run_streamlit():
    """Run Streamlit app"""
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Path to venv python
    if sys.platform == "win32":
        python_path = script_dir / "venv" / "Scripts" / "python.exe"
        streamlit_path = script_dir / "venv" / "Scripts" / "streamlit.exe"
    else:
        python_path = script_dir / "venv" / "bin" / "python"
        streamlit_path = script_dir / "venv" / "bin" / "streamlit"
    
    # Check if streamlit is installed
    try:
        subprocess.run([str(python_path), "-m", "pip", "show", "streamlit"], 
                      capture_output=True, check=True)
        print("✅ Streamlit is installed")
    except:
        print("📦 Installing streamlit...")
        subprocess.run([str(python_path), "-m", "pip", "install", "streamlit"])
    
    print("🚀 Starting Eldercare Communication Analyzer...")
    print("📍 Access at: http://localhost:8501")
    print("")
    print("Press Ctrl+C to stop the server")
    print("")
    
    # Run streamlit
    cmd = [
        str(streamlit_path),
        "run",
        "src/app/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#2196F3",
        "--theme.backgroundColor", "#FFFFFF",
        "--theme.secondaryBackgroundColor", "#F0F2F6",
        "--theme.textColor", "#262730"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit server stopped")

if __name__ == "__main__":
    run_streamlit()