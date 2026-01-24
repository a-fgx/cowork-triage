#!/usr/bin/env python3
"""
Launcher script for Cowork Triage UI

Usage:
    python start_ui.py
    OR
    ./start_ui.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("Starting Cowork Triage UI...")
    print("Access the UI at: http://localhost:8501")
    print("")

    ui_app_path = Path(__file__).parent / "ui_app.py"

    try:
        subprocess.run([
            "streamlit", "run", str(ui_app_path)
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down UI...")
        sys.exit(0)
    except FileNotFoundError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
