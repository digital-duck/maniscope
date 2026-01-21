#!/usr/bin/env python3
"""
Maniscope Evaluation Lab - Launch Script

Launch the Streamlit evaluation interface for benchmarking
reranking algorithms on MTEB datasets.

Usage:
    python run_app.py

Or directly with streamlit:
    streamlit run ui/Maniscope.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    app_path = Path(__file__).parent / "ui" / "Maniscope.py"

    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    print("🚀 Launching Maniscope Evaluation Lab...")
    print(f"📂 App location: {app_path}")
    print("💡 Press Ctrl+C to stop the server\n")

    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped. Thank you for using Maniscope!")
    except FileNotFoundError:
        print("\n❌ Error: Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
