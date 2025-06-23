#!/usr/bin/env python3
"""
Script to build PySlyde documentation locally.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    """Build the documentation."""
    # Get the project root directory
    project_root = Path(__file__).parent
    docs_dir = project_root / "docs"
    
    if not docs_dir.exists():
        print("Error: docs directory not found!")
        sys.exit(1)
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    # Install documentation dependencies
    print("Installing documentation dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to install documentation dependencies!")
        sys.exit(1)
    
    # Clean previous build
    build_dir = docs_dir / "_build"
    if build_dir.exists():
        print("Cleaning previous build...")
        shutil.rmtree(build_dir)
    
    # Build documentation
    print("Building documentation...")
    try:
        subprocess.run(["make", "html"], check=True)
        print("Documentation built successfully!")
        print(f"Open {build_dir / 'html' / 'index.html'} in your browser to view the documentation.")
    except subprocess.CalledProcessError:
        print("Error: Failed to build documentation!")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'make' command not found. Please install make or use sphinx-build directly.")
        sys.exit(1)

if __name__ == "__main__":
    main() 