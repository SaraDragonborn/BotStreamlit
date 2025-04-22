#!/usr/bin/env python
"""
Setup script for the trading bot application.
This will install all the required dependencies and help fix installation errors.
"""
import subprocess
import sys
import os
import time
import platform

def print_step(message):
    """Print a step message with formatting"""
    print("\n" + "=" * 80)
    print(f"STEP: {message}")
    print("=" * 80)

def run_command(command, description=None):
    """Run a shell command and handle errors"""
    if description:
        print(f"\n> {description}")
    
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False

def main():
    """Main setup function"""
    print_step("Setting up Trading Bot Application")
    
    # Determine the current operating system
    system = platform.system()
    print(f"Detected operating system: {system}")
    
    # Step 1: Install core dependencies
    print_step("Installing core dependencies")
    success = run_command(
        [sys.executable, "-m", "pip", "install", "streamlit", "pandas", "numpy", "plotly"],
        "Installing essential packages"
    )
    
    if not success:
        print("Failed to install core dependencies. Trying an alternative approach...")
        success = run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrading pip"
        )
        if success:
            success = run_command(
                [sys.executable, "-m", "pip", "install", "streamlit", "pandas", "numpy", "plotly"],
                "Retrying core packages installation"
            )
    
    if not success:
        print("ERROR: Failed to install core dependencies. Please try manually:")
        print("    pip install streamlit pandas numpy plotly")
        return False
    
    # Step 2: Install additional dependencies
    print_step("Installing additional dependencies")
    additional_packages = [
        "yfinance", "requests", "scikit-learn", "python-dotenv", 
        "matplotlib", "arrow"
    ]
    
    success = run_command(
        [sys.executable, "-m", "pip", "install"] + additional_packages,
        "Installing additional packages"
    )
    
    if not success:
        print("Installing packages one by one instead...")
        all_succeeded = True
        for package in additional_packages:
            pkg_success = run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            )
            if not pkg_success:
                all_succeeded = False
                print(f"WARNING: Failed to install {package}")
        
        success = all_succeeded
    
    # Step 3: Check if setup was successful
    if success:
        print_step("Setup completed successfully!")
        print("\nYou can now run the trading bot with:")
        print("    cd streamlit_app")
        print("    streamlit run Home.py")
    else:
        print_step("Setup completed with some issues")
        print("\nSome packages may not have been installed correctly.")
        print("Please check the error messages above.")
        print("You can try installing the missing packages manually or refer to INSTALLER_ERROR_FIX.md for more help.")
    
    return success

if __name__ == "__main__":
    main()