"""
HR Assistant - Installation Helper
This script ensures all dependencies are properly installed
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("=" * 50)
    print("THE HR - HR Intelligent Assistant")
    print("Installation Helper")
    print("=" * 50)
    print()
    
    # Core dependencies
    packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
    ]
    
    # Optional but recommended
    optional_packages = [
        ("PyPDF2", "PyPDF2"),
    ]
    
    print("Installing core dependencies...")
    failed = []
    
    for name, package in packages:
        print(f"Installing {name}...", end=" ")
        if install_package(package):
            print("✓")
        else:
            print("✗")
            failed.append(name)
    
    print("\nInstalling optional dependencies...")
    for name, package in optional_packages:
        print(f"Installing {name} (optional for PDF support)...", end=" ")
        if install_package(package):
            print("✓")
        else:
            print("✗ (App will run without full PDF support)")
    
    print("\n" + "=" * 50)
    
    if failed:
        print(f"⚠️  Some packages failed to install: {', '.join(failed)}")
        print("You may need to install them manually.")
    else:
        print("✅ All core dependencies installed successfully!")
    
    print("\nTo run the application:")
    print("  python -m streamlit run hr_assistant_v2.py")
    print("\nOr simply:")
    print("  streamlit run hr_assistant_v2.py")
    print("\n" + "=" * 50)
    
    # Ask if user wants to run the app now
    response = input("\nWould you like to start the HR Assistant now? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting HR Assistant...")
        print("Admin credentials: username='hradmin', password='hrpass123'")
        print("\nOpening in browser...")
        subprocess.call([sys.executable, "-m", "streamlit", "run", "hr_assistant_v2.py"])

if __name__ == "__main__":
    main()
