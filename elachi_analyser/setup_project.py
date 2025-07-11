#!/usr/bin/env python3
"""
Elaichi Quality and Price Analyzer - Project Setup Script
This script sets up the complete project structure and runs the initial setup.
"""

import os
import sys
import subprocess
import platform

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'templates',
        'static',
        'static/css',
        'static/js',
        'models',
        'data'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ Created: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        sys.exit(1)
    print(f"   ✓ Python {sys.version.split()[0]} is compatible")

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("   ✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually:")
        print("   pip install -r requirements.txt")
        return False
    return True

def run_data_generation():
    """Generate the synthetic dataset"""
    print("📊 Generating synthetic Elaichi dataset...")
    try:
        subprocess.check_call([sys.executable, "generate_elaichi_data.py"])
        print("   ✓ Dataset generated successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to generate dataset")
        return False
    return True

def train_models():
    """Train the machine learning models"""
    print("🧠 Training machine learning models...")
    try:
        subprocess.check_call([sys.executable, "train_models.py"])
        print("   ✓ Models trained successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to train models")
        return False
    return True

def create_run_script():
    """Create a simple run script"""
    print("🚀 Creating run script...")
    
    if platform.system() == "Windows":
        run_script = """@echo off
echo Starting Elaichi Quality and Price Analyzer...
python app.py
pause
"""
        with open("run.bat", "w") as f:
            f.write(run_script)
        print("   ✓ Created run.bat for Windows")
    else:
        run_script = """#!/bin/bash
echo "Starting Elaichi Quality and Price Analyzer..."
python3 app.py
"""
        with open("run.sh", "w") as f:
            f.write(run_script)
        os.chmod("run.sh", 0o755)
        print("   ✓ Created run.sh for Unix/Linux/Mac")

def display_instructions():
    """Display final instructions"""
    print("\n" + "="*60)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Project Structure:")
    print("   ├── app.py                    # Flask web application")
    print("   ├── generate_elaichi_data.py  # Dataset generator")
    print("   ├── train_models.py           # Model training script")
    print("   ├── requirements.txt          # Python dependencies")
    print("   ├── templates/")
    print("   │   ├── index.html           # Main web interface")
    print("   │   ├── 404.html             # Error page")
    print("   │   └── 500.html             # Error page")
    print("   ├── *.pkl                    # Trained ML models")
    print("   └── elaichi_dataset.csv      # Generated dataset")
    
    print("\n🚀 How to Run:")
    if platform.system() == "Windows":
        print("   Double-click: run.bat")
        print("   Or command: python app.py")
    else:
        print("   Command: ./run.sh")
        print("   Or: python3 app.py")
    
    print("\n🌐 Access the Application:")
    print("   Open your browser and go to: http://localhost:5000")
    
    print("\n🔧 Features:")
    print("   ✓ Predict Elaichi quality (Low/Standard/Premium)")
    print("   ✓ Estimate market price per kg")
    print("   ✓ Interactive web interface")
    print("   ✓ Real-time validation")
    print("   ✓ Confidence levels for predictions")
    
    print("\n📚 Model Information:")
    print("   • Price Prediction: Random Forest Regressor")
    print("   • Quality Classification: Random Forest Classifier")
    print("   • Dataset: 1000 synthetic samples")
    print("   • Features: Moisture, Size, Color, Aroma, Oil Content")

def main():
    """Main setup function"""
    print("🟢 Elaichi Quality and Price Analyzer - Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directory structure
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Please install requirements manually and run setup again")
        return
    
    # Generate dataset
    if not run_data_generation():
        print("\n⚠️  Please run generate_elaichi_data.py manually")
        return
    
    # Train models
    if not train_models():
        print("\n⚠️  Please run train_models.py manually")
        return
    
    # Create run script
    create_run_script()
    
    # Display final instructions
    display_instructions()

if __name__ == "__main__":
    main()