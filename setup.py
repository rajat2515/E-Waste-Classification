#!/usr/bin/env python3
"""
E-Waste Classification System Setup Script

This script helps users set up the E-Waste Classification System quickly.
It handles environment setup, dependency installation, and initial configuration.

Usage:
    python setup.py [--install-deps] [--create-dirs] [--download-sample-data]

Author: E-Waste Classification Team
Date: 2024
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    banner = """
    ♻️  E-WASTE CLASSIFICATION SYSTEM SETUP ♻️
    ================================================
    
    Welcome to the E-Waste Classification System!
    This setup script will help you get started quickly.
    
    ================================================
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "data/processed",
        "data/augmented",
        "models",
        "models/checkpoints",
        "models/saved_models",
        "models/logs",
        "notebooks"
    ]
    
    created_count = 0
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_count += 1
            print(f"   ✅ Created: {directory}")
        else:
            print(f"   ℹ️  Exists: {directory}")
    
    print(f"📁 Directory setup complete! ({created_count} new directories created)")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found!")
            return False
        
        # Install dependencies
        print("   Installing packages from requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print("❌ Failed to install dependencies!")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed."""
    print("🔍 Verifying installation...")
    
    required_packages = [
        "tensorflow",
        "keras",
        "numpy",
        "pandas",
        "matplotlib",
        "streamlit",
        "pillow",
        "opencv-python",
        "scikit-learn",
        "pyyaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Please install them manually or run with --install-deps")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_config():
    """Check configuration file."""
    print("⚙️  Checking configuration...")
    
    config_path = "config/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Check essential config sections
        required_sections = ['model', 'training', 'data', 'paths', 'classes']
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing config sections: {', '.join(missing_sections)}")
            return False
        
        print("✅ Configuration file is valid!")
        print(f"   Model: {config['model']['name']}")
        print(f"   Classes: {len(config['classes'])}")
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def create_sample_script():
    """Create a sample usage script."""
    print("📝 Creating sample usage script...")
    
    sample_script = '''#!/usr/bin/env python3
"""
Sample usage script for E-Waste Classification System
"""

import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from predictor import EWastePredictor

def main():
    print("🚀 E-Waste Classification System - Sample Usage")
    
    # Initialize data preprocessor
    print("\\n1. Data Preprocessing Example:")
    preprocessor = DataPreprocessor()
    
    # Check if dataset exists
    import os
    if os.path.exists(preprocessor.data_dir):
        print(f"   ✅ Dataset found: {preprocessor.data_dir}")
        # Uncomment to run preprocessing
        # data_info = preprocessor.load_and_analyze_data()
        # print(f"   📊 Dataset analyzed: {len(data_info)} classes")
    else:
        print(f"   ⚠️  Dataset not found: {preprocessor.data_dir}")
        print("   Please add your dataset to the specified directory")
    
    # Model prediction example
    print("\\n2. Prediction Example:")
    print("   To use the predictor:")
    print("   predictor = EWastePredictor(model_path='path/to/model.h5')")
    print("   result = predictor.predict_single_image('path/to/image.jpg')")
    
    # Web app example
    print("\\n3. Web Application:")
    print("   To launch the web app:")
    print("   streamlit run app.py")
    
    print("\\n✅ Sample usage completed!")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("sample_usage.py", "w") as f:
            f.write(sample_script)
        print("✅ Sample script created: sample_usage.py")
        return True
    except Exception as e:
        print(f"❌ Error creating sample script: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    next_steps = """
    🎉 SETUP COMPLETE! 
    
    📋 Next Steps:
    
    1. 📊 Prepare Your Dataset:
       - Create a 'modified-dataset' folder in the project root
       - Organize images in train/val/test splits
       - Each split should have subfolders for each e-waste category
    
    2. 🚀 Quick Start Options:
       
       a) Web Application (Recommended):
          streamlit run app.py
       
       b) Complete Training Pipeline:
          python train_model.py
       
       c) Interactive Notebook:
          jupyter notebook notebooks/E-Waste_Classification_Demo.ipynb
       
       d) Sample Usage:
          python sample_usage.py
    
    3. 📚 Learn More:
       - Check README.md for detailed documentation
       - Explore the notebooks/ directory for tutorials
       - Review config/config.yaml for customization options
    
    4. 🆘 Need Help?
       - Check the documentation in README.md
       - Review the sample scripts and notebooks
       - Ensure your dataset follows the expected structure
    
    ================================================
    🌍 Ready to classify e-waste for a sustainable future! ♻️
    ================================================
    """
    print(next_steps)

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='E-Waste Classification System Setup')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--create-dirs', action='store_true', help='Create directories')
    parser.add_argument('--skip-verification', action='store_true', help='Skip package verification')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    success = True
    
    # Run setup steps
    if args.all or args.create_dirs:
        success &= create_directories()
    
    if args.all or args.install_deps:
        success &= install_dependencies()
    
    if not args.skip_verification:
        success &= verify_installation()
    
    # Always check config and create sample script
    success &= check_config()
    success &= create_sample_script()
    
    if success:
        print_next_steps()
        return 0
    else:
        print("\n❌ Setup completed with some issues. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)