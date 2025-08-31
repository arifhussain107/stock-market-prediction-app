#!/usr/bin/env python3
"""
Startup script for Stock Prediction Web Application
"""
import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'torch', 'yfinance', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}/")

def main():
    print("🚀 Stock Prediction Web Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create directories
    create_directories()
    
    print("\n📱 Starting web application...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("🔄 The app will automatically reload when you make changes")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Import and run Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all files are in the correct location")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Application failed: {e}")

if __name__ == "__main__":
    main()
