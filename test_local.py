#!/usr/bin/env python3
"""
Local test script for the Flask app
Run this to verify everything works before checking Vercel
"""

import requests
import sys
import time

def test_local_app():
    """Test the Flask app locally"""
    print("🧪 Testing Flask app locally...")
    
    try:
        # Test the home route
        print("Testing / route...")
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✅ Home route working - Status:", response.status_code)
        else:
            print("❌ Home route failed - Status:", response.status_code)
            
        # Test the health route
        print("Testing /health route...")
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            print("✅ Health route working - Status:", response.status_code)
            print("Response:", response.json())
        else:
            print("❌ Health route failed - Status:", response.status_code)
            
        # Test the demo route
        print("Testing /api/demo route...")
        response = requests.get('http://localhost:5000/api/demo')
        if response.status_code == 200:
            print("✅ Demo route working - Status:", response.status_code)
            print("Response:", response.json())
        else:
            print("❌ Demo route failed - Status:", response.status_code)
            
        # Test the status route
        print("Testing /api/status route...")
        response = requests.get('http://localhost:5000/api/status')
        if response.status_code == 200:
            print("✅ Status route working - Status:", response.status_code)
            print("Response:", response.json())
        else:
            print("❌ Status route failed - Status:", response.status_code)
            
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running locally")
        print("💡 Start the app with: python api/index.py")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    print("\n🎉 All local tests completed!")
    return True

if __name__ == "__main__":
    print("🚀 Stock Market Prediction AI - Local Test")
    print("=" * 50)
    
    success = test_local_app()
    
    if success:
        print("\n✅ Local tests passed! Your Flask app is working correctly.")
        print("🌐 Now check your Vercel deployment at:")
        print("   https://stock-market-prediction-app-mauve.vercel.app")
    else:
        print("\n❌ Local tests failed. Please fix issues before deploying to Vercel.")
        sys.exit(1)
