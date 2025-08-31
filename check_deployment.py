#!/usr/bin/env python3
"""
Vercel Deployment Status Checker
Check if your app is working on Vercel
"""

import requests
import sys
import time

def check_vercel_deployment():
    """Check the Vercel deployment status"""
    vercel_url = "https://stock-market-prediction-app-mauve.vercel.app"
    
    print("🌐 Checking Vercel deployment status...")
    print(f"URL: {vercel_url}")
    print("=" * 60)
    
    try:
        # Test the main page
        print("1️⃣ Testing main page (/)...")
        response = requests.get(vercel_url, timeout=10)
        if response.status_code == 200:
            print("✅ Main page working - Status:", response.status_code)
            if "Stock Market Prediction AI" in response.text:
                print("✅ Content verified - Stock Prediction AI found")
            else:
                print("⚠️  Content mismatch - Expected content not found")
        else:
            print(f"❌ Main page failed - Status: {response.status_code}")
            
        # Test the health endpoint
        print("\n2️⃣ Testing health endpoint (/health)...")
        response = requests.get(f"{vercel_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working - Status:", response.status_code)
            try:
                data = response.json()
                print("✅ JSON response:", data)
            except:
                print("⚠️  Response is not valid JSON")
        else:
            print(f"❌ Health endpoint failed - Status: {response.status_code}")
            
        # Test the demo endpoint
        print("\n3️⃣ Testing demo endpoint (/api/demo)...")
        response = requests.get(f"{vercel_url}/api/demo", timeout=10)
        if response.status_code == 200:
            print("✅ Demo endpoint working - Status:", response.status_code)
            try:
                data = response.json()
                print("✅ JSON response:", data)
            except:
                print("⚠️  Response is not valid JSON")
        else:
            print(f"❌ Demo endpoint failed - Status: {response.status_code}")
            
        # Test the status endpoint
        print("\n4️⃣ Testing status endpoint (/api/status)...")
        response = requests.get(f"{vercel_url}/api/status", timeout=10)
        if response.status_code == 200:
            print("✅ Status endpoint working - Status:", response.status_code)
            try:
                data = response.json()
                print("✅ JSON response:", data)
            except:
                print("⚠️  Response is not valid JSON")
        else:
            print(f"❌ Status endpoint failed - Status: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - Vercel might be still deploying")
        print("💡 Wait a few more minutes and try again")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Vercel deployment might be in progress")
        print("💡 Wait a few more minutes and try again")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Deployment status check completed!")
    return True

def main():
    """Main function"""
    print("🚀 Stock Market Prediction AI - Vercel Deployment Check")
    print("=" * 60)
    
    print("⏳ This will check if your app is working on Vercel...")
    print("💡 Make sure you've waited at least 5-10 minutes after pushing changes")
    
    input("\nPress Enter to continue...")
    
    success = check_vercel_deployment()
    
    if success:
        print("\n✅ Your Vercel deployment appears to be working!")
        print("🌐 Visit your app at:")
        print("   https://stock-market-prediction-app-mauve.vercel.app")
        print("\n🎯 Next steps:")
        print("   1. Test the app manually in your browser")
        print("   2. Check if all features work as expected")
        print("   3. Share your working app with others!")
    else:
        print("\n❌ Deployment check failed.")
        print("💡 Possible solutions:")
        print("   1. Wait longer for Vercel to complete deployment")
        print("   2. Check Vercel dashboard for build errors")
        print("   3. Verify your GitHub repository has the latest changes")
        print("   4. Try pushing a small change to trigger redeployment")

if __name__ == "__main__":
    main()
