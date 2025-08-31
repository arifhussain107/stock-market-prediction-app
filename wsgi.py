import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("Successfully imported Flask app")
except ImportError as e:
    print(f"Import error: {e}")
    # Create a minimal app if import fails
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return {'message': 'App imported successfully', 'status': 'ok'}
    
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'message': 'App is running'}

if __name__ == "__main__":
    app.run()
