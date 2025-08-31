from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
import time
from datetime import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store current job status
current_job = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'results': None,
    'error': None,
    'start_time': None,
    'config': None,
    'trainer': None
}

# Cache for API responses to reduce redundant calls
response_cache = {}
cache_timeout = 0.5  # Cache responses for 0.5 seconds

# Job lock to prevent race conditions
job_lock = threading.Lock()

@app.route('/health')
def health_check():
    """Health check endpoint for Vercel"""
    return jsonify({'status': 'healthy', 'message': 'Stock Prediction App is running'})

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/available_stocks')
def available_stocks():
    """Get list of popular stock symbols"""
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson'},
        {'symbol': 'V', 'name': 'Visa Inc.'},
        {'symbol': 'WMT', 'name': 'Walmart Inc.'},
        {'symbol': 'PG', 'name': 'Procter & Gamble Co.'},
        {'symbol': 'UNH', 'name': 'UnitedHealth Group Inc.'},
        {'symbol': 'HD', 'name': 'The Home Depot Inc.'}
    ]
    return jsonify(stocks)

@app.route('/api/demo')
def demo():
    """Demo endpoint to show the app is working"""
    return jsonify({
        'message': 'Stock Prediction App is working!',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })

# Create necessary directories for production
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
