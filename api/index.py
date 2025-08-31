from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'message': 'Stock Prediction App is running!',
        'status': 'success',
        'endpoints': [
            '/health',
            '/api/demo',
            '/api/available_stocks'
        ]
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'App is running'})

@app.route('/api/demo')
def demo():
    return jsonify({
        'message': 'Stock Prediction App is working!',
        'status': 'success'
    })

@app.route('/api/available_stocks')
def available_stocks():
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'}
    ]
    return jsonify(stocks)

# Export the Flask app for Vercel
app.debug = False
