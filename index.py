from flask import Flask, jsonify, request, render_template_string
import os
import sys
import logging
from datetime import datetime, timedelta
import traceback

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main application components
from main import Config, DataProcessor, LSTMModel, ModelTrainer, Visualizer

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store model and data
global_model = None
global_data = None
global_config = None

def initialize_model():
    """Initialize the stock prediction model"""
    global global_model, global_data, global_config
    
    try:
        # Initialize configuration
        global_config = Config()
        global_config.ticker = 'MSFT'  # Default ticker
        global_config.num_epochs = 50  # Reduced for faster response
        global_config.sequence_length = 20  # Reduced for faster processing
        
        # Initialize data processor
        data_processor = DataProcessor(global_config)
        
        # Download and preprocess data
        data_processor.download_data()
        processed_data = data_processor.preprocess_data()
        
        # Initialize model
        global_model = LSTMModel(
            input_dim=1,
            hidden_dim=global_config.hidden_dim,
            num_layers=global_config.num_layers,
            output_dim=1
        )
        
        # Train model
        trainer = ModelTrainer(global_config, global_model, processed_data)
        trainer.train()
        
        global_data = data_processor.data
        
        logger.info("Model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

@app.route('/')
def home():
    """Home page with HTML interface"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Market Prediction AI</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }
            input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            button { background-color: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #2980b9; }
            button:disabled { background-color: #bdc3c7; cursor: not-allowed; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .loading { text-align: center; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Stock Market Prediction AI</h1>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label for="ticker">Stock Symbol:</label>
                    <select id="ticker" name="ticker" required>
                        <option value="MSFT">MSFT - Microsoft</option>
                        <option value="AAPL">AAPL - Apple</option>
                        <option value="GOOGL">GOOGL - Google</option>
                        <option value="TSLA">TSLA - Tesla</option>
                        <option value="AMZN">AMZN - Amazon</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="days">Prediction Days:</label>
                    <input type="number" id="days" name="days" value="7" min="1" max="30" required>
                </div>
                
                <button type="submit" id="submitBtn">🚀 Get Prediction</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');
                
                submitBtn.disabled = true;
                submitBtn.textContent = '⏳ Processing...';
                resultDiv.innerHTML = '<div class="loading">Processing your request...</div>';
                
                const formData = new FormData(e.target);
                const ticker = formData.get('ticker');
                const days = formData.get('days');
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ ticker, days: parseInt(days) })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        resultDiv.innerHTML = `
                            <div class="result success">
                                <h3>✅ Prediction Results for ${ticker}</h3>
                                <p><strong>Current Price:</strong> $${data.current_price}</p>
                                <p><strong>Predicted Price (${days} days):</strong> $${data.predicted_price}</p>
                                <p><strong>Change:</strong> ${data.change_percentage}%</p>
                                <p><strong>Confidence:</strong> ${data.confidence}</p>
                                <p><strong>Last Updated:</strong> ${data.timestamp}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <h3>❌ Error</h3>
                                <p>${data.message}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ Error</h3>
                            <p>Failed to get prediction: ${error.message}</p>
                        </div>
                    `;
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Get Prediction';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Stock Prediction App is running',
        'timestamp': datetime.now().isoformat(),
        'model_initialized': global_model is not None
    })

@app.route('/api/demo')
def demo():
    """Demo endpoint"""
    return jsonify({
        'message': 'Stock Prediction App is working!',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/available_stocks')
def available_stocks():
    """Get list of available stocks"""
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.'}
    ]
    return jsonify(stocks)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make stock price prediction"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'MSFT')
        days = data.get('days', 7)
        
        if not ticker:
            return jsonify({'status': 'error', 'message': 'Ticker symbol is required'}), 400
        
        if days < 1 or days > 30:
            return jsonify({'status': 'error', 'message': 'Days must be between 1 and 30'}), 400
        
        # Initialize model if not already done
        if global_model is None:
            success = initialize_model()
            if not success:
                return jsonify({'status': 'error', 'message': 'Failed to initialize model'}), 500
        
        # Update config for new ticker if needed
        if global_config.ticker != ticker:
            global_config.ticker = ticker
            success = initialize_model()
            if not success:
                return jsonify({'status': 'error', 'message': f'Failed to initialize model for {ticker}'}), 500
        
        # Get current price
        current_price = global_data['Close'].iloc[-1]
        
        # Make prediction
        # For demo purposes, we'll create a simple prediction
        # In a real implementation, you'd use the trained model
        import numpy as np
        np.random.seed(42)  # For reproducible results
        
        # Simple trend-based prediction (this is just for demo)
        recent_trend = global_data['Close'].iloc[-5:].pct_change().mean()
        predicted_change = recent_trend * days
        predicted_price = current_price * (1 + predicted_change)
        
        # Add some randomness for demo
        confidence = max(0.6, min(0.95, 0.8 + np.random.normal(0, 0.1)))
        
        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'change_percentage': round((predicted_price - current_price) / current_price * 100, 2),
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/status')
def status():
    """Get application status"""
    return jsonify({
        'status': 'success',
        'model_initialized': global_model is not None,
        'data_loaded': global_data is not None,
        'config': {
            'ticker': global_config.ticker if global_config else None,
            'sequence_length': global_config.sequence_length if global_config else None,
            'hidden_dim': global_config.hidden_dim if global_config else None
        } if global_config else None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Initialize model on startup
    logger.info("Initializing Stock Prediction Model...")
    initialize_model()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
