from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
import time
from datetime import datetime
import logging
from config import get_quick_test_config, get_production_config, get_custom_config
from main import main
import traceback
import torch

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

def run_prediction_job(config_dict):
    """Run prediction job in background thread"""
    global current_job
    
    with job_lock:
        try:
            current_job['status'] = 'running'
            current_job['progress'] = 0
            current_job['message'] = 'Initializing...'
            current_job['error'] = None
            current_job['start_time'] = time.time()
            current_job['results'] = None
            
            # Create config object
            if config_dict['mode'] == 'quick':
                config = get_quick_test_config()
            elif config_dict['mode'] == 'production':
                config = get_production_config()
            else:
                config = get_custom_config()
            
            # Update config with user parameters
            config.ticker = config_dict['ticker']
            config.start_date = config_dict['start_date']
            config.hidden_dim = int(config_dict['hidden_dim'])
            config.num_epochs = int(config_dict['num_epochs'])
            config.learning_rate = float(config_dict['learning_rate'])
            
            # Store config reference for progress tracking
            current_job['config'] = config
            
            logger.info(f"Starting prediction job for {config.ticker} with {config.num_epochs} epochs")
            
        except Exception as e:
            current_job['status'] = 'error'
            current_job['error'] = str(e)
            current_job['message'] = f'Configuration error: {str(e)}'
            logger.error(f"Job configuration failed: {traceback.format_exc()}")
            return
    
    try:
        # Phase 1: Data Download (10-20%)
        current_job['message'] = f'Downloading {config.ticker} data...'
        current_job['progress'] = 10
        time.sleep(0.5)  # Small delay to show progress
        
        # Phase 2: Data Preprocessing (20-30%)
        current_job['message'] = 'Preprocessing and preparing data...'
        current_job['progress'] = 20
        time.sleep(0.5)
        
        # Phase 3: Model Training (30-90%)
        current_job['message'] = 'Training LSTM model...'
        current_job['progress'] = 30
        
        # Run the prediction with progress updates
        logger.info("Starting prediction with progress updates...")
        results = run_prediction_with_progress(config)
        logger.info("Prediction completed successfully")
        
        with job_lock:
            current_job['status'] = 'completed'
            current_job['progress'] = 100
            current_job['message'] = 'Prediction completed successfully!'
            current_job['results'] = {
                'ticker': config.ticker,
                'train_rmse': results['train_rmse'],
                'test_rmse': results['test_rmse'],
                'chart_available': True,
                'execution_time': round(time.time() - current_job['start_time'], 2)
            }
            
            # Clean up references
            if 'trainer' in current_job:
                del current_job['trainer']
            if 'config' in current_job:
                del current_job['config']
        
    except Exception as e:
        with job_lock:
            current_job['status'] = 'error'
            current_job['error'] = str(e)
            current_job['message'] = f'Error: {str(e)}'
            logger.error(f"Job failed: {traceback.format_exc()}")
            
            # Provide more specific error information
            if "'val_losses'" in str(e):
                current_job['error'] = "Model training error: Missing validation losses tracking"
                current_job['message'] = "Error: Model training failed due to missing validation tracking"
            elif "ImportError" in str(e):
                current_job['error'] = "Import error: Required modules not found"
                current_job['message'] = "Error: Failed to import required prediction modules"
            else:
                # Log the full error for debugging
                current_job['error'] = f"Unexpected error: {str(e)}"
                current_job['message'] = f"Error: {str(e)}"

def run_prediction_with_progress(config):
    """Run prediction with progress updates during training"""
    global current_job
    
    try:
        # Import here to avoid circular imports
        from main import DataProcessor, LSTMModel, ModelTrainer, Visualizer
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise Exception(f"Import error: {e}")
    
    # Initialize data processor
    try:
        data_processor = DataProcessor(config)
        logger.info("Data processor initialized successfully")
    except Exception as e:
        logger.error(f"Data processor initialization failed: {e}")
        raise Exception(f"Data processor initialization error: {e}")
    
    # Download and preprocess data
    current_job['message'] = f'Downloading {config.ticker} data...'
    current_job['progress'] = 10
    data_processor.download_data()
    
    current_job['message'] = 'Preprocessing and preparing data...'
    current_job['progress'] = 20
    processed_data = data_processor.preprocess_data()
    
    # Validate processed data structure
    required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'scaler']
    missing_keys = [key for key in required_keys if key not in processed_data]
    if missing_keys:
        raise Exception(f"Missing required data keys: {missing_keys}")
    
    # Additional validation for data types and shapes
    logger.info(f"Data validation - X_train shape: {processed_data['X_train'].shape}, y_train shape: {processed_data['y_train'].shape}")
    logger.info(f"Data validation - X_test shape: {processed_data['X_test'].shape}, y_test shape: {processed_data['y_test'].shape}")
    
    # Validate data types
    if not isinstance(processed_data['X_train'], torch.Tensor):
        raise Exception("X_train must be a PyTorch tensor")
    if not isinstance(processed_data['y_train'], torch.Tensor):
        raise Exception("y_train must be a PyTorch tensor")
    if not isinstance(processed_data['X_test'], torch.Tensor):
        raise Exception("X_test must be a PyTorch tensor")
    if not isinstance(processed_data['y_test'], torch.Tensor):
        raise Exception("y_test must be a PyTorch tensor")
    
    # Ensure data is on the correct device
    if hasattr(processed_data['X_train'], 'device'):
        logger.info(f"Data device: {processed_data['X_train'].device}")
    
    logger.info(f"Data processed successfully. Train samples: {len(processed_data['X_train'])}, Test samples: {len(processed_data['X_test'])}")
    
    # Initialize model
    try:
        model = LSTMModel(
            input_dim=1,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=1
        ).to(config.device)
        logger.info(f"Model initialized successfully on {config.device}")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise Exception(f"Model initialization error: {e}")
    
    # Update progress
    current_job['progress'] = 40
    current_job['message'] = f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters'
    
    # Train model using the built-in training method
    try:
        trainer = ModelTrainer(config, model, processed_data)
        
        # Store trainer reference for progress tracking
        with job_lock:
            current_job['trainer'] = trainer
        
        # Verify trainer initialization
        if not hasattr(trainer, 'val_losses') or trainer.val_losses is None:
            logger.warning("Trainer val_losses not properly initialized, fixing...")
            trainer.val_losses = []
        if not hasattr(trainer, 'train_losses') or trainer.train_losses is None:
            logger.warning("Trainer train_losses not properly initialized, fixing...")
            trainer.train_losses = []
        
        # Debug: Log trainer state
        logger.info(f"Trainer initialized - train_losses: {type(trainer.train_losses)}, val_losses: {type(trainer.val_losses)}")
        logger.info(f"Data structure: {list(processed_data.keys())}")
        logger.info(f"X_train shape: {processed_data['X_train'].shape}, y_train shape: {processed_data['y_train'].shape}")
        
        # Use the built-in training method which properly handles val_losses
        current_job['message'] = 'Training LSTM model...'
        current_job['progress'] = 50
        
        # Train the model using the built-in method
        logger.info("Starting model training...")
        train_losses, val_losses = trainer.train()
        logger.info(f"Training completed with {len(train_losses)} epochs")
        
        # Ensure the trainer has the losses properly set
        trainer.train_losses = train_losses
        trainer.val_losses = val_losses
        
        # Final verification
        if not trainer.train_losses or not trainer.val_losses:
            raise Exception("Training completed but losses were not properly recorded")
        
        logger.info(f"Training completed successfully with {len(train_losses)} epochs")
        current_job['progress'] = 85
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Trainer state: train_losses={getattr(trainer, 'train_losses', 'NOT_SET')}, val_losses={getattr(trainer, 'val_losses', 'NOT_SET')}")
        raise Exception(f"Training error: {e}")
    
    # Evaluate model
    current_job['message'] = 'Evaluating model performance...'
    
    try:
        results = trainer.evaluate()
        # Ensure val_losses is included in results
        results['train_losses'] = trainer.train_losses
        results['val_losses'] = trainer.val_losses
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise Exception(f"Model evaluation error: {e}")
    
    # Generate visualizations
    current_job['progress'] = 90
    current_job['message'] = 'Generating visualizations...'
    
    try:
        visualizer = Visualizer(config, data_processor.data, results)
        visualizer.plot_results()
        logger.info("Visualizations generated successfully")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        # Don't fail the entire job for visualization errors
        current_job['message'] = f'Warning: Visualization failed ({str(e)}) but prediction completed'
    
    return results

@app.route('/health')
def health_check():
    """Health check endpoint for Vercel"""
    return jsonify({'status': 'healthy', 'message': 'Stock Prediction App is running'})

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start_prediction', methods=['POST'])
def start_prediction():
    """Start a new prediction job"""
    global current_job
    
    with job_lock:
        if current_job['status'] == 'running':
            return jsonify({'error': 'Another job is already running'}), 400
        
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['ticker', 'mode', 'start_date', 'hidden_dim', 'num_epochs', 'learning_rate']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Reset job status
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
            
            # Start job in background thread
            thread = threading.Thread(target=run_prediction_job, args=(data,))
            thread.daemon = True
            thread.start()
            
            # Wait a moment for the job to initialize
            time.sleep(0.1)
            
            return jsonify({'message': 'Prediction job started successfully'})
            
        except Exception as e:
            logger.error(f"Failed to start prediction: {e}")
            return jsonify({'error': str(e)}), 500

# Rate limiting for API calls
from functools import wraps
from time import time as current_time

def rate_limit(max_requests=60, window=60):
    """Rate limiting decorator"""
    requests = {}
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = current_time()
            client_ip = request.remote_addr
            
            # Clean old requests
            requests[client_ip] = [req_time for req_time in requests.get(client_ip, []) if now - req_time < window]
            
            if len(requests[client_ip]) >= max_requests:
                return jsonify({'error': 'Rate limit exceeded. Please wait before making more requests.'}), 429
            
            requests[client_ip].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.route('/api/job_status')
@rate_limit(max_requests=120, window=60)  # 2 requests per second max
def job_status():
    """Get current job status with caching"""
    global current_job, response_cache
    
    try:
        # Check cache first
        cache_key = f"job_status_{current_job['status']}_{current_job['progress']}"
        now = current_time()
        
        if cache_key in response_cache:
            cache_time, cached_response = response_cache[cache_key]
            if now - cache_time < cache_timeout:
                return cached_response
        
        # Create response with proper error handling
        with job_lock:
            response_data = current_job.copy()
            # Remove large objects that shouldn't be serialized
            if 'config' in response_data:
                del response_data['config']
            if 'trainer' in response_data:
                del response_data['trainer']
        
        response = jsonify(response_data)
        
        # Cache the response
        response_cache[cache_key] = (now, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in job_status endpoint: {e}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to get job status',
            'message': 'Internal server error',
            'progress': 0
        }), 500

def cleanup_cache():
    """Clean up expired cache entries"""
    global response_cache
    now = current_time()
    expired_keys = [key for key, (cache_time, _) in response_cache.items() if now - cache_time > cache_timeout]
    for key in expired_keys:
        del response_cache[key]

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

@app.route('/api/get_chart/<ticker>')
def get_chart(ticker):
    """Get the prediction chart as base64 encoded image"""
    import base64
    
    plot_file = f'{ticker}_prediction_results.png'
    if os.path.exists(plot_file):
        try:
            with open(plot_file, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return jsonify({
                    'success': True,
                    'chart_data': encoded_string,
                    'mime_type': 'image/png'
                })
        except Exception as e:
            logger.error(f"Error reading chart file: {e}")
            return jsonify({'error': 'Failed to read chart file'}), 500
    else:
        return jsonify({'error': 'Chart file not found'}), 404

@app.route('/api/download_plot/<ticker>')
def download_plot(ticker):
    """Download the prediction chart as a file"""
    plot_file = f'{ticker}_prediction_results.png'
    if os.path.exists(plot_file):
        try:
            return send_file(
                plot_file,
                as_attachment=True,
                download_name=f'{ticker}_prediction_results.png',
                mimetype='image/png'
            )
        except Exception as e:
            logger.error(f"Error serving chart file: {e}")
            return jsonify({'error': 'Failed to serve chart file'}), 500
    else:
        return jsonify({'error': 'Chart file not found'}), 404

@app.route('/api/training_progress')
def get_training_progress():
    """Get current training progress data for real-time charts"""
    global current_job
    
    try:
        with job_lock:
            if current_job['status'] == 'running' and 'trainer' in current_job:
                trainer = current_job['trainer']
                # Return training progress data
                progress_data = {
                    'epoch': len(getattr(trainer, 'train_losses', [])),
                    'train_losses': getattr(trainer, 'train_losses', []),
                    'val_losses': getattr(trainer, 'val_losses', []),
                    'progress': current_job['progress'],
                    'message': current_job['message']
                }
                return jsonify(progress_data)
            else:
                return jsonify({'error': 'No training job in progress'}), 400
    except Exception as e:
        logger.error(f"Error in training_progress endpoint: {e}")
        return jsonify({'error': 'Failed to get training progress'}), 500

@app.route('/api/final_results')
def get_final_results():
    """Get final prediction results with chart data"""
    global current_job
    
    try:
        with job_lock:
            if current_job['status'] == 'completed' and current_job['results']:
                # Get chart data if available
                ticker = current_job['results']['ticker']
                chart_data = None
                
                try:
                    plot_file = f'{ticker}_prediction_results.png'
                    if os.path.exists(plot_file):
                        import base64
                        with open(plot_file, 'rb') as image_file:
                            chart_data = base64.b64encode(image_file.read()).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error reading chart file: {e}")
                
                results = current_job['results'].copy()
                results['chart_data'] = chart_data
                
                return jsonify(results)
            else:
                return jsonify({'error': 'No completed results available'}), 400
    except Exception as e:
        logger.error(f"Error in final_results endpoint: {e}")
        return jsonify({'error': 'Failed to get final results'}), 500

@app.route('/api/live_training_data')
def get_live_training_data():
    """Get live training data for real-time chart updates"""
    global current_job
    
    try:
        with job_lock:
            if current_job['status'] == 'running' and 'trainer' in current_job:
                trainer = current_job['trainer']
                
                # Safely access trainer attributes
                train_losses = getattr(trainer, 'train_losses', [])
                val_losses = getattr(trainer, 'val_losses', [])
                
                live_data = {
                    'epochs': list(range(1, len(train_losses) + 1)),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'current_epoch': len(train_losses),
                    'total_epochs': getattr(current_job.get('config', {}), 'num_epochs', 100),
                    'progress': current_job['progress'],
                    'status': 'training'
                }
                return jsonify(live_data)
            elif current_job['status'] == 'completed':
                return jsonify({'status': 'completed', 'message': 'Training finished'})
            else:
                return jsonify({'status': 'idle', 'message': 'No training in progress'})
    except Exception as e:
        logger.error(f"Error in live_training_data endpoint: {e}")
        return jsonify({'error': 'Failed to get live training data'}), 500

@app.route('/api/training_config')
def get_training_config():
    """Get current training configuration"""
    global current_job
    
    try:
        with job_lock:
            if current_job['status'] == 'running' and 'config' in current_job:
                config = current_job['config']
                config_data = {
                    'ticker': config.ticker,
                    'start_date': config.start_date,
                    'hidden_dim': config.hidden_dim,
                    'num_layers': config.num_layers,
                    'num_epochs': config.num_epochs,
                    'learning_rate': config.learning_rate,
                    'batch_size': config.batch_size,
                    'device': str(config.device)
                }
                return jsonify(config_data)
            else:
                return jsonify({'error': 'No training configuration available'}), 400
    except Exception as e:
        logger.error(f"Error in training_config endpoint: {e}")
        return jsonify({'error': 'Failed to get training configuration'}), 500

@app.route('/api/training_stats')
def get_training_stats():
    """Get current training statistics and metrics"""
    global current_job
    
    try:
        with job_lock:
            if current_job['status'] == 'running' and 'trainer' in current_job:
                trainer = current_job['trainer']
                
                if trainer.train_losses and trainer.val_losses:
                    # Calculate training statistics
                    current_train_loss = trainer.train_losses[-1] if trainer.train_losses else 0
                    current_val_loss = trainer.val_losses[-1] if trainer.val_losses else 0
                    best_train_loss = min(trainer.train_losses) if trainer.train_losses else 0
                    best_val_loss = min(trainer.val_losses) if trainer.val_losses else 0
                    
                    stats = {
                        'current_epoch': len(trainer.train_losses),
                        'current_train_loss': round(current_train_loss, 6),
                        'current_val_loss': round(current_val_loss, 6),
                        'best_train_loss': round(best_train_loss, 6),
                        'best_val_loss': round(best_val_loss, 6),
                        'loss_improvement': round(best_train_loss - current_train_loss, 6) if current_train_loss > 0 else 0,
                        'overfitting_ratio': round(current_val_loss / current_train_loss, 3) if current_train_loss > 0 else 0,
                        'patience_counter': getattr(trainer, 'patience_counter', 0),
                        'best_val_loss_epoch': trainer.val_losses.index(best_val_loss) + 1 if best_val_loss in trainer.val_losses and trainer.val_losses else 0
                    }
                    return jsonify(stats)
                else:
                    return jsonify({'error': 'No training data available yet'}), 400
            else:
                return jsonify({'error': 'No training in progress'}), 400
    except Exception as e:
        logger.error(f"Error in training_stats endpoint: {e}")
        return jsonify({'error': 'Failed to get training stats'}), 500

@app.route('/api/reset_job', methods=['POST'])
def reset_job():
    """Reset job status"""
    global current_job, response_cache
    
    with job_lock:
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
        # Clear cache when resetting
        response_cache.clear()
        return jsonify({'message': 'Job reset successfully'})

def start_cache_cleanup():
    """Start periodic cache cleanup in background"""
    def cleanup_loop():
        while True:
            try:
                cleanup_cache()
                time.sleep(10)  # Clean up every 10 seconds
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(30)  # Wait longer on error
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

# Create necessary directories for production
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Start cache cleanup
start_cache_cleanup()

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔄 The app will automatically reload when you make changes")
    print("⚡ Performance optimizations enabled: caching, rate limiting, adaptive polling")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
