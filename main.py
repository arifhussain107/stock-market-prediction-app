import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
import os
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the stock prediction model"""
    def __init__(self):
        self.ticker = 'MSFT'
        self.start_date = '2020-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.sequence_length = 30
        self.train_split = 0.8
        self.hidden_dim = 64
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = 42
        # New caching and optimization settings
        self.enable_cache = True
        self.cache_dir = 'data_cache'
        self.cache_expiry_days = 1
        self.parallel_download = True
        self.max_workers = 4
        self.early_stopping_patience = 10 # Added for early stopping
        self.l2_reg = 0.001 # Added for L2 regularization

class DataProcessor:
    """Handles data downloading, preprocessing, and preparation with optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.data = None
        self.processed_data = None
        
        # Create cache directory
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _get_cache_key(self):
        """Generate cache key for the current data request"""
        cache_string = f"{self.config.ticker}_{self.config.start_date}_{self.config.end_date}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self):
        """Get cache file path"""
        cache_key = self._get_cache_key()
        return os.path.join(self.config.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path):
        """Check if cache file is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        # Check if cache is expired
        file_time = os.path.getmtime(cache_path)
        cache_age = time.time() - file_time
        max_age = self.config.cache_expiry_days * 24 * 3600  # Convert days to seconds
        
        return cache_age < max_age
    
    def _load_from_cache(self, cache_path):
        """Load data from cache"""
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Data loaded from cache: {cache_path}")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, data, cache_path):
        """Save data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                logger.info(f"Data saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def download_data(self):
        """Download stock data from Yahoo Finance with caching and optimizations"""
        try:
            # Check cache first
            if self.config.enable_cache:
                cache_path = self._get_cache_path()
                if self._is_cache_valid(cache_path):
                    cached_data = self._load_from_cache(cache_path)
                    if cached_data is not None:
                        self.data = cached_data
                        logger.info(f"Using cached data for {self.config.ticker}")
                        return self.data
            
            logger.info(f"Downloading data for {self.config.ticker} from {self.config.start_date} to {self.config.end_date}")
            
            # Use optimized yfinance download with progress bar disabled
            self.data = yf.download(
                self.config.ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                threads=True,  # Enable threading
                prepost=False,  # Disable pre/post market data for speed
                auto_adjust=True,  # Auto-adjust for splits/dividends
                repair=True  # Repair common data issues
            )
            
            if self.data.empty:
                raise ValueError(f"No data downloaded for {self.config.ticker}")
            
            # Save to cache
            if self.config.enable_cache:
                cache_path = self._get_cache_path()
                self._save_to_cache(self.data, cache_path)
                
            logger.info(f"Successfully downloaded {len(self.data)} data points")
            return self.data
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def download_multiple_stocks(self, tickers):
        """Download data for multiple stocks in parallel"""
        if not self.config.parallel_download:
            return {ticker: self._download_single_stock(ticker) for ticker in tickers}
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._download_single_stock, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    logger.error(f"Error downloading {ticker}: {e}")
                    results[ticker] = None
        
        return results
    
    def _download_single_stock(self, ticker):
        """Download data for a single stock"""
        try:
            data = yf.download(
                ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                threads=False,
                prepost=False,
                auto_adjust=True,
                repair=True
            )
            return data if not data.empty else None
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return None
    
    def preprocess_data(self):
        """Preprocess the downloaded data with optimizations"""
        try:
            if self.data is None:
                raise ValueError("No data available. Please download data first.")
            
            # Create a copy for processing
            df = self.data.copy()
            
            # Handle missing values more efficiently
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Use only closing prices for now (can be extended to multiple features)
            prices = df['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Create sequences more efficiently
            X, y = self._create_sequences_optimized(scaled_prices)
            
            # Split into train and test sets
            train_size = int(self.config.train_split * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Convert to PyTorch tensors
            X_train = torch.from_numpy(X_train).float().to(self.config.device)
            y_train = torch.from_numpy(y_train).float().to(self.config.device)
            X_test = torch.from_numpy(X_test).float().to(self.config.device)
            y_test = torch.from_numpy(y_test).float().to(self.config.device)
            
            self.processed_data = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'scaler': self.scaler
            }
            
            logger.info(f"Data preprocessed: Train set: {len(X_train)}, Test set: {len(X_test)}")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _create_sequences_optimized(self, data):
        """Create sequences for LSTM training more efficiently"""
        # Use vectorized operations instead of loops
        n = len(data) - self.config.sequence_length
        X = np.array([data[i:i + self.config.sequence_length] for i in range(n)])
        y = data[self.config.sequence_length:].reshape(-1, 1)
        return X, y

class LSTMModel(nn.Module):
    """LSTM-based stock price prediction model"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output
        out = out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class ModelTrainer:
    """Handles model training and evaluation with optimizations"""
    
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _create_data_loader(self, X, y, batch_size):
        """Create DataLoader for efficient batching"""
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Keep at 0 for Windows compatibility
            pin_memory=True if self.config.device.type == 'cuda' else False
        )
    
    def train(self):
        """Train the model with optimizations"""
        logger.info("Starting model training...")
        
        self.model.train()
        
        # Create data loaders
        train_loader = self._create_data_loader(self.data['X_train'], self.data['y_train'], self.config.batch_size)
        val_loader = self._create_data_loader(self.data['X_test'], self.data['y_test'], self.config.batch_size)
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch [{epoch}/{self.config.num_epochs}], "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"LR: {current_lr:.6f}")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        return self.train_losses, self.val_losses
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _validate(self):
        """Calculate validation loss (legacy method for compatibility)"""
        return self._validate_epoch(self._create_data_loader(self.data['X_test'], self.data['y_test'], self.config.batch_size))
    
    def evaluate(self):
        """Evaluate the model on test data"""
        logger.info("Evaluating model...")
        
        self.model.eval()
        with torch.no_grad():
            # Get predictions in batches to handle large datasets
            y_train_pred = self._predict_in_batches(self.data['X_train'])
            y_test_pred = self._predict_in_batches(self.data['X_test'])
            
            # Convert to numpy and inverse transform
            y_train_pred = self.data['scaler'].inverse_transform(y_train_pred.cpu().numpy())
            y_train_actual = self.data['scaler'].inverse_transform(self.data['y_train'].cpu().numpy())
            y_test_pred = self.data['scaler'].inverse_transform(y_test_pred.cpu().numpy())
            y_test_actual = self.data['scaler'].inverse_transform(self.data['y_test'].cpu().numpy())
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
            
            logger.info(f"Train RMSE: {train_rmse:.4f}")
            logger.info(f"Test RMSE: {test_rmse:.4f}")
            
            return {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred,
                    'actual_train': y_train_actual,
                    'actual_test': y_test_actual
                }
            }
    
    def _predict_in_batches(self, X):
        """Make predictions in batches to handle large datasets"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                batch_X = X[i:i + self.config.batch_size]
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred)
        
        self.model.train()
        return torch.cat(predictions, dim=0)

class Visualizer:
    """Handles plotting and visualization"""
    
    def __init__(self, config, data, results):
        self.config = config
        self.data = data
        self.results = results
        
    def plot_results(self):
        """Plot the training results and predictions"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Plot 1: Training and validation loss
            train_losses = self.results.get('train_losses', [])
            val_losses = self.results.get('val_losses', [])
            
            if train_losses and val_losses:
                # Ensure both arrays have the same length
                min_length = min(len(train_losses), len(val_losses))
                if min_length > 0:
                    # Plot training losses
                    axes[0].plot(range(min_length), train_losses[:min_length], label='Training Loss', color='blue')
                    
                    # Plot validation losses (every 10 epochs for clarity, but ensure we don't exceed array bounds)
                    val_indices = list(range(0, min_length, 10))
                    if val_indices and val_indices[-1] >= min_length:
                        val_indices[-1] = min_length - 1
                    if val_indices:
                        axes[0].plot(val_indices, [val_losses[i] for i in val_indices if i < len(val_losses)], 
                                     label='Validation Loss', color='red', marker='o')
                
                axes[0].set_title('Training and Validation Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True)
            else:
                axes[0].text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Training and Validation Loss')
            
            # Plot 2: Stock price predictions
            if 'predictions' in self.results and 'test' in self.results['predictions']:
                try:
                    test_dates = self.data['Close'].iloc[-len(self.results['predictions']['test']):].index
                    
                    axes[1].plot(test_dates, self.results['predictions']['actual_test'], 
                                 label='Actual Price', color='blue', linewidth=2)
                    axes[1].plot(test_dates, self.results['predictions']['test'], 
                                 label='Predicted Price', color='red', linewidth=2)
                    axes[1].set_title(f'{self.config.ticker} Stock Price Prediction')
                    axes[1].set_xlabel('Date')
                    axes[1].set_ylabel('Price ($)')
                    axes[1].legend()
                    axes[1].grid(True)
                except Exception as e:
                    logger.error(f"Error plotting predictions: {e}")
                    axes[1].text(0.5, 0.5, f'Error plotting predictions: {str(e)}', ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Stock Price Prediction')
            else:
                axes[1].text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Stock Price Prediction')
            
            # Plot 3: Prediction errors
            if 'predictions' in self.results and 'test' in self.results['predictions'] and 'test_rmse' in self.results:
                try:
                    test_dates = self.data['Close'].iloc[-len(self.results['predictions']['test']):].index
                    errors = np.abs(self.results['predictions']['actual_test'] - self.results['predictions']['test'])
                    axes[2].plot(test_dates, errors, color='green', linewidth=1)
                    axes[2].axhline(y=self.results['test_rmse'], color='red', linestyle='--', 
                                    label=f'RMSE: {self.results["test_rmse"]:.4f}')
                    axes[2].set_title('Prediction Errors')
                    axes[2].set_xlabel('Date')
                    axes[2].set_ylabel('Absolute Error')
                    axes[2].legend()
                    axes[2].grid(True)
                except Exception as e:
                    logger.error(f"Error plotting errors: {e}")
                    axes[2].text(0.5, 0.5, f'Error plotting errors: {str(e)}', ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title('Prediction Errors')
            else:
                axes[2].text(0.5, 0.5, 'No error data available', ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('Prediction Errors')
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'{self.config.ticker}_prediction_results.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved as {plot_filename}")
            
            # Close the figure to free memory (don't show in web app)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error in plot_results: {e}")
            # Try to save a simple error plot
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Plot Generation Error')
                plot_filename = f'{self.config.ticker}_prediction_results.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Error plot saved as {plot_filename}")
            except Exception as save_error:
                logger.error(f"Failed to save error plot: {save_error}")

def main(config=None):
    """Main function to run the stock prediction program"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize configuration if none provided
    if config is None:
        config = Config()
    
    try:
        logger.info("Starting Stock Prediction Program")
        logger.info(f"Using device: {config.device}")
        
        # Initialize data processor
        data_processor = DataProcessor(config)
        
        # Download and preprocess data
        data_processor.download_data()
        processed_data = data_processor.preprocess_data()
        
        # Initialize model
        model = LSTMModel(
            input_dim=1,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=1
        ).to(config.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        trainer = ModelTrainer(config, model, processed_data)
        train_losses, val_losses = trainer.train()
        
        # Evaluate model
        results = trainer.evaluate()
        results['train_losses'] = train_losses
        results['val_losses'] = val_losses
        
        # Visualize results
        visualizer = Visualizer(config, data_processor.data, results)
        visualizer.plot_results()
        
        logger.info("Program completed successfully!")
        
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()