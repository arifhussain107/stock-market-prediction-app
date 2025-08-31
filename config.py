"""
Configuration file for Stock Prediction Program
"""
import torch
from datetime import datetime

class Config:
    """Configuration class for the stock prediction model"""
    
    def __init__(self):
        # Stock data configuration
        self.ticker = 'MSFT'  # Stock symbol
        self.start_date = '2020-01-01'  # Start date for data
        self.end_date = datetime.now().strftime('%Y-%m-%d')  # End date (today)
        
        # Data processing configuration
        self.sequence_length = 30  # Number of time steps to look back
        self.train_split = 0.8  # Training data split ratio
        self.feature_columns = ['Close']  # Features to use (can be extended)
        
        # Model architecture configuration
        self.hidden_dim = 64  # LSTM hidden dimension
        self.num_layers = 2  # Number of LSTM layers
        self.dropout = 0.2  # Dropout rate
        self.bidirectional = False  # Use bidirectional LSTM
        
        # Training configuration
        self.learning_rate = 0.001  # Learning rate
        self.num_epochs = 100  # Number of training epochs
        self.batch_size = 32  # Batch size
        self.early_stopping_patience = 10  # Early stopping patience
        self.l2_reg = 0.001  # L2 regularization
        
        # System configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = 42  # Random seed for reproducibility
        self.num_workers = 0  # Number of workers for data loading
        
        # Logging and output configuration
        self.log_level = 'INFO'
        self.save_model = True  # Save trained model
        self.model_save_path = 'models/'
        self.results_save_path = 'results/'
        
        # New caching and optimization settings
        self.enable_cache = True
        self.cache_dir = 'data_cache'
        self.cache_expiry_days = 1
        self.parallel_download = True
        self.max_workers = 4
        
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("STOCK PREDICTION CONFIGURATION")
        print("=" * 50)
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("=" * 50)

# Default configuration
default_config = Config()

# Example configurations for different use cases
def get_quick_test_config():
    """Configuration for quick testing"""
    config = Config()
    config.num_epochs = 20
    config.hidden_dim = 32
    config.sequence_length = 20
    return config

def get_production_config():
    """Configuration for production use"""
    config = Config()
    config.num_epochs = 500
    config.hidden_dim = 128
    config.num_layers = 3
    config.dropout = 0.3
    config.early_stopping_patience = 20
    return config

def get_custom_config(**kwargs):
    """Create custom configuration"""
    config = Config()
    config.update(**kwargs)
    return config
