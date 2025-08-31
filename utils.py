"""
Utility functions for Stock Prediction Program
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import os
from datetime import datetime

def setup_logging(log_level='INFO'):
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_directories(*dirs):
    """Create directories if they don't exist"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def save_model(model, save_path):
    """Save the trained model"""
    create_directories(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def load_model(model, load_path):
    """Load a trained model"""
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    return model
