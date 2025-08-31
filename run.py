#!/usr/bin/env python3
"""
Simple run script for Stock Prediction Program
"""
import sys
import os
import argparse
from config import get_quick_test_config, get_production_config, get_custom_config

def main():
    parser = argparse.ArgumentParser(description='Stock Prediction Program Runner')
    parser.add_argument('--mode', choices=['quick', 'custom', 'production', 'default'], 
                       default='default', help='Run mode')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Stock ticker symbol')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date for data')
    
    args = parser.parse_args()
    
    print("🚀 Stock Prediction Program")
    print("=" * 40)
    
    if args.mode == 'quick':
        print("Running Quick Test Mode...")
        config = get_quick_test_config()
        config.ticker = args.ticker
        config.start_date = args.start_date
        
    elif args.mode == 'custom':
        print("Running Custom Mode...")
        config = get_custom_config(
            ticker=args.ticker,
            start_date=args.start_date,
            hidden_dim=args.hidden,
            num_epochs=args.epochs
        )
        
    elif args.mode == 'production':
        print("Running Production Mode...")
        config = get_production_config()
        config.ticker = args.ticker
        config.start_date = args.start_date
        
    else:  # default
        print("Running Default Mode...")
        config = get_custom_config(
            ticker=args.ticker,
            start_date=args.start_date,
            hidden_dim=args.hidden,
            num_epochs=args.epochs
        )
    
    print(f"Configuration:")
    print(f"  Ticker: {config.ticker}")
    print(f"  Start Date: {config.start_date}")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    print("=" * 40)
    
    # Import and run main program
    try:
        from main import main as run_prediction
        run_prediction(config)
        print("✅ Program completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Program failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
