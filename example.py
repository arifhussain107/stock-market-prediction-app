"""
Example script demonstrating how to use the Stock Prediction Program
"""
from main import main
from config import get_quick_test_config, get_production_config, get_custom_config
import logging

def run_quick_test():
    """Run a quick test with minimal training"""
    print("Running Quick Test...")
    config = get_quick_test_config()
    config.ticker = 'AAPL'  # Change to Apple stock
    config.print_config()
    
    try:
        main()
        print("Quick test completed successfully!")
    except Exception as e:
        print(f"Quick test failed: {e}")

def run_custom_prediction():
    """Run with custom configuration"""
    print("Running Custom Prediction...")
    
    # Custom configuration for Tesla stock
    config = get_custom_config(
        ticker='TSLA',
        start_date='2021-01-01',
        hidden_dim=128,
        num_epochs=150,
        learning_rate=0.0005
    )
    config.print_config()
    
    try:
        main()
        print("Custom prediction completed successfully!")
    except Exception as e:
        print(f"Custom prediction failed: {e}")

def run_production_model():
    """Run with production-ready configuration"""
    print("Running Production Model...")
    config = get_production_config()
    config.ticker = 'GOOGL'  # Google stock
    config.print_config()
    
    try:
        main()
        print("Production model completed successfully!")
    except Exception as e:
        print(f"Production model failed: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Stock Prediction Program Examples")
    print("=" * 40)
    
    # Choose which example to run
    choice = input("Choose example (1: Quick Test, 2: Custom, 3: Production): ")
    
    if choice == '1':
        run_quick_test()
    elif choice == '2':
        run_custom_prediction()
    elif choice == '3':
        run_production_model()
    else:
        print("Invalid choice. Running default example...")
        main()
