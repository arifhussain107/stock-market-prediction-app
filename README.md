# Stock Market Prediction App

A machine learning application that predicts stock prices using LSTM (Long Short-Term Memory) neural networks.

## Features

- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **LSTM Model**: Deep learning model for time series prediction
- **Data Visualization**: Interactive charts showing actual vs predicted prices
- **Performance Metrics**: RMSE evaluation for model accuracy
- **Scalable Architecture**: Supports different stocks and time periods

## Technologies Used

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **LSTM**: Neural network architecture for sequence prediction
- **Yahoo Finance**: Stock data source
- **Matplotlib**: Data visualization
- **Scikit-learn**: Data preprocessing and evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arifhussain107/stock-market-prediction-app.git
cd stock-market-prediction-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. The app will:
   - Download MSFT stock data from 2020-01-01
   - Train an LSTM model on historical data
   - Display training progress
   - Show prediction results and error analysis
   - Generate interactive charts

## Model Architecture

- **Input**: 30-day sequence of stock prices
- **LSTM Layers**: 2 layers with 32 hidden dimensions
- **Output**: Predicted stock price for the next day
- **Training**: 200 epochs with Adam optimizer

## Configuration

You can modify the following parameters in `main.py`:
- `ticker`: Stock symbol (default: MSFT)
- `seq_length`: Sequence length for prediction (default: 30)
- `hidden_dim`: LSTM hidden dimensions (default: 32)
- `num_layers`: Number of LSTM layers (default: 2)
- `num_epochs`: Training epochs (default: 200)

## Performance

The model provides:
- Training RMSE metrics
- Test RMSE metrics
- Visual comparison of actual vs predicted prices
- Error analysis charts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Disclaimer

This application is for educational purposes only. Stock predictions are not guaranteed and should not be used as financial advice. Always consult with financial professionals before making investment decisions.
