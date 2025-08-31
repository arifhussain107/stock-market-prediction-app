# Stock Prediction Program - Professional Template

A comprehensive, production-ready template for stock price prediction using LSTM neural networks with PyTorch.

## 🚀 Features

- **Modular Architecture**: Clean, organized code structure with separate classes for different responsibilities
- **Professional LSTM Model**: Advanced LSTM implementation with dropout and configurable layers
- **Comprehensive Data Processing**: Automated data downloading, preprocessing, and sequence creation
- **Flexible Configuration**: Easy-to-modify configuration system for different use cases
- **Professional Logging**: Comprehensive logging system for debugging and monitoring
- **Advanced Visualization**: Multiple plots showing training progress, predictions, and errors
- **Error Handling**: Robust error handling throughout the pipeline
- **Model Persistence**: Save and load trained models
- **Performance Metrics**: Multiple evaluation metrics (RMSE, MAE, R², etc.)

## 📁 Project Structure

```
stock_prediction_program/
├── main.py              # Main program with complete pipeline
├── config.py            # Configuration management
├── utils.py             # Utility functions
├── example.py           # Example usage scripts
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # Directory for saved models (auto-created)
├── results/            # Directory for results (auto-created)
└── logs/               # Log files (auto-created)
```

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch, yfinance, sklearn; print('All packages installed successfully!')"
   ```

## 🚀 Quick Start

### Basic Usage

Run the program with default settings (Microsoft stock):

```bash
python main.py
```

### Example Scripts

Run different examples:

```bash
python example.py
```

Choose from:
- Quick Test (20 epochs, small model)
- Custom Configuration (Tesla stock, custom parameters)
- Production Model (500 epochs, large model)

## ⚙️ Configuration

### Basic Configuration

Modify `config.py` or use the built-in configuration functions:

```python
from config import get_custom_config

# Custom configuration
config = get_custom_config(
    ticker='AAPL',           # Stock symbol
    start_date='2020-01-01', # Start date
    hidden_dim=128,          # LSTM hidden size
    num_epochs=200,          # Training epochs
    learning_rate=0.001      # Learning rate
)
```

### Available Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `ticker` | Stock symbol | 'MSFT' | Any valid stock symbol |
| `start_date` | Data start date | '2020-01-01' | YYYY-MM-DD format |
| `sequence_length` | Lookback period | 30 | 10-100 |
| `hidden_dim` | LSTM hidden size | 64 | 32-512 |
| `num_layers` | LSTM layers | 2 | 1-5 |
| `num_epochs` | Training epochs | 100 | 20-1000 |
| `learning_rate` | Learning rate | 0.001 | 0.0001-0.01 |

## 🔧 Customization

### Adding New Features

1. **New Data Sources**: Modify `DataProcessor.download_data()`
2. **Additional Features**: Extend `Config.feature_columns`
3. **New Models**: Create new classes inheriting from `nn.Module`
4. **Custom Metrics**: Add functions to `utils.py`

### Example: Multi-Feature Model

```python
# In config.py
config.feature_columns = ['Close', 'Volume', 'High', 'Low']

# In DataProcessor.preprocess_data()
features = df[config.feature_columns].values
scaled_features = self.scaler.fit_transform(features)
```

## 📊 Understanding the Output

The program generates:

1. **Training Progress**: Real-time loss values and validation metrics
2. **Performance Metrics**: RMSE, MAE, R² scores
3. **Visualizations**:
   - Training/validation loss curves
   - Actual vs. predicted stock prices
   - Prediction error analysis
4. **Saved Files**:
   - Model weights
   - Result plots
   - Log files

## 🎯 Best Practices

### For Quick Testing
- Use `get_quick_test_config()`
- Set `num_epochs = 20-50`
- Use smaller `hidden_dim = 32-64`

### For Production
- Use `get_production_config()`
- Set `num_epochs = 500+`
- Use larger `hidden_dim = 128-256`
- Enable early stopping

### For Research
- Experiment with different architectures
- Try various feature combinations
- Test different sequence lengths

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `batch_size` or `hidden_dim`
   - Use CPU: `config.device = 'cpu'`

2. **Poor Predictions**:
   - Increase `num_epochs`
   - Adjust `learning_rate`
   - Try different `sequence_length`

3. **Data Download Issues**:
   - Check internet connection
   - Verify stock symbol is valid
   - Try different date ranges

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster training
2. **Data Quality**: Use longer date ranges for better training
3. **Hyperparameter Tuning**: Experiment with different configurations
4. **Regularization**: Use dropout and L2 regularization to prevent overfitting

## 🔮 Future Enhancements

- [ ] Support for multiple stocks simultaneously
- [ ] Advanced technical indicators
- [ ] Ensemble methods
- [ ] Real-time prediction API
- [ ] Web interface
- [ ] Backtesting framework

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on the project repository
4. Check the example scripts for usage patterns

---

**Happy Trading! 📈💰**
