# 🌐 Stock Prediction Web Application

A beautiful, user-friendly web interface for your Stock Prediction AI program!

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
python start_webapp.py
```

### Option 2: Run Flask Directly
```bash
python app.py
```

### Option 3: Use Flask Command
```bash
flask run
```

## 🌐 Access the Web App

Once running, open your browser and go to:
**http://localhost:5000**

## ✨ Features

### 🎯 **Easy Configuration**
- **Stock Symbol**: Enter any valid stock ticker (AAPL, MSFT, TSLA, etc.)
- **Date Range**: Choose start date for historical data
- **Prediction Modes**:
  - **Quick Test**: 20 epochs, small model (fast testing)
  - **Custom**: User-defined parameters
  - **Production**: 500 epochs, large model (high accuracy)

### ⚙️ **Advanced Parameters**
- **Hidden Dimensions**: LSTM layer size (32-512)
- **Training Epochs**: Number of training iterations (20-1000)
- **Learning Rate**: Model learning speed (0.0001-0.01)

### 📊 **Real-time Monitoring**
- Live progress bar during training
- Real-time status updates
- Training and validation loss tracking

### 📈 **Results & Visualization**
- Training vs Test RMSE metrics
- Downloadable prediction plots
- Professional charts and analysis

## 🎨 **Beautiful UI**

- **Modern Design**: Clean, professional interface
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Elements**: Real-time updates and progress tracking
- **User-Friendly**: Intuitive form controls and validation

## 🔧 **How It Works**

1. **Configure**: Set your stock symbol and parameters
2. **Submit**: Click "Start Prediction" to begin
3. **Monitor**: Watch real-time progress and status
4. **Results**: View metrics and download visualizations
5. **Download**: Get your prediction plots as PNG files

## 📁 **File Structure**

```
stock_prediction_program/
├── app.py                 # Flask web application
├── start_webapp.py       # Easy startup script
├── templates/
│   └── index.html        # Web interface template
├── main.py               # Core prediction logic
├── config.py             # Configuration management
├── utils.py              # Utility functions
└── requirements.txt      # Python dependencies
```

## 🛠️ **Installation**

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python -c "import flask, torch, yfinance; print('✅ All packages installed!')"
   ```

## 🚀 **Usage Examples**

### Quick Test (Apple Stock)
1. Enter: `AAPL`
2. Select: Quick Test mode
3. Click: Start Prediction
4. Wait: ~2-3 minutes for results

### Production Model (Tesla Stock)
1. Enter: `TSLA`
2. Select: Production mode
3. Click: Start Prediction
4. Wait: ~10-15 minutes for results

### Custom Configuration
1. Enter: `GOOGL`
2. Select: Custom mode
3. Set: Hidden Dim = 128, Epochs = 200
4. Click: Start Prediction

## 🔍 **Troubleshooting**

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Kill process using port 5000
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   ```

2. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Flask Not Found**:
   ```bash
   pip install flask
   ```

### Debug Mode

For detailed error information, run:
```bash
python app.py
```

## 📱 **Mobile Support**

The web interface is fully responsive and works great on:
- 📱 Smartphones
- 📱 Tablets
- 💻 Desktop computers
- 🖥️ Laptops

## 🎯 **Best Practices**

### For Quick Testing
- Use Quick Test mode
- Choose popular stocks (AAPL, MSFT, GOOGL)
- Start with recent data (last 2 years)

### For Production Use
- Use Production mode
- Choose stocks with good data history
- Use longer date ranges for better training

### For Research
- Experiment with different parameters
- Try various stock symbols
- Compare different configurations

## 🔮 **Future Enhancements**

- [ ] User accounts and saved predictions
- [ ] Multiple stock comparison
- [ ] Advanced technical indicators
- [ ] Real-time stock data
- [ ] Export to Excel/CSV
- [ ] Email notifications
- [ ] API endpoints for developers

## 📞 **Support**

If you encounter issues:

1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check the console for error messages
4. Ensure you have internet connection for stock data

## 🎉 **Ready to Start?**

Run the startup script and begin predicting stock prices with AI!

```bash
python start_webapp.py
```

**Happy Trading! 📈💰**
