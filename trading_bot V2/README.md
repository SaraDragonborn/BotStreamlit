# AI-Powered Trading Bot

A comprehensive AI-powered trading platform that leverages advanced machine learning and real-time market analysis to provide intelligent trading strategies and portfolio management.

## Features

### Core Features
- **Multiple Market Support**: Trade US and Indian stocks, Forex, and Cryptocurrencies
- **Day Trading**: Handle up to 100 stocks simultaneously with automatic trade triggering
- **Strategy Builder**: Create and customize trading strategies without coding knowledge
- **Backtesting**: Test strategies against historical data before deploying
- **Portfolio Management**: Track and analyze your trading portfolio
- **Technical Indicators**: Over 40 technical indicators for advanced analysis
- **AI Analysis**: FinGPT for news sentiment analysis and FinRL for reinforcement learning

### Indian Market Trading
- **NSE/BSE Support**: Trade on Indian stock exchanges using Angel One API
- **Indian Market Symbols**: Support for Nifty, BankNifty, and Indian equities
- **Specialized Strategies**: Trading strategies optimized for Indian markets
- **Real-time Data**: Real-time market feeds from Indian exchanges

[View Indian Market Trading Documentation](./INDIAN_MARKET_TRADING.md)

## Getting Started

### Prerequisites
- Alpaca API key and secret for US market trading
- Angel One API credentials for Indian market trading (optional)
- Python 3.8+ for running the application

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r streamlit_app/requirements.txt
   ```
3. Set up your API keys in the Settings page or as environment variables

### Running the Application
```bash
cd streamlit_app
streamlit run Home.py
```

## Usage

### Setting up API Credentials
1. Go to the Settings page
2. Enter your Alpaca API credentials for US markets
3. Enter your Angel One API credentials for Indian markets (optional)
4. Test the connections

### Deploying to Streamlit Cloud
For easy access from anywhere, you can deploy the application to Streamlit Cloud:

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Set up the following:
   - Main file path: `streamlit_app/Home.py`
   - Python version: 3.8 or higher
   - Add your API keys as secrets in the Streamlit Cloud dashboard
4. Deploy the application
5. Access your trading bot from any device with an internet connection

### Creating a Strategy
1. Go to the Strategies page
2. Click "Create New Strategy"
3. Select strategy type (Technical, AI-Powered, or Indian Market)
4. Configure parameters
5. Save the strategy

### Running a Backtest
1. Go to the Strategies page
2. Select a strategy
3. Click "Backtest"
4. Configure backtest parameters
5. Run the backtest

### Starting Trading
1. Go to the Strategies page
2. Select an active strategy
3. Click "Start Trading"
4. Monitor your positions in the Portfolio page

## Testing Indian Market Functionality

To test the Indian market trading functionality:
```bash
./run_indian_market_test.sh
```

## Documentation

- [Indian Market Trading Guide](./INDIAN_MARKET_TRADING.md)
- [Day Trading Guide](./streamlit_app/DAY_TRADING_GUIDE.md)

## Technology Stack

- **Streamlit**: For interactive web application
- **Python**: For backend logic and data processing
- **Alpaca API**: For US market data and trading execution
- **Angel One API**: For Indian market data and trading execution
- **Plotly and Numpy**: For advanced data visualization
- **Machine Learning**: For strategy optimization
- **Real-time Data Processing**: For market analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial data provided by Alpaca, Angel One, and Yahoo Finance
- AI models from FinGPT and FinRL