# AI Trading Bot - Streamlit Application

A comprehensive AI-powered trading platform built with Streamlit that provides automated trading strategies, portfolio management, and intelligent market analysis.

## Core Features

- **Direct Alpaca API Integration**: Seamlessly connects to Alpaca for paper and live trading
- **Portfolio Management**: Track positions, orders, and performance
- **Strategy Builder**: Create, backtest, and deploy trading strategies
- **AI-Powered Analysis**: Leverage FinGPT, FinRL, and DeepTradeBot for advanced market insights
- **User-Friendly Interface**: No coding required to use any features

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run Home.py
   ```

3. Enter your Alpaca API credentials in the Settings page
4. Start exploring the dashboard and features

## API Connections

The application uses direct API connections to:

- **Alpaca Markets**: For trading stocks, ETFs, and crypto
- **Alpha Vantage**: For additional market data (optional)
- **FinRL**: For reinforcement learning trading strategies
- **DeepTradeBot**: For deep learning-based trade signals

## Deployment Options

### Local Deployment

Simply run with Streamlit:
```
streamlit run Home.py
```

### Streamlit Cloud Deployment

Follow the deployment guide in `deploy_to_streamlit_cloud.md` for step-by-step instructions.

## Directory Structure

- `Home.py`: Main application entry point
- `pages/`: Application pages (Strategies, Portfolio, Analysis, Settings)
- `utils/`: Utility modules for API connections, data processing, and visualization
- `data/`: Sample data and strategy templates
- `assets/`: Static assets and images

## Important Notes

- By default, the application runs in paper trading mode
- All API credentials are stored in Streamlit's session state
- For Streamlit Cloud deployment, use the environment secrets management system

## License

© 2025 AI Trading Bot