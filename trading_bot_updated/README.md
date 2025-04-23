# Multi-Asset Algorithmic Trading Bot

An advanced AI-powered trading platform supporting multiple asset classes (US stocks, Indian stocks, cryptocurrencies, forex) with a focus on algorithmic trading strategies, risk management, and performance tracking.

## Features

- **Multi-Asset Support**: Trade US stocks (via Alpaca), Indian stocks (via Angel One), with planned support for crypto and forex
- **Multiple Strategy Modules**: 
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - Trend Following
  - Mean Reversion
  - Breakout Detection
- **Comprehensive Architecture**:
  - Data Collection
  - Strategy Selection
  - Risk Management
  - Trade Execution
  - Performance Tracking
- **Backtesting**: Test strategies on historical data before deploying
- **User Interface**: Interactive Streamlit dashboard

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and fill in your API credentials:
   ```
   cp .env.template .env
   ```

## API Keys

The trading bot requires API keys to function:

1. **Alpaca API** (for US stocks): Sign up at [Alpaca Markets](https://app.alpaca.markets/)
2. **Angel One API** (for Indian stocks): Sign up at [Angel One SmartAPI](https://smartapi.angelbroking.com/)

## Getting Started

1. Set up your environment and API keys
2. Run the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```
3. Configure your strategies and watchlist
4. Start with paper trading before using real funds

## Usage

The application provides several tabs:

1. **Dashboard**: Overview of portfolio performance and market conditions
2. **Strategy Configuration**: Select and customize trading strategies
3. **Backtest**: Test strategies on historical data
4. **Trading**: Execute trades manually or enable automated trading
5. **Performance**: Track strategy and portfolio performance
6. **Settings**: Configure API keys and preferences

## Risk Warning

- Trading involves substantial risk of loss
- Always test strategies with paper trading first
- The bot is provided as-is with no guarantee of profits
- Never invest money you cannot afford to lose

## License

MIT License