# AI Trading Bot - User Guide

Welcome to the AI Trading Bot! This guide will help you navigate the features and functionality of your trading bot.

## Getting Started

### Installation

1. Ensure you have Python 3.9+ installed
2. Install required packages: `pip install -r requirements.txt`
3. Run the app: `streamlit run Home.py`

### API Connection

1. Create an [Alpaca](https://alpaca.markets/) account if you don't have one
2. Generate API keys from your Alpaca dashboard
3. Enter your API key and secret in the sidebar
4. Select either Paper Trading (simulated) or Live Trading
5. Test the connection to verify your credentials

## Dashboard

The dashboard provides an overview of your trading activity:

- **Portfolio Value**: Current value of all your positions
- **Available Cash**: Funds available for trading
- **Today's P&L**: Profit and loss for the current trading day
- **Active Positions**: Number of open positions

You'll also see:
- **Portfolio Composition**: Pie chart showing asset allocation
- **Recent Signals**: Trading signals from AI analysis
- **Recent Orders**: Last 5 executed orders 
- **Market News**: Latest financial news
- **Active Strategies**: Currently running trading strategies

## Pages

### Strategies

Create, edit, backtest and deploy trading strategies:

1. **Active Strategies**: View and manage existing strategies
2. **Strategy Builder**: Create new strategies using:
   - Technical indicators (Moving Averages, RSI, MACD)
   - AI-powered models (Sentiment Analysis, Deep Learning)
3. **Backtesting**: Test strategies on historical data to evaluate performance

### Portfolio

Manage your trading positions and orders:

1. **Positions**: View and manage open positions
2. **Orders**: Track recent orders and their status
3. **Trade**: Place new orders manually
4. **Performance**: Analyze your portfolio performance metrics
   - Equity curve
   - Drawdown chart
   - Monthly returns heatmap

### AI Analysis

Leverage AI for trading insights:

1. **News & Sentiment**: Analyze news sentiment for specific symbols
2. **Trading Signals**: Get AI-generated trading recommendations
3. **AI Models**: Train and deploy custom AI trading models:
   - FinGPT for sentiment analysis
   - DeepTradeBot for technical analysis
   - FinRL for reinforcement learning strategies

### Settings

Configure your trading environment:

1. **API Connections**: Manage API keys and connections
2. **Trading Preferences**: Set default trading parameters
3. **Notifications**: Configure alerts and notification methods
4. **Data Sources**: Choose market data providers
5. **Account**: User profile and security settings

## Features

### Technical Analysis

Utilize popular technical indicators:
- Moving Averages (SMA, EMA, WMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Support and resistance levels

### AI Capabilities

The bot includes advanced AI capabilities:
- **FinGPT**: Analyze news sentiment and generate summaries
- **DeepTradeBot**: Deep learning for pattern recognition
- **FinRL**: Reinforcement learning for adaptive strategies

### Risk Management

Control your trading risk:
- Set maximum position sizes
- Configure stop-loss and take-profit levels
- Set daily loss limits
- Create portfolio allocation rules

## Trading Modes

### Paper Trading

Practice trading with simulated money:
- Test strategies without financial risk
- Validate trading ideas in real market conditions
- Gain confidence before live trading

### Live Trading

Trade with real funds:
- Execute the same strategies from paper trading
- Implement proper risk management
- Monitor performance with real-time data

## Tips for Success

1. **Start with Paper Trading**: Always test strategies before using real money
2. **Use Backtesting**: Validate strategies with historical data
3. **Manage Risk**: Never risk more than you can afford to lose
4. **Diversify**: Run multiple strategies across different symbols
5. **Monitor Performance**: Regularly review your trading results

## Troubleshooting

- **Connection Issues**: Check your API keys and internet connection
- **Data Delays**: Market data may have slight delays based on your data provider
- **Strategy Errors**: Check the logs for specific error messages
- **Performance Issues**: Close other applications to free up resources

## Getting Help

For additional support:
- Check the documentation in the GitHub repository
- Review the API documentation for [Alpaca](https://alpaca.markets/docs/)
- Join trading communities for strategy ideas