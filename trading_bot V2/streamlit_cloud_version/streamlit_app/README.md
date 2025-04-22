# AI Trading Bot - Streamlit UI

A comprehensive AI-powered trading bot with a user-friendly Streamlit interface. This application integrates multiple AI models including FinGPT for sentiment analysis, DeepTradeBot for technical analysis, and FinRL for reinforcement learning-based trading strategies.

## Features

- **Dashboard**: Overview of portfolio performance, positions, and recent trading signals
- **Strategy Management**: Create, edit, backtest, and deploy custom trading strategies
- **Portfolio Management**: Track positions, orders, and performance metrics
- **AI Analysis**: Sentiment analysis, trading signals, and model training
- **Account Settings**: Configure API connections, trading preferences, and notifications

## Getting Started

### Prerequisites

- Python 3.9+
- Streamlit
- Pandas, Numpy, Plotly
- Alpaca API account (for trading capabilities)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd trading-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with your API keys:
   ```
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_API_SECRET=your_alpaca_secret
   ```

### Running the App

Run the Streamlit app:
```
streamlit run streamlit_app/Home.py
```

Or:
```
cd streamlit_app
streamlit run Home.py
```

The app will be available at http://localhost:8501

## Deployment to Streamlit Cloud

To deploy this application to Streamlit Cloud for easy access:

1. Create a Streamlit Cloud account at https://streamlit.io/cloud

2. Connect your GitHub repository to Streamlit Cloud

3. Deploy the app:
   - Select the repository
   - Set the main file path to `streamlit_app/Home.py`
   - Add your secrets (Alpaca API keys) in the Streamlit Cloud dashboard

4. Your app will be deployed with a public URL (e.g., `https://yourusername-trading-bot.streamlit.app`)

## Modules

- **Home**: Main dashboard with portfolio overview
- **Strategies**: Create and manage trading strategies, run backtests
- **Portfolio**: Manage positions and orders, analyze performance
- **AI Analysis**: Get sentiment analysis, trading signals, and train models
- **Settings**: Configure API connections and preferences

## API Integration

This app integrates with:

- **Alpaca API**: For executing trades and getting market data
- **FinGPT**: For sentiment analysis and news processing
- **DeepTradeBot**: For technical analysis and pattern recognition
- **FinRL**: For reinforcement learning trading strategies

## Security Notes

- API keys are stored securely as environment variables
- When deployed to Streamlit Cloud, use their secrets management
- The app is configured for both paper trading (simulation) and live trading

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Powered by Alpaca API
- Integrates FinGPT, DeepTradeBot, and FinRL for AI capabilities