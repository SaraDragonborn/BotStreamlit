# StocBot: Intraday Trading Bot for Indian Stocks

An automated trading bot for intraday trading on Indian stock markets using Angel One API. The bot implements two time-tested trading strategies and automatically switches between them based on market conditions.

## Features

- **Market Condition Analysis**: Uses ADX indicator to detect trending or sideways market
- **Strategy Switching**: Automatically selects the optimal strategy based on market conditions
- **Moving Average Crossover Strategy**: For trending markets
- **RSI Reversal Strategy**: For sideways/ranging markets
- **Risk Management**: Built-in stop-loss, take-profit, and position sizing
- **Backtesting**: Comprehensive backtesting to evaluate strategy performance
- **Logging**: Detailed logging of trades and system activity
- **Notifications**: Real-time trade and status notifications via Telegram

## Strategies

1. **Moving Average Crossover**
   - Uses 15 and 50 period Exponential Moving Averages (EMAs)
   - Generates buy signals when the short-term EMA crosses above the long-term EMA
   - Generates sell signals when the short-term EMA crosses below the long-term EMA
   - Best suited for trending markets

2. **RSI Reversal**
   - Uses Relative Strength Index (RSI) with a 14-period lookback
   - Generates buy signals when RSI is oversold (below 30) and starts to rise with increased volume
   - Generates sell signals when RSI is overbought (above 70) and starts to fall with increased volume
   - Best suited for sideways/ranging markets

## Getting Started

### Prerequisites

- Python 3.8 or later
- Angel One trading account with API access
- Telegram bot for notifications (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API credentials:
   ```
   # API credentials
   ANGEL_ONE_API_KEY=your_api_key
   ANGEL_ONE_CLIENT_ID=your_client_id
   ANGEL_ONE_CLIENT_PIN=your_client_pin
   
   # Telegram bot settings (optional)
   TELEGRAM_TOKEN=your_telegram_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   
   # Trading parameters (optional, can use defaults)
   CAPITAL=100000
   CAPITAL_PER_TRADE=5000
   MAX_POSITIONS=3
   STOP_LOSS_PERCENT=1.5
   TARGET_PERCENT=3.0
   ```

### Usage

#### Live Trading

```bash
python main.py
```

#### Backtesting

```bash
python main.py --backtest --symbol RELIANCE --start-date 2023-01-01 --end-date 2023-12-31
```

#### Using Custom Watchlist

```bash
python main.py --watchlist path/to/watchlist.txt
```

The watchlist file should contain one stock symbol per line.

## Configuration

All configuration parameters can be set in the `.env` file or passed as command-line arguments. Key parameters include:

- `CAPITAL`: Total trading capital
- `CAPITAL_PER_TRADE`: Maximum capital to allocate per trade
- `MAX_POSITIONS`: Maximum number of simultaneous positions
- `STOP_LOSS_PERCENT`: Default stop-loss percentage
- `TARGET_PERCENT`: Default take-profit percentage
- `MARKET_START_TIME`: Market opening time
- `MARKET_END_TIME`: Market closing time
- `TRADE_EXIT_TIME`: Time to exit all positions

Strategy-specific parameters can also be configured:
- Moving Average Crossover: `MA_SHORT_WINDOW`, `MA_LONG_WINDOW`, `MA_USE_EMA`
- RSI Reversal: `RSI_PERIOD`, `RSI_OVERSOLD`, `RSI_OVERBOUGHT`, `RSI_VOLUME_FACTOR`
- Market Condition: `ADX_PERIOD`, `TREND_THRESHOLD`, `SIDEWAYS_THRESHOLD`

## Directory Structure

```
trading-bot/
├── api/                  # API integrations
│   └── angel_api.py      # Angel One API connector
├── strategies/           # Trading strategies
│   ├── strategy_base.py  # Base strategy class
│   ├── moving_average_crossover.py
│   ├── rsi_reversal.py
│   └── market_condition.py
├── utils/                # Utility functions
│   ├── logger.py         # Logging utilities
│   └── telegram.py       # Telegram notifications
├── data/                 # Data storage
├── logs/                 # Log files
├── .env                  # Environment variables
├── config.py             # Configuration management
├── backtest.py           # Backtesting functionality
├── trading_bot.py        # Main bot class
└── main.py               # Entry point
```

## Disclaimer

This trading bot is provided for educational and research purposes only. It is not intended to be used as financial advice or a recommendation to trade real money. Trading stocks involves risk, and you may lose some or all of your investment. Always consult with a qualified financial advisor before making investment decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.