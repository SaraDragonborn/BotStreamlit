# Multi-Asset Trading Bot

An AI-powered algorithmic trading platform that supports multiple asset classes including US stocks, Indian stocks, cryptocurrencies, and forex. The system uses adaptive strategy selection and robust risk management to execute trades across different markets.

## Features

- **Multi-Asset Support**: Trade US stocks (via Alpaca), Indian stocks (via Angel One), cryptocurrencies, and forex
- **Smart Strategy Selection**: Dynamically selects the best trading strategy based on market conditions
- **Advanced Risk Management**: Portfolio-level risk controls with position sizing and drawdown management
- **Multiple Trading Strategies**:
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - Trend Following with ADX
  - Mean Reversion with Bollinger Bands
  - Breakout Detection with Volume Confirmation
- **Backtesting**: Test strategies on historical data to evaluate performance
- **Configurable**: Highly customizable through configuration files and environment variables

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-asset-trading-bot.git
   cd multi-asset-trading-bot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and configuration
   ```

## Usage

### Configuration

1. Update your API credentials in the `.env` file
2. Customize the trading parameters in `config.py` if needed

### Running the Bot

For paper trading (simulation mode):
```bash
python main.py --paper
```

For live trading (use with caution):
```bash
python main.py --live
```

### Backtesting

To backtest a strategy on historical data:
```bash
python backtest.py --strategy moving_average_crossover --symbol AAPL --start 2022-01-01 --end 2023-01-01
```

## Architecture

The trading bot follows a modular architecture:

- `api/` - API adapters for different brokers/exchanges
- `strategies/` - Trading strategy implementations
- `utils/` - Utility functions and helpers
- `data/` - Data storage and processing
- `logs/` - Log files
- `models/` - ML model files (if applicable)

Core components:
- Strategy Selector ("Brain") - Chooses the optimal strategy
- Risk Manager - Handles position sizing and risk controls
- API Adapters - Interfaces with brokers/exchanges
- Data Collectors - Retrieves market data

## Adding New Strategies

To add a new trading strategy:

1. Create a new file in the `strategies/` directory
2. Implement the `StrategyBase` abstract class
3. Register your strategy in the strategy selector

Example:
```python
from strategies.strategy_base import StrategyBase

class MyCustomStrategy(StrategyBase):
    def __init__(self):
        super().__init__(
            name="My Custom Strategy",
            description="Description of the strategy"
        )
    
    def _define_parameters(self) -> None:
        # Define strategy parameters
        self.parameters = {
            "param1": {
                "type": "int",
                "description": "Description",
                "default": 10,
                "min": 1,
                "max": 100
            }
        }
    
    def generate_signals(self, data, parameters=None):
        # Your strategy logic here
        # Return data with signal column added
```

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

Never risk money you cannot afford to lose. Always test thoroughly on paper trading before deploying with real funds.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.