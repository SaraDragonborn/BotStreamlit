"""
Configuration Module
=======================================
Central configuration for the trading bot.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API Configuration
ALPACA_CONFIG = {
    "api_key": os.environ.get("ALPACA_API_KEY", ""),
    "api_secret": os.environ.get("ALPACA_API_SECRET", ""),
    "paper_trading": os.environ.get("ALPACA_PAPER", "true").lower() == "true",
    "base_url": os.environ.get("ALPACA_BASE_URL", "")
}

ANGEL_CONFIG = {
    "api_key": os.environ.get("ANGEL_API_KEY", ""),
    "client_id": os.environ.get("ANGEL_CLIENT_ID", ""),
    "password": os.environ.get("ANGEL_PASSWORD", ""),
    "totp_key": os.environ.get("ANGEL_TOTP_KEY", ""),
    "api_secret": os.environ.get("ANGEL_API_SECRET", "")
}

# Asset Classes
ASSET_CLASSES = {
    "stocks": True,
    "crypto": False,
    "forex": False,
    "options": False,
    "futures": False
}

# Stock Universe
US_STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "HD", "BAC", "MA", "DIS", "ADBE", "CRM", "NFLX", "INTC",
    "VZ", "KO", "T", "PFE", "MRK", "CSCO", "CMCSA", "PEP", "ABT", "CVX"
]

INDIAN_STOCK_UNIVERSE = [
    "RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ", "ICICIBANK-EQ", "KOTAKBANK-EQ",
    "HINDUNILVR-EQ", "SBIN-EQ", "BAJFINANCE-EQ", "BHARTIARTL-EQ", "ITC-EQ", "AXISBANK-EQ",
    "ASIANPAINT-EQ", "HCLTECH-EQ", "WIPRO-EQ", "MARUTI-EQ", "TITAN-EQ", "ULTRACEMCO-EQ",
    "SUNPHARMA-EQ", "TATASTEEL-EQ", "BAJAJFINSV-EQ", "TATAMOTORS-EQ"
]

# Crypto Universe
CRYPTO_UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD", "DOT/USD", "DOGE/USD"]

# Forex Universe
FOREX_UNIVERSE = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]

# Trading Configuration
TRADING_CONFIG = {
    "trading_hours": {
        "us_stocks": {
            "start": "09:30",  # Eastern Time
            "end": "16:00"
        },
        "indian_stocks": {
            "start": "09:15",  # Indian Standard Time
            "end": "15:30"
        },
        "crypto": {
            "start": "00:00",  # 24/7 trading
            "end": "23:59"
        },
        "forex": {
            "start": "00:00",  # Nearly 24/5 trading
            "end": "23:59"
        }
    },
    "paper_trading": True,
    "max_positions": 10,
    "default_order_type": "market",
    "max_portfolio_risk": 0.02,  # Maximum 2% daily portfolio risk
    "max_position_risk": 0.01,   # Maximum 1% risk per position
    "daily_trading_limit": 0.1   # Maximum 10% of portfolio value traded per day
}

# Strategy Configuration
STRATEGY_CONFIG = {
    "strategy_selection_mode": "adaptive",  # 'performance', 'rotation', 'ensemble', 'adaptive'
    "lookback_periods": 30,
    "performance_metric": "sharpe_ratio",  # 'sharpe_ratio', 'total_return', 'profit_factor'
    "strategy_weights": {
        "moving_average_crossover": 1.0,
        "rsi": 1.0,
        "trend_following": 1.0,
        "mean_reversion": 1.0,
        "breakout": 1.0
    },
    "strategy_parameters": {
        "moving_average_crossover": {
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "ma_type": "EMA"
        },
        "rsi": {
            "rsi_period": 14,
            "overbought_threshold": 70,
            "oversold_threshold": 30
        },
        "trend_following": {
            "ema_period": 21,
            "adx_period": 14,
            "adx_threshold": 25
        },
        "mean_reversion": {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        },
        "breakout": {
            "channel_period": 20,
            "volume_factor": 1.5,
            "lookback_periods": 2,
            "atr_period": 14,
            "atr_multiplier": 1.0
        }
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.environ.get("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "file_rotation": {
        "maxBytes": 10 * 1024 * 1024,  # 10 MB
        "backupCount": 5
    }
}

# Database Configuration (if needed)
DATABASE_CONFIG = {
    "enabled": False,
    "type": "sqlite",
    "path": str(DATA_DIR / "trading_bot.db"),
    "connection_string": ""
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "initial_capital": 100000,
    "commission": 0.001,  # 0.1% commission per trade
    "slippage": 0.001,    # 0.1% slippage
    "benchmark": "SPY"    # Benchmark for comparison
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    "email": {
        "enabled": False,
        "smtp_server": os.environ.get("SMTP_SERVER", ""),
        "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
        "sender_email": os.environ.get("SENDER_EMAIL", ""),
        "receiver_email": os.environ.get("RECEIVER_EMAIL", ""),
        "password": os.environ.get("EMAIL_PASSWORD", "")
    },
    "telegram": {
        "enabled": False,
        "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID", "")
    }
}

# Function to load custom configuration from a file
def load_custom_config(filepath: str) -> Dict:
    """
    Load custom configuration from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the configuration file
        
    Returns:
    --------
    Dict
        Custom configuration
    """
    try:
        with open(filepath, 'r') as f:
            custom_config = json.load(f)
        return custom_config
    except Exception as e:
        logging.error(f"Error loading custom configuration: {str(e)}")
        return {}

# Function to get a unified configuration
def get_config(custom_config_path: Optional[str] = None) -> Dict:
    """
    Get a unified configuration, optionally merged with custom configuration.
    
    Parameters:
    -----------
    custom_config_path : str, optional
        Path to a custom configuration file
        
    Returns:
    --------
    Dict
        Unified configuration
    """
    config = {
        "alpaca": ALPACA_CONFIG,
        "angel": ANGEL_CONFIG,
        "asset_classes": ASSET_CLASSES,
        "us_stocks": US_STOCK_UNIVERSE,
        "indian_stocks": INDIAN_STOCK_UNIVERSE,
        "crypto": CRYPTO_UNIVERSE,
        "forex": FOREX_UNIVERSE,
        "trading": TRADING_CONFIG,
        "strategy": STRATEGY_CONFIG,
        "logging": LOGGING_CONFIG,
        "database": DATABASE_CONFIG,
        "backtest": BACKTEST_CONFIG,
        "notification": NOTIFICATION_CONFIG
    }
    
    # Merge custom configuration if provided
    if custom_config_path:
        custom_config = load_custom_config(custom_config_path)
        
        # Deep merge custom config into base config
        for key, value in custom_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Merge nested dictionaries
                config[key].update(value)
            else:
                # Replace top-level keys
                config[key] = value
    
    return config

# Function to validate the configuration
def validate_config(config: Dict) -> List[str]:
    """
    Validate the configuration.
    
    Parameters:
    -----------
    config : Dict
        Configuration to validate
        
    Returns:
    --------
    List[str]
        List of validation errors, empty if valid
    """
    errors = []
    
    # Validate API keys if using live trading
    if not config["trading"]["paper_trading"]:
        if not all([config["alpaca"]["api_key"], config["alpaca"]["api_secret"]]) and config["asset_classes"]["stocks"]:
            errors.append("Missing Alpaca API credentials for live US stock trading")
        
        if not all([config["angel"]["api_key"], config["angel"]["client_id"], config["angel"]["password"]]) and config["asset_classes"]["stocks"]:
            errors.append("Missing Angel One API credentials for live Indian stock trading")
    
    # Validate trading config
    if config["trading"]["max_portfolio_risk"] > 0.05:
        errors.append("Portfolio risk too high (> 5%)")
    
    if config["trading"]["max_position_risk"] > 0.02:
        errors.append("Position risk too high (> 2%)")
    
    # Validate strategy config
    if not config["strategy"]["strategy_selection_mode"] in ["performance", "rotation", "ensemble", "adaptive"]:
        errors.append("Invalid strategy selection mode")
    
    return errors

# Get the default configuration
config = get_config()

# Validate the configuration
validation_errors = validate_config(config)
if validation_errors:
    for error in validation_errors:
        logging.warning(f"Configuration Warning: {error}")

# Export default configuration
default_config = config