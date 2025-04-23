"""
Configuration Module
=======================================
Manages configuration for the trading bot application.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Default config
DEFAULT_CONFIG = {
    # API credentials
    'ANGEL_ONE_API_KEY': os.getenv('ANGEL_ONE_API_KEY', ''),
    'ANGEL_ONE_CLIENT_ID': os.getenv('ANGEL_ONE_CLIENT_ID', ''),
    'ANGEL_ONE_CLIENT_PIN': os.getenv('ANGEL_ONE_CLIENT_PIN', ''),
    'ANGEL_ONE_TOKEN': os.getenv('ANGEL_ONE_TOKEN', ''),
    
    # Alpaca API credentials (optional)
    'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY', ''),
    'ALPACA_API_SECRET': os.getenv('ALPACA_API_SECRET', ''),
    'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
    
    # Telegram bot settings
    'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN', ''),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', ''),
    
    # Trading parameters
    'CAPITAL': float(os.getenv('CAPITAL', '100000')),
    'CAPITAL_PER_TRADE': float(os.getenv('CAPITAL_PER_TRADE', '5000')),
    'MAX_POSITIONS': int(os.getenv('MAX_POSITIONS', '3')),
    'STOP_LOSS_PERCENT': float(os.getenv('STOP_LOSS_PERCENT', '1.5')),
    'TARGET_PERCENT': float(os.getenv('TARGET_PERCENT', '3.0')),
    'TRAILING_STOP_THRESHOLD': float(os.getenv('TRAILING_STOP_THRESHOLD', '1.5')),
    
    # Market timing
    'MARKET_START_TIME': os.getenv('MARKET_START_TIME', '09:30:00'),
    'MARKET_END_TIME': os.getenv('MARKET_END_TIME', '15:00:00'),
    'TRADE_EXIT_TIME': os.getenv('TRADE_EXIT_TIME', '15:00:00'),
    
    # Default watchlist (if not loaded from file)
    'DEFAULT_WATCHLIST': [
        'RELIANCE',
        'HDFCBANK',
        'INFY',
        'ICICIBANK',
        'TCS',
        'KOTAKBANK',
        'HINDUNILVR',
        'ITC',
        'AXISBANK',
        'SBIN'
    ],
    
    # Strategy parameters
    'STRATEGY_PARAMS': {
        'moving_average_crossover': {
            'short_window': int(os.getenv('MA_SHORT_WINDOW', '15')),
            'long_window': int(os.getenv('MA_LONG_WINDOW', '50')),
            'use_ema': os.getenv('MA_USE_EMA', 'true').lower() == 'true'
        },
        'rsi_reversal': {
            'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
            'oversold_threshold': int(os.getenv('RSI_OVERSOLD', '30')),
            'overbought_threshold': int(os.getenv('RSI_OVERBOUGHT', '70')),
            'volume_factor': float(os.getenv('RSI_VOLUME_FACTOR', '1.5'))
        },
        'market_condition': {
            'index_symbol': os.getenv('INDEX_SYMBOL', 'NIFTY'),
            'adx_period': int(os.getenv('ADX_PERIOD', '14')),
            'trend_threshold': int(os.getenv('TREND_THRESHOLD', '25')),
            'sideways_threshold': int(os.getenv('SIDEWAYS_THRESHOLD', '20'))
        }
    },
    
    # Logging settings
    'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
    'LOG_DIR': os.getenv('LOG_DIR', 'logs'),
    
    # Data directory
    'DATA_DIR': os.getenv('DATA_DIR', 'data'),
}

# Global config
_CONFIG = DEFAULT_CONFIG.copy()

def load_config(file_path: Optional[str] = None) -> Dict:
    """
    Load configuration from file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to config file
        
    Returns:
    --------
    dict
        Loaded configuration
    """
    global _CONFIG
    
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                loaded_config = json.load(f)
                
                # Merge loaded config with default config
                for key, value in loaded_config.items():
                    _CONFIG[key] = value
        except Exception as e:
            print(f"Error loading config from {file_path}: {str(e)}")
    
    return _CONFIG

def save_config(file_path: str) -> bool:
    """
    Save configuration to file.
    
    Parameters:
    -----------
    file_path : str
        Path to save config file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(_CONFIG, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error saving config to {file_path}: {str(e)}")
        return False

def get(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Parameters:
    -----------
    key : str
        Configuration key
    default : any, optional
        Default value if key not found
        
    Returns:
    --------
    any
        Configuration value
    """
    return _CONFIG.get(key, default)

def set(key: str, value: Any) -> None:
    """
    Set a configuration value.
    
    Parameters:
    -----------
    key : str
        Configuration key
    value : any
        Configuration value
    """
    _CONFIG[key] = value

def load_watchlist(file_path: Optional[str] = None) -> List[str]:
    """
    Load watchlist from file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to watchlist file
        
    Returns:
    --------
    list
        Watchlist
    """
    if not file_path:
        return get('DEFAULT_WATCHLIST', [])
    
    try:
        with open(file_path, 'r') as f:
            watchlist = [line.strip() for line in f.readlines() if line.strip()]
        return watchlist
    except Exception as e:
        print(f"Error loading watchlist from {file_path}: {str(e)}")
        return get('DEFAULT_WATCHLIST', [])

def create_env_template() -> bool:
    """
    Create a template .env file.
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    template_content = """# API credentials
ANGEL_ONE_API_KEY=
ANGEL_ONE_CLIENT_ID=
ANGEL_ONE_CLIENT_PIN=
ANGEL_ONE_TOKEN=

# Alpaca API credentials (optional)
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Telegram bot settings
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=

# Trading parameters
CAPITAL=100000
CAPITAL_PER_TRADE=5000
MAX_POSITIONS=3
STOP_LOSS_PERCENT=1.5
TARGET_PERCENT=3.0
TRAILING_STOP_THRESHOLD=1.5

# Market timing
MARKET_START_TIME=09:30:00
MARKET_END_TIME=15:00:00
TRADE_EXIT_TIME=15:00:00

# Strategy parameters
MA_SHORT_WINDOW=15
MA_LONG_WINDOW=50
MA_USE_EMA=true

RSI_PERIOD=14
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70
RSI_VOLUME_FACTOR=1.5

INDEX_SYMBOL=NIFTY
ADX_PERIOD=14
TREND_THRESHOLD=25
SIDEWAYS_THRESHOLD=20

# Logging settings
LOG_LEVEL=INFO
LOG_DIR=logs

# Data directory
DATA_DIR=data
"""
    
    try:
        with open('.env.template', 'w') as f:
            f.write(template_content)
        
        return True
    except Exception as e:
        print(f"Error creating .env template: {str(e)}")
        return False

# Create .env template if it doesn't exist
if not os.path.exists('.env.template'):
    create_env_template()