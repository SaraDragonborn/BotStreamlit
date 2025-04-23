"""
Logger Module
=======================================
Custom logging setup for the trading bot.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import datetime
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: str = None) -> logging.Logger:
    """
    Set up a logger with the specified name and file.
    
    Parameters:
    -----------
    name : str
        Logger name
    log_file : str, optional
        Log file path (if None, logs to console only)
    level : str, optional
        Log level (if None, uses level from environment variable)
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Get log level from environment or use default
    log_level_str = level or os.environ.get('LOG_LEVEL', 'INFO')
    numeric_level = getattr(logging, log_level_str.upper(), None)
    
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_trade_logger():
    """
    Get a logger for trade-related logs.
    
    Returns:
    --------
    logging.Logger
        Trade logger
    """
    return setup_logger('trade_logger', 'logs/trades.log')

def get_strategy_logger():
    """
    Get a logger for strategy-related logs.
    
    Returns:
    --------
    logging.Logger
        Strategy logger
    """
    return setup_logger('strategy_logger', 'logs/strategies.log')

def get_performance_logger():
    """
    Get a logger for performance-related logs.
    
    Returns:
    --------
    logging.Logger
        Performance logger
    """
    return setup_logger('performance_logger', 'logs/performance.log')

# Create global logger instances
main_logger = setup_logger('main', 'logs/main.log')
trade_logger = get_trade_logger()
strategy_logger = get_strategy_logger()
performance_logger = get_performance_logger()
api_logger = setup_logger('api', 'logs/api.log')