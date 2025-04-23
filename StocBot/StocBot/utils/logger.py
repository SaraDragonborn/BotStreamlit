"""
Logger Module
=======================================
Manages logging for the trading bot application.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import config

class TradeLogger:
    """
    Trade Logger for tracking trades.
    
    Logs trade information to file.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the Trade Logger.
        
        Parameters:
        -----------
        log_dir : str, optional
            Log directory
        """
        self.log_dir = log_dir or config.get('LOG_DIR', 'logs')
        self.trade_log_file = os.path.join(self.log_dir, 'trades.log')
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Log a trade.
        
        Parameters:
        -----------
        trade_data : dict
            Trade data
        """
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        try:
            # Write to log file
            with open(self.trade_log_file, 'a') as f:
                json.dump(trade_data, f)
                f.write('\n')
        except Exception as e:
            print(f"Error logging trade: {str(e)}")
    
    def get_trades(self, limit: Optional[int] = None) -> list:
        """
        Get trades from log file.
        
        Parameters:
        -----------
        limit : int, optional
            Limit number of trades to return
            
        Returns:
        --------
        list
            List of trades
        """
        trades = []
        
        try:
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                trade = json.loads(line.strip())
                                trades.append(trade)
                            except:
                                pass
        except Exception as e:
            print(f"Error getting trades: {str(e)}")
        
        # Sort trades by timestamp (newest first)
        trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Limit number of trades
        if limit:
            trades = trades[:limit]
        
        return trades

def setup_logger(name: str, level: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger.
    
    Parameters:
    -----------
    name : str
        Logger name
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Log file path
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    # Get log level from config if not provided
    if not level:
        level = config.get('LOG_LEVEL', 'INFO')
    
    # Convert level string to logging level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Create default log file in log directory
        log_dir = config.get('LOG_DIR', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Global trade logger instance
trade_logger = TradeLogger()