"""
Logger Module
=======================================
Custom logger for tracking bot operations, trades, and errors.
"""

import os
import logging
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import config

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with the specified name.
    
    Parameters:
    -----------
    name : str
        Logger name
    log_file : str, optional
        Log file name (if None, uses default from config)
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        # Set log level from config
        log_level_str = config.LOG_CONFIG.get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str)
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=config.LOG_CONFIG.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt=config.LOG_CONFIG.get('date_format', '%Y-%m-%d %H:%M:%S')
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if log file is specified
        if log_file is None:
            log_file = config.LOG_CONFIG.get('log_file', 'trading_bot.log')
        
        if log_file:
            # Ensure log file path is absolute
            if not os.path.isabs(log_file):
                log_file = os.path.join('logs', log_file)
            
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.LOG_CONFIG.get('max_log_file_size', 10 * 1024 * 1024),  # Default: 10 MB
                backupCount=config.LOG_CONFIG.get('backup_count', 5)  # Default: 5 backups
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# Global logger
main_logger = setup_logger('trading_bot')

class TradeLogger:
    """
    Trade Logger for tracking trade operations.
    
    Attributes:
    -----------
    logger : logging.Logger
        Logger instance
    trade_history_file : str
        File to store trade history
    trade_history : List[Dict]
        List of trade records
    """
    
    def __init__(self, trade_history_file: Optional[str] = None):
        """
        Initialize the Trade Logger.
        
        Parameters:
        -----------
        trade_history_file : str, optional
            File to store trade history (if None, uses default from config)
        """
        self.logger = setup_logger('trade_logger', 'trades.log')
        
        # Set trade history file
        if trade_history_file is None:
            data_dir = config.DATA_STORAGE.get('data_dir', 'data')
            self.trade_history_file = os.path.join(data_dir, 'trade_history.json')
        else:
            self.trade_history_file = trade_history_file
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.trade_history_file), exist_ok=True)
        
        # Load trade history if file exists
        self.trade_history = []
        self._load_trade_history()
        
        self.logger.info(f"Trade Logger initialized with history file: {self.trade_history_file}")
    
    def _load_trade_history(self):
        """Load trade history from file."""
        if os.path.exists(self.trade_history_file):
            try:
                with open(self.trade_history_file, 'r') as f:
                    self.trade_history = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading trade history: {str(e)}")
    
    def _save_trade_history(self):
        """Save trade history to file."""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trade history: {str(e)}")
    
    def log_trade(self, trade_data: Dict) -> str:
        """
        Log a trade.
        
        Parameters:
        -----------
        trade_data : Dict
            Trade data
            
        Returns:
        --------
        str
            Trade ID
        """
        # Generate a trade ID if not provided
        if 'trade_id' not in trade_data:
            trade_data['trade_id'] = f"trade_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.trade_history) + 1}"
        
        # Add timestamp if not provided
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Log trade
        trade_type = trade_data.get('order_type', 'UNKNOWN')
        symbol = trade_data.get('symbol', 'UNKNOWN')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        
        self.logger.info(f"TRADE: {trade_type} {quantity} {symbol} @ {price}")
        
        # Add to trade history
        self.trade_history.append(trade_data)
        self._save_trade_history()
        
        return trade_data['trade_id']
    
    def log_trade_update(self, trade_id: str, update_data: Dict) -> bool:
        """
        Update a trade record.
        
        Parameters:
        -----------
        trade_id : str
            Trade ID
        update_data : Dict
            Updated trade data
            
        Returns:
        --------
        bool
            True if trade was updated, False otherwise
        """
        # Find the trade by ID
        for i, trade in enumerate(self.trade_history):
            if trade.get('trade_id') == trade_id:
                # Log the update
                symbol = trade.get('symbol', 'UNKNOWN')
                self.logger.info(f"TRADE UPDATE: {symbol} (ID: {trade_id})")
                
                # Update trade data
                self.trade_history[i].update(update_data)
                
                # Add update timestamp
                if 'updated_at' not in update_data:
                    self.trade_history[i]['updated_at'] = datetime.now().isoformat()
                
                # Save trade history
                self._save_trade_history()
                
                return True
        
        self.logger.warning(f"Trade not found for update: {trade_id}")
        return False
    
    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """
        Get a trade by ID.
        
        Parameters:
        -----------
        trade_id : str
            Trade ID
            
        Returns:
        --------
        Dict or None
            Trade data or None if not found
        """
        for trade in self.trade_history:
            if trade.get('trade_id') == trade_id:
                return trade
        
        return None
    
    def get_trades(self, 
                  symbol: Optional[str] = None, 
                  asset_type: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> List[Dict]:
        """
        Get trades with optional filters.
        
        Parameters:
        -----------
        symbol : str, optional
            Filter by symbol
        asset_type : str, optional
            Filter by asset type
        start_date : str, optional
            Filter by start date (ISO format)
        end_date : str, optional
            Filter by end date (ISO format)
            
        Returns:
        --------
        List[Dict]
            Filtered trade list
        """
        filtered_trades = self.trade_history.copy()
        
        # Apply symbol filter
        if symbol:
            filtered_trades = [trade for trade in filtered_trades if trade.get('symbol') == symbol]
        
        # Apply asset type filter
        if asset_type:
            filtered_trades = [trade for trade in filtered_trades if trade.get('asset_type') == asset_type]
        
        # Apply date filters
        if start_date:
            filtered_trades = [trade for trade in filtered_trades 
                              if 'timestamp' in trade and trade['timestamp'] >= start_date]
        
        if end_date:
            filtered_trades = [trade for trade in filtered_trades 
                              if 'timestamp' in trade and trade['timestamp'] <= end_date]
        
        return filtered_trades

class PerformanceLogger:
    """
    Performance Logger for tracking bot performance.
    
    Attributes:
    -----------
    logger : logging.Logger
        Logger instance
    performance_file : str
        File to store performance data
    performance_data : Dict
        Performance data
    """
    
    def __init__(self, performance_file: Optional[str] = None):
        """
        Initialize the Performance Logger.
        
        Parameters:
        -----------
        performance_file : str, optional
            File to store performance data (if None, uses default from config)
        """
        self.logger = setup_logger('performance_logger', 'performance.log')
        
        # Set performance file
        if performance_file is None:
            data_dir = config.DATA_STORAGE.get('data_dir', 'data')
            self.performance_file = os.path.join(data_dir, 'performance.json')
        else:
            self.performance_file = performance_file
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
        
        # Initialize performance data
        self.performance_data = {
            'portfolio_history': [],
            'asset_performance': {},
            'strategy_performance': {},
            'daily_returns': {},
            'monthly_returns': {},
            'overall_metrics': {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_loss': 0,
                'net_profit': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Load performance data if file exists
        self._load_performance_data()
        
        self.logger.info(f"Performance Logger initialized with file: {self.performance_file}")
    
    def _load_performance_data(self):
        """Load performance data from file."""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    loaded_data = json.load(f)
                    # Update performance data with loaded data
                    self.performance_data.update(loaded_data)
            except Exception as e:
                self.logger.error(f"Error loading performance data: {str(e)}")
    
    def _save_performance_data(self):
        """Save performance data to file."""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")
    
    def update_portfolio_value(self, timestamp: str, portfolio_value: float, 
                              cash: float, positions_value: float) -> None:
        """
        Update portfolio value.
        
        Parameters:
        -----------
        timestamp : str
            Timestamp (ISO format)
        portfolio_value : float
            Total portfolio value
        cash : float
            Cash value
        positions_value : float
            Value of positions
        """
        # Add portfolio value to history
        portfolio_entry = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions_value': positions_value
        }
        
        self.performance_data['portfolio_history'].append(portfolio_entry)
        
        # Extract date (YYYY-MM-DD) from timestamp
        date = timestamp.split('T')[0] if 'T' in timestamp else timestamp
        
        # Extract month (YYYY-MM) from date
        month = date[:7]
        
        # Update daily and monthly returns
        if len(self.performance_data['portfolio_history']) > 1:
            # Get previous portfolio value
            prev_entry = self.performance_data['portfolio_history'][-2]
            prev_value = prev_entry['portfolio_value']
            
            # Calculate daily return
            daily_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
            
            # Update daily returns
            if date not in self.performance_data['daily_returns']:
                self.performance_data['daily_returns'][date] = daily_return
            
            # Update monthly returns
            if month not in self.performance_data['monthly_returns']:
                # Find the last day of the previous month
                prev_month_entries = [entry for entry in self.performance_data['portfolio_history'][:-1] 
                                    if entry['timestamp'].startswith(month)]
                
                if prev_month_entries:
                    first_entry = prev_month_entries[0]
                    first_value = first_entry['portfolio_value']
                    
                    # Calculate monthly return
                    monthly_return = (portfolio_value / first_value) - 1 if first_value > 0 else 0
                    
                    # Update monthly returns
                    self.performance_data['monthly_returns'][month] = monthly_return
        
        # Log and save
        self.logger.info(f"Updated portfolio value: {portfolio_value}")
        self._save_performance_data()
    
    def update_asset_performance(self, asset_type: str, symbol: str, 
                               profit_loss: float, trade_count: int) -> None:
        """
        Update asset performance.
        
        Parameters:
        -----------
        asset_type : str
            Asset type
        symbol : str
            Asset symbol
        profit_loss : float
            Profit/loss amount
        trade_count : int
            Number of trades
        """
        # Initialize asset performance if not exists
        if asset_type not in self.performance_data['asset_performance']:
            self.performance_data['asset_performance'][asset_type] = {}
        
        if symbol not in self.performance_data['asset_performance'][asset_type]:
            self.performance_data['asset_performance'][asset_type][symbol] = {
                'total_profit_loss': 0,
                'trade_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update asset performance
        asset_perf = self.performance_data['asset_performance'][asset_type][symbol]
        asset_perf['total_profit_loss'] += profit_loss
        asset_perf['trade_count'] += trade_count
        
        if profit_loss > 0:
            asset_perf['win_count'] += 1
        elif profit_loss < 0:
            asset_perf['loss_count'] += 1
        
        asset_perf['last_updated'] = datetime.now().isoformat()
        
        # Log and save
        self.logger.info(f"Updated performance for {symbol} ({asset_type}): {profit_loss}")
        self._save_performance_data()
    
    def update_strategy_performance(self, strategy_name: str, asset_type: str, 
                                  profit_loss: float, trade_count: int) -> None:
        """
        Update strategy performance.
        
        Parameters:
        -----------
        strategy_name : str
            Strategy name
        asset_type : str
            Asset type
        profit_loss : float
            Profit/loss amount
        trade_count : int
            Number of trades
        """
        # Initialize strategy performance if not exists
        if strategy_name not in self.performance_data['strategy_performance']:
            self.performance_data['strategy_performance'][strategy_name] = {}
        
        if asset_type not in self.performance_data['strategy_performance'][strategy_name]:
            self.performance_data['strategy_performance'][strategy_name][asset_type] = {
                'total_profit_loss': 0,
                'trade_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update strategy performance
        strategy_perf = self.performance_data['strategy_performance'][strategy_name][asset_type]
        strategy_perf['total_profit_loss'] += profit_loss
        strategy_perf['trade_count'] += trade_count
        
        if profit_loss > 0:
            strategy_perf['win_count'] += 1
        elif profit_loss < 0:
            strategy_perf['loss_count'] += 1
        
        strategy_perf['last_updated'] = datetime.now().isoformat()
        
        # Log and save
        self.logger.info(f"Updated performance for {strategy_name} ({asset_type}): {profit_loss}")
        self._save_performance_data()
    
    def update_overall_metrics(self, trade_data: Optional[Dict] = None) -> None:
        """
        Update overall performance metrics.
        
        Parameters:
        -----------
        trade_data : Dict, optional
            New trade data to incorporate in metrics
        """
        metrics = self.performance_data['overall_metrics']
        
        # Update metrics with new trade if provided
        if trade_data:
            profit_loss = trade_data.get('profit_loss', 0)
            
            metrics['total_trades'] += 1
            
            if profit_loss > 0:
                metrics['profitable_trades'] += 1
                metrics['total_profit'] += profit_loss
            else:
                metrics['losing_trades'] += 1
                metrics['total_loss'] += abs(profit_loss)
            
            metrics['net_profit'] = metrics['total_profit'] - metrics['total_loss']
        
        # Calculate win rate
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
        
        # Calculate profit factor
        if metrics['total_loss'] > 0:
            metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss']
        
        # Calculate maximum drawdown
        if self.performance_data['portfolio_history']:
            values = [entry['portfolio_value'] for entry in self.performance_data['portfolio_history']]
            peak = values[0]
            max_drawdown = 0
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            metrics['max_drawdown'] = max_drawdown
        
        # Calculate Sharpe ratio (simplified)
        if len(self.performance_data['daily_returns']) > 0:
            returns = list(self.performance_data['daily_returns'].values())
            avg_return = sum(returns) / len(returns)
            std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            
            if std_dev > 0:
                metrics['sharpe_ratio'] = (avg_return / std_dev) * (252 ** 0.5)  # Annualized
        
        # Update timestamp
        metrics['last_updated'] = datetime.now().isoformat()
        
        # Log and save
        self.logger.info(f"Updated overall metrics: win rate = {metrics['win_rate']:.2f}, net profit = {metrics['net_profit']}")
        self._save_performance_data()
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary.
        
        Returns:
        --------
        Dict
            Performance summary
        """
        return {
            'overall_metrics': self.performance_data['overall_metrics'],
            'portfolio_value': self.performance_data['portfolio_history'][-1]['portfolio_value'] if self.performance_data['portfolio_history'] else 0,
            'top_assets': self._get_top_assets(5),
            'top_strategies': self._get_top_strategies(3),
            'monthly_returns': self.performance_data['monthly_returns'],
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_top_assets(self, limit: int = 5) -> List[Dict]:
        """
        Get top performing assets.
        
        Parameters:
        -----------
        limit : int, default=5
            Number of top assets to return
            
        Returns:
        --------
        List[Dict]
            Top assets by profit/loss
        """
        all_assets = []
        
        for asset_type, assets in self.performance_data['asset_performance'].items():
            for symbol, data in assets.items():
                all_assets.append({
                    'symbol': symbol,
                    'asset_type': asset_type,
                    'profit_loss': data['total_profit_loss'],
                    'trade_count': data['trade_count']
                })
        
        # Sort by profit/loss (descending)
        all_assets.sort(key=lambda x: x['profit_loss'], reverse=True)
        
        return all_assets[:limit]
    
    def _get_top_strategies(self, limit: int = 3) -> List[Dict]:
        """
        Get top performing strategies.
        
        Parameters:
        -----------
        limit : int, default=3
            Number of top strategies to return
            
        Returns:
        --------
        List[Dict]
            Top strategies by profit/loss
        """
        all_strategies = []
        
        for strategy_name, asset_types in self.performance_data['strategy_performance'].items():
            # Calculate total profit/loss across all asset types
            total_profit_loss = sum(data['total_profit_loss'] for asset_type, data in asset_types.items())
            total_trade_count = sum(data['trade_count'] for asset_type, data in asset_types.items())
            
            all_strategies.append({
                'strategy': strategy_name,
                'profit_loss': total_profit_loss,
                'trade_count': total_trade_count
            })
        
        # Sort by profit/loss (descending)
        all_strategies.sort(key=lambda x: x['profit_loss'], reverse=True)
        
        return all_strategies[:limit]


# Create instances of loggers
trade_logger = TradeLogger()
performance_logger = PerformanceLogger()