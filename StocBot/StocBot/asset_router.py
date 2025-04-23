"""
Asset Router
=======================================
Detects asset type and routes to appropriate data fetcher and trade executor.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import config
from logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class AssetRouter:
    """
    Asset Router detects asset type and routes to appropriate data fetcher and trade executor.
    
    Attributes:
    -----------
    data_fetchers : Dict
        Dictionary of data fetchers for different asset types
    trade_executors : Dict
        Dictionary of trade executors for different asset types
    """
    
    def __init__(self):
        """Initialize the Asset Router."""
        self.data_fetchers = {}
        self.trade_executors = {}
        
        # Initialize data fetchers and trade executors dictionary
        # These will be populated when they are registered
        logger.info("Asset Router initialized")
    
    def register_data_fetcher(self, asset_type: str, data_fetcher):
        """
        Register a data fetcher for a specific asset type.
        
        Parameters:
        -----------
        asset_type : str
            Asset type
        data_fetcher : object
            Data fetcher object
        """
        self.data_fetchers[asset_type] = data_fetcher
        logger.info(f"Registered data fetcher for {asset_type}")
    
    def register_trade_executor(self, asset_type: str, trade_executor):
        """
        Register a trade executor for a specific asset type.
        
        Parameters:
        -----------
        asset_type : str
            Asset type
        trade_executor : object
            Trade executor object
        """
        self.trade_executors[asset_type] = trade_executor
        logger.info(f"Registered trade executor for {asset_type}")
    
    def detect_asset_type(self, symbol: str) -> Optional[str]:
        """
        Detect asset type from symbol.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        str or None
            Asset type or None if not detected
        """
        # First, check if the asset is in the config
        asset_config = config.get_asset_config(symbol)
        if asset_config:
            return asset_config['type']
        
        # If not in config, try to detect from symbol pattern
        if self._is_indian_stock_pattern(symbol):
            return config.ASSET_TYPE_INDIAN_STOCK
        elif self._is_us_stock_pattern(symbol):
            return config.ASSET_TYPE_US_STOCK
        elif self._is_crypto_pattern(symbol):
            return config.ASSET_TYPE_CRYPTO
        elif self._is_forex_pattern(symbol):
            return config.ASSET_TYPE_FOREX
        elif self._is_commodity_pattern(symbol):
            return config.ASSET_TYPE_COMMODITY
        
        # Cannot determine asset type
        logger.warning(f"Unable to detect asset type for symbol: {symbol}")
        return None
    
    def detect_exchange(self, symbol: str) -> Optional[str]:
        """
        Detect exchange from symbol.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        str or None
            Exchange name or None if not detected
        """
        # Check if the asset is in the config
        asset_config = config.get_asset_config(symbol)
        if asset_config:
            return asset_config['exchange']
        
        # Default exchanges based on asset type
        asset_type = self.detect_asset_type(symbol)
        if asset_type == config.ASSET_TYPE_INDIAN_STOCK:
            return 'NSE'
        elif asset_type == config.ASSET_TYPE_US_STOCK:
            return 'NASDAQ'  # Default to NASDAQ for US stocks
        elif asset_type == config.ASSET_TYPE_CRYPTO:
            return 'BINANCE'  # Default to Binance for crypto
        elif asset_type == config.ASSET_TYPE_FOREX:
            return 'OANDA'
        elif asset_type == config.ASSET_TYPE_COMMODITY:
            return 'MCX'
        
        return None
    
    def get_data_fetcher(self, symbol: str):
        """
        Get appropriate data fetcher for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        object or None
            Data fetcher object or None if not available
        """
        asset_type = self.detect_asset_type(symbol)
        if asset_type and asset_type in self.data_fetchers:
            return self.data_fetchers[asset_type]
        
        logger.error(f"No data fetcher available for {symbol} (type: {asset_type})")
        return None
    
    def get_trade_executor(self, symbol: str):
        """
        Get appropriate trade executor for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        object or None
            Trade executor object or None if not available
        """
        asset_type = self.detect_asset_type(symbol)
        if asset_type and asset_type in self.trade_executors:
            return self.trade_executors[asset_type]
        
        logger.error(f"No trade executor available for {symbol} (type: {asset_type})")
        return None
    
    def fetch_data(self, symbol: str, timeframe: str = '1d', limit: int = 100) -> Optional[Dict]:
        """
        Fetch data for a symbol using the appropriate data fetcher.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
        timeframe : str, default='1d'
            Data timeframe
        limit : int, default=100
            Number of data points to fetch
            
        Returns:
        --------
        dict or None
            Fetched data or None if error
        """
        data_fetcher = self.get_data_fetcher(symbol)
        if data_fetcher:
            try:
                data = data_fetcher.fetch_data(symbol, timeframe, limit)
                logger.info(f"Fetched data for {symbol} ({timeframe}): {len(data)} records")
                return data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                return None
        
        return None
    
    def execute_trade(self, symbol: str, order_type: str, quantity: float, 
                     price: Optional[float] = None, stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None) -> Dict:
        """
        Execute a trade for a symbol using the appropriate trade executor.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
        order_type : str
            Order type ('BUY', 'SELL')
        quantity : float
            Quantity to trade
        price : float, optional
            Limit price (if None, use market price)
        stop_loss : float, optional
            Stop loss price
        take_profit : float, optional
            Take profit price
            
        Returns:
        --------
        dict
            Trade execution result
        """
        trade_executor = self.get_trade_executor(symbol)
        if trade_executor:
            try:
                result = trade_executor.execute_trade(
                    symbol=symbol,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                logger.info(f"Executed {order_type} trade for {symbol}: {quantity} @ {price}")
                return result
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {str(e)}")
                return {
                    'success': False,
                    'error': str(e),
                    'symbol': symbol,
                    'order_type': order_type
                }
        
        return {
            'success': False,
            'error': "No trade executor available",
            'symbol': symbol,
            'order_type': order_type
        }
    
    def _is_indian_stock_pattern(self, symbol: str) -> bool:
        """
        Check if symbol follows Indian stock pattern.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        bool
            True if symbol follows Indian stock pattern
        """
        # Indian stocks are typically uppercase without special characters
        # and are in the NSE/BSE list
        # For simplicity, we check if it's all caps and no special characters
        return symbol.isupper() and re.match(r'^[A-Z]+$', symbol) is not None
    
    def _is_us_stock_pattern(self, symbol: str) -> bool:
        """
        Check if symbol follows US stock pattern.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        bool
            True if symbol follows US stock pattern
        """
        # US stocks are typically 1-5 uppercase letters
        return re.match(r'^[A-Z]{1,5}$', symbol) is not None
    
    def _is_crypto_pattern(self, symbol: str) -> bool:
        """
        Check if symbol follows cryptocurrency pattern.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        bool
            True if symbol follows cryptocurrency pattern
        """
        # Crypto symbols typically have a format like BTC/USDT or BTC/INR
        return re.match(r'^[A-Z]+/[A-Z]+$', symbol) is not None
    
    def _is_forex_pattern(self, symbol: str) -> bool:
        """
        Check if symbol follows forex pattern.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        bool
            True if symbol follows forex pattern
        """
        # Forex pairs typically have a format like EUR/USD or USD/INR
        return re.match(r'^[A-Z]{3}/[A-Z]{3}$', symbol) is not None
    
    def _is_commodity_pattern(self, symbol: str) -> bool:
        """
        Check if symbol follows commodity pattern.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
            
        Returns:
        --------
        bool
            True if symbol follows commodity pattern
        """
        # Commodities are typically descriptive words like GOLD, SILVER, CRUDEOIL
        commodity_keywords = ['GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS', 'COPPER']
        return symbol in commodity_keywords

# Create a global instance of the asset router
asset_router = AssetRouter()