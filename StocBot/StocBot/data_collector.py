"""
Data Collector Module
=======================================
Collects and prepares data for trading strategies.
"""

import os
import datetime
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union, Any, Tuple
from api.alpaca_api import alpaca_api
from api.angel_api import angel_api
from utils.logger import get_trade_logger
from config import get_config

config = get_config()
logger = get_trade_logger()

class DataCollector:
    """
    Collects market data from various sources and prepares it for analysis.
    
    Supports:
    - US stocks via Alpaca
    - Indian stocks via Angel One
    - (Future) Crypto and forex markets
    """
    
    def __init__(self):
        """Initialize the Data Collector."""
        self.data_cache = {}  # Cache for recent data
        self.cache_expiry = {}  # Expiry timestamps for cached data
        
        # Cache duration in seconds for different timeframes
        self.cache_duration = {
            '1m': 60,          # 1 minute
            '5m': 5 * 60,      # 5 minutes
            '15m': 15 * 60,    # 15 minutes
            '30m': 30 * 60,    # 30 minutes
            '1h': 60 * 60,     # 1 hour
            '1d': 24 * 60 * 60  # 1 day
        }
        
        logger.info("Data Collector initialized")
    
    def get_stock_data(self, 
                      symbol: str, 
                      timeframe: str = '1d', 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: int = 100,
                      source: str = 'auto') -> pd.DataFrame:
        """
        Get stock price data.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        timeframe : str, default='1d'
            Data timeframe ('1m', '5m', '15m', '30m', '1h', '1d')
        start_date : str, optional
            Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date (format: 'YYYY-MM-DD')
        limit : int, default=100
            Number of bars to retrieve if no dates specified
        source : str, default='auto'
            Data source ('alpaca', 'angel', 'yfinance', 'auto')
            
        Returns:
        --------
        pandas.DataFrame
            Price data with columns: datetime, open, high, low, close, volume
        """
        # Set default end date to today if not provided
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Set default start date based on timeframe and limit if not provided
        if not start_date:
            if timeframe == '1d':
                days = limit * 1.5  # Add buffer for weekends/holidays
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            elif timeframe == '1h':
                days = limit / 24 * 1.5
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            else:
                days = limit / 100  # Approximate for intraday
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}_{source}"
        if cache_key in self.data_cache and datetime.datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
            logger.info(f"Using cached data for {symbol} ({timeframe})")
            return self.data_cache[cache_key]
        
        # Determine which market/source to use
        if source == 'auto':
            # Simple heuristic: use Angel for Indian stocks, Alpaca for US
            if '-EQ' in symbol:
                source = 'angel'
            else:
                source = 'alpaca'
        
        # Get data from appropriate source
        df = pd.DataFrame()
        
        try:
            if source == 'alpaca':
                # Convert timeframe to Alpaca format
                alpaca_timeframe_map = {
                    '1m': '1Min',
                    '5m': '5Min',
                    '15m': '15Min',
                    '30m': '30Min',
                    '1h': '1Hour',
                    '1d': '1Day'
                }
                alpaca_timeframe = alpaca_timeframe_map.get(timeframe, '1Day')
                
                df = alpaca_api.get_bars(
                    symbol=symbol,
                    timeframe=alpaca_timeframe,
                    start=start_date,
                    end=end_date,
                    limit=limit
                )
                
            elif source == 'angel':
                # Convert timeframe to Angel format
                angel_timeframe_map = {
                    '1m': 'ONE_MINUTE',
                    '5m': 'FIVE_MINUTE',
                    '15m': 'FIFTEEN_MINUTE',
                    '30m': 'THIRTY_MINUTE',
                    '1h': 'ONE_HOUR',
                    '1d': 'ONE_DAY'
                }
                angel_timeframe = angel_timeframe_map.get(timeframe, 'ONE_DAY')
                
                df = angel_api.get_historical_data(
                    symbol=symbol,
                    exchange='NSE',
                    interval=angel_timeframe,
                    from_date=f"{start_date} 00:00:00",
                    to_date=f"{end_date} 23:59:59"
                )
                
            elif source == 'yfinance':
                # Convert timeframe to yfinance format
                yf_timeframe_map = {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '1h',
                    '1d': '1d'
                }
                yf_timeframe = yf_timeframe_map.get(timeframe, '1d')
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period=None,
                    interval=yf_timeframe,
                    start=start_date,
                    end=end_date
                )
                
                # Reformat columns to match expected format
                df.reset_index(inplace=True)
                df.rename(columns={
                    'Date': 'datetime',
                    'Datetime': 'datetime',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
            
            else:
                logger.error(f"Invalid data source: {source}")
                return pd.DataFrame()
            
            # Check if data was successfully retrieved
            if df.empty:
                logger.warning(f"No data retrieved for {symbol} from {source}")
                return pd.DataFrame()
            
            # Ensure datetime column exists and is named correctly
            if 'datetime' not in df.columns and df.index.name == 'datetime':
                df.reset_index(inplace=True)
            elif 'datetime' not in df.columns:
                first_col = df.columns[0]
                if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                    df.rename(columns={first_col: 'datetime'}, inplace=True)
            
            # Store in cache with expiry
            self.data_cache[cache_key] = df
            
            # Set cache expiry based on timeframe
            expiry_seconds = self.cache_duration.get(timeframe, 300)  # Default 5 minutes
            self.cache_expiry[cache_key] = datetime.datetime.now().timestamp() + expiry_seconds
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} from {source}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_data_with_indicators(self,
                                     symbol: str,
                                     timeframe: str = '1d',
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     source: str = 'auto',
                                     indicators: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Get stock price data with technical indicators.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        timeframe : str, default='1d'
            Data timeframe ('1m', '5m', '15m', '30m', '1h', '1d')
        start_date : str, optional
            Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date (format: 'YYYY-MM-DD')
        source : str, default='auto'
            Data source ('alpaca', 'angel', 'yfinance', 'auto')
        indicators : List[Dict], optional
            List of indicators to add, each a dict with:
            - 'name': indicator name
            - 'params': parameters dict
            
        Returns:
        --------
        pandas.DataFrame
            Price data with additional indicator columns
        """
        # Get raw price data
        df = self.get_stock_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source
        )
        
        if df.empty:
            return df
        
        # Add requested indicators
        if indicators:
            for indicator in indicators:
                indicator_name = indicator.get('name', '')
                params = indicator.get('params', {})
                
                try:
                    # Add SMA
                    if indicator_name.lower() == 'sma':
                        period = params.get('period', 20)
                        price_col = params.get('price_column', 'close')
                        df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
                    
                    # Add EMA
                    elif indicator_name.lower() == 'ema':
                        period = params.get('period', 20)
                        price_col = params.get('price_column', 'close')
                        df[f'ema_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
                    
                    # Add RSI
                    elif indicator_name.lower() == 'rsi':
                        period = params.get('period', 14)
                        price_col = params.get('price_column', 'close')
                        
                        delta = df[price_col].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        
                        avg_gain = gain.rolling(window=period).mean()
                        avg_loss = loss.rolling(window=period).mean()
                        
                        rs = avg_gain / avg_loss
                        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                    
                    # Add Bollinger Bands
                    elif indicator_name.lower() in ('bollinger', 'bollinger_bands', 'bb'):
                        period = params.get('period', 20)
                        std_dev = params.get('std_dev', 2)
                        price_col = params.get('price_column', 'close')
                        
                        df[f'bb_middle_{period}'] = df[price_col].rolling(window=period).mean()
                        df[f'bb_std_{period}'] = df[price_col].rolling(window=period).std()
                        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (df[f'bb_std_{period}'] * std_dev)
                        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (df[f'bb_std_{period}'] * std_dev)
                    
                    # Add MACD
                    elif indicator_name.lower() == 'macd':
                        fast_period = params.get('fast_period', 12)
                        slow_period = params.get('slow_period', 26)
                        signal_period = params.get('signal_period', 9)
                        price_col = params.get('price_column', 'close')
                        
                        fast_ema = df[price_col].ewm(span=fast_period, adjust=False).mean()
                        slow_ema = df[price_col].ewm(span=slow_period, adjust=False).mean()
                        df[f'macd_line'] = fast_ema - slow_ema
                        df[f'macd_signal'] = df[f'macd_line'].ewm(span=signal_period, adjust=False).mean()
                        df[f'macd_histogram'] = df[f'macd_line'] - df[f'macd_signal']
                    
                    # Add ATR (Average True Range)
                    elif indicator_name.lower() == 'atr':
                        period = params.get('period', 14)
                        
                        high = df['high']
                        low = df['low']
                        close = df['close'].shift(1)
                        
                        tr1 = high - low
                        tr2 = abs(high - close)
                        tr3 = abs(low - close)
                        
                        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                        df[f'atr_{period}'] = tr.rolling(window=period).mean()
                    
                    # Add Stochastic Oscillator
                    elif indicator_name.lower() in ('stochastic', 'stoch'):
                        k_period = params.get('k_period', 14)
                        d_period = params.get('d_period', 3)
                        
                        low_min = df['low'].rolling(window=k_period).min()
                        high_max = df['high'].rolling(window=k_period).max()
                        
                        df[f'stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
                        df[f'stoch_d'] = df[f'stoch_k'].rolling(window=d_period).mean()
                        
                except Exception as e:
                    logger.error(f"Error calculating indicator {indicator_name}: {str(e)}")
        
        return df
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str],
                               timeframe: str = '1d',
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               source: str = 'auto') -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple stocks.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        timeframe : str, default='1d'
            Data timeframe ('1m', '5m', '15m', '30m', '1h', '1d')
        start_date : str, optional
            Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date (format: 'YYYY-MM-DD')
        source : str, default='auto'
            Data source ('alpaca', 'angel', 'yfinance', 'auto')
            
        Returns:
        --------
        Dict[str, pandas.DataFrame]
            Dictionary of dataframes with symbols as keys
        """
        result = {}
        
        for symbol in symbols:
            df = self.get_stock_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                source=source
            )
            
            if not df.empty:
                result[symbol] = df
        
        logger.info(f"Retrieved data for {len(result)}/{len(symbols)} symbols")
        return result
    
    def calculate_correlation_matrix(self, 
                                   symbols: List[str],
                                   timeframe: str = '1d',
                                   days: int = 30,
                                   source: str = 'auto') -> pd.DataFrame:
        """
        Calculate correlation matrix between multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        timeframe : str, default='1d'
            Data timeframe ('1d' recommended)
        days : int, default=30
            Number of days to use for correlation
        source : str, default='auto'
            Data source ('alpaca', 'angel', 'yfinance', 'auto')
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        # Set date range
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Get data for all symbols
        data = self.get_multiple_stocks_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source
        )
        
        if not data:
            logger.warning("No data available for correlation calculation")
            return pd.DataFrame()
        
        # Extract close prices and calculate returns
        returns_data = {}
        
        for symbol, df in data.items():
            if not df.empty and len(df) > 1:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        if not returns_data:
            logger.warning("No return data available for correlation calculation")
            return pd.DataFrame()
        
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    def get_sector_data(self, symbols: List[str], source: str = 'yfinance') -> Dict[str, str]:
        """
        Get sector information for a list of symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        source : str, default='yfinance'
            Data source ('yfinance' supported currently)
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping symbols to sectors
        """
        sector_data = {}
        
        if source == 'yfinance':
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    sector = info.get('sector', 'Unknown')
                    industry = info.get('industry', 'Unknown')
                    
                    sector_data[symbol] = f"{sector}"
                    logger.debug(f"Sector for {symbol}: {sector} ({industry})")
                    
                except Exception as e:
                    logger.error(f"Error getting sector data for {symbol}: {str(e)}")
                    sector_data[symbol] = "Unknown"
        else:
            logger.warning(f"Sector data source {source} not supported")
            
        return sector_data
    
    def get_market_data(self, market: str = 'US') -> Dict:
        """
        Get overall market data.
        
        Parameters:
        -----------
        market : str, default='US'
            Market to get data for ('US', 'India')
            
        Returns:
        --------
        Dict
            Market data and status
        """
        market_data = {}
        
        try:
            if market == 'US':
                # Get S&P 500 data
                spy_data = self.get_stock_data(
                    symbol='SPY',
                    timeframe='1d',
                    limit=10,
                    source='yfinance'
                )
                
                if not spy_data.empty:
                    # Calculate daily change
                    latest = spy_data.iloc[-1]
                    prev = spy_data.iloc[-2]
                    daily_change = (latest['close'] / prev['close'] - 1) * 100
                    
                    market_data['index'] = 'S&P 500 ETF'
                    market_data['price'] = latest['close']
                    market_data['change_percent'] = daily_change
                    market_data['direction'] = 'up' if daily_change > 0 else 'down'
                
                # Check if market is open
                if config['alpaca']['api_key']:
                    clock = alpaca_api.get_clock()
                    market_data['is_open'] = clock.get('success', False) and clock.get('clock', {}).get('is_open', False)
            
            elif market == 'India':
                # Get Nifty 50 data (using a proxy)
                nifty_data = self.get_stock_data(
                    symbol='^NSEI',  # Nifty 50 index
                    timeframe='1d',
                    limit=10,
                    source='yfinance'
                )
                
                if not nifty_data.empty:
                    # Calculate daily change
                    latest = nifty_data.iloc[-1]
                    prev = nifty_data.iloc[-2]
                    daily_change = (latest['close'] / prev['close'] - 1) * 100
                    
                    market_data['index'] = 'NIFTY 50'
                    market_data['price'] = latest['close']
                    market_data['change_percent'] = daily_change
                    market_data['direction'] = 'up' if daily_change > 0 else 'down'
                
                # We don't have direct access to check if Indian market is open
                # Using a simple time-based check (9:15 AM - 3:30 PM IST on weekdays)
                now = datetime.datetime.now()
                india_time = now + datetime.timedelta(hours=5, minutes=30)  # Crude IST conversion
                weekday = india_time.weekday()
                hour = india_time.hour
                minute = india_time.minute
                
                is_weekday = weekday < 5  # Monday to Friday
                is_trading_hours = (
                    (hour > 9 or (hour == 9 and minute >= 15)) and  # After 9:15 AM
                    (hour < 15 or (hour == 15 and minute <= 30))     # Before 3:30 PM
                )
                
                market_data['is_open'] = is_weekday and is_trading_hours
            
            else:
                logger.warning(f"Market data for {market} not supported")
            
        except Exception as e:
            logger.error(f"Error getting market data for {market}: {str(e)}")
        
        return market_data

# Create global instance
data_collector = DataCollector()