"""
Alpaca API Adapter
=======================================
Provides interface to Alpaca API for trading US stocks.
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import alpaca_trade_api as tradeapi

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlpacaAPI:
    """
    Alpaca API adapter for trading US stocks.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_secret: Optional[str] = None,
                paper_trading: Optional[bool] = None,
                base_url: Optional[str] = None):
        """
        Initialize the Alpaca API adapter.
        
        Parameters:
        -----------
        api_key : str, optional
            Alpaca API key (if None, uses environment variable)
        api_secret : str, optional
            Alpaca API secret (if None, uses environment variable)
        paper_trading : bool, optional
            Whether to use paper trading (if None, uses environment variable)
        base_url : str, optional
            Alpaca API base URL (if None, uses appropriate URL based on paper_trading)
        """
        # Use provided values or get from environment
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        
        # Parse paper trading setting
        if paper_trading is None:
            paper_str = os.environ.get('ALPACA_PAPER', 'true')
            self.paper_trading = paper_str.lower() == 'true'
        else:
            self.paper_trading = paper_trading
        
        # Validate required credentials
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API key and secret are required")
            raise ValueError("Alpaca API key and secret are required")
        
        # Set base URL
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
        
        # Initialize API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            api_version='v2'
        )
        
        logger.info(f"Alpaca API initialized with paper trading: {self.paper_trading}")
    
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
        --------
        Dict
            Account information
        """
        try:
            account = self.api.get_account()
            
            # Convert to dictionary
            account_info = {
                'id': account.id,
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'created_at': account.created_at.isoformat() if hasattr(account.created_at, 'isoformat') else account.created_at,
                'updated_at': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'account_info': account_info
            }
            
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_positions(self) -> Dict:
        """
        Get all open positions.
        
        Returns:
        --------
        Dict
            Open positions
        """
        try:
            positions = self.api.list_positions()
            
            # Convert to dictionary
            positions_data = []
            
            for position in positions:
                positions_data.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'current_price': float(position.current_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': position.side,
                    'exchange': position.exchange
                })
            
            return {
                'success': True,
                'positions': positions_data
            }
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_position(self, symbol: str) -> Dict:
        """
        Get position for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict
            Position information
        """
        try:
            position = self.api.get_position(symbol)
            
            # Convert to dictionary
            position_data = {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': position.side,
                'exchange': position.exchange
            }
            
            return {
                'success': True,
                'position': position_data
            }
            
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_orders(self, status: Optional[str] = None) -> Dict:
        """
        Get orders with optional status filter.
        
        Parameters:
        -----------
        status : str, optional
            Order status filter ('open', 'closed', 'all')
            
        Returns:
        --------
        Dict
            Orders
        """
        try:
            orders = self.api.list_orders(status=status)
            
            # Convert to dictionary
            orders_data = []
            
            for order in orders:
                orders_data.append({
                    'id': order.id,
                    'client_order_id': order.client_order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty) if hasattr(order, 'filled_qty') else 0,
                    'type': order.type,
                    'time_in_force': order.time_in_force,
                    'limit_price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                    'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                    'status': order.status,
                    'created_at': order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else order.created_at,
                    'updated_at': order.updated_at.isoformat() if hasattr(order.updated_at, 'isoformat') else order.updated_at,
                    'submitted_at': order.submitted_at.isoformat() if hasattr(order.submitted_at, 'isoformat') else order.submitted_at,
                    'filled_at': order.filled_at.isoformat() if hasattr(order.filled_at, 'isoformat') else order.filled_at
                })
            
            return {
                'success': True,
                'orders': orders_data
            }
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order(self, order_id: str) -> Dict:
        """
        Get a specific order by ID.
        
        Parameters:
        -----------
        order_id : str
            Order ID
            
        Returns:
        --------
        Dict
            Order information
        """
        try:
            order = self.api.get_order(order_id)
            
            # Convert to dictionary
            order_data = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'filled_qty': float(order.filled_qty) if hasattr(order, 'filled_qty') else 0,
                'type': order.type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                'status': order.status,
                'created_at': order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else order.created_at,
                'updated_at': order.updated_at.isoformat() if hasattr(order.updated_at, 'isoformat') else order.updated_at,
                'submitted_at': order.submitted_at.isoformat() if hasattr(order.submitted_at, 'isoformat') else order.submitted_at,
                'filled_at': order.filled_at.isoformat() if hasattr(order.filled_at, 'isoformat') else order.filled_at
            }
            
            return {
                'success': True,
                'order': order_data
            }
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def place_order(self, 
                  symbol: str, 
                  side: str, 
                  qty: float,
                  type: str = 'market',
                  time_in_force: str = 'day',
                  limit_price: Optional[float] = None,
                  stop_price: Optional[float] = None,
                  client_order_id: Optional[str] = None,
                  extended_hours: bool = False) -> Dict:
        """
        Place an order.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        side : str
            Order side ('buy' or 'sell')
        qty : float
            Number of shares
        type : str, default='market'
            Order type ('market', 'limit', 'stop', 'stop_limit')
        time_in_force : str, default='day'
            Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
        limit_price : float, optional
            Limit price for limit and stop-limit orders
        stop_price : float, optional
            Stop price for stop and stop-limit orders
        client_order_id : str, optional
            Client-specified order ID
        extended_hours : bool, default=False
            Whether to allow trading during extended hours
            
        Returns:
        --------
        Dict
            Order result
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                extended_hours=extended_hours
            )
            
            # Convert to dictionary
            order_data = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'type': order.type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                'status': order.status,
                'created_at': order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else order.created_at
            }
            
            return {
                'success': True,
                'order': order_data
            }
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def place_bracket_order(self, 
                          symbol: str, 
                          side: str, 
                          qty: float,
                          take_profit_price: float,
                          stop_loss_price: float,
                          limit_price: Optional[float] = None,
                          time_in_force: str = 'day',
                          client_order_id: Optional[str] = None) -> Dict:
        """
        Place a bracket order (entry + take profit + stop loss).
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        side : str
            Order side ('buy' or 'sell')
        qty : float
            Number of shares
        take_profit_price : float
            Take profit price
        stop_loss_price : float
            Stop loss price
        limit_price : float, optional
            Limit price for the entry order (if None, use market order)
        time_in_force : str, default='day'
            Time in force
        client_order_id : str, optional
            Client-specified order ID prefix
            
        Returns:
        --------
        Dict
            Order result
        """
        try:
            # Determine entry order type
            order_type = 'limit' if limit_price is not None else 'market'
            
            # Place bracket order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                client_order_id=client_order_id,
                order_class='bracket',
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_loss_price}
            )
            
            # Convert to dictionary
            order_data = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'type': order.type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                'status': order.status,
                'order_class': 'bracket',
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'created_at': order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else order.created_at
            }
            
            return {
                'success': True,
                'order': order_data
            }
            
        except Exception as e:
            logger.error(f"Error placing bracket order for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            Order ID
            
        Returns:
        --------
        Dict
            Cancellation result
        """
        try:
            self.api.cancel_order(order_id)
            
            return {
                'success': True,
                'message': f"Order {order_id} cancelled"
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_all_orders(self) -> Dict:
        """
        Cancel all open orders.
        
        Returns:
        --------
        Dict
            Cancellation result
        """
        try:
            self.api.cancel_all_orders()
            
            return {
                'success': True,
                'message': "All orders cancelled"
            }
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_bars(self, 
               symbol: str, 
               timeframe: str = '1Day', 
               start: Optional[str] = None,
               end: Optional[str] = None,
               limit: int = 100,
               adjustment: str = 'raw') -> pd.DataFrame:
        """
        Get historical price bars.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        timeframe : str, default='1Day'
            Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day', etc.)
        start : str, optional
            Start date/time (ISO 8601 format)
        end : str, optional
            End date/time (ISO 8601 format)
        limit : int, default=100
            Maximum number of bars to return
        adjustment : str, default='raw'
            Price adjustment ('raw', 'split', 'dividend', 'all')
            
        Returns:
        --------
        pandas.DataFrame
            Historical price bars
        """
        try:
            # Set default end to now if not provided
            if end is None:
                end = datetime.now().isoformat()
            
            # Set default start based on timeframe if not provided
            if start is None:
                if timeframe == '1Day':
                    # Default to 100 trading days
                    start = (datetime.now() - timedelta(days=140)).isoformat()
                elif timeframe == '1Hour':
                    # Default to 100 trading hours
                    start = (datetime.now() - timedelta(days=10)).isoformat()
                else:
                    # Default to 100 timeframe units (approximation)
                    start = (datetime.now() - timedelta(days=5)).isoformat()
            
            # Get bars
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=adjustment
            ).df
            
            # If the dataframe is empty, return empty DataFrame
            if bars.empty:
                return pd.DataFrame()
            
            # Rename columns to match expected format
            bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            
            # Reset index to make timestamp a column
            bars.reset_index(inplace=True)
            bars.rename(columns={'timestamp': 'datetime'}, inplace=True)
            
            return bars
            
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_last_quote(self, symbol: str) -> Dict:
        """
        Get the last quote for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict
            Last quote
        """
        try:
            quote = self.api.get_last_quote(symbol)
            
            quote_data = {
                'symbol': symbol,
                'ask_price': float(quote.ask_price),
                'ask_size': int(quote.ask_size),
                'bid_price': float(quote.bid_price),
                'bid_size': int(quote.bid_size),
                'timestamp': quote.timestamp.isoformat() if hasattr(quote.timestamp, 'isoformat') else quote.timestamp
            }
            
            return {
                'success': True,
                'quote': quote_data
            }
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_last_trade(self, symbol: str) -> Dict:
        """
        Get the last trade for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict
            Last trade
        """
        try:
            trade = self.api.get_last_trade(symbol)
            
            trade_data = {
                'symbol': symbol,
                'price': float(trade.price),
                'size': int(trade.size),
                'timestamp': trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else trade.timestamp
            }
            
            return {
                'success': True,
                'trade': trade_data
            }
            
        except Exception as e:
            logger.error(f"Error getting last trade for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_clock(self) -> Dict:
        """
        Get market clock information.
        
        Returns:
        --------
        Dict
            Market clock information
        """
        try:
            clock = self.api.get_clock()
            
            clock_data = {
                'timestamp': clock.timestamp.isoformat() if hasattr(clock.timestamp, 'isoformat') else clock.timestamp,
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat() if hasattr(clock.next_open, 'isoformat') else clock.next_open,
                'next_close': clock.next_close.isoformat() if hasattr(clock.next_close, 'isoformat') else clock.next_close
            }
            
            return {
                'success': True,
                'clock': clock_data
            }
            
        except Exception as e:
            logger.error(f"Error getting clock: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_calendar(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict:
        """
        Get market calendar.
        
        Parameters:
        -----------
        start : str, optional
            Start date (YYYY-MM-DD format)
        end : str, optional
            End date (YYYY-MM-DD format)
            
        Returns:
        --------
        Dict
            Market calendar
        """
        try:
            # Default to current month if not specified
            if start is None:
                start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            
            if end is None:
                # Last day of current month
                end_date = datetime.now().replace(day=28) + timedelta(days=4)
                end_date = end_date - timedelta(days=end_date.day)
                end = end_date.strftime('%Y-%m-%d')
            
            calendar = self.api.get_calendar(start=start, end=end)
            
            calendar_data = []
            for day in calendar:
                calendar_data.append({
                    'date': day.date.isoformat() if hasattr(day.date, 'isoformat') else day.date,
                    'open': day.open.isoformat() if hasattr(day.open, 'isoformat') else day.open,
                    'close': day.close.isoformat() if hasattr(day.close, 'isoformat') else day.close,
                    'session_open': day.session_open.isoformat() if hasattr(day.session_open, 'isoformat') else day.session_open,
                    'session_close': day.session_close.isoformat() if hasattr(day.session_close, 'isoformat') else day.session_close
                })
            
            return {
                'success': True,
                'calendar': calendar_data
            }
            
        except Exception as e:
            logger.error(f"Error getting calendar: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
        --------
        bool
            True if market is open, False otherwise
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking if market is open: {str(e)}")
            return False
    
    def get_last_price(self, symbol: str) -> Dict:
        """
        Get the latest price for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict
            Price information with structure:
            {
                'success': bool,
                'price': float,
                'timestamp': str
            }
        """
        try:
            # Get latest bar data
            bars = self.api.get_latest_bar(symbol)
            
            if bars:
                price = float(bars.c)  # close price
                timestamp = bars.t.isoformat() if hasattr(bars.t, 'isoformat') else str(bars.t)
                
                return {
                    'success': True,
                    'price': price,
                    'timestamp': timestamp
                }
            else:
                return {
                    'success': False,
                    'error': f"No price data available for {symbol}"
                }
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Create a global instance
alpaca_api = AlpacaAPI()