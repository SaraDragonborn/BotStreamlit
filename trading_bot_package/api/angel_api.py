"""
Angel One API Adapter
=======================================
Provides interface to Angel One API for trading Indian stocks.
"""

import os
import logging
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AngelOneAPI:
    """
    Angel One API adapter for trading Indian stocks.
    """
    
    # API Endpoints
    BASE_URL = "https://apiconnect.angelbroking.com"
    LOGIN_URL = f"{BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword"
    TOKEN_URL = f"{BASE_URL}/rest/auth/angelbroking/jwt/v1/generateTokens"
    PROFILE_URL = f"{BASE_URL}/rest/secure/angelbroking/user/v1/getProfile"
    FUNDS_URL = f"{BASE_URL}/rest/secure/angelbroking/user/v1/getRMS"
    PLACE_ORDER_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/placeOrder"
    MODIFY_ORDER_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/modifyOrder"
    CANCEL_ORDER_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/cancelOrder"
    ORDER_BOOK_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/getOrderBook"
    TRADE_BOOK_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/getTradeBook"
    POSITION_BOOK_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/getPosition"
    HOLDING_URL = f"{BASE_URL}/rest/secure/angelbroking/portfolio/v1/getHolding"
    HISTORICAL_URL = f"{BASE_URL}/rest/secure/angelbroking/historical/v1/getCandleData"
    EXCHANGE_INFO_URL = f"{BASE_URL}/rest/secure/angelbroking/market/v1/exchangeInfo"
    LTP_DATA_URL = f"{BASE_URL}/rest/secure/angelbroking/market/v1/getLTP"
    MARKET_DATA_URL = f"{BASE_URL}/rest/secure/angelbroking/market/v1/marketData"

    def __init__(self, 
                api_key: Optional[str] = None, 
                client_id: Optional[str] = None,
                password: Optional[str] = None,
                totp_key: Optional[str] = None,
                api_secret: Optional[str] = None):
        """
        Initialize the Angel One API adapter.
        
        Parameters:
        -----------
        api_key : str, optional
            Angel One API key (if None, uses environment variable)
        client_id : str, optional
            Angel One client ID (if None, uses environment variable)
        password : str, optional
            Angel One password (if None, uses environment variable)
        totp_key : str, optional
            Angel One TOTP key for two-factor authentication (if None, uses environment variable)
        api_secret : str, optional
            Angel One API secret (if None, uses environment variable)
        """
        # Use provided values or get from environment
        self.api_key = api_key or os.environ.get('ANGEL_API_KEY')
        self.client_id = client_id or os.environ.get('ANGEL_CLIENT_ID')
        self.password = password or os.environ.get('ANGEL_PASSWORD')
        self.totp_key = totp_key or os.environ.get('ANGEL_TOTP_KEY')
        self.api_secret = api_secret or os.environ.get('ANGEL_API_SECRET')
        
        # Validate required credentials
        if not self.api_key or not self.client_id or not self.password:
            logger.error("Angel One API key, client ID, and password are required")
            raise ValueError("Angel One API key, client ID, and password are required")
        
        # Auth tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        # Session for API calls
        self.session = requests.Session()
        
        # Default headers
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': '127.0.0.1',
            'X-MACAddress': '00:00:00:00:00:00',
            'X-PrivateKey': self.api_key
        }
        
        # Try to authenticate if all credentials are provided
        if self.totp_key:
            try:
                self.login()
            except Exception as e:
                logger.error(f"Error during initial authentication: {str(e)}")
        else:
            logger.warning("TOTP key not provided, authentication will be required before making API calls")
    
    def login(self) -> bool:
        """
        Login to Angel One and generate access token.
        
        Returns:
        --------
        bool
            True if login successful, False otherwise
        """
        try:
            # For now, we'll simulate the login process
            # In production, this would use the actual Angel One API
            
            logger.info("Simulating Angel One API login (this is a placeholder)")
            
            # Simulate successful login
            self.access_token = "simulated_access_token"
            self.refresh_token = "simulated_refresh_token"
            self.token_expiry = datetime.now() + timedelta(hours=23)
            
            # Update headers with token
            self.headers['Authorization'] = f"Bearer {self.access_token}"
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging in: {str(e)}")
            return False
    
    def _check_token(self) -> bool:
        """
        Check if access token is valid and refresh if necessary.
        
        Returns:
        --------
        bool
            True if token is valid, False otherwise
        """
        # If no token or expired, login again
        if not self.access_token or (self.token_expiry and datetime.now() > self.token_expiry):
            return self.login()
        
        return True
    
    def _make_api_call(self, 
                     method: str, 
                     endpoint: str, 
                     payload: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Dict:
        """
        Make an API call to Angel One.
        
        Parameters:
        -----------
        method : str
            HTTP method ('GET', 'POST', 'PUT', 'DELETE')
        endpoint : str
            API endpoint URL
        payload : dict, optional
            Request payload
        params : dict, optional
            Query parameters
            
        Returns:
        --------
        dict
            API response
        """
        # Check token
        if not self._check_token():
            raise Exception("Failed to authenticate")
        
        # This is a simulation function
        logger.info(f"Simulating API call: {method} {endpoint}")
        
        # Simulate successful response
        return {'status': True, 'message': 'Success', 'data': self._get_simulated_data(endpoint, payload, params)}
    
    def _get_simulated_data(self, endpoint: str, payload: Optional[Dict], params: Optional[Dict]) -> Any:
        """
        Get simulated data for endpoints.
        
        Parameters:
        -----------
        endpoint : str
            API endpoint URL
        payload : dict, optional
            Request payload
        params : dict, optional
            Query parameters
            
        Returns:
        --------
        Any
            Simulated data
        """
        # Simulate different endpoint responses
        if endpoint == self.PROFILE_URL:
            return {
                'clientcode': self.client_id,
                'name': 'Simulated User',
                'email': 'user@example.com',
                'mobileno': '9999999999',
                'exchanges': ['NSE', 'BSE', 'MCX'],
                'products': ['DELIVERY', 'INTRADAY', 'MARGIN'],
                'lastlogintime': datetime.now().isoformat()
            }
        
        elif endpoint == self.FUNDS_URL:
            return {
                'availablecash': 100000.0,
                'utilisedmargin': 25000.0,
                'net': 75000.0,
                'availablecarrymargin': 50000.0
            }
        
        elif endpoint == self.PLACE_ORDER_URL:
            return {
                'orderid': f"simulated_order_{int(time.time())}"
            }
        
        elif endpoint == self.CANCEL_ORDER_URL:
            return {
                'orderid': payload.get('orderid') if payload else 'unknown'
            }
        
        elif endpoint == self.ORDER_BOOK_URL:
            return [
                {
                    'orderid': f"simulated_order_1",
                    'tradingsymbol': 'RELIANCE-EQ',
                    'symboltoken': '123456',
                    'exchange': 'NSE',
                    'transactiontype': 'BUY',
                    'producttype': 'DELIVERY',
                    'ordertype': 'MARKET',
                    'quantity': 10,
                    'status': 'COMPLETE',
                    'orderdisplaystatustr': 'Completed',
                    'updatetime': datetime.now().isoformat(),
                    'averageprice': 2500.0,
                    'filledqty': 10
                },
                {
                    'orderid': f"simulated_order_2",
                    'tradingsymbol': 'TATAMOTORS-EQ',
                    'symboltoken': '789012',
                    'exchange': 'NSE',
                    'transactiontype': 'SELL',
                    'producttype': 'DELIVERY',
                    'ordertype': 'LIMIT',
                    'quantity': 5,
                    'status': 'PENDING',
                    'orderdisplaystatustr': 'Pending',
                    'updatetime': datetime.now().isoformat(),
                    'averageprice': 0.0,
                    'filledqty': 0
                }
            ]
        
        elif endpoint == self.POSITION_BOOK_URL:
            return [
                {
                    'tradingsymbol': 'RELIANCE-EQ',
                    'symboltoken': '123456',
                    'exchange': 'NSE',
                    'producttype': 'DELIVERY',
                    'netqty': 10,
                    'averageprice': 2500.0,
                    'mtm': 500.0,
                    'ltp': 2550.0
                },
                {
                    'tradingsymbol': 'HDFCBANK-EQ',
                    'symboltoken': '456789',
                    'exchange': 'NSE',
                    'producttype': 'DELIVERY',
                    'netqty': 5,
                    'averageprice': 1700.0,
                    'mtm': -100.0,
                    'ltp': 1680.0
                }
            ]
        
        elif endpoint == self.HOLDING_URL:
            return [
                {
                    'tradingsymbol': 'RELIANCE-EQ',
                    'exchange': 'NSE',
                    'symboltoken': '123456',
                    'quantity': 10,
                    'costprice': 2500.0,
                    'ltp': 2550.0,
                    'pnl': 500.0
                },
                {
                    'tradingsymbol': 'HDFCBANK-EQ',
                    'exchange': 'NSE',
                    'symboltoken': '456789',
                    'quantity': 5,
                    'costprice': 1700.0,
                    'ltp': 1680.0,
                    'pnl': -100.0
                }
            ]
        
        elif endpoint == self.HISTORICAL_URL:
            # Generate some random candle data
            candles = []
            end_date = datetime.now()
            
            if payload and 'interval' in payload:
                interval = payload['interval']
                days_to_generate = 30 if interval == 'ONE_DAY' else 5
                
                start_date = end_date - timedelta(days=days_to_generate)
                current_date = start_date
                
                while current_date <= end_date:
                    # Skip weekends
                    if current_date.weekday() < 5:  # Monday to Friday
                        base_price = 100 + np.random.normal(0, 2)
                        
                        candle = [
                            current_date.strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
                            base_price - np.random.uniform(0, 1),  # open
                            base_price + np.random.uniform(0, 2),  # high
                            base_price - np.random.uniform(0, 2),  # low
                            base_price + np.random.normal(0, 0.5),  # close
                            int(np.random.uniform(1000, 10000))  # volume
                        ]
                        candles.append(candle)
                    
                    # Increment date based on interval
                    if interval == 'ONE_DAY':
                        current_date += timedelta(days=1)
                    elif interval == 'ONE_HOUR':
                        current_date += timedelta(hours=1)
                    elif interval == 'FIFTEEN_MINUTE':
                        current_date += timedelta(minutes=15)
                    else:
                        current_date += timedelta(days=1)
            
            return candles
        
        elif endpoint == self.LTP_DATA_URL:
            symbol = payload.get('tradingsymbol', 'UNKNOWN') if payload else 'UNKNOWN'
            return {
                'tradingsymbol': symbol,
                'exchange': payload.get('exchange', 'NSE') if payload else 'NSE',
                'ltp': np.random.uniform(100, 5000)
            }
        
        # Default return for unhandled endpoints
        return {}
    
    def get_profile(self) -> Dict:
        """
        Get user profile information.
        
        Returns:
        --------
        dict
            User profile information
        """
        return self._make_api_call('GET', self.PROFILE_URL)
    
    def get_funds(self) -> Dict:
        """
        Get account funds and limits.
        
        Returns:
        --------
        dict
            Account funds and limits
        """
        return self._make_api_call('GET', self.FUNDS_URL)
    
    def place_order(self, 
                  symbol: str, 
                  exchange: str = 'NSE', 
                  transaction_type: str = 'BUY', 
                  quantity: int = 1,
                  order_type: str = 'MARKET',
                  price: Optional[float] = None,
                  trigger_price: Optional[float] = None,
                  variety: str = 'NORMAL',
                  product: str = 'DELIVERY') -> Dict:
        """
        Place an order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'RELIANCE-EQ')
        exchange : str, default='NSE'
            Exchange ('NSE', 'BSE', 'NFO', 'MCX', etc.)
        transaction_type : str, default='BUY'
            Transaction type ('BUY', 'SELL')
        quantity : int, default=1
            Order quantity
        order_type : str, default='MARKET'
            Order type ('MARKET', 'LIMIT', 'SL', 'SL-M')
        price : float, optional
            Order price (required for LIMIT orders)
        trigger_price : float, optional
            Trigger price (required for SL and SL-M orders)
        variety : str, default='NORMAL'
            Order variety ('NORMAL', 'AMO', 'CO', 'BO')
        product : str, default='DELIVERY'
            Product type ('DELIVERY', 'CARRYFORWARD', 'INTRADAY', 'MARGIN')
            
        Returns:
        --------
        dict
            Order response
        """
        # Validate parameters
        if order_type == 'LIMIT' and price is None:
            raise ValueError("Price is required for LIMIT orders")
        
        if order_type in ['SL', 'SL-M'] and trigger_price is None:
            raise ValueError("Trigger price is required for SL and SL-M orders")
        
        # Prepare order payload
        payload = {
            'variety': variety,
            'tradingsymbol': symbol,
            'symboltoken': '123456',  # Simulated token
            'transactiontype': transaction_type,
            'exchange': exchange,
            'ordertype': order_type,
            'producttype': product,
            'duration': 'DAY',
            'quantity': quantity
        }
        
        # Add price if provided
        if price is not None:
            payload['price'] = price
        
        # Add trigger price if provided
        if trigger_price is not None:
            payload['triggerprice'] = trigger_price
        
        return self._make_api_call('POST', self.PLACE_ORDER_URL, payload)
    
    def modify_order(self, 
                   order_id: str, 
                   quantity: Optional[int] = None,
                   price: Optional[float] = None,
                   trigger_price: Optional[float] = None,
                   order_type: Optional[str] = None) -> Dict:
        """
        Modify an existing order.
        
        Parameters:
        -----------
        order_id : str
            Order ID to modify
        quantity : int, optional
            New quantity
        price : float, optional
            New price
        trigger_price : float, optional
            New trigger price
        order_type : str, optional
            New order type
            
        Returns:
        --------
        dict
            Modify order response
        """
        # Prepare payload
        payload = {
            'variety': 'NORMAL',  # Only NORMAL variety can be modified
            'orderid': order_id
        }
        
        # Add optional parameters if provided
        if quantity is not None:
            payload['quantity'] = quantity
        
        if price is not None:
            payload['price'] = price
        
        if trigger_price is not None:
            payload['triggerprice'] = trigger_price
        
        if order_type is not None:
            payload['ordertype'] = order_type
        
        return self._make_api_call('POST', self.MODIFY_ORDER_URL, payload)
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            Order ID to cancel
            
        Returns:
        --------
        dict
            Cancel order response
        """
        payload = {
            'variety': 'NORMAL',  # Assuming NORMAL variety
            'orderid': order_id
        }
        
        return self._make_api_call('POST', self.CANCEL_ORDER_URL, payload)
    
    def get_order_book(self) -> Dict:
        """
        Get order book.
        
        Returns:
        --------
        dict
            Order book
        """
        return self._make_api_call('GET', self.ORDER_BOOK_URL)
    
    def get_trade_book(self) -> Dict:
        """
        Get trade book.
        
        Returns:
        --------
        dict
            Trade book
        """
        return self._make_api_call('GET', self.TRADE_BOOK_URL)
    
    def get_positions(self) -> Dict:
        """
        Get positions.
        
        Returns:
        --------
        dict
            Positions
        """
        return self._make_api_call('GET', self.POSITION_BOOK_URL)
    
    def get_holdings(self) -> Dict:
        """
        Get holdings.
        
        Returns:
        --------
        dict
            Holdings
        """
        return self._make_api_call('GET', self.HOLDING_URL)
    
    def get_historical_data(self, 
                          symbol: str, 
                          exchange: str = 'NSE',
                          interval: str = 'ONE_DAY',
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical candle data.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'RELIANCE-EQ')
        exchange : str, default='NSE'
            Exchange ('NSE', 'BSE', 'NFO', 'MCX', etc.)
        interval : str, default='ONE_DAY'
            Candle interval ('ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY')
        from_date : str, optional
            From date (format: 'YYYY-MM-DD HH:MM:SS')
        to_date : str, optional
            To date (format: 'YYYY-MM-DD HH:MM:SS')
            
        Returns:
        --------
        pandas.DataFrame
            Historical candle data
        """
        # Set default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if not from_date:
            # Default to 30 days for daily candles, 1 day for intraday
            if interval == 'ONE_DAY':
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare payload
        payload = {
            'exchange': exchange,
            'symboltoken': '123456',  # Simulated token
            'interval': interval,
            'fromdate': from_date,
            'todate': to_date
        }
        
        response = self._make_api_call('POST', self.HISTORICAL_URL, payload)
        
        if not response.get('status', False):
            logger.error(f"Failed to get historical data: {response.get('message', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert response to DataFrame
        data = response.get('data', [])
        if not data:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert OHLCV columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_ltp(self, symbol: str, exchange: str = 'NSE') -> Dict:
        """
        Get last traded price.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'RELIANCE-EQ')
        exchange : str, default='NSE'
            Exchange ('NSE', 'BSE', 'NFO', 'MCX', etc.)
            
        Returns:
        --------
        dict
            Last traded price
        """
        payload = {
            'exchange': exchange,
            'tradingsymbol': symbol,
            'symboltoken': '123456'  # Simulated token
        }
        
        return self._make_api_call('POST', self.LTP_DATA_URL, payload)
    
    def get_market_data(self, 
                      symbols: List[Dict[str, str]], 
                      mode: str = 'FULL') -> Dict:
        """
        Get real-time market data.
        
        Parameters:
        -----------
        symbols : List[Dict[str, str]]
            List of symbol dictionaries with 'exchange' and 'tradingsymbol' keys
        mode : str, default='FULL'
            Mode ('FULL', 'LTP', 'QUOTE')
            
        Returns:
        --------
        dict
            Market data
        """
        # Add token to symbols
        for symbol in symbols:
            if 'symboltoken' not in symbol:
                symbol['symboltoken'] = '123456'  # Simulated token
        
        payload = {
            'mode': mode,
            'exchangeTokens': symbols
        }
        
        return self._make_api_call('POST', self.MARKET_DATA_URL, payload)

# Create a global instance
angel_api = AngelOneAPI()