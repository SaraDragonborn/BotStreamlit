"""
Angel One API Integration for Indian Markets

This module provides a complete interface to interact with the Angel One API for trading,
market data retrieval, and account management for Indian markets (NSE/BSE).

The API allows for:
1. Authentication and login to Angel One
2. Getting historical data for Indian stocks
3. Placing, modifying, and canceling orders
4. Getting current positions and portfolio
5. Real-time market data through the WebSocket feed
"""

import requests
import json
import pandas as pd
import time
import hashlib
import logging
from datetime import datetime, timedelta
import os
import pyotp
import streamlit as st
from typing import Dict, List, Optional, Union, Any
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('angel_one_api')

class AngelOneAPI:
    """
    Angel One API class for Indian markets
    """
    def __init__(self, client_id=None, password=None, api_key=None, totp_key=None, historical_key=None, market_feed_key=None):
        """
        Initialize the Angel One API with credentials
        
        Args:
            client_id (str): Angel One client ID
            password (str): Angel One password
            api_key (str): Angel One API key 
            totp_key (str): Angel One TOTP key for 2FA
            historical_key (str): Angel One data API key for historical data
            market_feed_key (str): Angel One market feed API key
        """
        # Try to get credentials from session state, then environment variables if not provided
        self.client_id = client_id or st.session_state.get('ANGEL_ONE_CLIENT_ID', os.environ.get('ANGEL_ONE_CLIENT_ID'))
        self.password = password or st.session_state.get('ANGEL_ONE_PASSWORD', os.environ.get('ANGEL_ONE_PASSWORD'))
        self.api_key = api_key or st.session_state.get('ANGEL_ONE_API_KEY', os.environ.get('ANGEL_ONE_API_KEY'))
        self.totp_key = totp_key or st.session_state.get('ANGEL_ONE_TOTP_KEY', os.environ.get('ANGEL_ONE_TOTP_KEY'))
        self.historical_key = historical_key or st.session_state.get('ANGEL_ONE_HISTORICAL_API_KEY', os.environ.get('ANGEL_ONE_HISTORICAL_API_KEY'))
        self.market_feed_key = market_feed_key or st.session_state.get('ANGEL_ONE_MARKET_FEED_API_KEY', os.environ.get('ANGEL_ONE_MARKET_FEED_API_KEY'))
        
        # Session variables
        self.session = requests.Session()
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        self.symbols_list = None
        self.last_login_time = None
        self.authenticated = False
        
        # API endpoints
        self.base_url = "https://apiconnect.angelbroking.com"
        self.smart_api_url = f"{self.base_url}/rest/auth/angelbroking/user/v1"
        self.data_api_url = f"{self.base_url}/rest/secure/angelbroking/historical/v1"
        self.order_api_url = f"{self.base_url}/rest/secure/angelbroking/order/v1"
        self.market_api_url = f"{self.base_url}/rest/secure/angelbroking/market/v1"
        self.ws_url = "wss://wsfeeds.angelbroking.com/NestHtml5Mobile/socket/stream"
        
        # Headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "192.168.0.1",
            "X-ClientPublicIP": "182.76.19.22",
            "X-MACAddress": "00:00:00:00:00:00",
            "X-PrivateKey": self.api_key
        }
        
        # Try automatic login if credentials are available
        if self.client_id and self.password and self.api_key and self.totp_key:
            try:
                self.login()
            except Exception as e:
                logger.warning(f"Auto-login failed: {e}")
                
    def login(self) -> Dict:
        """
        Login to Angel One API using client ID, password and TOTP
        
        Returns:
            Dict: Response containing auth token and other session details
        """
        if not self.client_id or not self.password or not self.api_key:
            raise ValueError("Client ID, password and API key are required for login")
        
        # Generate TOTP
        totp = None
        if self.totp_key:
            totp = pyotp.TOTP(self.totp_key).now()
        else:
            raise ValueError("TOTP key is required for 2FA authentication")
        
        login_data = {
            "clientcode": self.client_id,
            "password": self.password,
            "totp": totp
        }
        
        response = self.session.post(
            f"{self.smart_api_url}/login", 
            json=login_data,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Login failed: {response.text}")
        
        data = response.json()
        if data['status'] != True:
            raise Exception(f"Login error: {data['message']}")
        
        # Update auth tokens
        self.auth_token = data['data']['jwtToken']
        self.refresh_token = data['data']['refreshToken']
        self.feed_token = data['data']['feedToken']
        self.headers['Authorization'] = f"Bearer {self.auth_token}"
        
        self.last_login_time = datetime.now()
        self.authenticated = True
        
        logger.info("Successfully logged in to Angel One")
        return data
    
    def logout(self) -> Dict:
        """
        Logout from Angel One API
        
        Returns:
            Dict: Logout response
        """
        if not self.authenticated:
            return {"status": True, "message": "Already logged out"}
        
        response = self.session.post(
            f"{self.smart_api_url}/logout",
            headers=self.headers
        )
        
        data = response.json()
        if data['status']:
            self.auth_token = None
            self.refresh_token = None
            self.feed_token = None
            self.authenticated = False
            logger.info("Successfully logged out from Angel One")
        
        return data
    
    def _check_auth(self):
        """Check authentication and re-login if needed"""
        if not self.authenticated:
            self.login()
        elif self.last_login_time and (datetime.now() - self.last_login_time).total_seconds() > 21600:  # 6 hours
            logger.info("Auth token expired, refreshing...")
            self.login()
    
    def get_profile(self) -> Dict:
        """
        Get user profile information
        
        Returns:
            Dict: User profile details
        """
        self._check_auth()
        
        response = self.session.get(
            f"{self.smart_api_url}/getProfile",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get profile: {response.text}")
        
        return response.json()
    
    def get_funds(self) -> Dict:
        """
        Get available funds and margins
        
        Returns:
            Dict: Funds and margin details
        """
        self._check_auth()
        
        response = self.session.get(
            f"{self.smart_api_url}/getRMS",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get funds: {response.text}")
        
        return response.json()
    
    def get_historical_data(self, 
                           symbol: str, 
                           exchange: str = "NSE", 
                           interval: str = "ONE_DAY",
                           from_date: str = None, 
                           to_date: str = None) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol
        
        Args:
            symbol (str): Trading symbol (e.g., "RELIANCE-EQ", "NIFTY")
            exchange (str): Exchange (NSE, BSE, etc.)
            interval (str): Candle interval (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY)
            from_date (str): Start date in format "YYYY-MM-DD"
            to_date (str): End date in format "YYYY-MM-DD"
            
        Returns:
            pd.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]
        """
        self._check_auth()
        
        # Set default dates if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
            
        # Format dates for API
        from_date = f"{from_date} 09:15"
        to_date = f"{to_date} 15:30"
        
        payload = {
            "symboltoken": self._get_token(symbol, exchange),
            "exchange": exchange,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        
        response = self.session.post(
            f"{self.data_api_url}/getCandleData",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get historical data: {response.text}")
        
        data = response.json()
        if not data['status']:
            raise Exception(f"API error: {data['message']}")
        
        # Create DataFrame
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        else:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_quote(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """
        Get current market quotes for symbols
        
        Args:
            symbols (List[str]): List of trading symbols
            exchange (str): Exchange (NSE, BSE, etc.)
            
        Returns:
            Dict: Quotes for the requested symbols
        """
        self._check_auth()
        
        payload = {
            "mode": "FULL",
            "exchangeTokens": {
                exchange: [self._get_token(symbol, exchange) for symbol in symbols]
            }
        }
        
        response = self.session.post(
            f"{self.market_api_url}/quote",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get quotes: {response.text}")
        
        return response.json()
    
    def place_order(self, 
                   symbol: str, 
                   exchange: str = "NSE", 
                   transaction_type: str = "BUY",
                   quantity: int = 1, 
                   price: float = 0, 
                   order_type: str = "MARKET",
                   product: str = "INTRADAY") -> Dict:
        """
        Place a new order
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            transaction_type (str): BUY or SELL
            quantity (int): Number of shares/units
            price (float): Order price (for LIMIT orders)
            order_type (str): MARKET, LIMIT, SL, SL-M
            product (str): INTRADAY, DELIVERY, MARGIN
            
        Returns:
            Dict: Order response with order ID
        """
        self._check_auth()
        
        # Validate order parameters
        transaction_type = transaction_type.upper()
        if transaction_type not in ["BUY", "SELL"]:
            raise ValueError("Transaction type must be BUY or SELL")
        
        order_type = order_type.upper()
        if order_type not in ["MARKET", "LIMIT", "SL", "SL-M"]:
            raise ValueError("Order type must be MARKET, LIMIT, SL, or SL-M")
        
        product = product.upper()
        if product not in ["INTRADAY", "DELIVERY", "MARGIN"]:
            raise ValueError("Product type must be INTRADAY, DELIVERY, or MARGIN")
        
        if product == "INTRADAY":
            product = "I"  # Angel API uses I for INTRADAY
        elif product == "DELIVERY":
            product = "D"  # Angel API uses D for DELIVERY
        elif product == "MARGIN":
            product = "M"  # Angel API uses M for MARGIN
        
        payload = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": self._get_token(symbol, exchange),
            "transactiontype": transaction_type,
            "exchange": exchange,
            "ordertype": order_type,
            "producttype": product,
            "duration": "DAY",
            "quantity": str(quantity)
        }
        
        # Add price for LIMIT/SL orders
        if order_type in ["LIMIT", "SL", "SL-M"]:
            payload["price"] = str(price)
        
        # Add trigger price for SL/SL-M orders
        if order_type in ["SL", "SL-M"]:
            payload["triggerprice"] = str(price)
        
        response = self.session.post(
            f"{self.order_api_url}/placeOrder",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to place order: {response.text}")
        
        return response.json()
    
    def modify_order(self, order_id: str, **kwargs) -> Dict:
        """
        Modify an existing order
        
        Args:
            order_id (str): Order ID to modify
            **kwargs: Parameters to modify (quantity, price, trigger_price, order_type)
            
        Returns:
            Dict: Response with status of modification
        """
        self._check_auth()
        
        payload = {
            "variety": "NORMAL",
            "orderid": order_id
        }
        
        # Add parameters to be modified
        for key, value in kwargs.items():
            if key == "quantity":
                payload["quantity"] = str(value)
            elif key == "price":
                payload["price"] = str(value)
            elif key == "trigger_price":
                payload["triggerprice"] = str(value)
            elif key == "order_type":
                payload["ordertype"] = value.upper()
        
        response = self.session.post(
            f"{self.order_api_url}/modifyOrder",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to modify order: {response.text}")
        
        return response.json()
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            Dict: Response with status of cancellation
        """
        self._check_auth()
        
        payload = {
            "variety": "NORMAL",
            "orderid": order_id
        }
        
        response = self.session.post(
            f"{self.order_api_url}/cancelOrder",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to cancel order: {response.text}")
        
        return response.json()
    
    def get_order_book(self) -> List[Dict]:
        """
        Get order book (list of all orders)
        
        Returns:
            List[Dict]: List of orders
        """
        self._check_auth()
        
        response = self.session.get(
            f"{self.order_api_url}/getOrderBook",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get order book: {response.text}")
        
        data = response.json()
        if not data.get('status'):
            raise Exception(f"API error: {data.get('message')}")
            
        return data.get('data', [])
    
    def get_trade_book(self) -> List[Dict]:
        """
        Get trade book (list of all executed trades)
        
        Returns:
            List[Dict]: List of trades
        """
        self._check_auth()
        
        response = self.session.get(
            f"{self.order_api_url}/getTradeBook",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get trade book: {response.text}")
        
        data = response.json()
        if not data.get('status'):
            raise Exception(f"API error: {data.get('message')}")
            
        return data.get('data', [])
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions
        
        Returns:
            List[Dict]: List of current positions
        """
        self._check_auth()
        
        response = self.session.get(
            f"{self.order_api_url}/getPosition",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get positions: {response.text}")
        
        data = response.json()
        if not data.get('status'):
            raise Exception(f"API error: {data.get('message')}")
            
        return data.get('data', [])
    
    def get_holdings(self) -> List[Dict]:
        """
        Get holdings (long-term portfolio)
        
        Returns:
            List[Dict]: List of holdings
        """
        self._check_auth()
        
        response = self.session.get(
            f"{self.order_api_url}/getHolding",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get holdings: {response.text}")
        
        data = response.json()
        if not data.get('status'):
            raise Exception(f"API error: {data.get('message')}")
            
        return data.get('data', [])
    
    def _get_token(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Get token for a symbol
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            str: Symbol token
        """
        # Load symbols list if not already loaded
        if not self.symbols_list:
            self._load_symbols()
        
        # Search for the symbol
        if self.symbols_list is not None:
            filtered = self.symbols_list[(self.symbols_list['name'] == symbol) & 
                                        (self.symbols_list['exch_seg'] == exchange)]
            if not filtered.empty:
                return filtered.iloc[0]['token']
        
        # If symbol not found, return the symbol itself
        # This works with numeric tokens for some indices
        if symbol.isdigit():
            return symbol
            
        # For Nifty and Bank Nifty, return fixed tokens
        if symbol == "NIFTY 50" or symbol == "NIFTY":
            return "26000"  # NIFTY token
        elif symbol == "BANKNIFTY" or symbol == "NIFTY BANK":
            return "26009"  # BANKNIFTY token
        
        raise ValueError(f"Symbol {symbol} not found in {exchange}")
    
    def _load_symbols(self):
        """Load all symbols and tokens"""
        try:
            # Angel One API gives a token file
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                self.symbols_list = pd.DataFrame(data)
                logger.info(f"Loaded {len(self.symbols_list)} symbols")
            else:
                logger.error(f"Failed to load symbols: {response.text}")
        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
            logger.error(traceback.format_exc())
    
    def search_symbols(self, query: str, exchange: str = None) -> pd.DataFrame:
        """
        Search for symbols matching a query
        
        Args:
            query (str): Search query
            exchange (str, optional): Filter by exchange
            
        Returns:
            pd.DataFrame: Matching symbols
        """
        # Load symbols list if not already loaded
        if not self.symbols_list:
            self._load_symbols()
        
        if self.symbols_list is None:
            return pd.DataFrame()
        
        # Filter by query
        query = query.upper()
        result = self.symbols_list[
            (self.symbols_list['name'].str.upper().str.contains(query)) |
            (self.symbols_list['symbol'].str.upper().str.contains(query))
        ]
        
        # Filter by exchange if provided
        if exchange:
            result = result[result['exch_seg'] == exchange]
        
        # Return top 50 results
        return result.head(50)

# Convenience function to check if Angel One credentials are available
def check_angel_one_credentials():
    """Check if Angel One API credentials are available in session state or environment variables"""
    credentials = {
        'ANGEL_ONE_CLIENT_ID': st.session_state.get('ANGEL_ONE_CLIENT_ID', os.environ.get('ANGEL_ONE_CLIENT_ID')),
        'ANGEL_ONE_PASSWORD': st.session_state.get('ANGEL_ONE_PASSWORD', os.environ.get('ANGEL_ONE_PASSWORD')),
        'ANGEL_ONE_API_KEY': st.session_state.get('ANGEL_ONE_API_KEY', os.environ.get('ANGEL_ONE_API_KEY')),
        'ANGEL_ONE_HISTORICAL_API_KEY': st.session_state.get('ANGEL_ONE_HISTORICAL_API_KEY', os.environ.get('ANGEL_ONE_HISTORICAL_API_KEY')),
        'ANGEL_ONE_MARKET_FEED_API_KEY': st.session_state.get('ANGEL_ONE_MARKET_FEED_API_KEY', os.environ.get('ANGEL_ONE_MARKET_FEED_API_KEY')),
    }
    
    # Check if all required credentials are available
    missing = [key for key, value in credentials.items() if not value]
    
    if missing:
        return False, missing
    return True, []

# Test Angel One connection
def test_angel_one_connection(client_id=None, password=None, api_key=None, totp_key=None):
    """
    Test connection to Angel One API
    
    Args:
        client_id (str, optional): Angel One client ID. Defaults to None.
        password (str, optional): Angel One password. Defaults to None.
        api_key (str, optional): Angel One API key. Defaults to None.
        totp_key (str, optional): Angel One TOTP key. Defaults to None.
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Use provided credentials or get from session state
        client_id = client_id or st.session_state.get('angel_one_client_id')
        password = password or st.session_state.get('angel_one_password')
        api_key = api_key or st.session_state.get('angel_one_api_key')
        totp_key = totp_key or st.session_state.get('angel_one_totp_key', '')
        
        # Check if credentials are provided
        if not client_id or not password or not api_key:
            return False, "Required credentials missing. Please provide client ID, password, and API key."
            
        # Create API client
        api = AngelOneAPI(client_id=client_id, password=password, api_key=api_key, totp_key=totp_key)
        
        # Try to login if not authenticated
        if not api.authenticated:
            try:
                api.login()
            except Exception as e:
                return False, f"Authentication failed: {str(e)}"
        
        # Try to get basic account info
        try:
            profile = api.get_profile()
            if profile and profile.get('status'):
                return True, f"Successfully connected to Angel One! Welcome, {profile.get('data', {}).get('name', 'User')}."
            else:
                return False, "Connected but couldn't retrieve account information."
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
            
    except Exception as e:
        logger.error(f"Angel One connection test failed: {e}")
        logger.error(traceback.format_exc())
        return False, f"Connection test failed: {str(e)}"

# Initialize Angel One API with credentials from session state or environment
def initialize_angel_one_api():
    """Initialize and return Angel One API instance with available credentials"""
    # Check if credentials are available
    has_credentials, missing = check_angel_one_credentials()
    
    if not has_credentials:
        # If credentials are missing, store them in session state
        st.session_state['ANGEL_ONE_CLIENT_ID'] = st.session_state.get('ANGEL_ONE_CLIENT_ID', os.environ.get('ANGEL_ONE_CLIENT_ID'))
        st.session_state['ANGEL_ONE_PASSWORD'] = st.session_state.get('ANGEL_ONE_PASSWORD', os.environ.get('ANGEL_ONE_PASSWORD'))
        st.session_state['ANGEL_ONE_API_KEY'] = st.session_state.get('ANGEL_ONE_API_KEY', os.environ.get('ANGEL_ONE_API_KEY'))
        st.session_state['ANGEL_ONE_HISTORICAL_API_KEY'] = st.session_state.get('ANGEL_ONE_HISTORICAL_API_KEY', os.environ.get('ANGEL_ONE_HISTORICAL_API_KEY'))
        st.session_state['ANGEL_ONE_MARKET_FEED_API_KEY'] = st.session_state.get('ANGEL_ONE_MARKET_FEED_API_KEY', os.environ.get('ANGEL_ONE_MARKET_FEED_API_KEY'))
    
    # Initialize API client
    api = AngelOneAPI()
    return api