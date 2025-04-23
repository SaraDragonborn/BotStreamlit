"""
Angel One API Module
=======================================
Connects to Angel One API for trading.
"""

import os
import hashlib
import time
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import config
from utils.logger import setup_logger

# Set up logger
logger = setup_logger('angel_api')

class AngelOneAPI:
    """
    Angel One API connector.
    
    Provides methods to interact with Angel One API for trading.
    
    Attributes:
    -----------
    api_key : str
        Angel One API key
    client_id : str
        Angel One client ID
    client_pin : str
        Angel One client PIN
    token : str
        Angel One token
    base_url : str
        Angel One API base URL
    headers : dict
        Request headers
    """
    
    def __init__(self):
        """
        Initialize the Angel One API connector.
        """
        self.api_key = config.get('ANGEL_ONE_API_KEY', '')
        self.client_id = config.get('ANGEL_ONE_CLIENT_ID', '')
        self.client_pin = config.get('ANGEL_ONE_CLIENT_PIN', '')
        self.token = config.get('ANGEL_ONE_TOKEN', '')
        self.base_url = "https://apiconnect.angelbroking.com"
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
        self._authenticated = False
    
    def authenticate(self) -> bool:
        """
        Authenticate with Angel One API.
        
        Returns:
        --------
        bool
            True if authentication was successful, False otherwise
        """
        if not self.api_key or not self.client_id or not self.client_pin:
            logger.error("API key, client ID, or client PIN not set")
            return False
        
        # If we already have a token, use it
        if self.token:
            self.headers['Authorization'] = f"Bearer {self.token}"
            self._authenticated = True
            return True
        
        url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        
        try:
            # Hash the client PIN
            pin_hash = hashlib.sha256(self.client_pin.encode()).hexdigest()
            
            # Login payload
            payload = {
                'clientcode': self.client_id,
                'password': pin_hash
            }
            
            # Send login request
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    # Get token
                    self.token = data['data']['jwtToken']
                    
                    # Save token to config
                    config.set('ANGEL_ONE_TOKEN', self.token)
                    
                    # Update headers
                    self.headers['Authorization'] = f"Bearer {self.token}"
                    
                    logger.info("Authentication successful")
                    self._authenticated = True
                    return True
                else:
                    logger.error(f"Authentication failed: {data['message']}")
                    return False
            else:
                logger.error(f"Authentication failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error authenticating: {str(e)}")
            return False
    
    def get_profile(self) -> Optional[Dict]:
        """
        Get user profile.
        
        Returns:
        --------
        dict or None
            User profile or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile"
        
        try:
            # Send request
            response = requests.get(url, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    return data['data']
                else:
                    logger.error(f"Failed to get profile: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get profile: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting profile: {str(e)}")
            return None
    
    def get_funds(self) -> Optional[Dict]:
        """
        Get user funds.
        
        Returns:
        --------
        dict or None
            User funds or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getRMS"
        
        try:
            # Send request
            response = requests.get(url, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    return data['data']
                else:
                    logger.error(f"Failed to get funds: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get funds: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting funds: {str(e)}")
            return None
    
    def get_quote(self, symbol: str, exchange: str = 'NSE') -> Optional[Dict]:
        """
        Get quote for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        exchange : str, default='NSE'
            Exchange (NSE, BSE, NFO, etc.)
            
        Returns:
        --------
        dict or None
            Quote data or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/market/v1/quote"
        
        try:
            # Quote payload
            payload = {
                'mode': 'FULL',
                'exchangeTokens': {
                    exchange: [symbol]
                }
            }
            
            # Send request
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    return data['data'][exchange][symbol]
                else:
                    logger.error(f"Failed to get quote: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get quote: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = '5minute',
                          from_date: Optional[str] = None, to_date: Optional[str] = None,
                          exchange: str = 'NSE') -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        timeframe : str, default='5minute'
            Data timeframe (1minute, 5minute, 15minute, 30minute, 60minute, 1day)
        from_date : str, optional
            From date (format: 'YYYY-MM-DD')
        to_date : str, optional
            To date (format: 'YYYY-MM-DD')
        exchange : str, default='NSE'
            Exchange (NSE, BSE, NFO, etc.)
            
        Returns:
        --------
        pandas.DataFrame or None
            Historical price data or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        # Default dates if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
        
        try:
            # Map timeframe to Angel One timeframe
            timeframe_map = {
                '1minute': 'ONE_MINUTE',
                '5minute': 'FIVE_MINUTE',
                '15minute': 'FIFTEEN_MINUTE',
                '30minute': 'THIRTY_MINUTE',
                '60minute': 'ONE_HOUR',
                '1day': 'ONE_DAY'
            }
            
            angel_timeframe = timeframe_map.get(timeframe, 'FIVE_MINUTE')
            
            # Convert dates to Angel One format
            from_datetime = datetime.strptime(from_date, '%Y-%m-%d')
            to_datetime = datetime.strptime(to_date, '%Y-%m-%d') + timedelta(days=1, microseconds=-1)
            
            from_timestamp = int(from_datetime.timestamp())
            to_timestamp = int(to_datetime.timestamp())
            
            # Historical data payload
            payload = {
                'symbolToken': symbol,
                'exchange': exchange,
                'fromDate': from_timestamp,
                'toDate': to_timestamp,
                'candleTimeframe': angel_timeframe
            }
            
            # Send request
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    # Convert data to DataFrame
                    candles = data['data']
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    
                    # Set timestamp as index
                    df = df.set_index('timestamp')
                    
                    return df
                else:
                    logger.error(f"Failed to get historical data: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get historical data: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def place_order(self, symbol: str, quantity: int, side: str, 
                   order_type: str = 'MARKET', price: Optional[float] = None,
                   stop_loss: Optional[float] = None, target: Optional[float] = None,
                   exchange: str = 'NSE') -> Optional[str]:
        """
        Place an order.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        quantity : int
            Number of shares
        side : str
            Order side ('BUY' or 'SELL')
        order_type : str, default='MARKET'
            Order type ('MARKET' or 'LIMIT')
        price : float, optional
            Limit price (required for LIMIT orders)
        stop_loss : float, optional
            Stop loss percentage
        target : float, optional
            Target percentage
        exchange : str, default='NSE'
            Exchange (NSE, BSE, NFO, etc.)
            
        Returns:
        --------
        str or None
            Order ID or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder"
        
        try:
            # Validate parameters
            if order_type == 'LIMIT' and price is None:
                logger.error("Price is required for LIMIT orders")
                return None
            
            if side not in ['BUY', 'SELL']:
                logger.error(f"Invalid side: {side}")
                return None
            
            # Order payload
            payload = {
                'variety': 'NORMAL',
                'tradingsymbol': symbol,
                'symboltoken': symbol,
                'transactiontype': side,
                'exchange': exchange,
                'ordertype': order_type,
                'producttype': 'INTRADAY',
                'duration': 'DAY',
                'quantity': quantity
            }
            
            # Add price for LIMIT orders
            if order_type == 'LIMIT':
                payload['price'] = price
            
            # Send request
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    order_id = data['data']['orderid']
                    
                    # Place stop loss order if requested
                    if stop_loss and side == 'BUY':
                        self._place_stop_loss_order(symbol, quantity, price or self.get_quote(symbol)['ltp'], stop_loss, exchange)
                    
                    # Place target order if requested
                    if target and side == 'BUY':
                        self._place_target_order(symbol, quantity, price or self.get_quote(symbol)['ltp'], target, exchange)
                    
                    logger.info(f"Order placed successfully: {order_id}")
                    return order_id
                else:
                    logger.error(f"Failed to place order: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to place order: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def _place_stop_loss_order(self, symbol: str, quantity: int, entry_price: float, 
                              stop_loss_percent: float, exchange: str = 'NSE') -> Optional[str]:
        """
        Place a stop loss order.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        quantity : int
            Number of shares
        entry_price : float
            Entry price
        stop_loss_percent : float
            Stop loss percentage
        exchange : str, default='NSE'
            Exchange (NSE, BSE, NFO, etc.)
            
        Returns:
        --------
        str or None
            Order ID or None if error
        """
        url = f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder"
        
        try:
            # Calculate stop loss price
            stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
            
            # Round to 2 decimal places
            stop_loss_price = round(stop_loss_price, 2)
            
            # Stop loss order payload
            payload = {
                'variety': 'STOPLOSS',
                'tradingsymbol': symbol,
                'symboltoken': symbol,
                'transactiontype': 'SELL',
                'exchange': exchange,
                'ordertype': 'STOPLOSS_LIMIT',
                'producttype': 'INTRADAY',
                'duration': 'DAY',
                'quantity': quantity,
                'triggerprice': stop_loss_price,
                'price': stop_loss_price
            }
            
            # Send request
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    order_id = data['data']['orderid']
                    logger.info(f"Stop loss order placed successfully: {order_id}")
                    return order_id
                else:
                    logger.error(f"Failed to place stop loss order: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to place stop loss order: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing stop loss order: {str(e)}")
            return None
    
    def _place_target_order(self, symbol: str, quantity: int, entry_price: float, 
                           target_percent: float, exchange: str = 'NSE') -> Optional[str]:
        """
        Place a target order.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        quantity : int
            Number of shares
        entry_price : float
            Entry price
        target_percent : float
            Target percentage
        exchange : str, default='NSE'
            Exchange (NSE, BSE, NFO, etc.)
            
        Returns:
        --------
        str or None
            Order ID or None if error
        """
        url = f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder"
        
        try:
            # Calculate target price
            target_price = entry_price * (1 + target_percent / 100)
            
            # Round to 2 decimal places
            target_price = round(target_price, 2)
            
            # Target order payload
            payload = {
                'variety': 'NORMAL',
                'tradingsymbol': symbol,
                'symboltoken': symbol,
                'transactiontype': 'SELL',
                'exchange': exchange,
                'ordertype': 'LIMIT',
                'producttype': 'INTRADAY',
                'duration': 'DAY',
                'quantity': quantity,
                'price': target_price
            }
            
            # Send request
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    order_id = data['data']['orderid']
                    logger.info(f"Target order placed successfully: {order_id}")
                    return order_id
                else:
                    logger.error(f"Failed to place target order: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to place target order: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing target order: {str(e)}")
            return None
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get order status.
        
        Parameters:
        -----------
        order_id : str
            Order ID
            
        Returns:
        --------
        dict or None
            Order status or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook"
        
        try:
            # Send request
            response = requests.get(url, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    # Find order with matching ID
                    orders = data['data']
                    for order in orders:
                        if order['orderid'] == order_id:
                            return order
                    
                    logger.error(f"Order not found: {order_id}")
                    return None
                else:
                    logger.error(f"Failed to get order status: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get order status: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return None
    
    def get_orders(self) -> Optional[List[Dict]]:
        """
        Get all orders.
        
        Returns:
        --------
        list or None
            List of orders or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook"
        
        try:
            # Send request
            response = requests.get(url, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    return data['data']
                else:
                    logger.error(f"Failed to get orders: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get orders: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return None
    
    def get_positions(self) -> Optional[List[Dict]]:
        """
        Get all positions.
        
        Returns:
        --------
        list or None
            List of positions or None if error
        """
        if not self._authenticated:
            if not self.authenticate():
                return None
        
        url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getPosition"
        
        try:
            # Send request
            response = requests.get(url, headers=self.headers)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] and data['message'] == 'SUCCESS':
                    return data['data']
                else:
                    logger.error(f"Failed to get positions: {data['message']}")
                    return None
            else:
                logger.error(f"Failed to get positions: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None
    
    def square_off_position(self, symbol: str, exchange: str = 'NSE') -> bool:
        """
        Square off a position.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        exchange : str, default='NSE'
            Exchange (NSE, BSE, NFO, etc.)
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._authenticated:
            if not self.authenticate():
                return False
        
        # Get positions
        positions = self.get_positions()
        
        if not positions:
            logger.error("Failed to get positions")
            return False
        
        # Find position with matching symbol
        position = None
        for pos in positions:
            if pos['tradingsymbol'] == symbol and pos['exchange'] == exchange:
                position = pos
                break
        
        if not position:
            logger.error(f"Position not found: {symbol}")
            return False
        
        # Get quantity to square off
        quantity = int(position['netqty'])
        
        if quantity == 0:
            logger.info(f"Position already squared off: {symbol}")
            return True
        
        # Determine transaction type
        side = 'SELL' if quantity > 0 else 'BUY'
        quantity = abs(quantity)
        
        # Place order to square off
        order_id = self.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type='MARKET',
            exchange=exchange
        )
        
        if order_id:
            logger.info(f"Position squared off successfully: {symbol}")
            return True
        else:
            logger.error(f"Failed to square off position: {symbol}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all positions.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._authenticated:
            if not self.authenticate():
                return False
        
        # Get positions
        positions = self.get_positions()
        
        if not positions:
            logger.info("No positions to close")
            return True
        
        success = True
        
        # Square off each position
        for position in positions:
            symbol = position['tradingsymbol']
            exchange = position['exchange']
            
            if int(position['netqty']) != 0:
                if not self.square_off_position(symbol, exchange):
                    success = False
        
        return success