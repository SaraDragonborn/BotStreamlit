"""
Trade Executor Module
=======================================
Executes trades based on signals from strategies.
"""

import os
import time
import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from api.alpaca_api import alpaca_api
from api.angel_api import angel_api
from utils.logger import get_trade_logger
from config import get_config

config = get_config()
logger = get_trade_logger()

class TradeExecutor:
    """
    Trade Executor component.
    
    Executes trades on different markets based on signals:
    - US stocks via Alpaca
    - Indian stocks via Angel One
    - (Future) Crypto and forex markets
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize the Trade Executor.
        
        Parameters:
        -----------
        paper_trading : bool, default=True
            Whether to use paper trading (simulated) or live trading
        """
        self.paper_trading = paper_trading
        self.order_history = []
        self.pending_orders = {}
        
        logger.info(f"Trade Executor initialized with paper trading: {paper_trading}")
        
        # Initialize connection to brokers
        if not paper_trading:
            logger.warning("LIVE TRADING ENABLED - Real funds will be used!")
    
    def execute_order(self, 
                    symbol: str,
                    order_type: str,
                    quantity: float,
                    side: str,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = 'day',
                    market: str = 'US') -> Dict:
        """
        Execute an order.
        
        Parameters:
        -----------
        symbol : str
            Symbol to trade
        order_type : str
            Order type ('market', 'limit', 'stop', 'stop_limit')
        quantity : float
            Quantity to trade
        side : str
            Order side ('buy', 'sell')
        limit_price : float, optional
            Limit price for limit orders
        stop_price : float, optional
            Stop price for stop orders
        time_in_force : str, default='day'
            Time in force ('day', 'gtc', 'ioc')
        market : str, default='US'
            Market to trade ('US', 'India')
            
        Returns:
        --------
        Dict
            Order result
        """
        # Check if paper trading
        if not self.paper_trading:
            logger.warning(f"LIVE ORDER: {side.upper()} {quantity} {symbol} at {limit_price or 'MARKET'}")
        
        try:
            result = {}
            
            # Execute order in the appropriate market
            if market == 'US':
                # Use Alpaca API for US stocks
                result = alpaca_api.place_order(
                    symbol=symbol,
                    side=side,
                    qty=quantity,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price
                )
                
            elif market == 'India':
                # Use Angel One API for Indian stocks
                # Map order parameters to Angel One format
                angel_order_type = {
                    'market': 'MARKET',
                    'limit': 'LIMIT',
                    'stop': 'SL',
                    'stop_limit': 'SL-M'
                }.get(order_type, 'MARKET')
                
                angel_side = {
                    'buy': 'BUY',
                    'sell': 'SELL'
                }.get(side, 'BUY')
                
                result = angel_api.place_order(
                    symbol=symbol,
                    exchange='NSE',
                    transaction_type=angel_side,
                    quantity=int(quantity),
                    order_type=angel_order_type,
                    price=limit_price,
                    trigger_price=stop_price,
                    product='DELIVERY'
                )
            
            # Record order in history
            order_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': symbol,
                'market': market,
                'type': order_type,
                'side': side,
                'quantity': quantity,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'time_in_force': time_in_force,
                'paper_trading': self.paper_trading,
                'status': 'submitted',
                'raw_result': result
            }
            
            # Extract order ID
            if result.get('success', False):
                if market == 'US':
                    order_id = result.get('order', {}).get('id')
                    status = result.get('order', {}).get('status')
                elif market == 'India':
                    order_id = result.get('data', {}).get('orderid')
                    status = 'submitted'  # Angel API doesn't return status in the response
                
                order_record['order_id'] = order_id
                order_record['status'] = status
                
                # Add to pending orders if not filled
                if status not in ['filled', 'completed']:
                    self.pending_orders[order_id] = order_record
                
                logger.info(f"Order submitted: {side.upper()} {quantity} {symbol} ({order_id})")
            else:
                order_record['status'] = 'failed'
                order_record['error'] = result.get('error', 'Unknown error')
                logger.error(f"Order failed: {side.upper()} {quantity} {symbol} - {order_record['error']}")
            
            self.order_history.append(order_record)
            
            return order_record
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            
            # Record failed order
            order_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': symbol,
                'market': market,
                'type': order_type,
                'side': side,
                'quantity': quantity,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'time_in_force': time_in_force,
                'paper_trading': self.paper_trading,
                'status': 'failed',
                'error': str(e)
            }
            
            self.order_history.append(order_record)
            
            return order_record
    
    def execute_bracket_order(self, 
                            symbol: str,
                            quantity: float,
                            side: str,
                            entry_price: Optional[float] = None,
                            take_profit_price: float = None,
                            stop_loss_price: float = None,
                            market: str = 'US') -> Dict:
        """
        Execute a bracket order (entry + take profit + stop loss).
        
        Parameters:
        -----------
        symbol : str
            Symbol to trade
        quantity : float
            Quantity to trade
        side : str
            Order side ('buy', 'sell')
        entry_price : float, optional
            Entry price (if None, use market order)
        take_profit_price : float
            Take profit price
        stop_loss_price : float
            Stop loss price
        market : str, default='US'
            Market to trade ('US', 'India')
            
        Returns:
        --------
        Dict
            Order result
        """
        try:
            result = {}
            
            # Execute order in the appropriate market
            if market == 'US':
                # For US stocks, Alpaca supports bracket orders directly
                order_type = 'limit' if entry_price is not None else 'market'
                
                result = alpaca_api.place_bracket_order(
                    symbol=symbol,
                    side=side,
                    qty=quantity,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    limit_price=entry_price
                )
                
            elif market == 'India':
                # For Indian stocks, we need to create separate orders
                # First, place the main entry order
                angel_order_type = 'LIMIT' if entry_price is not None else 'MARKET'
                angel_side = 'BUY' if side.lower() == 'buy' else 'SELL'
                
                entry_result = angel_api.place_order(
                    symbol=symbol,
                    exchange='NSE',
                    transaction_type=angel_side,
                    quantity=int(quantity),
                    order_type=angel_order_type,
                    price=entry_price,
                    product='DELIVERY'
                )
                
                result = {
                    'entry_order': entry_result,
                    'take_profit_order': None,
                    'stop_loss_order': None
                }
                
                # We would need to track the entry order and place the take profit and stop loss
                # orders after it's filled, which requires order tracking and state management
                # For simplicity, we'll leave this as future enhancement
                
                logger.warning("Full bracket order support for Indian markets requires order tracking")
            
            # Record order in history
            order_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': symbol,
                'market': market,
                'type': 'bracket',
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'paper_trading': self.paper_trading,
                'status': 'submitted',
                'raw_result': result
            }
            
            # Extract order ID
            if market == 'US' and result.get('success', False):
                order_id = result.get('order', {}).get('id')
                status = result.get('order', {}).get('status')
                
                order_record['order_id'] = order_id
                order_record['status'] = status
                
                logger.info(f"Bracket order submitted: {side.upper()} {quantity} {symbol} ({order_id})")
            
            self.order_history.append(order_record)
            
            return order_record
            
        except Exception as e:
            logger.error(f"Error executing bracket order: {str(e)}")
            
            # Record failed order
            order_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': symbol,
                'market': market,
                'type': 'bracket',
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'paper_trading': self.paper_trading,
                'status': 'failed',
                'error': str(e)
            }
            
            self.order_history.append(order_record)
            
            return order_record
    
    def cancel_order(self, order_id: str, market: str = 'US') -> Dict:
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            Order ID to cancel
        market : str, default='US'
            Market ('US', 'India')
            
        Returns:
        --------
        Dict
            Cancellation result
        """
        try:
            result = {}
            
            # Cancel order in the appropriate market
            if market == 'US':
                result = alpaca_api.cancel_order(order_id)
            elif market == 'India':
                result = angel_api.cancel_order(order_id)
            
            # Update order record
            if order_id in self.pending_orders:
                self.pending_orders[order_id]['status'] = 'canceled'
                
                if not result.get('success', False):
                    self.pending_orders[order_id]['error'] = result.get('error', 'Unknown error')
                
                # Remove from pending orders
                canceled_order = self.pending_orders.pop(order_id)
                
                # Update in order history
                for i, order in enumerate(self.order_history):
                    if order.get('order_id') == order_id:
                        self.order_history[i] = canceled_order
                        break
            
            logger.info(f"Order {order_id} canceled")
            
            return {
                'success': result.get('success', False),
                'order_id': order_id,
                'message': result.get('message', ''),
                'error': result.get('error', '')
            }
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    def cancel_all_orders(self, market: str = 'US') -> Dict:
        """
        Cancel all open orders.
        
        Parameters:
        -----------
        market : str, default='US'
            Market ('US', 'India')
            
        Returns:
        --------
        Dict
            Cancellation result
        """
        try:
            result = {}
            
            # Cancel all orders in the appropriate market
            if market == 'US':
                result = alpaca_api.cancel_all_orders()
            elif market == 'India':
                # Angel One doesn't have a direct cancel all orders endpoint
                # We would need to get all open orders and cancel them individually
                orders = angel_api.get_order_book()
                
                canceled = []
                failed = []
                
                if orders.get('status', False):
                    open_orders = [
                        order for order in orders.get('data', [])
                        if order.get('status') in ['pending', 'open', 'trigger pending']
                    ]
                    
                    for order in open_orders:
                        order_id = order.get('orderid')
                        cancel_result = angel_api.cancel_order(order_id)
                        
                        if cancel_result.get('status', False):
                            canceled.append(order_id)
                        else:
                            failed.append(order_id)
                
                result = {
                    'success': True,
                    'canceled_orders': canceled,
                    'failed_orders': failed
                }
            
            # Update pending orders
            for order_id in list(self.pending_orders.keys()):
                if self.pending_orders[order_id]['market'] == market:
                    self.pending_orders[order_id]['status'] = 'canceled'
                    
                    # Remove from pending orders
                    canceled_order = self.pending_orders.pop(order_id)
                    
                    # Update in order history
                    for i, order in enumerate(self.order_history):
                        if order.get('order_id') == order_id:
                            self.order_history[i] = canceled_order
                            break
            
            logger.info(f"All orders canceled for market: {market}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error canceling all orders: {str(e)}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str, market: str = 'US') -> Dict:
        """
        Get order status.
        
        Parameters:
        -----------
        order_id : str
            Order ID
        market : str, default='US'
            Market ('US', 'India')
            
        Returns:
        --------
        Dict
            Order status
        """
        try:
            result = {}
            
            # Get order status from the appropriate market
            if market == 'US':
                result = alpaca_api.get_order(order_id)
            elif market == 'India':
                # Angel One doesn't have a direct get order endpoint
                orders = angel_api.get_order_book()
                
                if orders.get('status', False):
                    for order in orders.get('data', []):
                        if order.get('orderid') == order_id:
                            result = {
                                'success': True,
                                'order': order
                            }
                            break
            
            # Update pending order status if found
            if order_id in self.pending_orders and result.get('success', False):
                order_data = result.get('order', {})
                
                if market == 'US':
                    status = order_data.get('status')
                elif market == 'India':
                    # Map Angel One status to common format
                    angel_status = order_data.get('status', '')
                    status_map = {
                        'complete': 'filled',
                        'rejected': 'rejected',
                        'cancelled': 'canceled',
                        'pending': 'open'
                    }
                    status = status_map.get(angel_status.lower(), 'unknown')
                
                self.pending_orders[order_id]['status'] = status
                
                # Remove from pending if terminal status
                if status in ['filled', 'rejected', 'canceled']:
                    updated_order = self.pending_orders.pop(order_id)
                    
                    # Update in order history
                    for i, order in enumerate(self.order_history):
                        if order.get('order_id') == order_id:
                            self.order_history[i] = updated_order
                            break
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {str(e)}")
            
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    def update_pending_orders(self) -> None:
        """Update status of all pending orders."""
        for order_id in list(self.pending_orders.keys()):
            market = self.pending_orders[order_id]['market']
            self.get_order_status(order_id, market)
    
    def execute_signal(self, 
                     signal: Dict, 
                     risk_info: Dict, 
                     market: str = 'US',
                     wait_for_fill: bool = False) -> Dict:
        """
        Execute a trade based on a signal and risk parameters.
        
        Parameters:
        -----------
        signal : Dict
            Signal information
            - symbol: stock symbol
            - direction: 'buy' or 'sell'
            - signal_type: type of signal
            - strength: signal strength (0-1)
        risk_info : Dict
            Risk information
            - entry_price: entry price
            - stop_price: stop loss price
            - take_profit_price: take profit price
            - shares: number of shares
            - risk_amount: risk amount
        market : str, default='US'
            Market to trade ('US', 'India')
        wait_for_fill : bool, default=False
            Whether to wait for the order to fill
            
        Returns:
        --------
        Dict
            Execution result
        """
        try:
            symbol = signal.get('symbol')
            direction = signal.get('direction', 'buy').lower()
            signal_type = signal.get('signal_type', 'unspecified')
            signal_strength = signal.get('strength', 1.0)
            
            entry_price = risk_info.get('entry_price')
            stop_price = risk_info.get('stop_price')
            take_profit_price = risk_info.get('take_profit_price')
            shares = risk_info.get('shares', 0)
            
            # Validate signal
            if not symbol or shares <= 0:
                logger.error(f"Invalid signal or risk info: {symbol}, shares={shares}")
                return {
                    'success': False,
                    'error': 'Invalid signal or risk info'
                }
            
            # Log signal
            logger.info(f"Executing {direction} signal for {symbol} ({signal_type}), "
                       f"shares={shares}, entry={entry_price}, stop={stop_price}")
            
            # Determine order type
            if entry_price is None:
                order_type = 'market'
            else:
                order_type = 'limit'
            
            # If take profit and stop loss are provided, use bracket order
            if take_profit_price is not None and stop_price is not None:
                order_result = self.execute_bracket_order(
                    symbol=symbol,
                    quantity=shares,
                    side=direction,
                    entry_price=entry_price,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_price,
                    market=market
                )
            else:
                # Otherwise, use regular order
                order_result = self.execute_order(
                    symbol=symbol,
                    order_type=order_type,
                    quantity=shares,
                    side=direction,
                    limit_price=entry_price,
                    market=market
                )
            
            # Wait for fill if requested
            if wait_for_fill and order_result.get('success', False):
                order_id = order_result.get('order_id')
                
                if order_id:
                    for _ in range(10):  # Wait up to 10 seconds
                        status_result = self.get_order_status(order_id, market)
                        
                        if status_result.get('success', False):
                            status = status_result.get('order', {}).get('status')
                            
                            if status in ['filled', 'completed']:
                                logger.info(f"Order {order_id} filled")
                                order_result['status'] = 'filled'
                                break
                            elif status in ['rejected', 'canceled']:
                                logger.warning(f"Order {order_id} {status}")
                                order_result['status'] = status
                                break
                        
                        time.sleep(1)
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing signal: {str(e)}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_position(self, symbol: str, market: str = 'US') -> Dict:
        """
        Get position for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol
        market : str, default='US'
            Market ('US', 'India')
            
        Returns:
        --------
        Dict
            Position information
        """
        try:
            if market == 'US':
                return alpaca_api.get_position(symbol)
            elif market == 'India':
                positions = angel_api.get_positions()
                
                if positions.get('status', False):
                    for position in positions.get('data', []):
                        if position.get('tradingsymbol') == symbol:
                            return {
                                'success': True,
                                'position': position
                            }
                
                return {
                    'success': False,
                    'error': f"No position found for {symbol}"
                }
            
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_all_positions(self, market: str = 'US') -> Dict:
        """
        Get all positions.
        
        Parameters:
        -----------
        market : str, default='US'
            Market ('US', 'India')
            
        Returns:
        --------
        Dict
            All positions
        """
        try:
            if market == 'US':
                return alpaca_api.get_positions()
            elif market == 'India':
                return angel_api.get_positions()
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            
            return {
                'success': False,
                'error': str(e)
            }

# Create global instance
trade_executor = TradeExecutor(paper_trading=True)