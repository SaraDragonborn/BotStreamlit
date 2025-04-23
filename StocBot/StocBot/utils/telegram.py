"""
Telegram Notification Module
=======================================
Sends trade notifications to Telegram.
"""

import os
import requests
from typing import Optional
import config
from utils.logger import setup_logger

# Set up logger
logger = setup_logger('telegram')

def send_telegram_message(message: str) -> bool:
    """
    Send a message to Telegram.
    
    Parameters:
    -----------
    message : str
        Message to send
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Get Telegram token and chat ID from config
    token = config.get('TELEGRAM_TOKEN', '')
    chat_id = config.get('TELEGRAM_CHAT_ID', '')
    
    # Check if token and chat ID are set
    if not token or not chat_id:
        logger.warning("Telegram token or chat ID not set")
        return False
    
    # Telegram API URL
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    try:
        # Send message
        response = requests.post(
            url,
            json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
        )
        
        # Check response
        if response.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Failed to send Telegram message: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}")
        return False

def send_trade_notification(action: str, symbol: str, quantity: int, price: float, strategy: str) -> bool:
    """
    Send a trade notification to Telegram.
    
    Parameters:
    -----------
    action : str
        Trade action ('BUY' or 'SELL')
    symbol : str
        Stock symbol
    quantity : int
        Number of shares
    price : float
        Price per share
    strategy : str
        Strategy name
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    message = f"*TRADE ALERT*\n\n"
    message += f"Action: *{action}*\n"
    message += f"Symbol: *{symbol}*\n"
    message += f"Quantity: {quantity}\n"
    message += f"Price: {price:.2f}\n"
    message += f"Total: {quantity * price:.2f}\n"
    message += f"Strategy: {strategy}\n"
    message += f"Time: {os.popen('date').read().strip()}\n"
    
    return send_telegram_message(message)

def send_status_notification(message: str) -> bool:
    """
    Send a status notification to Telegram.
    
    Parameters:
    -----------
    message : str
        Status message
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    full_message = f"*STATUS UPDATE*\n\n{message}"
    
    return send_telegram_message(full_message)

def send_error_notification(message: str) -> bool:
    """
    Send an error notification to Telegram.
    
    Parameters:
    -----------
    message : str
        Error message
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    full_message = f"*ERROR*\n\n{message}"
    
    return send_telegram_message(full_message)

def send_alert_notification(symbol: str, price: float, condition: str, value: float) -> bool:
    """
    Send an alert notification to Telegram.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    price : float
        Current price
    condition : str
        Alert condition
    value : float
        Alert value
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    message = f"*PRICE ALERT*\n\n"
    message += f"Symbol: *{symbol}*\n"
    message += f"Price: {price:.2f}\n"
    message += f"Alert: {condition} {value:.2f}\n"
    message += f"Time: {os.popen('date').read().strip()}\n"
    
    return send_telegram_message(message)