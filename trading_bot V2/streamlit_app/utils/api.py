import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# Base URLs for API
BASE_URL = "http://localhost:5000/api"
ALPACA_ENDPOINTS = {
    'paper': 'https://paper-api.alpaca.markets',
    'live': 'https://api.alpaca.markets'
}

def test_alpaca_connection(api_key, api_secret):
    """Test connection to Alpaca API"""
    try:
        if not api_key or not api_secret:
            return False, "API key and secret are required"
        
        # In a production app, we would set the headers with API key/secret
        # For now, we're assuming the server handles authentication
        endpoint = f"{BASE_URL}/trading/alpaca/test"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return True, "Connected successfully"
        else:
            return False, f"Connection failed: {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_account_info():
    """Get Alpaca account information"""
    try:
        endpoint = f"{BASE_URL}/trading/alpaca/account"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching account info: {str(e)}")
        return None

def get_positions():
    """Get current positions"""
    try:
        endpoint = f"{BASE_URL}/trading/alpaca/positions"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching positions: {str(e)}")
        return []

def get_orders():
    """Get recent orders"""
    try:
        endpoint = f"{BASE_URL}/trading/alpaca/orders"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching orders: {str(e)}")
        return []

def get_assets():
    """Get available trading assets"""
    try:
        endpoint = f"{BASE_URL}/trading/alpaca/assets"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching assets: {str(e)}")
        return []

def place_order(symbol, qty, side, type="market", time_in_force="day", limit_price=None, stop_price=None):
    """Place a new order"""
    try:
        endpoint = f"{BASE_URL}/trading/alpaca/orders"
        
        # Build order parameters
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force
        }
        
        # Add optional parameters
        if limit_price and type == "limit":
            order_data["limit_price"] = limit_price
        if stop_price and type == "stop" or type == "stop_limit":
            order_data["stop_price"] = stop_price
        
        response = requests.post(endpoint, json=order_data)
        
        if response.status_code == 200 or response.status_code == 201:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        print(f"Error placing order: {str(e)}")
        return {"error": str(e)}

def get_asset_price(symbol):
    """Get current price for an asset"""
    try:
        endpoint = f"{BASE_URL}/trading/alpaca/quote/{symbol}"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('latestPrice', 0)
        else:
            return 0
    except Exception as e:
        print(f"Error fetching price: {str(e)}")
        return 0

def get_historical_data(symbol, timeframe="1D", start_date=None, end_date=None, limit=100):
    """Get historical price data for a symbol"""
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        endpoint = f"{BASE_URL}/trading/alpaca/bars?symbol={symbol}&timeframe={timeframe}&start={start_date}&end={end_date}&limit={limit}"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")
        return []

# FinGPT API functions
def get_fingpt_models():
    """Get available FinGPT models"""
    try:
        endpoint = f"{BASE_URL}/fingpt/models"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return []

def get_fingpt_news(symbol, days=7, limit=5):
    """Get financial news from FinGPT"""
    try:
        endpoint = f"{BASE_URL}/fingpt/news?ticker={symbol}&days={days}&limit={limit}"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def analyze_fingpt_sentiment(texts):
    """Analyze sentiment of financial texts"""
    try:
        endpoint = f"{BASE_URL}/fingpt/sentiment"
        payload = {"texts": texts}
        
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return []

def get_fingpt_summary(texts, max_length=150):
    """Generate summary from financial texts"""
    try:
        endpoint = f"{BASE_URL}/fingpt/summary"
        payload = {"texts": texts, "maxLength": max_length}
        
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"summary": "Unable to generate summary"}
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return {"summary": f"Error: {str(e)}"}

def get_fingpt_signals(symbol, days=30):
    """Get trading signals from FinGPT"""
    try:
        endpoint = f"{BASE_URL}/fingpt/signals/{symbol}?days={days}"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching signals: {str(e)}")
        return None

def analyze_ticker(symbol, days=30):
    """Get comprehensive ticker analysis"""
    try:
        endpoint = f"{BASE_URL}/fingpt/analyze/{symbol}?days={days}"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error analyzing ticker: {str(e)}")
        return None

# DeepTradeBot and FinRL API functions
def get_available_strategies():
    """Get available trading strategies"""
    try:
        endpoint = f"{BASE_URL}/deeptradebot/strategies"
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching strategies: {str(e)}")
        return []

def create_strategy(strategy_data):
    """Create a new trading strategy"""
    try:
        endpoint = f"{BASE_URL}/deeptradebot/strategies"
        response = requests.post(endpoint, json=strategy_data)
        
        if response.status_code == 200 or response.status_code == 201:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        print(f"Error creating strategy: {str(e)}")
        return {"error": str(e)}

def backtest_strategy(strategy_id, symbol, start_date, end_date, initial_capital=10000):
    """Backtest a trading strategy"""
    try:
        endpoint = f"{BASE_URL}/deeptradebot/backtest"
        payload = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital
        }
        
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        print(f"Error backtesting strategy: {str(e)}")
        return {"error": str(e)}

def train_model(model_type, symbol, start_date, end_date, params=None):
    """Train an AI model for trading"""
    try:
        if model_type == "deeptradebot":
            endpoint = f"{BASE_URL}/deeptradebot/train"
        elif model_type == "finrl":
            endpoint = f"{BASE_URL}/finrl/train"
        else:
            return {"error": "Invalid model type"}
            
        payload = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "params": params or {}
        }
        
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return {"error": str(e)}