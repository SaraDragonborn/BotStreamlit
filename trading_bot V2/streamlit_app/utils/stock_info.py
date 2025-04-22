"""
Utility functions for retrieving and displaying stock information
"""
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime
import os

# Create a cache for stock info
if 'stock_info_cache' not in st.session_state:
    st.session_state.stock_info_cache = {}

def get_stock_info(symbol):
    """
    Retrieves basic information about a stock symbol including name, exchange, and current price
    
    Args:
        symbol (str): The stock symbol to look up
    
    Returns:
        dict: A dictionary containing the stock's information or None if not found
    """
    # Check cache first to avoid unnecessary API calls
    if symbol in st.session_state.stock_info_cache:
        # If cache entry is less than 5 min old, use it
        cache_entry = st.session_state.stock_info_cache[symbol]
        cache_time = cache_entry.get('timestamp', 0)
        if time.time() - cache_time < 300:  # 5 minutes
            return cache_entry
    
    # Check if this is an Indian market symbol
    is_indian_market = False
    if "-EQ" in symbol or symbol in ["NIFTY 50", "BANKNIFTY"]:
        is_indian_market = True
    
    try:
        # Use Angel One API for Indian market symbols if available
        if is_indian_market and 'angel_one_api_key' in st.session_state and st.session_state.angel_one_api_key:
            from utils.fixed_api import get_indian_data
            
            # Get stock data from Angel One
            try:
                # Attempt to get basic information for Indian stocks
                stock_info = {
                    'symbol': symbol,
                    'name': get_indian_stock_name(symbol),
                    'exchange': 'NSE' if '-EQ' in symbol else 'NSE INDEX',
                    'current_price': 0,  # Will be updated below if available
                    'currency': 'INR',
                    'timestamp': time.time(),
                    'day_change': 0,
                    'market_cap': 0,
                    'logo_url': '',
                    'sector': 'Indian Market',
                    'industry': 'Indian Market'
                }
                
                # Try to get latest price data from Angel One
                try:
                    # Get last day's data (interval="ONE_DAY", days=5)
                    df = get_indian_data(symbol, exchange="NSE", interval="ONE_DAY", days=5)
                    if not df.empty:
                        # Get most recent price
                        last_row = df.iloc[-1]
                        stock_info['current_price'] = last_row['close']
                        
                        # Calculate day change if possible
                        if len(df) > 1:
                            prev_close = df.iloc[-2]['close']
                            if prev_close > 0:
                                day_change = (last_row['close'] - prev_close) / prev_close * 100
                                stock_info['day_change'] = day_change
                except Exception as e:
                    st.warning(f"Could not retrieve current price for {symbol}: {str(e)}")
                
                # Cache the info
                st.session_state.stock_info_cache[symbol] = stock_info
                return stock_info
            
            except Exception as e:
                st.warning(f"Error retrieving Indian market data for {symbol}: {str(e)}")
                # Fall back to yfinance or predefined data
        
        # For non-Indian symbols or if Angel One fails, use yfinance
        # Get basic info using yfinance
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Check if we got valid data
        if not info or 'longName' not in info:
            # For Indian symbols if yfinance doesn't have data, use predefined info
            if is_indian_market:
                return get_hardcoded_indian_stock_info(symbol)
            return None
        
        # Create a dictionary with essential info
        stock_info = {
            'symbol': symbol,
            'name': info.get('longName', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'currency': info.get('currency', 'USD'),
            'timestamp': time.time(),
            'day_change': info.get('regularMarketChangePercent', 0),
            'market_cap': info.get('marketCap', 0),
            'logo_url': info.get('logo_url', ''),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
        
        # Store in cache
        st.session_state.stock_info_cache[symbol] = stock_info
        
        return stock_info
    
    except Exception as e:
        print(f"Error retrieving stock info for {symbol}: {str(e)}")
        # Last resort for Indian market symbols - hardcoded data
        if is_indian_market:
            return get_hardcoded_indian_stock_info(symbol)
        return None

def get_indian_stock_name(symbol):
    """Get the full name of an Indian stock from its symbol"""
    
    indian_symbols = {
        "NIFTY 50": "NIFTY 50 Index",
        "BANKNIFTY": "Bank NIFTY Index",
        "RELIANCE-EQ": "Reliance Industries Ltd.",
        "TCS-EQ": "Tata Consultancy Services Ltd.",
        "HDFCBANK-EQ": "HDFC Bank Ltd.",
        "INFY-EQ": "Infosys Ltd.",
        "ICICIBANK-EQ": "ICICI Bank Ltd.",
        "KOTAKBANK-EQ": "Kotak Mahindra Bank Ltd.",
        "HINDUNILVR-EQ": "Hindustan Unilever Ltd.",
        "SBIN-EQ": "State Bank of India",
        "BAJFINANCE-EQ": "Bajaj Finance Ltd.",
        "BHARTIARTL-EQ": "Bharti Airtel Ltd.",
        "ITC-EQ": "ITC Ltd.",
        "ASIANPAINT-EQ": "Asian Paints Ltd.",
        "MARUTI-EQ": "Maruti Suzuki India Ltd.",
        "TITAN-EQ": "Titan Company Ltd."
    }
    
    return indian_symbols.get(symbol, f"{symbol} (Indian)")

def get_hardcoded_indian_stock_info(symbol):
    """Return hardcoded information for Indian market symbols when API fails"""
    
    base_info = {
        'symbol': symbol,
        'name': get_indian_stock_name(symbol),
        'exchange': 'NSE' if '-EQ' in symbol else 'NSE INDEX',
        'currency': 'INR',
        'timestamp': time.time(),
        'day_change': 0.0,
        'market_cap': 0,
        'logo_url': '',
        'sector': 'Indian Market',
        'industry': 'Indian Market'
    }
    
    # Add approximate current prices for well-known symbols
    prices = {
        "NIFTY 50": 22500,
        "BANKNIFTY": 48500,
        "RELIANCE-EQ": 2900,
        "TCS-EQ": 3800,
        "HDFCBANK-EQ": 1450,
        "INFY-EQ": 1500,
        "ICICIBANK-EQ": 1050,
        "KOTAKBANK-EQ": 1750,
        "HINDUNILVR-EQ": 2300,
        "SBIN-EQ": 750,
        "BAJFINANCE-EQ": 6800,
        "BHARTIARTL-EQ": 1200,
        "ITC-EQ": 430,
        "ASIANPAINT-EQ": 2800,
        "MARUTI-EQ": 10500,
        "TITAN-EQ": 3200
    }
    
    base_info['current_price'] = prices.get(symbol, 1000)  # Default price if symbol not found
    
    return base_info

def display_stock_info(symbol, container=None):
    """
    Displays basic stock information in a streamlit container
    
    Args:
        symbol (str): The stock symbol to display
        container: Optional streamlit container to render in. If None, renders directly to the current context.
    
    Returns:
        The stock info dictionary or None if not found
    """
    target = container if container else st
    
    if not symbol:
        return None
    
    with target:
        with st.spinner(f"Loading info for {symbol}..."):
            stock_info = get_stock_info(symbol)
            
            if not stock_info:
                st.warning(f"Could not find information for symbol: {symbol}")
                return None
            
            # Build the display
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {stock_info['name']} ({symbol})")
                st.markdown(f"**Exchange:** {stock_info['exchange']}")
                
                if 'sector' in stock_info and stock_info['sector'] != 'Unknown':
                    st.markdown(f"**Sector:** {stock_info['sector']}")
                
            with col2:
                price = stock_info['current_price']
                currency = stock_info['currency']
                day_change = stock_info['day_change']
                
                if price > 0:
                    st.markdown(f"**Price:** {price:.2f} {currency}")
                    
                    # Format day change with color
                    if day_change > 0:
                        st.markdown(f"**Change:** <span style='color:green'>+{day_change:.2f}%</span>", unsafe_allow_html=True)
                    elif day_change < 0:
                        st.markdown(f"**Change:** <span style='color:red'>{day_change:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Change:** 0.00%")
                else:
                    st.warning("Price data unavailable")
            
            # Add a separator
            st.markdown("---")
            
            return stock_info

def display_mini_stock_info(symbol):
    """
    Displays a compact version of stock info inline
    
    Args:
        symbol (str): The stock symbol to display
    
    Returns:
        The stock info dictionary or None if not found
    """
    if not symbol:
        return None
    
    stock_info = get_stock_info(symbol)
    if not stock_info:
        st.write(f"{symbol}: Information not available")
        return None
    
    price = stock_info['current_price']
    day_change = stock_info['day_change']
    
    # Format with color based on change
    if day_change > 0:
        st.markdown(f"**{symbol}** ({stock_info['exchange']}) - {stock_info['name']} - **{price:.2f}** <span style='color:green'>▲{day_change:.2f}%</span>", unsafe_allow_html=True)
    elif day_change < 0:
        st.markdown(f"**{symbol}** ({stock_info['exchange']}) - {stock_info['name']} - **{price:.2f}** <span style='color:red'>▼{abs(day_change):.2f}%</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**{symbol}** ({stock_info['exchange']}) - {stock_info['name']} - **{price:.2f}** ━ 0.00%", unsafe_allow_html=True)
    
    return stock_info

def search_stock(query):
    """
    Search for stocks matching the query
    
    Args:
        query (str): Search string for stock names or symbols
    
    Returns:
        list: List of matching stock symbols and names
    """
    query = query.upper().strip()
    
    try:
        if len(query) < 2:
            return []
        
        result = []
        
        # Check for Indian market symbols first
        indian_symbols = {
            "NIFTY 50": "NIFTY 50 Index",
            "BANKNIFTY": "Bank NIFTY Index",
            "RELIANCE-EQ": "Reliance Industries Ltd.",
            "TCS-EQ": "Tata Consultancy Services Ltd.",
            "HDFCBANK-EQ": "HDFC Bank Ltd.",
            "INFY-EQ": "Infosys Ltd.",
            "ICICIBANK-EQ": "ICICI Bank Ltd.",
            "KOTAKBANK-EQ": "Kotak Mahindra Bank Ltd.",
            "HINDUNILVR-EQ": "Hindustan Unilever Ltd.",
            "SBIN-EQ": "State Bank of India",
            "BAJFINANCE-EQ": "Bajaj Finance Ltd.",
            "BHARTIARTL-EQ": "Bharti Airtel Ltd.",
            "ITC-EQ": "ITC Ltd.",
            "ASIANPAINT-EQ": "Asian Paints Ltd.",
            "MARUTI-EQ": "Maruti Suzuki India Ltd.",
            "TITAN-EQ": "Titan Company Ltd."
        }
        
        # Check if we should prioritize Indian market search
        indian_market_focus = False
        if 'angel_one_api_key' in st.session_state and st.session_state.angel_one_api_key:
            indian_market_focus = True
        
        # Add matching Indian symbols
        for symbol, name in indian_symbols.items():
            if query in symbol or query in name.upper():
                result.append({
                    'symbol': symbol,
                    'name': name,
                    'exchange': 'NSE' if '-EQ' in symbol else 'NSE INDEX'
                })
        
        # If we have Angel One API and Indian market focus, prioritize those results
        if indian_market_focus and result:
            # If we have Indian results, just return those first
            return result[:10]  # Limit to 10 results
        
        # Continue with US market search
        # This is a simplified version - in a real app you'd use a proper API for this
        try:
            tickers = yf.Tickers(f"{query}*")
            # Process tickers if available
        except:
            pass  # Just continue to fallbacks if this fails
        
        # Fallback to common stocks
        fallbacks = {
            'A': ['AAPL', 'AMZN', 'ADBE', 'AMAT'],
            'B': ['BA', 'BAC', 'BABA', 'BMY'],
            'C': ['CSCO', 'COST', 'CVX', 'CAT'],
            'D': ['DIS', 'DELL', 'DHR'],
            'F': ['FB', 'F', 'FDX'],
            'G': ['GOOGL', 'GE', 'GM', 'GILD'],
            'I': ['INTC', 'IBM', 'ISRG'],
            'M': ['MSFT', 'META', 'MCD', 'MMM'],
            'N': ['NFLX', 'NVDA', 'NKE', 'NOW'],
            'O': ['ORCL', 'OXY'],
            'P': ['PYPL', 'PFE', 'PEP'],
            'S': ['SPY', 'SBUX', 'SQ'],
            'T': ['TSLA', 'TWTR', 'T', 'TXN'],
            'U': ['UNH', 'UBER', 'UNP'],
            'V': ['V', 'VZ', 'VMW']
        }
        
        if query[0] in fallbacks:
            relevant = [s for s in fallbacks[query[0]] if query in s]
            for symbol in relevant[:10]:
                info = get_stock_info(symbol)
                if info:
                    result.append({
                        'symbol': symbol,
                        'name': info.get('name', 'Unknown'),
                        'exchange': info.get('exchange', '')
                    })
        
        # If using Angel One API and we have no results yet, try to get direct results from Angel One
        if indian_market_focus and not result and len(query) >= 3:
            try:
                from utils.fixed_api import get_indian_symbols
                indian_symbols_list = get_indian_symbols()
                
                for symbol in indian_symbols_list:
                    if query in symbol or query in get_indian_stock_name(symbol).upper():
                        result.append({
                            'symbol': symbol,
                            'name': get_indian_stock_name(symbol),
                            'exchange': 'NSE' if '-EQ' in symbol else 'NSE INDEX'
                        })
            except Exception as e:
                print(f"Error searching Indian symbols: {str(e)}")
        
        return result[:15]  # Limit to 15 results total
    
    except Exception as e:
        print(f"Error searching stocks: {str(e)}")
        return []