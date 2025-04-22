"""
Compatibility module for CCXT
This provides fallback functionality when ccxt is not available
"""

def get_crypto_data(symbol, timeframe="1d", limit=100):
    """
    Fallback function when ccxt is not available
    """
    import yfinance as yf
    # Try to get crypto data from yfinance instead
    crypto_symbol = symbol.replace("/", "-")
    try:
        df = yf.download(f"{crypto_symbol}", period="1mo")
        return df
    except:
        import pandas as pd
        import datetime
        # Return empty dataframe with the right structure
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

# Try to try:
    import ccxt
except ImportError:
    from utils.ccxt_handler import ccxt, ccxt_available, use fallbacks if not available
try:
    try:
    import ccxt
except ImportError:
    from utils.ccxt_handler import ccxt, ccxt_available
    # ccxt is available
    ccxt_available = True
except ImportError:
    # ccxt is not available, use fallbacks
    ccxt_available = False
    # Create dummy ccxt class
    class ccxt:
        class Exchange:
            pass
