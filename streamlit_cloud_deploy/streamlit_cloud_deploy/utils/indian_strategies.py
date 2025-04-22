import numpy as np
import pandas as pd

def nifty_momentum_strategy(data, window=10, threshold=2.0):
    """
    Nifty Momentum Strategy specifically designed for Indian markets
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    window : int
        Lookback window for momentum calculation
    threshold : float
        Momentum threshold to trigger signals (%)
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate momentum (rate of change)
    signals['momentum'] = signals['Close'].pct_change(periods=window) * 100
    
    # Generate signals
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['momentum'] > threshold, 1.0, 
                               np.where(signals['momentum'] < -threshold, -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def gap_trading_strategy(data, gap_threshold=1.5):
    """
    Gap Trading Strategy for Indian markets
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    gap_threshold : float
        Gap threshold to trigger signals (%)
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate overnight gaps
    signals['prev_close'] = signals['Close'].shift(1)
    signals['gap'] = ((signals['Open'] - signals['prev_close']) / signals['prev_close']) * 100
    
    # Generate signals
    # Buy on gap down, sell on gap up
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['gap'] < -gap_threshold, 1.0, 
                               np.where(signals['gap'] > gap_threshold, -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def volume_spike_strategy(data, window=20, volume_factor=2.0, price_change_threshold=1.0):
    """
    Volume Spike Strategy for Indian markets
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    window : int
        Lookback window for volume average
    volume_factor : float
        Factor to identify volume spike
    price_change_threshold : float
        Price change threshold to confirm spike (%)
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate volume metrics
    signals['volume_ma'] = signals['Volume'].rolling(window=window).mean()
    signals['volume_ratio'] = signals['Volume'] / signals['volume_ma']
    
    # Calculate price change
    signals['price_change'] = signals['Close'].pct_change() * 100
    
    # Generate signals
    # Buy on volume spike with positive price change
    # Sell on volume spike with negative price change
    signals['signal'] = 0.0
    signals['signal'] = np.where((signals['volume_ratio'] > volume_factor) & 
                               (signals['price_change'] > price_change_threshold), 1.0, 
                               np.where((signals['volume_ratio'] > volume_factor) & 
                                      (signals['price_change'] < -price_change_threshold), -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def intraday_momentum_reversal(data, window=60, overbought=70, oversold=30):
    """
    Intraday Momentum Reversal strategy for Indian markets
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data (typically intraday data)
    window : int
        RSI calculation window (in minutes for intraday)
    overbought : int
        RSI overbought threshold
    oversold : int
        RSI oversold threshold
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate RSI
    delta = signals['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    signals['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate time of day (assuming intraday data)
    if isinstance(signals.index[0], pd.Timestamp):
        signals['hour'] = signals.index.hour
        
        # More aggressive reversal during last hour of trading (3:00-3:30 PM in Indian markets)
        late_session = (signals['hour'] >= 15) & (signals['hour'] < 16)
        
        # Generate signals with more aggressive late session reversal
        signals['signal'] = 0.0
        signals['signal'] = np.where((signals['rsi'] < oversold) | 
                                   (late_session & (signals['rsi'] < oversold + 10)), 1.0, 
                                   np.where((signals['rsi'] > overbought) | 
                                          (late_session & (signals['rsi'] > overbought - 10)), -1.0, 0.0))
    else:
        # Fallback if datetime index is not available
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, 
                                   np.where(signals['rsi'] > overbought, -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def nifty_bank_nifty_divergence(nifty_data, bank_nifty_data, window=5):
    """
    Nifty and Bank Nifty Divergence Strategy
    
    Parameters:
    -----------
    nifty_data : DataFrame
        DataFrame with Nifty OHLCV data
    bank_nifty_data : DataFrame
        DataFrame with Bank Nifty OHLCV data
    window : int
        Lookback window for divergence calculation
        
    Returns:
    --------
    DataFrame
        DataFrame with signals for Nifty
    """
    # Make copies of the data
    nifty_signals = nifty_data.copy()
    bank_nifty = bank_nifty_data.copy()
    
    # Ensure both DataFrames have the same index
    common_dates = nifty_signals.index.intersection(bank_nifty.index)
    nifty_signals = nifty_signals.loc[common_dates]
    bank_nifty = bank_nifty.loc[common_dates]
    
    # Calculate percentage change
    nifty_signals['nifty_change'] = nifty_signals['Close'].pct_change(periods=window) * 100
    nifty_signals['bank_nifty_change'] = bank_nifty['Close'].pct_change(periods=window) * 100
    
    # Calculate divergence
    nifty_signals['divergence'] = nifty_signals['nifty_change'] - nifty_signals['bank_nifty_change']
    
    # Generate signals
    # Buy Nifty when it underperforms Bank Nifty significantly (mean reversion expectation)
    # Sell Nifty when it outperforms Bank Nifty significantly
    nifty_signals['signal'] = 0.0
    nifty_signals['signal'] = np.where(nifty_signals['divergence'] < -2.0, 1.0, 
                                     np.where(nifty_signals['divergence'] > 2.0, -1.0, 0.0))
    
    # Calculate positions
    nifty_signals['position'] = nifty_signals['signal'].diff()
    
    return nifty_signals

def option_expiry_effect(data, days_to_expiry=2):
    """
    Option Expiry Effect Strategy for Indian markets (typically for Thursday expiry)
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    days_to_expiry : int
        Number of days before expiry to trigger signals
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Check if index is datetime
    if isinstance(signals.index[0], pd.Timestamp):
        # Get day of week (0 = Monday, 4 = Friday)
        signals['day_of_week'] = signals.index.dayofweek
        
        # Calculate days to Thursday (3), assuming Thursday expiry
        signals['days_to_expiry'] = (3 - signals['day_of_week']) % 7
        signals.loc[signals['days_to_expiry'] == 0, 'days_to_expiry'] = 7  # For Thursdays, set to 7 (next expiry)
        
        # Generate signals
        # Buy n days before expiry, sell on expiry day
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['days_to_expiry'] == days_to_expiry, 1.0, 
                                   np.where(signals['days_to_expiry'] == 0, -1.0, 0.0))
    else:
        # Fallback if datetime index is not available
        signals['signal'] = 0.0
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def fii_dii_flow_strategy(data, fii_flow, dii_flow, window=5):
    """
    FII/DII Flow based Strategy for Indian markets
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    fii_flow : DataFrame
        DataFrame with FII daily flow data (index should match data)
    dii_flow : DataFrame
        DataFrame with DII daily flow data (index should match data)
    window : int
        Lookback window for flow aggregation
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Ensure indexes match and merge data
    common_dates = signals.index.intersection(fii_flow.index).intersection(dii_flow.index)
    signals = signals.loc[common_dates]
    
    # Add FII and DII flow data
    signals['fii_flow'] = fii_flow.loc[common_dates]['net_flow']
    signals['dii_flow'] = dii_flow.loc[common_dates]['net_flow']
    
    # Calculate cumulative flows
    signals['fii_cum_flow'] = signals['fii_flow'].rolling(window=window).sum()
    signals['dii_cum_flow'] = signals['dii_flow'].rolling(window=window).sum()
    
    # Generate signals
    # Buy when both FIIs and DIIs are buying
    # Sell when both FIIs and DIIs are selling
    signals['signal'] = 0.0
    signals['signal'] = np.where((signals['fii_cum_flow'] > 0) & (signals['dii_cum_flow'] > 0), 1.0, 
                               np.where((signals['fii_cum_flow'] < 0) & (signals['dii_cum_flow'] < 0), -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def open_high_low_strategy(data, atr_periods=14, multiplier=1.5):
    """
    Open-High-Low Strategy for Indian markets
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    atr_periods : int
        Periods for ATR calculation
    multiplier : float
        Multiplier for ATR to set thresholds
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate True Range
    signals['high_low'] = signals['High'] - signals['Low']
    signals['high_close'] = abs(signals['High'] - signals['Close'].shift(1))
    signals['low_close'] = abs(signals['Low'] - signals['Close'].shift(1))
    signals['tr'] = signals[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # Calculate ATR
    signals['atr'] = signals['tr'].rolling(window=atr_periods).mean()
    
    # Calculate thresholds
    signals['threshold'] = signals['atr'] * multiplier
    
    # Calculate Open-High and Open-Low differences
    signals['open_high'] = signals['High'] - signals['Open']
    signals['open_low'] = signals['Open'] - signals['Low']
    
    # Generate signals
    # Buy when price moves up significantly from open
    # Sell when price moves down significantly from open
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['open_high'] > signals['threshold'], 1.0, 
                               np.where(signals['open_low'] > signals['threshold'], -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

# Dictionary of Indian strategy functions
INDIAN_STRATEGY_FUNCTIONS = {
    "Nifty Momentum": nifty_momentum_strategy,
    "Gap Trading": gap_trading_strategy,
    "Volume Spike": volume_spike_strategy,
    "Intraday Momentum Reversal": intraday_momentum_reversal,
    "Nifty-Bank Nifty Divergence": nifty_bank_nifty_divergence,
    "Option Expiry Effect": option_expiry_effect,
    "FII-DII Flow Strategy": fii_dii_flow_strategy,
    "Open-High-Low Strategy": open_high_low_strategy
}