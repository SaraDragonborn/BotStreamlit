import numpy as np
import pandas as pd
import streamlit as st

def moving_average_crossover(data, fast_period=10, slow_period=50, ma_type="SMA"):
    """
    Moving Average Crossover strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    fast_period : int
        Fast moving average period
    slow_period : int
        Slow moving average period
    ma_type : str
        Moving average type (SMA, EMA, WMA)
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate fast and slow moving averages
    if ma_type == "SMA":
        signals['fast_ma'] = signals['Close'].rolling(window=fast_period, min_periods=1).mean()
        signals['slow_ma'] = signals['Close'].rolling(window=slow_period, min_periods=1).mean()
    elif ma_type == "EMA":
        signals['fast_ma'] = signals['Close'].ewm(span=fast_period, adjust=False).mean()
        signals['slow_ma'] = signals['Close'].ewm(span=slow_period, adjust=False).mean()
    elif ma_type == "WMA":
        signals['fast_ma'] = signals['Close'].rolling(window=fast_period, min_periods=1).apply(
            lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        signals['slow_ma'] = signals['Close'].rolling(window=slow_period, min_periods=1).apply(
            lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    
    # Generate signals: 1 when fast MA crosses above slow MA, -1 when fast MA crosses below slow MA
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['fast_ma'] > signals['slow_ma'], 1.0, 0.0)
    
    # Calculate positions (the position changes only when the signal changes)
    signals['position'] = signals['signal'].diff()
    
    return signals

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    """
    RSI strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    period : int
        RSI period
    overbought : int
        Overbought level
    oversold : int
        Oversold level
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate RSI
    delta = signals['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    signals['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals: 1 when RSI crosses below oversold, -1 when RSI crosses above overbought
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, 
                               np.where(signals['rsi'] > overbought, -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def macd_strategy(data, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    fast_period : int
        Fast EMA period
    slow_period : int
        Slow EMA period
    signal_period : int
        Signal line period
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate MACD
    signals['ema_fast'] = signals['Close'].ewm(span=fast_period, adjust=False).mean()
    signals['ema_slow'] = signals['Close'].ewm(span=slow_period, adjust=False).mean()
    signals['macd'] = signals['ema_fast'] - signals['ema_slow']
    signals['signal_line'] = signals['macd'].ewm(span=signal_period, adjust=False).mean()
    signals['histogram'] = signals['macd'] - signals['signal_line']
    
    # Generate signals: 1 when MACD crosses above signal line, -1 when MACD crosses below signal line
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['macd'] > signals['signal_line'], 1.0, 0.0)
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def bollinger_bands_strategy(data, period=20, std_dev=2.0):
    """
    Bollinger Bands strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    period : int
        Moving average period
    std_dev : float
        Number of standard deviations
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate Bollinger Bands
    signals['ma'] = signals['Close'].rolling(window=period).mean()
    signals['std'] = signals['Close'].rolling(window=period).std()
    signals['upper_band'] = signals['ma'] + (signals['std'] * std_dev)
    signals['lower_band'] = signals['ma'] - (signals['std'] * std_dev)
    
    # Generate signals: 1 when price crosses below lower band, -1 when price crosses above upper band
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['Close'] < signals['lower_band'], 1.0, 
                                np.where(signals['Close'] > signals['upper_band'], -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def mean_reversion_strategy(data, period=20, threshold=2.0):
    """
    Mean Reversion strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    period : int
        Moving average period
    threshold : float
        Deviation threshold (%)
        
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate mean and deviation
    signals['ma'] = signals['Close'].rolling(window=period).mean()
    signals['pct_diff'] = (signals['Close'] - signals['ma']) / signals['ma'] * 100
    
    # Generate signals: 1 when price is below MA by threshold%, -1 when price is above MA by threshold%
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['pct_diff'] < -threshold, 1.0, 
                                np.where(signals['pct_diff'] > threshold, -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def volume_breakout_strategy(data, ma_period=20, volume_factor=2.0):
    """
    Volume Breakout strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    ma_period : int
        Moving average period for volume
    volume_factor : float
        Factor to determine volume breakout
    
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate price and volume metrics
    signals['price_change'] = signals['Close'].pct_change()
    signals['volume_ma'] = signals['Volume'].rolling(window=ma_period).mean()
    
    # Generate signals
    # Buy when volume is high and price is up, sell when volume is high and price is down
    signals['signal'] = 0.0
    signals['signal'] = np.where((signals['Volume'] > signals['volume_ma'] * volume_factor) & 
                                (signals['price_change'] > 0), 1.0, 
                                np.where((signals['Volume'] > signals['volume_ma'] * volume_factor) & 
                                        (signals['price_change'] < 0), -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def support_resistance_strategy(data, window=20, threshold=0.03):
    """
    Support and Resistance strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    window : int
        Window for finding support/resistance
    threshold : float
        Threshold for price movement (%)
    
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Find potential support and resistance levels
    signals['rolling_low'] = signals['Low'].rolling(window=window, center=True).min()
    signals['rolling_high'] = signals['High'].rolling(window=window, center=True).max()
    
    # Calculate price relative to support and resistance
    signals['price_to_support'] = (signals['Close'] / signals['rolling_low']) - 1
    signals['price_to_resistance'] = (signals['Close'] / signals['rolling_high']) - 1
    
    # Generate signals
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['price_to_support'] < threshold, 1.0,  # Near support, buy
                                np.where(signals['price_to_resistance'] > -threshold, -1.0, 0.0))  # Near resistance, sell
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def pattern_recognition_strategy(data, window=14):
    """
    Simple pattern recognition strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    window : int
        Lookback window for patterns
    
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate some basic patterns
    signals['body'] = signals['Close'] - signals['Open']
    signals['shadow_upper'] = signals['High'] - np.maximum(signals['Open'], signals['Close'])
    signals['shadow_lower'] = np.minimum(signals['Open'], signals['Close']) - signals['Low']
    
    # Identify doji (small body relative to shadows)
    signals['doji'] = abs(signals['body']) < (0.1 * (signals['High'] - signals['Low']))
    
    # Identify bullish engulfing (current green candle engulfs previous red candle)
    signals['bullish_engulfing'] = (signals['body'].shift(1) < 0) & \
                                 (signals['body'] > 0) & \
                                 (signals['Open'] < signals['Close'].shift(1)) & \
                                 (signals['Close'] > signals['Open'].shift(1))
    
    # Identify bearish engulfing (current red candle engulfs previous green candle)
    signals['bearish_engulfing'] = (signals['body'].shift(1) > 0) & \
                                 (signals['body'] < 0) & \
                                 (signals['Open'] > signals['Close'].shift(1)) & \
                                 (signals['Close'] < signals['Open'].shift(1))
    
    # Generate signals
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['bullish_engulfing'], 1.0, 
                               np.where(signals['bearish_engulfing'], -1.0, 0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def multi_timeframe_strategy(data, fast_period=10, slow_period=30):
    """
    Multi-timeframe strategy combining short and long term signals
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    fast_period : int
        Fast period for short-term indicators
    slow_period : int
        Slow period for long-term indicators
    
    Returns:
    --------
    DataFrame
        DataFrame with signals
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Short-term trend (Fast EMA)
    signals['fast_ema'] = signals['Close'].ewm(span=fast_period, adjust=False).mean()
    
    # Long-term trend (Slow EMA)
    signals['slow_ema'] = signals['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # Short-term momentum (RSI)
    delta = signals['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    signals['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals:
    # Buy when price is above slow EMA and RSI is not overbought
    # Sell when price is below slow EMA and RSI is not oversold
    signals['signal'] = 0.0
    signals['signal'] = np.where((signals['Close'] > signals['slow_ema']) & (signals['rsi'] < 70), 1.0, 
                               np.where((signals['Close'] < signals['slow_ema']) & (signals['rsi'] > 30), -1.0, 
                                       0.0))
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def backtest_strategy(signals, initial_capital=100000.0):
    """
    Backtest a trading strategy
    
    Parameters:
    -----------
    signals : DataFrame
        DataFrame with signals
    initial_capital : float
        Initial capital
        
    Returns:
    --------
    DataFrame
        DataFrame with backtest results
    """
    # Make a copy of the signals DataFrame
    portfolio = signals.copy()
    
    # Calculate positions and holdings
    portfolio['shares'] = np.where(portfolio['position'] == 1, 
                                  initial_capital / portfolio['Close'], 
                                  np.where(portfolio['position'] == -1, 
                                          -initial_capital / portfolio['Close'], 
                                          0))
    
    portfolio['holdings'] = portfolio['shares'].cumsum() * portfolio['Close']
    
    # Calculate cash and total portfolio value
    portfolio['cash'] = initial_capital - (portfolio['shares'] * portfolio['Close']).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    
    # Calculate returns and performance metrics
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
    
    # Calculate drawdowns
    portfolio['max_total'] = portfolio['total'].cummax()
    portfolio['drawdown'] = (portfolio['total'] / portfolio['max_total'] - 1) * 100
    
    return portfolio

def calculate_performance_metrics(portfolio):
    """
    Calculate performance metrics for a backtest
    
    Parameters:
    -----------
    portfolio : DataFrame
        DataFrame with backtest results
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Calculate total return
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
    
    # Calculate annualized return
    days = (portfolio.index[-1] - portfolio.index[0]).days
    annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0%)
    daily_returns = portfolio['returns'].dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    # Calculate maximum drawdown
    max_drawdown = portfolio['drawdown'].min()
    
    # Calculate win rate
    trades = portfolio[portfolio['position'] != 0]
    wins = trades[trades['returns'] > 0]
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    
    # Calculate profit factor
    winning_trades = trades[trades['returns'] > 0]['returns'].sum()
    losing_trades = abs(trades[trades['returns'] < 0]['returns'].sum())
    profit_factor = winning_trades / losing_trades if losing_trades != 0 else float('inf')
    
    # Return performance metrics
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trades)
    }

def optimize_strategy(data, strategy_func, param_grid, initial_capital=100000.0):
    """
    Optimize strategy parameters using grid search
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with OHLCV data
    strategy_func : function
        Strategy function to optimize
    param_grid : dict
        Dictionary with parameter grids
    initial_capital : float
        Initial capital
        
    Returns:
    --------
    tuple
        Best parameters and performance metrics
    """
    best_params = None
    best_metrics = None
    best_return = -float('inf')
    
    # Generate all parameter combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(itertools.product(*values))
    
    # Try each parameter combination
    for params in param_combinations:
        # Create parameter dictionary
        param_dict = {keys[i]: params[i] for i in range(len(keys))}
        
        # Apply strategy with parameters
        signals = strategy_func(data, **param_dict)
        
        # Backtest strategy
        portfolio = backtest_strategy(signals, initial_capital)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(portfolio)
        
        # Update best parameters if better
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            best_params = param_dict
            best_metrics = metrics
    
    return best_params, best_metrics

# Dictionary of strategy functions
STRATEGY_FUNCTIONS = {
    "Moving Average Crossover": moving_average_crossover,
    "RSI Strategy": rsi_strategy,
    "MACD Crossover": macd_strategy, 
    "Bollinger Bands": bollinger_bands_strategy,
    "Mean Reversion": mean_reversion_strategy,
    "Volume Breakout": volume_breakout_strategy,
    "Support & Resistance": support_resistance_strategy,
    "Pattern Recognition": pattern_recognition_strategy,
    "Multi-Timeframe": multi_timeframe_strategy
}