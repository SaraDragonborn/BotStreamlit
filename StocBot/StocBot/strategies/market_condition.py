"""
Market Condition Analyzer Module
=======================================
Analyzes market conditions using ADX to determine trending or sideways markets.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Union, Any
import config

class MarketConditionAnalyzer:
    """
    Market Condition Analyzer.
    
    Analyzes market conditions using ADX indicator to determine if the market
    is trending or in a sideways pattern.
    
    Attributes:
    -----------
    params : dict
        Parameters including adx_period, trend_threshold, sideways_threshold
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the Market Condition Analyzer.
        
        Parameters:
        -----------
        params : dict, optional
            Parameters for the analyzer
        """
        # Get default parameters from config if not provided
        if params is None:
            params = config.get('STRATEGY_PARAMS', {}).get('market_condition', {})
        
        # Default parameters if not in config
        default_params = {
            'index_symbol': 'NIFTY',  # Index to use for market condition detection
            'adx_period': 14,      # ADX calculation period
            'trend_threshold': 25, # ADX threshold for trend detection
            'sideways_threshold': 20, # ADX threshold for sideways market detection
        }
        
        # Merge default parameters with provided parameters
        self.params = params or {}
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def analyze(self, data: pd.DataFrame) -> str:
        """
        Analyze market condition using ADX.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        str
            Market condition ('trending', 'sideways', or 'neutral')
        """
        if data is None or data.empty:
            return 'neutral'
        
        # Calculate ADX
        adx_period = self.params['adx_period']
        adx = ta.trend.ADXIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=adx_period,
            fillna=True
        )
        
        # Get ADX values
        data['adx'] = adx.adx()
        
        # Get the latest ADX value
        latest_adx = data['adx'].iloc[-1]
        
        # Determine market condition
        trend_threshold = self.params['trend_threshold']
        sideways_threshold = self.params['sideways_threshold']
        
        if latest_adx >= trend_threshold:
            return 'trending'
        elif latest_adx <= sideways_threshold:
            return 'sideways'
        else:
            return 'neutral'
    
    def recommend_strategy(self, data: pd.DataFrame) -> str:
        """
        Recommend a strategy based on market condition.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        str
            Recommended strategy ('moving_average_crossover' or 'rsi_reversal')
        """
        condition = self.analyze(data)
        
        if condition == 'trending':
            return 'moving_average_crossover'
        elif condition == 'sideways':
            return 'rsi_reversal'
        else:
            # Default to EMA crossover as a safe choice
            return 'moving_average_crossover'