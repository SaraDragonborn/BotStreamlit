import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Strategy template definitions
STRATEGY_TEMPLATES = {
    # Traditional Moving Average Strategies
    "moving_average_crossover": {
        "name": "Moving Average Crossover",
        "description": "A strategy that generates buy and sell signals based on the crossover of two moving averages of different periods.",
        "type": "Technical",
        "parameters": {
            "ma_type": "Simple",
            "fast_period": 9,
            "slow_period": 21
        },
        "trading_rules": {
            "buy_condition": "Fast MA crosses above Slow MA",
            "sell_condition": "Fast MA crosses below Slow MA"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "ema_crossover": {
        "name": "EMA Crossover",
        "description": "Uses Exponential Moving Average crossovers which give more weight to recent prices for faster signals.",
        "type": "Technical",
        "parameters": {
            "fast_period": 8,
            "slow_period": 21
        },
        "trading_rules": {
            "buy_condition": "Fast EMA crosses above Slow EMA",
            "sell_condition": "Fast EMA crosses below Slow EMA"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "triple_ma_strategy": {
        "name": "Triple Moving Average",
        "description": "Uses three moving averages to confirm trends and generate more reliable signals.",
        "type": "Technical",
        "parameters": {
            "short_period": 5,
            "medium_period": 20,
            "long_period": 50,
            "ma_type": "Exponential"
        },
        "trading_rules": {
            "buy_condition": "Short MA > Medium MA > Long MA (aligned in uptrend)",
            "sell_condition": "Short MA < Medium MA < Long MA (aligned in downtrend)"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 4,
            "take_profit": 12
        }
    },
    "hull_ma_strategy": {
        "name": "Hull Moving Average",
        "description": "Uses the Hull Moving Average which reduces lag and improves smoothness for better trend identification.",
        "type": "Technical",
        "parameters": {
            "hma_period": 16,
            "signal_length": 1
        },
        "trading_rules": {
            "buy_condition": "HMA slope turns positive",
            "sell_condition": "HMA slope turns negative"
        },
        "risk": {
            "position_size": 12,
            "stop_loss": 6,
            "take_profit": 18
        }
    },
    
    # Oscillator-Based Strategies
    "rsi_strategy": {
        "name": "RSI Momentum",
        "description": "Uses the Relative Strength Index (RSI) to identify overbought and oversold conditions for generating signals.",
        "type": "Technical",
        "parameters": {
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30
        },
        "trading_rules": {
            "buy_condition": "RSI crosses above Oversold level",
            "sell_condition": "RSI crosses below Overbought level"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 3,
            "take_profit": 9
        }
    },
    "rsi_trendline": {
        "name": "RSI Trendline Strategy",
        "description": "Identifies trendlines on RSI for early trend reversal signals.",
        "type": "Technical",
        "parameters": {
            "rsi_period": 14,
            "trend_strength": 3
        },
        "trading_rules": {
            "buy_condition": "RSI breaks above downward trendline",
            "sell_condition": "RSI breaks below upward trendline"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 10
        }
    },
    "stochastic_strategy": {
        "name": "Stochastic Oscillator",
        "description": "Uses stochastic oscillator to identify potential reversal points in price.",
        "type": "Technical",
        "parameters": {
            "k_period": 14,
            "d_period": 3,
            "overbought": 80,
            "oversold": 20
        },
        "trading_rules": {
            "buy_condition": "%K crosses above %D while both below oversold level",
            "sell_condition": "%K crosses below %D while both above overbought level"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 4,
            "take_profit": 12
        }
    },
    "macd_strategy": {
        "name": "MACD Strategy",
        "description": "Uses the Moving Average Convergence Divergence (MACD) indicator to identify momentum changes and generate trading signals.",
        "type": "Technical",
        "parameters": {
            "fast_ema": 12,
            "slow_ema": 26,
            "signal_period": 9
        },
        "trading_rules": {
            "buy_condition": "MACD crosses above Signal Line",
            "sell_condition": "MACD crosses below Signal Line"
        },
        "risk": {
            "position_size": 12,
            "stop_loss": 4,
            "take_profit": 12
        }
    },
    "macd_histogram": {
        "name": "MACD Histogram Strategy",
        "description": "Focuses on MACD histogram changes to detect early momentum shifts.",
        "type": "Technical",
        "parameters": {
            "fast_ema": 12,
            "slow_ema": 26,
            "signal_period": 9,
            "hist_trigger": 2
        },
        "trading_rules": {
            "buy_condition": "Histogram turns positive after consecutive increases",
            "sell_condition": "Histogram turns negative after consecutive decreases"
        },
        "risk": {
            "position_size": 12,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "rsi_macd_combined": {
        "name": "RSI+MACD Combined",
        "description": "Combines RSI and MACD for signal confirmation, requiring both indicators to align.",
        "type": "Technical",
        "parameters": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_threshold": 50
        },
        "trading_rules": {
            "buy_condition": "MACD crosses above Signal Line while RSI > 50",
            "sell_condition": "MACD crosses below Signal Line while RSI < 50"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 4,
            "take_profit": 12
        }
    },
    
    # Trend and Volatility Strategies
    "bollinger_bands": {
        "name": "Bollinger Bands Strategy",
        "description": "Uses Bollinger Bands to identify price breakouts and potential reversals based on volatility.",
        "type": "Technical",
        "parameters": {
            "ma_period": 20,
            "band_stddev": 2.0,
            "ma_type": "Simple"
        },
        "trading_rules": {
            "buy_condition": "Price touches lower band then moves inward",
            "sell_condition": "Price touches upper band then moves inward"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "donchian_channel": {
        "name": "Donchian Channel Breakout",
        "description": "Generates signals when price breaks out of the highest high or lowest low over a period.",
        "type": "Technical",
        "parameters": {
            "period": 20,
            "atr_multiplier": 1.5
        },
        "trading_rules": {
            "buy_condition": "Price breaks above upper channel line",
            "sell_condition": "Price breaks below lower channel line"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "keltner_channel": {
        "name": "Keltner Channel Strategy",
        "description": "Similar to Bollinger Bands but uses Average True Range for volatility-based channels.",
        "type": "Technical",
        "parameters": {
            "ema_period": 20,
            "atr_period": 10,
            "atr_multiplier": 2.0
        },
        "trading_rules": {
            "buy_condition": "Price breaks above upper channel",
            "sell_condition": "Price breaks below lower channel"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 6,
            "take_profit": 12
        }
    },
    "atr_trailing_stop": {
        "name": "ATR Trailing Stop Strategy",
        "description": "Uses Average True Range to set dynamic trailing stops that adapt to market volatility.",
        "type": "Technical",
        "parameters": {
            "atr_period": 14,
            "atr_multiplier": 3.0,
            "trend_ma": 50
        },
        "trading_rules": {
            "buy_condition": "Price > MA and prior close < trailing stop",
            "sell_condition": "Price < MA and prior close > trailing stop"
        },
        "risk": {
            "position_size": 10,
            "use_atr_for_stop": True,
            "take_profit": 15
        }
    },
    
    # Volume-Based Strategies
    "volume_breakout": {
        "name": "Volume Breakout Strategy",
        "description": "Identifies price breakouts confirmed by abnormally high trading volume.",
        "type": "Technical",
        "parameters": {
            "price_period": 20,
            "volume_period": 20,
            "volume_threshold": 2.0
        },
        "trading_rules": {
            "buy_condition": "Price breaks above range high with volume > threshold",
            "sell_condition": "Price breaks below range low with volume > threshold"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "obv_strategy": {
        "name": "On-Balance Volume Strategy",
        "description": "Uses On-Balance Volume indicator to confirm price trends with volume.",
        "type": "Technical",
        "parameters": {
            "obv_signal_period": 10
        },
        "trading_rules": {
            "buy_condition": "OBV makes new high while price makes new high",
            "sell_condition": "OBV makes new low while price makes new low"
        },
        "risk": {
            "position_size": 12,
            "stop_loss": 4,
            "take_profit": 12
        }
    },
    "vwap_strategy": {
        "name": "VWAP Reversion Strategy",
        "description": "Uses Volume Weighted Average Price as a reference point for intraday mean reversion.",
        "type": "Technical",
        "parameters": {
            "standard_deviation": 2.0,
            "lookback_period": "intraday"
        },
        "trading_rules": {
            "buy_condition": "Price falls below VWAP - (std_dev * VWAP)",
            "sell_condition": "Price rises above VWAP + (std_dev * VWAP)"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 3,
            "take_profit": 6
        }
    },
    
    # Pattern-Based Strategies
    "candlestick_patterns": {
        "name": "Candlestick Pattern Strategy",
        "description": "Identifies common candlestick patterns for potential reversals or continuation signals.",
        "type": "Technical",
        "parameters": {
            "patterns": ["Engulfing", "Doji", "Hammer", "Morning Star", "Evening Star"],
            "confirmation_period": 3
        },
        "trading_rules": {
            "buy_condition": "Bullish pattern forms at support level",
            "sell_condition": "Bearish pattern forms at resistance level"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "harmonic_patterns": {
        "name": "Harmonic Pattern Strategy",
        "description": "Identifies harmonic price patterns (Gartley, Butterfly, Bat) for potential reversals.",
        "type": "Technical",
        "parameters": {
            "patterns": ["Gartley", "Butterfly", "Bat", "Crab", "Cypher"],
            "tolerance": 0.05
        },
        "trading_rules": {
            "buy_condition": "Completion of bullish harmonic pattern",
            "sell_condition": "Completion of bearish harmonic pattern"
        },
        "risk": {
            "position_size": 5,
            "stop_loss": 3,
            "take_profit": 12
        }
    },
    "chart_patterns": {
        "name": "Chart Pattern Strategy",
        "description": "Identifies classical chart patterns for trend continuation or reversal signals.",
        "type": "Technical",
        "parameters": {
            "patterns": ["Head and Shoulders", "Double Top/Bottom", "Triangle", "Flag", "Pennant"],
            "min_pattern_bars": 10
        },
        "trading_rules": {
            "buy_condition": "Bullish pattern breakout",
            "sell_condition": "Bearish pattern breakout"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    
    # Day Trading Strategies
    "scalping_vwap": {
        "name": "VWAP Scalping",
        "description": "Fast day trading strategy that uses VWAP as a reference for quick entries and exits",
        "type": "Intraday",
        "parameters": {
            "vwap_period": "1 day",
            "deviation": 0.15,
            "volume_filter": 1.5
        },
        "trading_rules": {
            "buy_condition": "Price dips below VWAP and bounces with increased volume",
            "sell_condition": "Price rises set percentage above entry or hits time limit"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 0.2,
            "take_profit": 0.5,
            "max_holding_time": "30 minutes"
        }
    },
    "opening_range_breakout": {
        "name": "Opening Range Breakout",
        "description": "Classic day trading strategy that trades breakouts from the opening price range",
        "type": "Intraday",
        "parameters": {
            "range_duration": "15 minutes",
            "breakout_threshold": 0.2,
            "confirmation_time": "2 minutes"
        },
        "trading_rules": {
            "buy_condition": "Price breaks above the defined opening range high",
            "sell_condition": "Price breaks below the defined opening range low or end of session"
        },
        "risk": {
            "position_size": 20,
            "stop_loss": 0.3,
            "take_profit": 1.0,
            "max_holding_time": "Day session"
        }
    },
    "price_action_reversal": {
        "name": "Price Action Reversal",
        "description": "Identifies intraday reversal patterns at support and resistance levels",
        "type": "Intraday",
        "parameters": {
            "lookback_bars": 10,
            "pattern_types": ["Pin Bar", "Engulfing", "Inside Bar"],
            "volume_confirmation": True
        },
        "trading_rules": {
            "buy_condition": "Bullish reversal pattern at support with volume confirmation",
            "sell_condition": "Bearish reversal pattern at resistance with volume confirmation"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 0.4,
            "take_profit": 1.2,
            "max_holding_time": "Day session"
        }
    },
    "momentum_scalping": {
        "name": "Momentum Scalping",
        "description": "Uses short-term momentum indicators to capture quick price moves",
        "type": "Intraday",
        "parameters": {
            "momentum_indicator": "RSI",
            "period": 7,
            "overbought": 75,
            "oversold": 25,
            "volume_filter": True
        },
        "trading_rules": {
            "buy_condition": "RSI crosses above oversold with increasing volume",
            "sell_condition": "RSI crosses above 60 or drops back below entry level"
        },
        "risk": {
            "position_size": 20,
            "stop_loss": 0.3,
            "take_profit": 0.6,
            "trailing_stop": True,
            "max_holding_time": "2 hours"
        }
    },
    "news_gap_trading": {
        "name": "News Gap Trading",
        "description": "Trades price gaps from overnight news or earnings announcements",
        "type": "Intraday",
        "parameters": {
            "gap_size_min": 2.0,
            "wait_time": "15 minutes",
            "volume_surge": 3.0
        },
        "trading_rules": {
            "buy_condition": "Up gap with strong first 15min candle and increasing volume",
            "sell_condition": "Down gap with weak first 15min candle and increasing volume"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 0.5,
            "take_profit": 1.5,
            "max_holding_time": "Day session"
        }
    },
    "support_resistance_bounce": {
        "name": "Support/Resistance Bounce",
        "description": "Day trades bounces off major support and resistance levels",
        "type": "Intraday",
        "parameters": {
            "level_lookback": "20 days",
            "proximity": 0.3,
            "bounce_confirmation": "2 candles"
        },
        "trading_rules": {
            "buy_condition": "Price approaches support and forms bullish pattern",
            "sell_condition": "Price approaches resistance and forms bearish pattern"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 0.4,
            "take_profit": 1.2,
            "max_holding_time": "Day session"
        }
    },
    
    # AI and Hybrid Strategies
    "sentiment_strategy": {
        "name": "News Sentiment",
        "description": "Uses AI to analyze news sentiment and generate trading signals based on market sentiment shifts.",
        "type": "AI-Powered",
        "parameters": {
            "model_type": "FinGPT Sentiment",
            "time_horizon": "Short-term (Days)",
            "signal_threshold": 0.7
        },
        "trading_rules": {
            "buy_condition": "Sentiment score above threshold",
            "sell_condition": "Sentiment score below negative threshold"
        },
        "risk": {
            "position_size": 5,
            "stop_loss": 7,
            "take_profit": 20
        }
    },
    "finrl_strategy": {
        "name": "FinRL Deep RL",
        "description": "Uses reinforcement learning to dynamically adapt to changing market conditions and optimize trading decisions.",
        "type": "AI-Powered",
        "parameters": {
            "model_type": "FinRL",
            "algorithm": "PPO",
            "training_episodes": 100,
            "reward_risk_ratio": 2.0
        },
        "trading_rules": {
            "buy_condition": "RL model returns buy action",
            "sell_condition": "RL model returns sell action"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 6,
            "take_profit": 18
        }
    },
    "combined_strategy": {
        "name": "Technical & Sentiment",
        "description": "Combines technical indicators with sentiment analysis for more robust trading signals.",
        "type": "Hybrid",
        "parameters": {
            "technical_weight": 0.6,
            "sentiment_weight": 0.4,
            "ma_period": 20,
            "rsi_period": 14
        },
        "trading_rules": {
            "buy_condition": "Combined score above threshold",
            "sell_condition": "Combined score below threshold"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "deeptradebot_strategy": {
        "name": "DeepTradeBot Neural Network",
        "description": "Uses deep learning neural networks to predict price movements based on historical patterns.",
        "type": "AI-Powered",
        "parameters": {
            "lookback_period": 50,
            "prediction_period": 10,
            "confidence_threshold": 0.75,
            "network_type": "LSTM"
        },
        "trading_rules": {
            "buy_condition": "Predicted price increase with confidence > threshold",
            "sell_condition": "Predicted price decrease with confidence > threshold"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "ensemble_strategy": {
        "name": "Machine Learning Ensemble",
        "description": "Uses ensemble of multiple machine learning models to generate more reliable signals.",
        "type": "AI-Powered",
        "parameters": {
            "models": ["Random Forest", "SVM", "Gradient Boosting", "Neural Network"],
            "features": 15,
            "voting_method": "Majority"
        },
        "trading_rules": {
            "buy_condition": "Majority of models predict price increase",
            "sell_condition": "Majority of models predict price decrease"
        },
        "risk": {
            "position_size": 5,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    
    # Advanced Statistical Strategies
    "mean_reversion": {
        "name": "Mean Reversion Strategy",
        "description": "Trades on the statistical principle that prices tend to revert to their mean over time.",
        "type": "Statistical",
        "parameters": {
            "ma_period": 50,
            "std_dev_threshold": 2.0,
            "exit_threshold": 0.5
        },
        "trading_rules": {
            "buy_condition": "Price below MA by std_dev_threshold standard deviations",
            "sell_condition": "Price reverts to within exit_threshold standard deviations of MA"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 10
        }
    },
    "pairs_trading": {
        "name": "Pairs Trading Strategy",
        "description": "Exploits the mean-reverting relationship between two correlated assets.",
        "type": "Statistical",
        "parameters": {
            "correlation_threshold": 0.8,
            "lookback_period": 60,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5
        },
        "trading_rules": {
            "buy_condition": "Spread widens beyond entry_z_score standard deviations",
            "sell_condition": "Spread reverts to exit_z_score standard deviations"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "maximum_holding_days": 10
        }
    },
    "volatility_breakout": {
        "name": "Volatility Breakout Strategy",
        "description": "Trades breakouts from periods of low volatility, anticipating strong directional moves.",
        "type": "Statistical",
        "parameters": {
            "volatility_period": 20,
            "volatility_threshold": 0.5,
            "breakout_multiple": 1.5
        },
        "trading_rules": {
            "buy_condition": "Price moves up by breakout_multiple * ATR from low volatility",
            "sell_condition": "Price moves down by breakout_multiple * ATR from low volatility"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "regime_switching": {
        "name": "Market Regime Switching Strategy",
        "description": "Adapts trading strategy based on detected market regime (trending, ranging, volatile).",
        "type": "Adaptive",
        "parameters": {
            "regime_detection_period": 50,
            "volatility_threshold": 1.5,
            "trend_strength_threshold": 25
        },
        "trading_rules": {
            "buy_condition": "Trend regime: Moving Average strategy, Range regime: RSI strategy",
            "sell_condition": "Varies by detected regime"
        },
        "risk": {
            "position_size": 10,
            "dynamic_stop_loss": True,
            "dynamic_take_profit": True
        }
    },
    
    # Indian Market Specific Strategies
    "nifty_bank_nifty_gap": {
        "name": "Nifty/Bank Nifty Gap Strategy",
        "description": "Day trading strategy specifically for Nifty and Bank Nifty indices focusing on gap fills",
        "type": "Indian Market",
        "parameters": {
            "gap_threshold": 0.5,
            "gap_type": "all",
            "pre_market_momentum": True
        },
        "trading_rules": {
            "buy_condition": "Down gap with first 15-minute candle showing bullish momentum",
            "sell_condition": "Up gap with first 15-minute candle showing bearish momentum"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 0.3,
            "take_profit": 0.8,
            "max_holding_time": "3 hours"
        }
    },
    "nse_open_high_low": {
        "name": "NSE Open-High-Low Strategy",
        "description": "Trades based on the relationship between opening price and first hour high/low",
        "type": "Indian Market",
        "parameters": {
            "observation_time": "First 60 minutes",
            "breakout_threshold": 0.2,
            "volume_confirmation": True
        },
        "trading_rules": {
            "buy_condition": "Price breaks above the first hour high with good volume",
            "sell_condition": "Price breaks below the first hour low with good volume"
        },
        "risk": {
            "position_size": 15,
            "stop_loss": 0.4,
            "take_profit": 1.0,
            "max_holding_time": "Day session"
        }
    },
    "india_vix_mean_reversion": {
        "name": "India VIX Mean Reversion",
        "description": "Uses India VIX (Volatility Index) for mean reversion trades on Nifty",
        "type": "Indian Market",
        "parameters": {
            "vix_period": 20,
            "std_dev": 2.0,
            "mean_calculation": "Simple Moving Average"
        },
        "trading_rules": {
            "buy_condition": "India VIX is above 2 standard deviations of its mean (high fear)",
            "sell_condition": "India VIX is below 2 standard deviations of its mean (low fear)"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 0.5,
            "take_profit": 1.5,
            "max_holding_time": "3 days"
        }
    },
    "fii_dii_flow": {
        "name": "FII/DII Flow Strategy",
        "description": "Trades based on Foreign/Domestic Institutional Investor flows in the Indian market",
        "type": "Indian Market",
        "parameters": {
            "flow_threshold": "₹1000 crore",
            "consecutive_days": 3,
            "sector_focus": "Index heavyweights"
        },
        "trading_rules": {
            "buy_condition": "Sustained FII buying for consecutive days above threshold",
            "sell_condition": "Sustained FII selling for consecutive days above threshold"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 0.6,
            "take_profit": 1.8,
            "max_holding_time": "5 days"
        }
    },
    "nse_pre_budget": {
        "name": "NSE Pre-Budget Strategy",
        "description": "Trades NSE stocks based on sector rotation before Indian budget announcements",
        "type": "Indian Market",
        "parameters": {
            "pre_budget_days": 10,
            "sectors": ["Infrastructure", "Banking", "Defense", "FMCG"],
            "momentum_threshold": 5.0
        },
        "trading_rules": {
            "buy_condition": "Sector showing above-threshold momentum in pre-budget period",
            "sell_condition": "Take profit before actual budget announcement"
        },
        "risk": {
            "position_size": 8,
            "stop_loss": 0.7,
            "take_profit": 2.1,
            "max_holding_time": "Until budget day"
        }
    },
    "nse_results_momentum": {
        "name": "NSE Earnings Results Momentum",
        "description": "Trades momentum after quarterly results announcements for NSE stocks",
        "type": "Indian Market",
        "parameters": {
            "results_surprise": ">5%",
            "volume_surge": ">100%",
            "entry_delay": "1 day"
        },
        "trading_rules": {
            "buy_condition": "Positive earnings surprise with gap up and volume surge",
            "sell_condition": "Negative earnings surprise with gap down and volume surge"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 0.5,
            "take_profit": 1.5,
            "trailing_stop": True,
            "max_holding_time": "5 days"
        }
    },
    
    # Options Strategies
    "options_momentum": {
        "name": "Options Momentum Strategy",
        "description": "Uses momentum indicators to time options purchases, particularly calls and puts.",
        "type": "Options",
        "parameters": {
            "momentum_indicator": "RSI",
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30,
            "option_type": "calls/puts",
            "days_to_expiration": 30
        },
        "trading_rules": {
            "buy_condition": "Buy calls when RSI is oversold and turning up",
            "sell_condition": "Buy puts when RSI is overbought and turning down"
        },
        "risk": {
            "position_size": 5,
            "max_loss": 100
        }
    },
    "iron_condor": {
        "name": "Iron Condor Options Strategy",
        "description": "Sells an out-of-the-money put spread and an out-of-the-money call spread to profit from low volatility.",
        "type": "Options",
        "parameters": {
            "iv_rank_threshold": 50,
            "days_to_expiration": 45,
            "delta_for_short_options": 0.16,
            "delta_for_long_options": 0.05,
            "profit_target": 50
        },
        "trading_rules": {
            "entry_condition": "IV Rank > iv_rank_threshold",
            "exit_condition": "Profit target reached or 21 days to expiration"
        },
        "risk": {
            "position_size": 3,
            "max_loss": 300
        }
    },
    
    # Commodity and Forex Specific Strategies
    "commodity_channel_index": {
        "name": "CCI Strategy",
        "description": "Uses the Commodity Channel Index to identify cyclical turns in commodities or currencies.",
        "type": "Commodity/Forex",
        "parameters": {
            "cci_period": 20,
            "overbought": 100,
            "oversold": -100
        },
        "trading_rules": {
            "buy_condition": "CCI crosses above oversold level",
            "sell_condition": "CCI crosses below overbought level"
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        }
    },
    "carry_trade": {
        "name": "Currency Carry Trade",
        "description": "Exploits interest rate differentials between currencies, going long high-yield and short low-yield.",
        "type": "Forex",
        "parameters": {
            "interest_differential_min": 2.0,
            "volatility_threshold": 10.0
        },
        "trading_rules": {
            "buy_condition": "Buy currency with high interest rate vs. low rate",
            "hold_condition": "Interest differential remains favorable" 
        },
        "risk": {
            "position_size": 10,
            "stop_loss": 8,
            "max_correlation": 0.7
        }
    }
}

# Default available symbols
DEFAULT_SYMBOLS = [
    # US Stocks
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "SPY", "QQQ", "JPM", 
    "BAC", "V", "JNJ", "PG", "KO",
    "DIS", "NFLX", "INTC", "AMD", "CSCO",
    
    # Indian NSE Stocks and Indices
    "NIFTY50.NS", "BANKNIFTY.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "HDFC.NS",
    "BAJFINANCE.NS", "SBIN.NS", "AXISBANK.NS", "BHARTIARTL.NS", "LT.NS",
    "ASIANPAINT.NS", "WIPRO.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ULTRACEMCO.NS", "INDUSINDBK.NS",
    "ADANIPORTS.NS", "BPCL.NS", "JSWSTEEL.NS", "HEROMOTOCO.NS", "DRREDDY.NS"
]

def save_strategies(strategies):
    """
    Save strategies to a JSON file
    
    Args:
        strategies: List of strategy dictionaries
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs("streamlit_app/data", exist_ok=True)
        
        # Save to JSON file
        with open("streamlit_app/data/strategies.json", "w") as f:
            json.dump(strategies, f)
            
        return True
    except Exception as e:
        st.error(f"Error saving strategies: {str(e)}")
        return False

def load_strategies():
    """
    Load strategies from JSON file or initialize with defaults if file doesn't exist
    
    Returns:
        List of strategy dictionaries
    """
    # Path to strategies JSON file
    file_path = "streamlit_app/data/strategies.json"
    
    # Check if file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading strategies: {str(e)}")
            return initialize_default_strategies()
    else:
        # Initialize with default strategies
        return initialize_default_strategies()

def initialize_default_strategies():
    """
    Initialize default strategies
    
    Returns:
        List of default strategy dictionaries
    """
    # Create sample strategies
    strategies = [
        {
            "id": 1,
            "name": "Moving Average Crossover",
            "description": "Classic strategy using crossover of fast and slow moving averages to generate buy/sell signals",
            "symbols": ["AAPL", "MSFT"],
            "type": "Technical",
            "status": "Active",
            "parameters": {
                "fast_period": 9,
                "slow_period": 21,
                "ma_type": "Simple"
            },
            "risk": {
                "position_size": 10,
                "stop_loss": 5,
                "take_profit": 15
            },
            "performance": {
                "win_rate": 62,
                "profit_factor": 1.8,
                "sharpe_ratio": 1.4,
                "max_drawdown": 12.5
            }
        },
        {
            "id": 2,
            "name": "RSI Momentum",
            "description": "Identifies overbought and oversold conditions using the Relative Strength Index",
            "symbols": ["SPY"],
            "type": "Technical",
            "status": "Active",
            "parameters": {
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30
            },
            "risk": {
                "position_size": 15,
                "stop_loss": 3,
                "take_profit": 9
            },
            "performance": {
                "win_rate": 58,
                "profit_factor": 1.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": 8.3
            }
        },
        {
            "id": 3,
            "name": "News Sentiment",
            "description": "Uses AI to analyze news sentiment and generate trading signals",
            "symbols": ["TSLA", "NVDA"],
            "type": "AI-Powered",
            "status": "Paused",
            "parameters": {
                "model_type": "FinGPT Sentiment",
                "time_horizon": "Short-term (Days)",
                "signal_threshold": 0.7
            },
            "risk": {
                "position_size": 5,
                "stop_loss": 7,
                "take_profit": 20
            },
            "performance": {
                "win_rate": 67,
                "profit_factor": 2.1,
                "sharpe_ratio": 1.7,
                "max_drawdown": 15.2
            }
        }
    ]
    
    # Save the strategies to file
    save_strategies(strategies)
    
    return strategies

def get_strategy_templates():
    """
    Get available strategy templates
    
    Returns:
        Dictionary of strategy templates
    """
    return STRATEGY_TEMPLATES

def get_strategy_by_id(strategies, strategy_id):
    """
    Get a strategy by ID
    
    Args:
        strategies: List of strategy dictionaries
        strategy_id: ID of the strategy to retrieve
    
    Returns:
        Strategy dictionary or None if not found
    """
    for strategy in strategies:
        if strategy["id"] == strategy_id:
            return strategy
    return None

def get_available_symbols():
    """
    Get available trading symbols
    
    Returns:
        List of available symbols
    """
    # This would typically come from an API call to get available symbols
    # For now, we'll return a default list
    return DEFAULT_SYMBOLS

def create_strategy(strategies, strategy_data):
    """
    Create a new strategy
    
    Args:
        strategies: List of strategy dictionaries
        strategy_data: Dictionary with strategy data
    
    Returns:
        Updated list of strategies
    """
    # Generate a new ID
    new_id = max([s["id"] for s in strategies]) + 1 if strategies else 1
    
    # Generate sample performance metrics
    np_random = np.random.RandomState(new_id)  # Consistent random numbers
    performance = {
        "win_rate": np_random.randint(55, 70),
        "profit_factor": round(np_random.uniform(1.2, 2.2), 1),
        "sharpe_ratio": round(np_random.uniform(0.8, 1.8), 1),
        "max_drawdown": round(np_random.uniform(5, 20), 1)
    }
    
    # Create new strategy with ID and performance
    new_strategy = {
        "id": new_id,
        "status": "Active",
        "performance": performance,
        **strategy_data
    }
    
    # Add to strategies list
    strategies.append(new_strategy)
    
    # Save strategies
    save_strategies(strategies)
    
    return strategies

def update_strategy(strategies, strategy_id, strategy_data):
    """
    Update an existing strategy
    
    Args:
        strategies: List of strategy dictionaries
        strategy_id: ID of the strategy to update
        strategy_data: Dictionary with updated strategy data
    
    Returns:
        Updated list of strategies, success flag, and message
    """
    for i, strategy in enumerate(strategies):
        if strategy["id"] == strategy_id:
            # Keep performance and status from existing strategy
            performance = strategy.get("performance", {})
            status = strategy.get("status", "Active")
            
            # Update the strategy
            strategies[i] = {
                "id": strategy_id,
                "status": status,
                "performance": performance,
                **strategy_data
            }
            
            # Save strategies
            success = save_strategies(strategies)
            
            return strategies, success, f"Strategy '{strategy_data['name']}' updated successfully"
    
    return strategies, False, f"Strategy with ID {strategy_id} not found"

def delete_strategy(strategies, strategy_id):
    """
    Delete a strategy
    
    Args:
        strategies: List of strategy dictionaries
        strategy_id: ID of the strategy to delete
    
    Returns:
        Updated list of strategies, success flag, and message
    """
    for i, strategy in enumerate(strategies):
        if strategy["id"] == strategy_id:
            # Remove the strategy
            del strategies[i]
            
            # Save strategies
            success = save_strategies(strategies)
            
            return strategies, success, f"Strategy deleted successfully"
    
    return strategies, False, f"Strategy with ID {strategy_id} not found"

def update_strategy_status(strategies, strategy_id, new_status):
    """
    Update a strategy's status
    
    Args:
        strategies: List of strategy dictionaries
        strategy_id: ID of the strategy to update
        new_status: New status ('Active' or 'Paused')
    
    Returns:
        Updated list of strategies, success flag, and message
    """
    for i, strategy in enumerate(strategies):
        if strategy["id"] == strategy_id:
            # Update the status
            strategies[i]["status"] = new_status
            
            # Save strategies
            success = save_strategies(strategies)
            
            return strategies, success, f"Strategy status updated to {new_status}"
    
    return strategies, False, f"Strategy with ID {strategy_id} not found"