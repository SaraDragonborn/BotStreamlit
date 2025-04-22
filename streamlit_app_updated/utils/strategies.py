import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Strategy template definitions
STRATEGY_TEMPLATES = {
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
    }
}

# Default available symbols
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "SPY", "QQQ", "JPM", 
    "BAC", "V", "JNJ", "PG", "KO",
    "DIS", "NFLX", "INTC", "AMD", "CSCO"
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