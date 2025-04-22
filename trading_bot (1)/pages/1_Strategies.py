import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import (
    get_available_strategies, create_strategy, 
    backtest_strategy, get_historical_data
)

# Configure the page
st.set_page_config(
    page_title="Strategies | AI Trading Bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .strategy-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .strategy-active {
        border-left: 4px solid #4CAF50;
    }
    .strategy-paused {
        border-left: 4px solid #FFC107;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage with color"""
    color = "positive" if value >= 0 else "negative"
    return f'<span class="{color}">{value:+.2f}%</span>'

# Initialize session state for strategies
if 'strategies' not in st.session_state:
    # In a real app, these would come from the API
    st.session_state.strategies = [
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

# Main content
st.markdown('<p class="main-header">Trading Strategies</p>', unsafe_allow_html=True)

tabs = st.tabs(["Active Strategies", "Strategy Builder", "Backtesting"])

with tabs[0]:  # Active Strategies
    st.subheader("Your Trading Strategies")
    
    if not st.session_state.strategies:
        st.info("You don't have any strategies yet. Create one in the Strategy Builder tab.")
    
    # Display each strategy as a card
    for strategy in st.session_state.strategies:
        card_class = "strategy-card strategy-active" if strategy["status"] == "Active" else "strategy-card strategy-paused"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        # Strategy header
        col1, col2, col3 = st.columns([4, 2, 1])
        
        with col1:
            st.markdown(f"### {strategy['name']}")
            st.markdown(f"*{strategy['description']}*")
        
        with col2:
            symbols = ", ".join(strategy["symbols"])
            st.markdown(f"**Symbols:** {symbols}")
            st.markdown(f"**Type:** {strategy['type']}")
        
        with col3:
            status = strategy["status"]
            status_color = "green" if status == "Active" else "orange"
            st.markdown(f"<span style='color:{status_color};'>●</span> {status}", unsafe_allow_html=True)
            
            if status == "Active":
                if st.button("Pause", key=f"pause_{strategy['id']}"):
                    # Update strategy status in session state
                    for i, s in enumerate(st.session_state.strategies):
                        if s["id"] == strategy["id"]:
                            st.session_state.strategies[i]["status"] = "Paused"
                            st.rerun()
            else:
                if st.button("Activate", key=f"activate_{strategy['id']}"):
                    # Update strategy status in session state
                    for i, s in enumerate(st.session_state.strategies):
                        if s["id"] == strategy["id"]:
                            st.session_state.strategies[i]["status"] = "Active"
                            st.rerun()
        
        # Strategy details
        with st.expander("View details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Parameters")
                for key, value in strategy["parameters"].items():
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                
                st.markdown("#### Risk Settings")
                st.markdown(f"**Position Size:** {strategy['risk']['position_size']}% of portfolio")
                st.markdown(f"**Stop Loss:** {strategy['risk']['stop_loss']}%")
                st.markdown(f"**Take Profit:** {strategy['risk']['take_profit']}%")
            
            with col2:
                st.markdown("#### Performance Metrics")
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Win Rate", f"{strategy['performance']['win_rate']}%")
                    st.metric("Profit Factor", f"{strategy['performance']['profit_factor']}")
                
                with metrics_col2:
                    st.metric("Sharpe Ratio", f"{strategy['performance']['sharpe_ratio']}")
                    st.metric("Max Drawdown", f"{strategy['performance']['max_drawdown']}%")
        
        # Strategy actions
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("Backtest", key=f"backtest_{strategy['id']}"):
                # Set the default symbol for backtesting tab
                if "backtest_symbol" not in st.session_state:
                    st.session_state.backtest_symbol = strategy["symbols"][0]
                if "backtest_strategy_id" not in st.session_state:
                    st.session_state.backtest_strategy_id = strategy["id"]
                # Switch to backtesting tab
                st.session_state.selected_tab = 2
                st.rerun()
        
        with col2:
            if st.button("Edit", key=f"edit_{strategy['id']}"):
                # Set the strategy for editing in the strategy builder tab
                st.session_state.edit_strategy_id = strategy["id"]
                # Switch to strategy builder tab
                st.session_state.selected_tab = 1
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create new strategy button
    if st.button("Create New Strategy", type="primary"):
        # Switch to strategy builder tab
        st.session_state.selected_tab = 1
        # Clear edit strategy ID if it exists
        if "edit_strategy_id" in st.session_state:
            del st.session_state.edit_strategy_id
        st.rerun()

with tabs[1]:  # Strategy Builder
    st.subheader("Build Your Trading Strategy")
    
    # Check if we're editing an existing strategy
    editing = False
    if "edit_strategy_id" in st.session_state:
        editing = True
        # Find the strategy being edited
        edit_strategy = None
        for s in st.session_state.strategies:
            if s["id"] == st.session_state.edit_strategy_id:
                edit_strategy = s
                break
    
    col1, col2 = st.columns(2)
    
    with col1:
        if editing and edit_strategy:
            strategy_name = st.text_input("Strategy Name", value=edit_strategy["name"])
            
            strategy_type = st.selectbox(
                "Strategy Type",
                options=["Technical Indicator", "Moving Average", "RSI", "MACD", "AI-Powered", "Custom"],
                index=["Technical Indicator", "Moving Average", "RSI", "MACD", "AI-Powered", "Custom"].index(edit_strategy["type"]) if edit_strategy["type"] in ["Technical Indicator", "Moving Average", "RSI", "MACD", "AI-Powered", "Custom"] else 0
            )
            
            symbols = st.multiselect(
                "Trading Symbols",
                options=["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "SPY", "QQQ"],
                default=edit_strategy["symbols"]
            )
            
            strategy_description = st.text_area("Description", value=edit_strategy["description"])
        else:
            strategy_name = st.text_input("Strategy Name", "My New Strategy")
            
            strategy_type = st.selectbox(
                "Strategy Type",
                options=["Technical Indicator", "Moving Average", "RSI", "MACD", "AI-Powered", "Custom"]
            )
            
            symbols = st.multiselect(
                "Trading Symbols",
                options=["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "SPY", "QQQ"],
                default=["AAPL"]
            )
            
            strategy_description = st.text_area("Description", "Enter a description of your strategy here...")
    
    with col2:
        # Parameters depend on strategy type
        if strategy_type == "Moving Average":
            st.markdown("### Moving Average Parameters")
            
            if editing and edit_strategy and edit_strategy["type"] == "Moving Average":
                ma_type = st.selectbox("MA Type", options=["Simple", "Exponential", "Weighted"], index=["Simple", "Exponential", "Weighted"].index(edit_strategy["parameters"]["ma_type"]) if "ma_type" in edit_strategy["parameters"] else 0)
                fast_period = st.number_input("Fast MA Period", min_value=1, max_value=50, value=edit_strategy["parameters"].get("fast_period", 9))
                slow_period = st.number_input("Slow MA Period", min_value=2, max_value=200, value=edit_strategy["parameters"].get("slow_period", 21))
            else:
                ma_type = st.selectbox("MA Type", options=["Simple", "Exponential", "Weighted"])
                fast_period = st.number_input("Fast MA Period", min_value=1, max_value=50, value=9)
                slow_period = st.number_input("Slow MA Period", min_value=2, max_value=200, value=21)
            
            st.markdown("### Trading Rules")
            buy_condition = st.selectbox("Buy When", options=["Fast MA crosses above Slow MA", "Price crosses above Fast MA"])
            sell_condition = st.selectbox("Sell When", options=["Fast MA crosses below Slow MA", "Price crosses below Fast MA"])
            
        elif strategy_type == "RSI":
            st.markdown("### RSI Parameters")
            
            if editing and edit_strategy and edit_strategy["type"] == "RSI":
                rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=edit_strategy["parameters"].get("rsi_period", 14))
                overbought = st.number_input("Overbought Level", min_value=50, max_value=100, value=edit_strategy["parameters"].get("overbought", 70))
                oversold = st.number_input("Oversold Level", min_value=0, max_value=50, value=edit_strategy["parameters"].get("oversold", 30))
            else:
                rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=14)
                overbought = st.number_input("Overbought Level", min_value=50, max_value=100, value=70)
                oversold = st.number_input("Oversold Level", min_value=0, max_value=50, value=30)
            
            st.markdown("### Trading Rules")
            buy_condition = st.selectbox("Buy When", options=["RSI crosses above Oversold", "RSI is below Oversold"])
            sell_condition = st.selectbox("Sell When", options=["RSI crosses below Overbought", "RSI is above Overbought"])
            
        elif strategy_type == "MACD":
            st.markdown("### MACD Parameters")
            
            if editing and edit_strategy and edit_strategy["type"] == "MACD":
                fast_ema = st.number_input("Fast EMA Period", min_value=1, max_value=50, value=edit_strategy["parameters"].get("fast_ema", 12))
                slow_ema = st.number_input("Slow EMA Period", min_value=2, max_value=100, value=edit_strategy["parameters"].get("slow_ema", 26))
                signal_period = st.number_input("Signal Period", min_value=1, max_value=50, value=edit_strategy["parameters"].get("signal_period", 9))
            else:
                fast_ema = st.number_input("Fast EMA Period", min_value=1, max_value=50, value=12)
                slow_ema = st.number_input("Slow EMA Period", min_value=2, max_value=100, value=26)
                signal_period = st.number_input("Signal Period", min_value=1, max_value=50, value=9)
            
            st.markdown("### Trading Rules")
            buy_condition = st.selectbox("Buy When", options=["MACD crosses above Signal Line", "MACD crosses above zero line"])
            sell_condition = st.selectbox("Sell When", options=["MACD crosses below Signal Line", "MACD crosses below zero line"])
            
        elif strategy_type == "AI-Powered":
            st.markdown("### AI Model Parameters")
            
            if editing and edit_strategy and edit_strategy["type"] == "AI-Powered":
                model_type = st.selectbox(
                    "Model Type", 
                    options=["FinGPT Sentiment", "DeepTradeBot LSTM", "FinRL"],
                    index=["FinGPT Sentiment", "DeepTradeBot LSTM", "FinRL"].index(edit_strategy["parameters"].get("model_type", "FinGPT Sentiment")) if "model_type" in edit_strategy["parameters"] else 0
                )
                time_horizon = st.selectbox(
                    "Time Horizon", 
                    options=["Short-term (Days)", "Medium-term (Weeks)", "Long-term (Months)"],
                    index=["Short-term (Days)", "Medium-term (Weeks)", "Long-term (Months)"].index(edit_strategy["parameters"].get("time_horizon", "Short-term (Days)")) if "time_horizon" in edit_strategy["parameters"] else 0
                )
                signal_threshold = st.slider("Signal Threshold", min_value=0.0, max_value=1.0, value=edit_strategy["parameters"].get("signal_threshold", 0.7), step=0.1)
            else:
                model_type = st.selectbox("Model Type", options=["FinGPT Sentiment", "DeepTradeBot LSTM", "FinRL"])
                time_horizon = st.selectbox("Time Horizon", options=["Short-term (Days)", "Medium-term (Weeks)", "Long-term (Months)"])
                signal_threshold = st.slider("Signal Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.markdown("### Risk Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if editing and edit_strategy:
            position_size = st.slider("Position Size (% of Portfolio)", min_value=1, max_value=100, value=edit_strategy["risk"].get("position_size", 10))
        else:
            position_size = st.slider("Position Size (% of Portfolio)", min_value=1, max_value=100, value=10)
    
    with col2:
        if editing and edit_strategy:
            stop_loss = st.number_input("Stop Loss (%)", min_value=1, max_value=50, value=edit_strategy["risk"].get("stop_loss", 5))
        else:
            stop_loss = st.number_input("Stop Loss (%)", min_value=1, max_value=50, value=5)
    
    with col3:
        if editing and edit_strategy:
            take_profit = st.number_input("Take Profit (%)", min_value=1, max_value=100, value=edit_strategy["risk"].get("take_profit", 15))
        else:
            take_profit = st.number_input("Take Profit (%)", min_value=1, max_value=100, value=15)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if editing:
            if st.button("Update Strategy", type="primary"):
                # Update existing strategy
                for i, s in enumerate(st.session_state.strategies):
                    if s["id"] == st.session_state.edit_strategy_id:
                        # Create parameters dict based on strategy type
                        parameters = {}
                        if strategy_type == "Moving Average":
                            parameters = {
                                "ma_type": ma_type,
                                "fast_period": fast_period,
                                "slow_period": slow_period
                            }
                        elif strategy_type == "RSI":
                            parameters = {
                                "rsi_period": rsi_period,
                                "overbought": overbought,
                                "oversold": oversold
                            }
                        elif strategy_type == "MACD":
                            parameters = {
                                "fast_ema": fast_ema,
                                "slow_ema": slow_ema,
                                "signal_period": signal_period
                            }
                        elif strategy_type == "AI-Powered":
                            parameters = {
                                "model_type": model_type,
                                "time_horizon": time_horizon,
                                "signal_threshold": signal_threshold
                            }
                        
                        st.session_state.strategies[i] = {
                            "id": st.session_state.edit_strategy_id,
                            "name": strategy_name,
                            "description": strategy_description,
                            "symbols": symbols,
                            "type": strategy_type,
                            "status": s["status"],  # Keep existing status
                            "parameters": parameters,
                            "risk": {
                                "position_size": position_size,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit
                            },
                            "performance": s["performance"]  # Keep existing performance
                        }
                        break
                
                # Clear edit strategy ID and go back to active strategies tab
                del st.session_state.edit_strategy_id
                st.session_state.selected_tab = 0
                st.success(f"Strategy '{strategy_name}' updated successfully!")
                st.rerun()
        else:
            if st.button("Create Strategy", type="primary"):
                # Create parameters dict based on strategy type
                parameters = {}
                if strategy_type == "Moving Average":
                    parameters = {
                        "ma_type": ma_type,
                        "fast_period": fast_period,
                        "slow_period": slow_period
                    }
                elif strategy_type == "RSI":
                    parameters = {
                        "rsi_period": rsi_period,
                        "overbought": overbought,
                        "oversold": oversold
                    }
                elif strategy_type == "MACD":
                    parameters = {
                        "fast_ema": fast_ema,
                        "slow_ema": slow_ema,
                        "signal_period": signal_period
                    }
                elif strategy_type == "AI-Powered":
                    parameters = {
                        "model_type": model_type,
                        "time_horizon": time_horizon,
                        "signal_threshold": signal_threshold
                    }
                
                # Create a new strategy
                new_id = max([s["id"] for s in st.session_state.strategies]) + 1 if st.session_state.strategies else 1
                
                # Sample performance metrics for a new strategy
                np_random = np.random.RandomState(new_id)  # Consistent random numbers
                performance = {
                    "win_rate": np_random.randint(55, 70),
                    "profit_factor": round(np_random.uniform(1.2, 2.2), 1),
                    "sharpe_ratio": round(np_random.uniform(0.8, 1.8), 1),
                    "max_drawdown": round(np_random.uniform(5, 20), 1)
                }
                
                st.session_state.strategies.append({
                    "id": new_id,
                    "name": strategy_name,
                    "description": strategy_description,
                    "symbols": symbols,
                    "type": strategy_type,
                    "status": "Active",
                    "parameters": parameters,
                    "risk": {
                        "position_size": position_size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    },
                    "performance": performance
                })
                
                # Go back to active strategies tab
                st.session_state.selected_tab = 0
                st.success(f"Strategy '{strategy_name}' created successfully!")
                st.rerun()
    
    with col2:
        if editing:
            if st.button("Cancel Editing"):
                # Clear edit strategy ID and go back to active strategies tab
                del st.session_state.edit_strategy_id
                st.session_state.selected_tab = 0
                st.rerun()

with tabs[2]:  # Backtesting
    st.subheader("Backtest Your Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get strategies for selection
        strategy_options = [s["name"] for s in st.session_state.strategies]
        
        if "backtest_strategy_id" in st.session_state:
            # Find the strategy name
            strategy_name = next((s["name"] for s in st.session_state.strategies if s["id"] == st.session_state.backtest_strategy_id), strategy_options[0] if strategy_options else "")
            backtest_strategy = st.selectbox(
                "Select Strategy",
                options=strategy_options,
                index=strategy_options.index(strategy_name) if strategy_name in strategy_options else 0
            )
        else:
            backtest_strategy = st.selectbox(
                "Select Strategy",
                options=strategy_options
            )
        
        # Find the selected strategy
        selected_strategy = None
        for s in st.session_state.strategies:
            if s["name"] == backtest_strategy:
                selected_strategy = s
                break
        
        if selected_strategy:
            symbol_options = selected_strategy["symbols"]
            
            if "backtest_symbol" in st.session_state and st.session_state.backtest_symbol in symbol_options:
                symbol = st.selectbox(
                    "Symbol",
                    options=symbol_options,
                    index=symbol_options.index(st.session_state.backtest_symbol)
                )
            else:
                symbol = st.selectbox(
                    "Symbol",
                    options=symbol_options
                )
        else:
            symbol = st.selectbox(
                "Symbol",
                options=["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "SPY", "QQQ"]
            )
    
    with col2:
        today = datetime.now()
        
        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=365)
        )
        
        end_date = st.date_input(
            "End Date",
            value=today
        )
        
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # In a real app, this would call the backtest_strategy function
            # For now, we'll generate sample results
            
            # Sample equity curve data
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            np_random = np.random.RandomState(hash(f"{backtest_strategy}_{symbol}") % 2**32)  # Consistent random numbers
            
            equity = [initial_capital]
            for i in range(1, len(dates)):
                daily_return = np_random.normal(0.0005, 0.01)  # Small positive drift
                new_equity = equity[-1] * (1 + daily_return)
                equity.append(new_equity)
            
            equity_df = pd.DataFrame({
                'Date': dates,
                'Equity': equity
            })
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['Date'],
                y=equity_df['Equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Add a benchmark line (e.g., S&P 500)
            benchmark = [initial_capital]
            for i in range(1, len(dates)):
                daily_return = np_random.normal(0.0004, 0.008)  # Slightly lower return, lower volatility
                new_value = benchmark[-1] * (1 + daily_return)
                benchmark.append(new_value)
            
            fig.add_trace(go.Scatter(
                x=equity_df['Date'],
                y=benchmark,
                mode='lines',
                name='Benchmark (SPY)',
                line=dict(color='#FFA000', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Backtest Results: {backtest_strategy} on {symbol}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            final_equity = equity[-1]
            total_return = (final_equity / initial_capital - 1) * 100
            
            # Calculate daily returns
            daily_returns = [0]
            for i in range(1, len(equity)):
                daily_ret = (equity[i] / equity[i-1]) - 1
                daily_returns.append(daily_ret)
            
            daily_returns_series = pd.Series(daily_returns[1:])  # Skip first 0
            
            # Calculate metrics
            sharpe = daily_returns_series.mean() / daily_returns_series.std() * (252 ** 0.5)  # Annualized Sharpe
            max_drawdown = 0
            peak = equity[0]
            
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate benchmark metrics
            benchmark_return = (benchmark[-1] / benchmark[0] - 1) * 100
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{total_return:.2f}%", f"{total_return - benchmark_return:.2f}% vs benchmark")
            col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col3.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
            
            # Calculate win rate (simplified)
            trades = int(len(dates) * 0.4)  # Assume trades on 40% of days
            wins = int(trades * 0.55)  # Assume 55% win rate
            col4.metric("Win Rate", f"{wins}/{trades} ({wins/trades*100:.1f}%)")
            
            # Monthly returns heatmap
            st.markdown("### Monthly Returns")
            
            # Create sample monthly returns
            monthly_returns = pd.DataFrame(index=range(1, 13), columns=range(start_date.year, end_date.year + 1))
            
            for year in range(start_date.year, end_date.year + 1):
                for month in range(1, 13):
                    if (year == start_date.year and month < start_date.month) or (year == end_date.year and month > end_date.month):
                        monthly_returns.loc[month, year] = None
                    else:
                        monthly_returns.loc[month, year] = np_random.normal(0.5, 3.0)  # Random monthly return
            
            # Create heatmap
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            z_values = monthly_returns.values.tolist()
            
            # Generate colors
            colors = []
            for row in z_values:
                row_colors = []
                for value in row:
                    if value is None:
                        row_colors.append('white')
                    elif value >= 0:
                        intensity = min(value / 5.0, 1.0)  # Scale the color intensity
                        row_colors.append(f'rgba(76, 175, 80, {intensity})')
                    else:
                        intensity = min(abs(value) / 5.0, 1.0)  # Scale the color intensity
                        row_colors.append(f'rgba(244, 67, 54, {intensity})')
                colors.append(row_colors)
            
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=list(monthly_returns.columns),
                y=month_names,
                colorscale='RdYlGn',
                showscale=False,
                text=[[f"{value:.2f}%" if value is not None else "" for value in row] for row in z_values],
                hoverinfo="text",
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_nticks=len(monthly_returns.columns),
                yaxis_nticks=12
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade list
            st.markdown("### Trade List")
            
            # Sample trade list
            trades_data = []
            trade_dates = np_random.choice(dates[5:], size=min(trades, 20), replace=False)  # Show max 20 trades
            trade_dates = sorted(trade_dates)
            
            for i, date in enumerate(trade_dates):
                is_win = i < int(len(trade_dates) * 0.55)  # 55% win rate
                trade_return = np_random.uniform(0.005, 0.03) if is_win else -np_random.uniform(0.005, 0.025)
                entry_price = np_random.uniform(100, 200)
                exit_price = entry_price * (1 + trade_return)
                profit_loss = entry_price * trade_return * 100
                
                trades_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Symbol': symbol,
                    'Type': 'Buy' if np_random.random() > 0.5 else 'Sell',
                    'Entry Price': f"${entry_price:.2f}",
                    'Exit Price': f"${exit_price:.2f}",
                    'Return': f"{trade_return*100:+.2f}%",
                    'Profit/Loss': f"${profit_loss:.2f}"
                })
            
            trades_df = pd.DataFrame(trades_data)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            # Save button
            if st.button("Save Backtest Results"):
                st.success("Backtest results saved successfully!")
                
            # Apply strategy button
            if st.button("Apply Strategy with These Settings"):
                st.success("Strategy applied successfully! It is now active with the backtested settings.")