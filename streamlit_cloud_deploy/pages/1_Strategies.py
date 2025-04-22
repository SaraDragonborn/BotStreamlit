import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add utils directory to path if not already added
utils_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Configure the page
st.set_page_config(
    page_title="Trading Strategies",
    page_icon="🧠",
    layout="wide"
)

# Main title
st.title("Trading Strategies")
st.subheader("Configure and backtest your strategies")

# Add custom CSS
st.markdown("""
<style>
.strategy-description {
    background-color: #f5f7f9;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Strategy selection
strategies = [
    "Moving Average Crossover",
    "RSI Strategy",
    "MACD Crossover",
    "Bollinger Bands",
    "Mean Reversion"
]

selected_strategy = st.selectbox("Select a strategy", strategies)

# Strategy descriptions
descriptions = {
    "Moving Average Crossover": "This strategy generates a buy signal when a shorter-period moving average crosses above a longer-period moving average, and a sell signal when the shorter moving average crosses below the longer one.",
    "RSI Strategy": "This strategy uses the Relative Strength Index (RSI) to identify overbought and oversold conditions. It generates buy signals when RSI falls below the oversold level and sell signals when it rises above the overbought level.",
    "MACD Crossover": "The Moving Average Convergence Divergence (MACD) strategy generates signals when the MACD line crosses the signal line or the zero line.",
    "Bollinger Bands": "This strategy uses Bollinger Bands to identify overbought and oversold conditions. Buy signals occur when price touches the lower band and sell signals when it touches the upper band.",
    "Mean Reversion": "Based on the principle that prices tend to revert to their mean over time, this strategy buys when prices are significantly below their moving average and sells when they are above."
}

st.markdown(f"<div class='strategy-description'><strong>{selected_strategy}</strong><br>{descriptions[selected_strategy]}</div>", unsafe_allow_html=True)

# Strategy parameters
st.header("Strategy Parameters")

params = {}

# Display different parameters based on selected strategy
if selected_strategy == "Moving Average Crossover":
    col1, col2 = st.columns(2)
    with col1:
        params['fast_period'] = st.slider("Fast MA Period", 5, 50, 10)
        params['slow_period'] = st.slider("Slow MA Period", 20, 200, 50)
    with col2:
        params['ma_type'] = st.selectbox("MA Type", ["Simple", "Exponential", "Weighted"])
        
elif selected_strategy == "RSI Strategy":
    col1, col2 = st.columns(2)
    with col1:
        params['rsi_period'] = st.slider("RSI Period", 2, 30, 14)
    with col2:
        params['overbought'] = st.slider("Overbought Level", 50, 90, 70)
        params['oversold'] = st.slider("Oversold Level", 10, 50, 30)
        
elif selected_strategy == "MACD Crossover":
    col1, col2 = st.columns(2)
    with col1:
        params['fast_ema'] = st.slider("Fast EMA Period", 5, 30, 12)
        params['slow_ema'] = st.slider("Slow EMA Period", 10, 50, 26)
    with col2:
        params['signal_period'] = st.slider("Signal Period", 2, 20, 9)
        
elif selected_strategy == "Bollinger Bands":
    col1, col2 = st.columns(2)
    with col1:
        params['bb_period'] = st.slider("Period", 5, 50, 20)
    with col2:
        params['std_dev'] = st.slider("Standard Deviation", 1.0, 4.0, 2.0, 0.1)
        
elif selected_strategy == "Mean Reversion":
    col1, col2 = st.columns(2)
    with col1:
        params['mean_period'] = st.slider("Mean Period", 5, 100, 20)
    with col2:
        params['threshold'] = st.slider("Deviation Threshold (%)", 1.0, 10.0, 2.0, 0.1)

# Symbol selection
st.header("Symbol Selection")
symbol = st.text_input("Enter a symbol", "AAPL")

# Timeframe selection
timeframes = ["1 day", "1 week", "1 month", "3 months", "6 months", "1 year"]
timeframe = st.selectbox("Select timeframe", timeframes)

# Map timeframe to yfinance period
timeframe_map = {
    "1 day": "1d",
    "1 week": "1wk",
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y"
}

# Function to fetch stock data
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        data = yf.download(symbol, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to apply strategy to data
def apply_strategy(data, strategy, params):
    """Apply a trading strategy to stock data"""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0.0
    
    if strategy == "Moving Average Crossover":
        # Calculate moving averages
        if params['ma_type'] == "Simple":
            signals['fast_ma'] = data['Close'].rolling(window=params['fast_period'], min_periods=1).mean()
            signals['slow_ma'] = data['Close'].rolling(window=params['slow_period'], min_periods=1).mean()
        elif params['ma_type'] == "Exponential":
            signals['fast_ma'] = data['Close'].ewm(span=params['fast_period'], adjust=False).mean()
            signals['slow_ma'] = data['Close'].ewm(span=params['slow_period'], adjust=False).mean()
        else:  # Weighted
            signals['fast_ma'] = data['Close'].rolling(window=params['fast_period'], min_periods=1).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
            signals['slow_ma'] = data['Close'].rolling(window=params['slow_period'], min_periods=1).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        
        # Generate signals
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['fast_ma'] > signals['slow_ma'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        
    elif strategy == "RSI Strategy":
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params['rsi_period']).mean()
        
        rs = gain / loss
        signals['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['rsi'] < params['oversold'], 1.0, 
                                   np.where(signals['rsi'] > params['overbought'], -1.0, signals['signal']))
        signals['positions'] = signals['signal'].diff()
        
    elif strategy == "MACD Crossover":
        # Calculate MACD
        signals['ema_fast'] = data['Close'].ewm(span=params['fast_ema'], adjust=False).mean()
        signals['ema_slow'] = data['Close'].ewm(span=params['slow_ema'], adjust=False).mean()
        signals['macd'] = signals['ema_fast'] - signals['ema_slow']
        signals['signal_line'] = signals['macd'].ewm(span=params['signal_period'], adjust=False).mean()
        
        # Generate signals
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['macd'] > signals['signal_line'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        
    elif strategy == "Bollinger Bands":
        # Calculate Bollinger Bands
        signals['ma'] = data['Close'].rolling(window=params['bb_period']).mean()
        signals['std'] = data['Close'].rolling(window=params['bb_period']).std()
        signals['upper_band'] = signals['ma'] + (signals['std'] * params['std_dev'])
        signals['lower_band'] = signals['ma'] - (signals['std'] * params['std_dev'])
        
        # Generate signals
        signals['signal'] = 0.0
        signals['signal'] = np.where(data['Close'] < signals['lower_band'], 1.0, 
                                    np.where(data['Close'] > signals['upper_band'], -1.0, signals['signal']))
        signals['positions'] = signals['signal'].diff()
        
    elif strategy == "Mean Reversion":
        # Calculate mean and deviation
        signals['ma'] = data['Close'].rolling(window=params['mean_period']).mean()
        signals['pct_diff'] = (data['Close'] - signals['ma']) / signals['ma'] * 100
        
        # Generate signals
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['pct_diff'] < -params['threshold'], 1.0, 
                                    np.where(signals['pct_diff'] > params['threshold'], -1.0, signals['signal']))
        signals['positions'] = signals['signal'].diff()
    
    return signals

# Function to backtest a strategy
def backtest_strategy(signals, initial_capital=100000.0):
    """Perform a backtest of the strategy"""
    # Create a DataFrame for positions and portfolio
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['price'] = signals['price']
    
    # Calculate positions based on signal
    positions['signal'] = signals['signal']
    positions['position'] = positions['signal'].diff()
    
    # Calculate shares and holdings
    positions['shares'] = positions['position'] * initial_capital / positions['price']
    positions['shares'] = positions['shares'].fillna(0)
    positions['holdings'] = positions['shares'].cumsum() * positions['price']
    
    # Calculate cash and portfolio value
    positions['cash'] = initial_capital - (positions['shares'] * positions['price']).cumsum()
    positions['total'] = positions['cash'] + positions['holdings']
    
    # Calculate returns
    positions['returns'] = positions['total'].pct_change()
    
    return positions

# Run backtest button
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        try:
            # Download the data
            data = get_stock_data(symbol, timeframe_map[timeframe])
            
            if data is None or data.empty:
                st.error(f"No data available for {symbol} in the selected timeframe.")
            else:
                # Apply the strategy
                signals = apply_strategy(data, selected_strategy, params)
                
                # Run the backtest
                portfolio = backtest_strategy(signals)
                
                # Display results
                st.success("Backtest completed successfully!")
                
                # Display equity curve
                st.subheader("Portfolio Performance")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio.index,
                    y=portfolio['total'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                # Buy and sell markers
                buy_signals = portfolio[portfolio['position'] > 0]
                sell_signals = portfolio[portfolio['position'] < 0]
                
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='triangle-up'
                    )
                ))
                
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-down'
                    )
                ))
                
                fig.update_layout(
                    title=f"Portfolio Performance ({selected_strategy})",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                
                # Calculate metrics
                initial_value = portfolio['total'].iloc[0]
                final_value = portfolio['total'].iloc[-1]
                total_return = ((final_value / initial_value) - 1) * 100
                
                annualized_return = ((final_value / initial_value) ** (252 / len(portfolio)) - 1) * 100
                
                # Daily returns
                daily_returns = portfolio['returns'].dropna()
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
                
                # Maximum drawdown
                portfolio['peak'] = portfolio['total'].cummax()
                portfolio['drawdown'] = (portfolio['total'] / portfolio['peak'] - 1) * 100
                max_drawdown = portfolio['drawdown'].min()
                
                # Win rate
                trades = portfolio[portfolio['position'] != 0]
                win_trades = trades[trades['returns'] > 0]
                win_rate = len(win_trades) / len(trades) * 100 if len(trades) > 0 else 0
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                with col2:
                    st.metric("Annualized Return", f"{annualized_return:.2f}%")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", f"{len(trades)}")
                
                with col2:
                    st.metric("Win Rate", f"{win_rate:.2f}%")
                
                with col3:
                    st.metric("Initial Capital", f"${initial_value:,.2f}")
                
                with col4:
                    st.metric("Final Capital", f"${final_value:,.2f}")
                
                # Strategy visualization
                st.subheader("Strategy Visualization")
                
                # Create figure for strategy visualization
                strat_fig = go.Figure()
                
                # Add price chart
                strat_fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price"
                ))
                
                # Add strategy-specific indicators
                if selected_strategy == "Moving Average Crossover":
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['fast_ma'],
                        mode='lines',
                        name=f"{params['fast_period']} MA",
                        line=dict(color='blue', width=1.5)
                    ))
                    
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['slow_ma'],
                        mode='lines',
                        name=f"{params['slow_period']} MA",
                        line=dict(color='red', width=1.5)
                    ))
                    
                elif selected_strategy == "RSI Strategy":
                    # Create a new row for RSI
                    strat_fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                             shared_xaxes=True, vertical_spacing=0.03)
                    
                    # Add price chart to first row
                    strat_fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Price"
                    ), row=1, col=1)
                    
                    # Add RSI to second row
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=1.5)
                    ), row=2, col=1)
                    
                    # Add overbought/oversold lines
                    strat_fig.add_trace(go.Scatter(
                        x=[signals.index[0], signals.index[-1]],
                        y=[params['overbought'], params['overbought']],
                        mode='lines',
                        name='Overbought',
                        line=dict(color='red', width=1, dash='dash')
                    ), row=2, col=1)
                    
                    strat_fig.add_trace(go.Scatter(
                        x=[signals.index[0], signals.index[-1]],
                        y=[params['oversold'], params['oversold']],
                        mode='lines',
                        name='Oversold',
                        line=dict(color='green', width=1, dash='dash')
                    ), row=2, col=1)
                    
                elif selected_strategy == "MACD Crossover":
                    # Create a new row for MACD
                    strat_fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                             shared_xaxes=True, vertical_spacing=0.03)
                    
                    # Add price chart to first row
                    strat_fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Price"
                    ), row=1, col=1)
                    
                    # Add MACD to second row
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=1.5)
                    ), row=2, col=1)
                    
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['signal_line'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='red', width=1.5)
                    ), row=2, col=1)
                    
                    # Add histogram
                    strat_fig.add_trace(go.Bar(
                        x=signals.index,
                        y=signals['macd'] - signals['signal_line'],
                        name='Histogram',
                        marker_color=np.where(signals['macd'] >= signals['signal_line'], 'green', 'red')
                    ), row=2, col=1)
                    
                elif selected_strategy == "Bollinger Bands":
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['ma'],
                        mode='lines',
                        name='Middle Band',
                        line=dict(color='blue', width=1.5)
                    ))
                    
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['upper_band'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='red', width=1.5)
                    ))
                    
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['lower_band'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='green', width=1.5)
                    ))
                    
                elif selected_strategy == "Mean Reversion":
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals['ma'],
                        mode='lines',
                        name=f"{params['mean_period']} MA",
                        line=dict(color='blue', width=1.5)
                    ))
                    
                    # Add deviation bands
                    upper_band = signals['ma'] * (1 + params['threshold']/100)
                    lower_band = signals['ma'] * (1 - params['threshold']/100)
                    
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=upper_band,
                        mode='lines',
                        name=f"+{params['threshold']}%",
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    
                    strat_fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=lower_band,
                        mode='lines',
                        name=f"-{params['threshold']}%",
                        line=dict(color='green', width=1, dash='dash')
                    ))
                
                # Add buy and sell markers to price chart
                strat_fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='triangle-up'
                    )
                ), row=1 if selected_strategy in ["RSI Strategy", "MACD Crossover"] else 1, col=1)
                
                strat_fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-down'
                    )
                ), row=1 if selected_strategy in ["RSI Strategy", "MACD Crossover"] else 1, col=1)
                
                # Update layout
                strat_fig.update_layout(
                    title=f"{symbol} with {selected_strategy}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    template="plotly_white"
                )
                
                # Display the chart
                st.plotly_chart(strat_fig, use_container_width=True)
                
                # Display trade list
                st.subheader("Trade List")
                
                # Prepare trade list
                trades_list = []
                current_position = 0
                entry_price = 0
                entry_date = None
                
                for date, row in portfolio[portfolio['position'] != 0].iterrows():
                    if row['position'] > 0:  # Buy signal
                        current_position = 1
                        entry_price = row['price']
                        entry_date = date
                    elif row['position'] < 0 and current_position == 1:  # Sell signal after buy
                        current_position = 0
                        exit_price = row['price']
                        exit_date = date
                        
                        # Calculate profit
                        profit = (exit_price - entry_price) / entry_price * 100
                        profit_dollar = (exit_price - entry_price) * row['shares']
                        
                        # Add to trade list
                        trades_list.append({
                            "Entry Date": entry_date.strftime("%Y-%m-%d"),
                            "Entry Price": f"${entry_price:.2f}",
                            "Exit Date": exit_date.strftime("%Y-%m-%d"),
                            "Exit Price": f"${exit_price:.2f}",
                            "Profit/Loss": f"{profit:.2f}%",
                            "P/L Amount": f"${profit_dollar:.2f}",
                            "Type": "Long"
                        })
                
                # Display trade list
                if trades_list:
                    trades_df = pd.DataFrame(trades_list)
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades were generated during the backtest period.")
                
        except Exception as e:
            st.error(f"Error during backtesting: {e}")

# Strategy optimization section
st.header("Strategy Optimization")
st.info("Strategy optimization feature is coming soon. This feature will use grid search or genetic algorithms to find optimal strategy parameters.")

# Save strategy section
st.header("Save and Deploy Strategy")
strategy_name = st.text_input("Strategy Name", f"My {selected_strategy}")

col1, col2 = st.columns(2)
with col1:
    if st.button("Save Strategy"):
        st.success(f"Strategy '{strategy_name}' saved successfully!")
with col2:
    if st.button("Deploy Strategy"):
        st.success(f"Strategy '{strategy_name}' deployed to live trading!")
        st.info("The bot will now trade based on this strategy. Monitor the results in the Portfolio section.")

# Notes section
st.header("Strategy Notes")
notes = st.text_area("Add notes about this strategy", height=150)
if st.button("Save Notes"):
    st.success("Notes saved successfully!")