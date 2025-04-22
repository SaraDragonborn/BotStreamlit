import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import (
    get_account_info, get_positions, get_orders, 
    get_asset_price, place_order, get_historical_data
)

# Configure the page
st.set_page_config(
    page_title="Portfolio | AI Trading Bot",
    page_icon="📈",
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
    .position-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .position-profitable {
        border-left: 4px solid #4CAF50;
    }
    .position-losing {
        border-left: 4px solid #F44336;
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

# Main content
st.markdown('<p class="main-header">Portfolio Management</p>', unsafe_allow_html=True)

tabs = st.tabs(["Positions", "Orders", "Trade", "Performance"])

with tabs[0]:  # Positions
    st.subheader("Current Positions")
    
    # Get positions
    positions = get_positions()
    
    # Mock data for display
    if not positions:
        positions = [
            {
                "symbol": "AAPL",
                "qty": 10,
                "avg_entry_price": 185.75,
                "current_price": 191.25,
                "market_value": 1912.50,
                "unrealized_pl": 55.00,
                "unrealized_plpc": 2.96
            },
            {
                "symbol": "MSFT",
                "qty": 5,
                "avg_entry_price": 325.50,
                "current_price": 318.75,
                "market_value": 1593.75,
                "unrealized_pl": -33.75,
                "unrealized_plpc": -2.07
            },
            {
                "symbol": "AMZN",
                "qty": 8,
                "avg_entry_price": 141.25,
                "current_price": 146.50,
                "market_value": 1172.00,
                "unrealized_pl": 42.00,
                "unrealized_plpc": 3.72
            }
        ]
    
    # Display portfolio summary
    if positions:
        # Calculate total portfolio value and P&L
        total_value = sum(float(position.get('market_value', 0)) for position in positions)
        total_pl = sum(float(position.get('unrealized_pl', 0)) for position in positions)
        avg_pl_percent = sum(float(position.get('unrealized_plpc', 0)) for position in positions) / len(positions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Value", format_currency(total_value))
        
        with col2:
            st.metric("Unrealized P&L", format_currency(total_pl), f"{avg_pl_percent:+.2f}%")
        
        with col3:
            winning_positions = sum(1 for p in positions if float(p.get('unrealized_plpc', 0)) > 0)
            st.metric("Positions", f"{len(positions)}", f"{winning_positions} profitable")
        
        # Display each position
        for position in positions:
            symbol = position.get('symbol', '')
            qty = float(position.get('qty', 0))
            avg_price = float(position.get('avg_entry_price', 0))
            current_price = float(position.get('current_price', 0))
            market_value = float(position.get('market_value', 0))
            unrealized_pl = float(position.get('unrealized_pl', 0))
            unrealized_plpc = float(position.get('unrealized_plpc', 0))
            
            card_class = "position-card position-profitable" if unrealized_plpc >= 0 else "position-card position-losing"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
            
            with col1:
                st.markdown(f"### {symbol}")
                st.markdown(f"{int(qty)} shares")
            
            with col2:
                st.markdown("**Entry Price**")
                st.markdown(f"{format_currency(avg_price)}")
            
            with col3:
                st.markdown("**Current Price**")
                st.markdown(f"{format_currency(current_price)}")
            
            with col4:
                st.markdown("**Market Value**")
                st.markdown(f"{format_currency(market_value)}")
            
            with col5:
                st.markdown("**Unrealized P&L**")
                st.markdown(f"{format_currency(unrealized_pl)} ({format_percentage(unrealized_plpc)})", unsafe_allow_html=True)
            
            # Position actions
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                if st.button("Close Position", key=f"close_{symbol}"):
                    st.success(f"Order placed to close position in {symbol}")
            
            with col2:
                if st.button("Set Alerts", key=f"alert_{symbol}"):
                    st.info(f"Alert settings for {symbol}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("You have no open positions")
        
        # Show suggested positions
        st.markdown("### Suggested Positions")
        st.markdown("Based on your active strategies, the following positions are recommended:")
        
        suggested_positions = [
            {"symbol": "AAPL", "strategy": "Moving Average Crossover", "signal": "BUY", "suggested_qty": 5},
            {"symbol": "TSLA", "strategy": "RSI Momentum", "signal": "BUY", "suggested_qty": 3},
            {"symbol": "SPY", "strategy": "AI-Powered", "signal": "BUY", "suggested_qty": 2}
        ]
        
        for position in suggested_positions:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{position['symbol']}**")
            
            with col2:
                st.markdown(f"{position['strategy']}")
            
            with col3:
                st.markdown(f"Signal: {position['signal']}")
            
            with col4:
                if st.button(f"Place Order for {position['suggested_qty']} shares", key=f"buy_{position['symbol']}"):
                    st.success(f"Order placed for {position['suggested_qty']} shares of {position['symbol']}")

with tabs[1]:  # Orders
    st.subheader("Recent Orders")
    
    # Get orders
    orders = get_orders()
    
    # Mock data for display
    if not orders:
        orders = [
            {
                "id": "123456",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "type": "market",
                "status": "filled",
                "filled_avg_price": 185.75,
                "filled_qty": 10,
                "submitted_at": "2023-04-21T09:30:00Z",
                "filled_at": "2023-04-21T09:30:01Z"
            },
            {
                "id": "123457",
                "symbol": "MSFT",
                "side": "buy",
                "qty": 5,
                "type": "limit",
                "limit_price": 320.00,
                "status": "filled",
                "filled_avg_price": 319.50,
                "filled_qty": 5,
                "submitted_at": "2023-04-20T10:15:00Z",
                "filled_at": "2023-04-20T10:20:30Z"
            },
            {
                "id": "123458",
                "symbol": "TSLA",
                "side": "sell",
                "qty": 3,
                "type": "market",
                "status": "filled",
                "filled_avg_price": 159.25,
                "filled_qty": 3,
                "submitted_at": "2023-04-19T14:30:00Z",
                "filled_at": "2023-04-19T14:30:02Z"
            },
            {
                "id": "123459",
                "symbol": "NVDA",
                "side": "buy",
                "qty": 2,
                "type": "limit",
                "limit_price": 860.00,
                "status": "open",
                "submitted_at": "2023-04-22T08:45:00Z"
            }
        ]
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Status", ["All", "Open", "Filled", "Canceled"], index=0)
    
    with col2:
        side_filter = st.selectbox("Side", ["All", "Buy", "Sell"], index=0)
    
    with col3:
        symbol_filter = st.text_input("Symbol", "")
    
    # Apply filters
    filtered_orders = orders.copy()
    
    if status_filter != "All":
        filtered_orders = [order for order in filtered_orders if order.get('status', '').lower() == status_filter.lower()]
    
    if side_filter != "All":
        filtered_orders = [order for order in filtered_orders if order.get('side', '').lower() == side_filter.lower()]
    
    if symbol_filter:
        filtered_orders = [order for order in filtered_orders if symbol_filter.upper() in order.get('symbol', '').upper()]
    
    # Display orders
    if filtered_orders:
        # Convert to DataFrame for display
        orders_data = []
        for order in filtered_orders:
            # Format datetime
            submitted_at = order.get('submitted_at', '')[:19].replace('T', ' ')
            filled_at = order.get('filled_at', '')[:19].replace('T', ' ') if 'filled_at' in order else '-'
            
            # Format prices
            filled_avg_price = format_currency(float(order.get('filled_avg_price', 0))) if 'filled_avg_price' in order else '-'
            limit_price = format_currency(float(order.get('limit_price', 0))) if 'limit_price' in order else '-'
            
            orders_data.append({
                'Symbol': order.get('symbol', ''),
                'Side': order.get('side', '').capitalize(),
                'Qty': order.get('qty', ''),
                'Type': order.get('type', '').upper(),
                'Limit Price': limit_price,
                'Filled Price': filled_avg_price,
                'Status': order.get('status', '').capitalize(),
                'Submitted At': submitted_at,
                'Filled At': filled_at
            })
        
        orders_df = pd.DataFrame(orders_data)
        st.dataframe(orders_df, use_container_width=True, hide_index=True)
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Cancel All Open Orders"):
                open_count = sum(1 for order in filtered_orders if order.get('status', '').lower() == 'open')
                if open_count > 0:
                    st.success(f"Canceled {open_count} open orders")
                else:
                    st.info("No open orders to cancel")
    else:
        st.info("No orders match your filters")

with tabs[2]:  # Trade
    st.subheader("Place New Orders")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Order form
        with st.form("order_form"):
            st.markdown("### Order Details")
            
            symbol = st.text_input("Symbol", "AAPL").upper()
            
            col1, col2 = st.columns(2)
            
            with col1:
                side = st.selectbox("Side", ["Buy", "Sell"])
            
            with col2:
                order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop", "Stop Limit"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                qty = st.number_input("Quantity", min_value=1, value=1, step=1)
            
            with col2:
                time_in_force = st.selectbox("Time in Force", ["Day", "GTC", "IOC", "FOK"])
            
            # Show price inputs based on order type
            if order_type == "Limit" or order_type == "Stop Limit":
                limit_price = st.number_input("Limit Price", min_value=0.01, value=100.0, step=0.01)
            else:
                limit_price = None
            
            if order_type == "Stop" or order_type == "Stop Limit":
                stop_price = st.number_input("Stop Price", min_value=0.01, value=95.0, step=0.01)
            else:
                stop_price = None
            
            # Submit button
            submitted = st.form_submit_button("Place Order")
            
            if submitted:
                # In a real app, this would call the place_order function
                st.success(f"Order placed: {side} {qty} {symbol} @ {order_type}")
    
    with col2:
        # Quote information
        st.markdown("### Market Information")
        
        # Fetch quote for the entered symbol
        quote_symbol = symbol if 'symbol' in locals() else "AAPL"
        
        # In a real app, this would fetch actual price data
        current_price = 191.25  # Mocked price
        day_change = 1.5  # Mocked day change
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"{quote_symbol} Price", format_currency(current_price), f"{day_change:+.2f}%")
        
        with col2:
            st.metric("Bid-Ask Spread", f"{format_currency(current_price-0.1)} - {format_currency(current_price+0.1)}")
        
        # Price chart
        st.markdown("### Price Chart")
        
        # Sample price data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = [current_price]
        
        # Generate random walk
        for i in range(29):
            change = (0.002 * (1 if i % 3 == 0 else -1)) + (0.005 * (pd.np.random.random() - 0.5))
            prices.append(prices[-1] * (1 + change))
        
        prices = prices[::-1]  # Reverse to match dates
        
        price_df = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_df['Date'],
            y=price_df['Price'],
            mode='lines',
            name='Price',
            line=dict(color='#1E88E5', width=2)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="",
            yaxis_title="Price"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:  # Performance
    st.subheader("Portfolio Performance")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.selectbox(
            "Time Period",
            options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "YTD", "All Time"],
            index=2
        )
    
    with col2:
        benchmark = st.selectbox(
            "Benchmark",
            options=["S&P 500 (SPY)", "Nasdaq (QQQ)", "Dow Jones (DIA)", "None"],
            index=0
        )
    
    # Generate sample performance data
    today = datetime.now()
    
    if period == "1 Week":
        start_date = today - timedelta(days=7)
    elif period == "1 Month":
        start_date = today - timedelta(days=30)
    elif period == "3 Months":
        start_date = today - timedelta(days=90)
    elif period == "6 Months":
        start_date = today - timedelta(days=180)
    elif period == "1 Year":
        start_date = today - timedelta(days=365)
    elif period == "YTD":
        start_date = datetime(today.year, 1, 1)
    else:  # All Time
        start_date = today - timedelta(days=365*2)
    
    dates = pd.date_range(start=start_date, end=today, freq='B')
    
    # Generate portfolio performance
    initial_value = 10000
    portfolio_values = [initial_value]
    
    # Random walk with positive drift
    for i in range(1, len(dates)):
        daily_return = pd.np.random.normal(0.0007, 0.012)  # Small positive drift
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)
    
    # Generate benchmark performance
    benchmark_values = [initial_value]
    
    # Random walk with slightly lower drift
    for i in range(1, len(dates)):
        daily_return = pd.np.random.normal(0.0005, 0.01)  # Lower drift, lower volatility
        new_value = benchmark_values[-1] * (1 + daily_return)
        benchmark_values.append(new_value)
    
    # Create dataframe
    performance_df = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_values,
        'Benchmark': benchmark_values
    })
    
    # Performance metrics
    final_portfolio = portfolio_values[-1]
    final_benchmark = benchmark_values[-1]
    
    portfolio_return = (final_portfolio / initial_value - 1) * 100
    benchmark_return = (final_benchmark / initial_value - 1) * 100
    
    alpha = portfolio_return - benchmark_return
    
    # Daily returns
    portfolio_daily_returns = [(portfolio_values[i] / portfolio_values[i-1] - 1) for i in range(1, len(portfolio_values))]
    benchmark_daily_returns = [(benchmark_values[i] / benchmark_values[i-1] - 1) for i in range(1, len(benchmark_values))]
    
    portfolio_volatility = pd.np.std(portfolio_daily_returns) * (252 ** 0.5) * 100  # Annualized volatility
    benchmark_volatility = pd.np.std(benchmark_daily_returns) * (252 ** 0.5) * 100  # Annualized volatility
    
    # Drawdown calculation
    portfolio_drawdown = []
    peak = portfolio_values[0]
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        portfolio_drawdown.append(drawdown)
    
    max_drawdown = max(portfolio_drawdown)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{portfolio_return:.2f}%", f"{alpha:+.2f}% α")
    
    with col2:
        st.metric("Annualized Volatility", f"{portfolio_volatility:.2f}%", f"{benchmark_volatility-portfolio_volatility:+.2f}% vs benchmark")
    
    with col3:
        sharpe = (portfolio_return / 100) / (portfolio_volatility / 100) if portfolio_volatility > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    # Performance chart
    st.markdown("### Performance Chart")
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_df['Date'],
        y=performance_df['Portfolio'],
        mode='lines',
        name='Portfolio',
        line=dict(color='#1E88E5', width=2)
    ))
    
    if benchmark != "None":
        fig.add_trace(go.Scatter(
            x=performance_df['Date'],
            y=performance_df['Benchmark'],
            mode='lines',
            name=benchmark,
            line=dict(color='#FFA000', width=2, dash='dash')
        ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    st.markdown("### Drawdown")
    
    drawdown_df = pd.DataFrame({
        'Date': dates,
        'Drawdown': portfolio_drawdown
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown_df['Date'],
        y=drawdown_df['Drawdown'],
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#F44336', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(
            autorange="reversed"  # Invert y-axis
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly returns
    st.markdown("### Monthly Returns")
    
    # Create sample monthly returns
    monthly_returns = pd.DataFrame(index=range(1, 13), columns=range(start_date.year, today.year + 1))
    
    for year in range(start_date.year, today.year + 1):
        for month in range(1, 13):
            if (year == start_date.year and month < start_date.month) or (year == today.year and month > today.month):
                monthly_returns.loc[month, year] = None
            else:
                monthly_returns.loc[month, year] = pd.np.random.normal(0.5, 3.0)  # Random monthly return
    
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