import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import json
from datetime import datetime, timedelta

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fixed_api import (
    get_account_info, get_positions, get_orders
)
from utils.api import (
    get_asset_price, place_order, get_historical_data
)
from utils.stock_info import display_stock_info, get_stock_info
from utils.stock_statistics import (
    get_stock_statistics, batch_get_statistics, 
    create_statistics_df, format_statistics_df
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

tabs = st.tabs(["Positions", "Orders", "Trade", "Watchlist", "Performance"])

with tabs[0]:  # Positions
    st.subheader("Current Positions")
    
    # Get positions
    positions = get_positions()
    
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
        st.markdown("Check the Signals Dashboard for trading opportunities based on your active strategies.")
        
        # Call to action to view signals
        st.markdown("#### Next Steps:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Signal Dashboard", key="goto_signals"):
                st.switch_page("pages/5_Signals_Dashboard.py")
        with col2:
            if st.button("Configure Trading Strategies", key="goto_strategies"):
                st.switch_page("pages/1_Strategies.py")

with tabs[1]:  # Orders
    st.subheader("Recent Orders")
    
    # Get orders
    orders = get_orders()
    
    # Empty by default
    if not orders:
        orders = []
    
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
            
            # Display stock info when a symbol is entered and form is not yet submitted
            if symbol:
                stock_info_placeholder = st.empty()
                with stock_info_placeholder:
                    stock_info = get_stock_info(symbol)
                    if stock_info:
                        st.markdown(f"""
                        **Company:** {stock_info['name']}  
                        **Exchange:** {stock_info['exchange']}  
                        **Current Price:** ${stock_info['current_price']:.2f} ({stock_info['day_change']:+.2f}%)
                        """)
                    else:
                        st.warning(f"No information found for symbol: {symbol}")
            
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
        
        # Get live stock info
        stock_info = get_stock_info(quote_symbol)
        
        if stock_info:
            current_price = stock_info['current_price']
            day_change = stock_info['day_change']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{quote_symbol} Price", format_currency(current_price), f"{day_change:+.2f}%")
            
            with col2:
                # Estimate bid-ask spread
                bid_price = current_price * 0.999  # Estimate 0.1% spread
                ask_price = current_price * 1.001
                st.metric("Bid-Ask Spread", f"{format_currency(bid_price)} - {format_currency(ask_price)}")
                
            # Show additional info
            st.markdown(f"""
            **Company:** {stock_info['name']}  
            **Exchange:** {stock_info['exchange']}  
            **Sector:** {stock_info.get('sector', 'N/A')}
            """)
        else:
            # Fallback if we can't get info
            st.info(f"Unable to retrieve current market data for {quote_symbol}")
        
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

with tabs[3]:  # Watchlist
    st.subheader("Stock Watchlist")
    
    # Add description
    st.markdown("""
    Monitor up to 100 stocks with comprehensive statistics. Add symbols to your watchlist to track important metrics 
    like price, day range, volume, technical indicators, and more.
    """)
    
    # Load existing watchlist from session state
    if 'watchlist_symbols' not in st.session_state:
        st.session_state.watchlist_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    if 'watchlist_exchanges' not in st.session_state:
        st.session_state.watchlist_exchanges = ['US', 'US', 'US', 'US', 'US']
    
    if 'watchlist_data' not in st.session_state:
        st.session_state.watchlist_data = []
    
    # Input for adding new symbols
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        new_symbol = st.text_input("Add Symbol to Watchlist", placeholder="Enter symbol (e.g., AAPL, HDFC-EQ)")
    
    with col2:
        exchange = st.selectbox(
            "Exchange", 
            options=["US", "NSE", "BSE"],
            index=0
        )
    
    with col3:
        if st.button("Add Symbol") and new_symbol:
            new_symbol = new_symbol.strip().upper()
            if new_symbol not in st.session_state.watchlist_symbols:
                st.session_state.watchlist_symbols.append(new_symbol)
                st.session_state.watchlist_exchanges.append(exchange)
                st.success(f"Added {new_symbol} to watchlist")
            else:
                st.warning(f"{new_symbol} is already in your watchlist")
    
    # Bulk add option
    with st.expander("Bulk Add Symbols"):
        bulk_symbols = st.text_area(
            "Enter multiple symbols (one per line)",
            placeholder="AAPL\nMSFT\nGOOGL"
        )
        bulk_exchange = st.selectbox(
            "Exchange for all symbols",
            options=["US", "NSE", "BSE"],
            index=0,
            key="bulk_exchange"
        )
        
        if st.button("Add All Symbols"):
            if bulk_symbols:
                symbols_list = [s.strip().upper() for s in bulk_symbols.split('\n') if s.strip()]
                new_count = 0
                for symbol in symbols_list:
                    if symbol and symbol not in st.session_state.watchlist_symbols:
                        st.session_state.watchlist_symbols.append(symbol)
                        st.session_state.watchlist_exchanges.append(bulk_exchange)
                        new_count += 1
                
                if new_count > 0:
                    st.success(f"Added {new_count} new symbols to your watchlist")
                else:
                    st.info("No new symbols were added (they may already be in your watchlist)")
    
    # Display the watchlist
    if st.session_state.watchlist_symbols:
        # Refresh button
        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col1:
            st.write(f"Tracking {len(st.session_state.watchlist_symbols)} symbols")
        with refresh_col2:
            if st.button("Refresh Data"):
                # Clear cached data to force refresh
                st.session_state.watchlist_data = []
                st.info("Refreshing stock data...")
        
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            filter_exchange = st.multiselect(
                "Filter by Exchange",
                options=["US", "NSE", "BSE", "All"],
                default=["All"]
            )
        
        with filter_col2:
            sort_by = st.selectbox(
                "Sort By",
                options=[
                    "Symbol", "Price", "Day Change %", "Volume", "Market Cap",
                    "Volatility", "RSI", "52W High", "52W Low"
                ],
                index=0
            )
        
        with filter_col3:
            sort_order = st.selectbox(
                "Order",
                options=["Ascending", "Descending"],
                index=1
            )
        
        # Fetch stock data if not already fetched
        if not st.session_state.watchlist_data and st.session_state.watchlist_symbols:
            with st.spinner("Fetching stock data..."):
                st.session_state.watchlist_data = batch_get_statistics(
                    st.session_state.watchlist_symbols, 
                    st.session_state.watchlist_exchanges
                )
        
        # Create and format dataframe
        if st.session_state.watchlist_data:
            # Create raw dataframe
            df = create_statistics_df(st.session_state.watchlist_data)
            
            # Apply exchange filter
            if "All" not in filter_exchange and df is not None and not df.empty:
                df = df[df['exchange'].isin(filter_exchange)]
            
            # Apply sorting
            if df is not None and not df.empty:
                sort_column_map = {
                    "Symbol": "symbol",
                    "Price": "current_price",
                    "Day Change %": "day_change",
                    "Volume": "volume",
                    "Market Cap": "market_cap",
                    "Volatility": "volatility",
                    "RSI": "rsi",
                    "52W High": "52w_high",
                    "52W Low": "52w_low"
                }
                
                sort_col = sort_column_map.get(sort_by, "symbol")
                ascending = sort_order == "Ascending"
                
                if sort_col in df.columns:
                    df = df.sort_values(by=sort_col, ascending=ascending)
            
            # Display the dataframe
            if df is not None and not df.empty:
                # Calculate day high and low (as a percentage of current price)
                if 'current_price' in df.columns and 'atr_20' in df.columns:
                    # Use ATR to estimate day range when actual high/low not available
                    df['day_high'] = df['current_price'] * (1 + df['atr_20'] / (2 * df['current_price']))
                    df['day_low'] = df['current_price'] * (1 - df['atr_20'] / (2 * df['current_price']))
                
                # Show a more detailed table with expandable rows
                with st.expander("View Full Statistics", expanded=True):
                    # Select columns to display prominently
                    display_cols = [
                        'symbol', 'name', 'current_price', 'day_change',
                        'day_high', 'day_low', 'volume', 'market_cap', 
                        'rsi', 'volatility', '52w_high', '52w_low'
                    ]
                    
                    # Select only columns that exist in the dataframe
                    display_cols = [col for col in display_cols if col in df.columns]
                    
                    # Create a formatted dataframe for display
                    display_df = format_statistics_df(df[display_cols])
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Actions menu for each stock
                col1, col2 = st.columns(2)
                
                with col1:
                    symbol_to_trade = st.selectbox(
                        "Select Symbol to Trade",
                        options=df['symbol'].tolist()
                    )
                    
                    stock_data = df[df['symbol'] == symbol_to_trade].iloc[0].to_dict() if not df.empty else {}
                    
                    if stock_data:
                        st.markdown(f"""
                        ### {stock_data.get('name', symbol_to_trade)} ({symbol_to_trade})
                        **Current Price:** ${stock_data.get('current_price', 0):.2f}  
                        **Day Range:** ${stock_data.get('day_low', 0):.2f} - ${stock_data.get('day_high', 0):.2f}  
                        **Volume:** {stock_data.get('volume', 0):,.0f}  
                        **Market Cap:** ${stock_data.get('market_cap', 0)/1e9:.2f}B
                        """)
                
                with col2:
                    st.markdown("### Quick Actions")
                    
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("Trade", key=f"trade_{symbol_to_trade}"):
                            # Switch to Trade tab and pre-fill symbol
                            st.session_state.selected_symbol = symbol_to_trade
                            st.experimental_set_query_params(tab="Trade", symbol=symbol_to_trade)
                            st.experimental_rerun()
                    
                    with action_col2:
                        if st.button("Analyze", key=f"analyze_{symbol_to_trade}"):
                            # Switch to AI Analysis page
                            st.session_state.analysis_symbol = symbol_to_trade
                            st.switch_page("pages/3_AI_Analysis.py")
                    
                    with action_col3:
                        if st.button("Remove", key=f"remove_{symbol_to_trade}"):
                            # Remove the symbol from watchlist
                            idx = st.session_state.watchlist_symbols.index(symbol_to_trade)
                            st.session_state.watchlist_symbols.pop(idx)
                            st.session_state.watchlist_exchanges.pop(idx)
                            st.session_state.watchlist_data = [d for d in st.session_state.watchlist_data if d.get('symbol') != symbol_to_trade]
                            st.success(f"Removed {symbol_to_trade} from watchlist")
                            st.experimental_rerun()
                
                # Technical Indicators Summary
                st.markdown("### Technical Indicators Summary")
                
                indicators_cols = st.columns(4)
                
                with indicators_cols[0]:
                    st.metric(
                        "RSI", 
                        f"{stock_data.get('rsi', 0):.2f}",
                        delta=f"{(stock_data.get('rsi', 50) - 50):.2f}" if 'rsi' in stock_data else None,
                        delta_color="inverse"  # High RSI is overbought (potentially negative)
                    )
                
                with indicators_cols[1]:
                    ma_50_delta = stock_data.get('current_price', 0) - stock_data.get('ma_50', 0)
                    delta_pct = (ma_50_delta / stock_data.get('ma_50', 1)) * 100 if stock_data.get('ma_50', 0) > 0 else 0
                    
                    st.metric(
                        "50-day MA", 
                        f"${stock_data.get('ma_50', 0):.2f}",
                        delta=f"{delta_pct:+.2f}%" if 'ma_50' in stock_data else None
                    )
                
                with indicators_cols[2]:
                    ma_200_delta = stock_data.get('current_price', 0) - stock_data.get('ma_200', 0)
                    delta_pct = (ma_200_delta / stock_data.get('ma_200', 1)) * 100 if stock_data.get('ma_200', 0) > 0 else 0
                    
                    st.metric(
                        "200-day MA", 
                        f"${stock_data.get('ma_200', 0):.2f}",
                        delta=f"{delta_pct:+.2f}%" if 'ma_200' in stock_data else None
                    )
                
                with indicators_cols[3]:
                    st.metric(
                        "52W Position", 
                        f"{stock_data.get('52w_percentile', 50):.2f}%",
                        delta=f"{stock_data.get('52w_percentile', 50) - 50:.2f}%" if '52w_percentile' in stock_data else None
                    )
            else:
                st.warning("No data available for the selected filters")
        else:
            st.info("Loading stock data... Please wait.")
    else:
        st.info("Your watchlist is empty. Add symbols to get started.")
    
    # Stock comparison
    with st.expander("Compare Stocks"):
        col1, col2 = st.columns(2)
        
        with col1:
            compare_stocks = st.multiselect(
                "Select Stocks to Compare",
                options=st.session_state.watchlist_symbols,
                default=st.session_state.watchlist_symbols[:min(5, len(st.session_state.watchlist_symbols))]
            )
        
        with col2:
            compare_metric = st.selectbox(
                "Comparison Metric",
                options=[
                    "Price", "Day Change %", "Market Cap", "P/E Ratio",
                    "Volume", "Volatility", "RSI", "52W Range"
                ]
            )
        
        if compare_stocks and st.session_state.watchlist_data:
            # Filter data for selected stocks
            comparison_data = [d for d in st.session_state.watchlist_data if d.get('symbol') in compare_stocks]
            
            if comparison_data:
                # Create a comparison chart
                metric_map = {
                    "Price": "current_price",
                    "Day Change %": "day_change",
                    "Market Cap": "market_cap",
                    "P/E Ratio": "pe_ratio",
                    "Volume": "volume",
                    "Volatility": "volatility",
                    "RSI": "rsi",
                    "52W Range": "52w_percentile"
                }
                
                metric_col = metric_map.get(compare_metric)
                
                if metric_col:
                    # Extract data for the selected metric
                    symbols = [d.get('symbol', 'Unknown') for d in comparison_data]
                    values = [d.get(metric_col, 0) for d in comparison_data]
                    
                    # Create bar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=symbols,
                        y=values,
                        marker_color='#1E88E5'
                    ))
                    
                    fig.update_layout(
                        title=f"Comparison by {compare_metric}",
                        xaxis_title="Symbol",
                        yaxis_title=compare_metric,
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create comparison table
                    comparison_df = pd.DataFrame(comparison_data)
                    if not comparison_df.empty:
                        display_cols = [
                            'symbol', 'name', 'current_price', 'day_change',
                            'market_cap', 'pe_ratio', 'volume', 'volatility', 
                            'rsi', '52w_percentile'
                        ]
                        
                        # Only include columns that exist in the dataframe
                        display_cols = [col for col in display_cols if col in comparison_df.columns]
                        
                        st.dataframe(
                            format_statistics_df(comparison_df[display_cols]),
                            use_container_width=True,
                            hide_index=True
                        )

with tabs[4]:  # Performance
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