import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import random
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.strategies import get_strategy_templates, get_available_symbols, load_strategies
from utils.stock_info import display_stock_info, display_mini_stock_info, get_stock_info

def generate_signal(strategy_name, strategy_type, symbol, interval="15m"):
    """Generate a realistic signal for a given strategy and symbol"""
    now = datetime.now()
    
    # Calculate signal strength
    if strategy_type in ["Technical", "Statistical", "Adaptive", "Intraday"]:
        strength = round(random.uniform(0.5, 0.99), 2)
    elif strategy_type in ["AI-Powered", "Hybrid"]:
        strength = round(random.uniform(0.6, 0.95), 2)
    elif strategy_type == "Indian Market":
        strength = round(random.uniform(0.55, 0.97), 2)
    else:
        strength = round(random.uniform(0.5, 0.9), 2)
    
    # Generate random action (biased slightly toward 'Buy')
    action_random = random.random()
    if action_random > 0.55:  # Slightly biased toward buys
        action = "Buy"
        color = "green"
    else:
        action = "Sell"
        color = "red"
    
    # Generate realistic price targets
    current_price = random.uniform(50, 5000)
    if ".NS" in symbol:  # Indian stocks typically have lower prices
        current_price = random.uniform(100, 3000)
    
    if action == "Buy":
        target_price = current_price * (1 + random.uniform(0.005, 0.03))
        stop_loss = current_price * (1 - random.uniform(0.003, 0.015))
    else:
        target_price = current_price * (1 - random.uniform(0.005, 0.03))
        stop_loss = current_price * (1 + random.uniform(0.003, 0.015))
    
    # Time to live for signal based on interval
    interval_hours = {"1m": 0.5, "5m": 2, "15m": 6, "30m": 12, "1h": 24, "4h": 48, "1d": 120}
    ttl_hours = interval_hours.get(interval, 24)
    expiry = now + timedelta(hours=ttl_hours)
    
    return {
        "strategy": strategy_name,
        "type": strategy_type,
        "symbol": symbol,
        "action": action,
        "strength": strength,
        "color": color,
        "price": round(current_price, 2),
        "target": round(target_price, 2),
        "stop_loss": round(stop_loss, 2),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "interval": interval,
        "expiry": expiry.strftime("%Y-%m-%d %H:%M:%S")
    }

def calculate_convergence(signals_df):
    """Calculate convergence metrics based on signals"""
    if signals_df.empty:
        return {"buy": 0, "sell": 0, "strength": 0, "direction": "Neutral", "confidence": "Low"}
    
    # Get buy vs sell counts
    buy_count = len(signals_df[signals_df['action'] == 'Buy'])
    sell_count = len(signals_df[signals_df['action'] == 'Sell'])
    total = buy_count + sell_count
    
    if total == 0:
        return {"buy": 0, "sell": 0, "strength": 0, "direction": "Neutral", "confidence": "Low"}
    
    # Calculate percentages
    buy_percent = (buy_count / total) * 100
    sell_percent = (sell_count / total) * 100
    
    # Calculate average signal strength
    avg_strength = signals_df['strength'].mean()
    
    # Determine direction and confidence
    if buy_percent > 70:
        direction = "Strong Buy"
        confidence = "High" if avg_strength > 0.8 else "Medium"
    elif buy_percent > 55:
        direction = "Buy"
        confidence = "Medium"
    elif sell_percent > 70:
        direction = "Strong Sell"
        confidence = "High" if avg_strength > 0.8 else "Medium"
    elif sell_percent > 55:
        direction = "Sell"
        confidence = "Medium"
    else:
        direction = "Neutral"
        confidence = "Low"
    
    return {
        "buy": round(buy_percent, 1),
        "sell": round(sell_percent, 1),
        "strength": round(avg_strength * 100, 1),
        "direction": direction,
        "confidence": confidence
    }

def generate_signals_for_symbol(symbol, all_strategies, active_strategies, strategy_templates, num_signals=15):
    """Generate comprehensive signals for a specific symbol"""
    signals = []
    intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    # Get active strategy names
    active_strategy_names = [s['name'] for s in active_strategies if s['status'] == 'Active']
    
    # Add signals for active user strategies
    for strategy in active_strategies:
        if strategy['status'] == 'Active' and symbol in strategy.get('symbols', []):
            for _ in range(random.randint(1, 3)):  # 1-3 signals per active strategy
                interval = random.choice(intervals)
                signals.append(generate_signal(
                    strategy['name'], 
                    strategy['type'], 
                    symbol, 
                    interval
                ))
    
    # Fill remaining signals with template strategies
    if len(signals) < num_signals:
        strategy_types = list(set([t['type'] for t in strategy_templates.values()]))
        remaining = num_signals - len(signals)
        
        for _ in range(remaining):
            # Pick a random strategy template
            strategy_key = random.choice(list(strategy_templates.keys()))
            strategy_info = strategy_templates[strategy_key]
            
            # Only use if not already an active user strategy
            if strategy_info['name'] not in active_strategy_names:
                interval = random.choice(intervals)
                signals.append(generate_signal(
                    strategy_info['name'], 
                    strategy_info['type'], 
                    symbol, 
                    interval
                ))
    
    # Convert to DataFrame and sort by timestamp (newest first)
    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values('timestamp', ascending=False)
    
    return df

def display_signals_dashboard():
    st.title("Trading Signals Dashboard")
    
    # Initialize session state for selected symbol if not exists
    if 'selected_symbol_signals' not in st.session_state:
        st.session_state.selected_symbol_signals = None
    if 'signals_generated' not in st.session_state:
        st.session_state.signals_generated = False
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    
    # Sidebar for symbol selection
    st.sidebar.header("Signal Settings")
    
    all_symbols = get_available_symbols()
    
    # Allow for custom symbol input
    custom_symbol = st.sidebar.text_input("Enter Custom Symbol:", "")
    
    # Display stock info when custom symbol is entered
    if custom_symbol:
        stock_info_container = st.sidebar.container()
        with stock_info_container:
            stock_info = get_stock_info(custom_symbol)
            if stock_info:
                st.markdown(f"""
                **{custom_symbol}**: {stock_info['name']}  
                **Exchange:** {stock_info['exchange']}  
                **Price:** ${stock_info['current_price']:.2f} ({stock_info['day_change']:+.2f}%)
                """)
            else:
                st.warning(f"No information found for symbol: {custom_symbol}")
    
    if custom_symbol and custom_symbol not in all_symbols:
        all_symbols = [custom_symbol] + all_symbols
    
    # Multi-select for symbols with a high default limit
    selected_symbols = st.sidebar.multiselect(
        "Select Symbols (up to 100):",
        all_symbols,
        default=all_symbols[:5] if all_symbols else []
    )
    
    # Limit to 100 symbols
    if len(selected_symbols) > 100:
        st.sidebar.warning("Maximum 100 symbols allowed. Only first 100 will be used.")
        selected_symbols = selected_symbols[:100]
    
    # Filter options
    st.sidebar.subheader("Filter Options")
    signal_types = ["All", "Technical", "AI-Powered", "Hybrid", "Statistical", "Intraday", "Indian Market", "Options"]
    selected_types = st.sidebar.multiselect("Strategy Types:", signal_types, default=["All"])
    
    signal_strengths = ["All", "Strong (>80%)", "Medium (60-80%)", "Weak (<60%)"]
    selected_strengths = st.sidebar.multiselect("Signal Strength:", signal_strengths, default=["All"])
    
    time_frames = ["All", "1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "4 hours", "1 day"]
    selected_frames = st.sidebar.multiselect("Time Frames:", time_frames, default=["All"])
    
    # Define interval map for use in filtering
    interval_map = {
        "1 minute": "1m", 
        "5 minutes": "5m", 
        "15 minutes": "15m", 
        "30 minutes": "30m", 
        "1 hour": "1h", 
        "4 hours": "4h", 
        "1 day": "1d"
    }
    
    # Generate signals button
    if st.sidebar.button("Generate Signals"):
        if not selected_symbols:
            st.sidebar.error("Please select at least one symbol.")
        else:
            with st.spinner("Generating signals for selected symbols..."):
                st.session_state.signals_generated = True
                st.session_state.last_update_time = datetime.now()
                
                # Get strategies and templates
                user_strategies = load_strategies()
                active_strategies = [s for s in user_strategies if s['status'] == 'Active']
                strategy_templates = get_strategy_templates()
                
                signals_by_symbol = {}
                for symbol in selected_symbols:
                    signals_df = generate_signals_for_symbol(
                        symbol, 
                        user_strategies, 
                        active_strategies, 
                        strategy_templates
                    )
                    signals_by_symbol[symbol] = signals_df
                
                st.session_state.signals_by_symbol = signals_by_symbol
                st.success(f"Signals generated for {len(selected_symbols)} symbols!")
    
    # Display last update time
    if st.session_state.last_update_time:
        st.sidebar.info(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content area
    if st.session_state.signals_generated:
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["Signal Summary", "Detailed Signals", "Signal Convergence"])
        
        # Get signals
        signals_by_symbol = st.session_state.signals_by_symbol
        
        # Create a flattened list of all signals for summary
        all_signals = []
        for symbol, signals_df in signals_by_symbol.items():
            if not signals_df.empty:
                all_signals.append(signals_df)
        
        if all_signals:
            all_signals_df = pd.concat(all_signals).reset_index(drop=True)
            
            # Apply filters
            if "All" not in selected_types:
                all_signals_df = all_signals_df[all_signals_df['type'].isin(selected_types)]
                
            if "All" not in selected_strengths:
                if "Strong (>80%)" in selected_strengths:
                    all_signals_df = all_signals_df[all_signals_df['strength'] > 0.8]
                elif "Medium (60-80%)" in selected_strengths:
                    all_signals_df = all_signals_df[(all_signals_df['strength'] > 0.6) & (all_signals_df['strength'] <= 0.8)]
                elif "Weak (<60%)" in selected_strengths:
                    all_signals_df = all_signals_df[all_signals_df['strength'] <= 0.6]
            
            if "All" not in selected_frames:
                selected_intervals = [interval_map[frame] for frame in selected_frames if frame in interval_map]
                all_signals_df = all_signals_df[all_signals_df['interval'].isin(selected_intervals)]
            
            # Tab 1: Summary
            with tab1:
                st.subheader("Signal Summary")
                
                # Show summary by action count
                buy_signals = len(all_signals_df[all_signals_df['action'] == 'Buy'])
                sell_signals = len(all_signals_df[all_signals_df['action'] == 'Sell'])
                total_signals = len(all_signals_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Buy Signals", buy_signals, f"{buy_signals/total_signals*100:.1f}%" if total_signals > 0 else "0%")
                with col2:
                    st.metric("Sell Signals", sell_signals, f"{sell_signals/total_signals*100:.1f}%" if total_signals > 0 else "0%")
                with col3:
                    st.metric("Total Signals", total_signals)
                
                # Top signals by strength
                st.subheader("Top Signals by Strength")
                
                top_signals = all_signals_df.sort_values('strength', ascending=False).head(10)
                if not top_signals.empty:
                    # Custom formatting for signal strength display
                    def highlight_action(val):
                        if val == 'Buy':
                            return 'background-color: rgba(0, 128, 0, 0.2); color: darkgreen'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.2); color: darkred'
                    
                    # Format signal strength as percentage
                    top_signals['Signal Strength'] = (top_signals['strength'] * 100).round(1).astype(str) + '%'
                    
                    # Format for display
                    display_df = top_signals[['symbol', 'action', 'Signal Strength', 'strategy', 'interval', 'timestamp']]
                    display_df.columns = ['Symbol', 'Action', 'Strength', 'Strategy', 'Timeframe', 'Generated At']
                    
                    # Display with styling
                    st.dataframe(display_df.style.applymap(highlight_action, subset=['Action']))
                
                # Signal distribution by type
                st.subheader("Signal Distribution by Type")
                if not all_signals_df.empty:
                    type_counts = all_signals_df['type'].value_counts().reset_index()
                    type_counts.columns = ['Strategy Type', 'Count']
                    st.bar_chart(type_counts.set_index('Strategy Type'))
            
            # Tab 2: Detailed Signals
            with tab2:
                st.subheader("Detailed Signals")
                
                # Symbol selector in the detailed view
                symbol_for_details = st.selectbox("Select Symbol for Detailed Signals:", selected_symbols)
                
                # Display stock info for selected symbol
                if symbol_for_details:
                    info_col1, info_col2 = st.columns([3, 1])
                    with info_col1:
                        stock_info = get_stock_info(symbol_for_details)
                        if stock_info:
                            st.markdown(f"""
                            ### {stock_info['name']}
                            **Exchange:** {stock_info['exchange']} | **Sector:** {stock_info.get('sector', 'N/A')}
                            """)
                    with info_col2:
                        if stock_info:
                            price = stock_info['current_price']
                            change = stock_info['day_change']
                            color = "green" if change >= 0 else "red"
                            st.markdown(f"""
                            ### ${price:.2f} <span style="color:{color};">({change:+.2f}%)</span>
                            """, unsafe_allow_html=True)
                    st.markdown("---")
                
                if symbol_for_details in signals_by_symbol:
                    signals_df = signals_by_symbol[symbol_for_details]
                    
                    # Apply the same filters as in the summary
                    if "All" not in selected_types:
                        signals_df = signals_df[signals_df['type'].isin(selected_types)]
                        
                    if "All" not in selected_strengths:
                        if "Strong (>80%)" in selected_strengths:
                            signals_df = signals_df[signals_df['strength'] > 0.8]
                        elif "Medium (60-80%)" in selected_strengths:
                            signals_df = signals_df[(signals_df['strength'] > 0.6) & (signals_df['strength'] <= 0.8)]
                        elif "Weak (<60%)" in selected_strengths:
                            signals_df = signals_df[signals_df['strength'] <= 0.6]
                    
                    if "All" not in selected_frames:
                        selected_intervals = [interval_map[frame] for frame in selected_frames if frame in interval_map]
                        signals_df = signals_df[signals_df['interval'].isin(selected_intervals)]
                    
                    if not signals_df.empty:
                        st.write(f"Found {len(signals_df)} signals for {symbol_for_details}")
                        
                        # Prepare data for display
                        display_columns = ['action', 'strength', 'price', 'target', 'stop_loss', 
                                          'strategy', 'type', 'interval', 'timestamp', 'expiry']
                        
                        display_df = signals_df[display_columns].copy()
                        display_df['strength'] = (display_df['strength'] * 100).round(1).astype(str) + '%'
                        
                        # Rename columns for better display
                        column_map = {
                            'action': 'Action',
                            'strength': 'Strength',
                            'price': 'Entry Price',
                            'target': 'Target Price',
                            'stop_loss': 'Stop Loss',
                            'strategy': 'Strategy',
                            'type': 'Type',
                            'interval': 'Timeframe',
                            'timestamp': 'Generated At',
                            'expiry': 'Valid Until'
                        }
                        display_df.rename(columns=column_map, inplace=True)
                        
                        # Define color function for action column
                        def color_action(val):
                            if val == 'Buy':
                                return 'background-color: rgba(0, 128, 0, 0.2); color: darkgreen'
                            else:
                                return 'background-color: rgba(255, 0, 0, 0.2); color: darkred'
                        
                        # Display table with styling
                        st.dataframe(display_df.style.applymap(color_action, subset=['Action']))
                    else:
                        st.info(f"No signals match the current filters for {symbol_for_details}.")
                else:
                    st.info(f"No signals available for {symbol_for_details}.")
            
            # Tab 3: Signal Convergence
            with tab3:
                st.subheader("Signal Convergence Analysis")
                
                # Symbol selector for convergence
                symbol_for_convergence = st.selectbox(
                    "Select Symbol for Convergence Analysis:", 
                    selected_symbols,
                    key="convergence_symbol"
                )
                
                # Display stock info for convergence analysis
                if symbol_for_convergence:
                    stock_info = get_stock_info(symbol_for_convergence)
                    if stock_info:
                        info_col1, info_col2 = st.columns([3, 1])
                        with info_col1:
                            st.markdown(f"""
                            ### {stock_info['name']}
                            **Exchange:** {stock_info['exchange']} | **Sector:** {stock_info.get('sector', 'N/A')}
                            """)
                        with info_col2:
                            price = stock_info['current_price']
                            change = stock_info['day_change']
                            color = "green" if change >= 0 else "red"
                            st.markdown(f"""
                            ### ${price:.2f} <span style="color:{color};">({change:+.2f}%)</span>
                            """, unsafe_allow_html=True)
                        st.markdown("---")
                
                if symbol_for_convergence in signals_by_symbol:
                    signals_df = signals_by_symbol[symbol_for_convergence]
                    
                    if not signals_df.empty:
                        # Apply filters
                        filtered_df = signals_df.copy()
                        if "All" not in selected_types:
                            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
                            
                        if "All" not in selected_strengths:
                            if "Strong (>80%)" in selected_strengths:
                                filtered_df = filtered_df[filtered_df['strength'] > 0.8]
                            elif "Medium (60-80%)" in selected_strengths:
                                filtered_df = filtered_df[(filtered_df['strength'] > 0.6) & (filtered_df['strength'] <= 0.8)]
                            elif "Weak (<60%)" in selected_strengths:
                                filtered_df = filtered_df[filtered_df['strength'] <= 0.6]
                        
                        if "All" not in selected_frames:
                            selected_intervals = [interval_map[frame] for frame in selected_frames if frame in interval_map]
                            filtered_df = filtered_df[filtered_df['interval'].isin(selected_intervals)]
                        
                        # Calculate convergence metrics
                        convergence = calculate_convergence(filtered_df)
                        
                        # Display convergence summary
                        st.subheader(f"Convergence for {symbol_for_convergence}")
                        
                        # Visual indicator
                        col1, col2, col3 = st.columns([1,3,1])
                        with col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; 
                                        background-color: {'rgba(0,128,0,0.2)' if convergence['direction'].startswith('Buy') else 
                                                            'rgba(255,0,0,0.2)' if convergence['direction'].startswith('Sell') else 
                                                            'rgba(200,200,200,0.2)'};
                                        border-radius: 10px; margin-bottom: 20px;">
                                <h2 style="margin: 0; color: {'darkgreen' if convergence['direction'].startswith('Buy') else 
                                                            'darkred' if convergence['direction'].startswith('Sell') else 
                                                            'gray'};">
                                    {convergence['direction']}
                                </h2>
                                <p style="margin: 5px 0 0 0;">
                                    {convergence['confidence']} Confidence | Signal Strength: {convergence['strength']}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Signal breakdown
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background-color: rgba(0,128,0,0.1); border-radius: 5px;">
                                <h3 style="margin: 0; color: darkgreen;">Buy Signals: {convergence['buy']}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background-color: rgba(255,0,0,0.1); border-radius: 5px;">
                                <h3 style="margin: 0; color: darkred;">Sell Signals: {convergence['sell']}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Type breakdown
                        st.subheader("Signal Breakdown by Type")
                        type_action_counts = filtered_df.groupby(['type', 'action']).size().unstack(fill_value=0)
                        
                        if not type_action_counts.empty:
                            # Calculate percentages
                            type_action_percent = type_action_counts.div(type_action_counts.sum(axis=1), axis=0) * 100
                            type_action_percent = type_action_percent.round(1)
                            
                            # Add total counts
                            type_action_counts['Total'] = type_action_counts.sum(axis=1)
                            
                            # Combine counts and percentages
                            result = type_action_counts.copy()
                            for col in ['Buy', 'Sell']:
                                if col in type_action_percent.columns:
                                    result[f'{col} %'] = type_action_percent[col]
                            
                            # Reorder columns
                            cols = []
                            for col in ['Buy', 'Buy %', 'Sell', 'Sell %']:
                                if col in result.columns or col.split()[0] in result.columns:
                                    cols.append(col)
                            cols.append('Total')
                            
                            # Display the result
                            display_df = result[cols]
                            display_df.index.name = 'Strategy Type'
                            display_df.reset_index(inplace=True)
                            
                            st.dataframe(display_df)
                        else:
                            st.info("No signal data available for breakdown.")
                        
                        # Timeframe breakdown
                        st.subheader("Signal Breakdown by Timeframe")
                        timeframe_counts = filtered_df['interval'].value_counts().reset_index()
                        timeframe_counts.columns = ['Timeframe', 'Count']
                        
                        timeframe_map = {
                            '1m': '1 minute',
                            '5m': '5 minutes',
                            '15m': '15 minutes',
                            '30m': '30 minutes',
                            '1h': '1 hour',
                            '4h': '4 hours',
                            '1d': '1 day'
                        }
                        timeframe_counts['Timeframe'] = timeframe_counts['Timeframe'].map(
                            lambda x: timeframe_map.get(x, x)
                        )
                        
                        st.bar_chart(timeframe_counts.set_index('Timeframe'))
                        
                    else:
                        st.info(f"No signals available for {symbol_for_convergence}.")
                else:
                    st.info(f"No signals available for {symbol_for_convergence}.")
        else:
            st.info("No signals generated. Please select symbols and click 'Generate Signals'.")
    else:
        # Initial state - instructions
        st.info("👈 Select symbols and click 'Generate Signals' in the sidebar to view trading signals")
        
        st.markdown("""
        ### Signals Dashboard Features:
        
        - **View signals** from all your active strategies for up to 100 stocks
        - **Filter signals** by type, strength, and timeframe
        - **Analyze signal convergence** to identify strong trading opportunities
        - **Compare signals** across different strategy types
        - **Track signal expiration** to know when to exit trades
        
        This dashboard combines technical indicators, AI predictions, and custom strategies to provide a comprehensive view of potential trading opportunities.
        """)

# Main function
def main():
    # Set page config
    st.set_page_config(
        page_title="Trading Signals Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Display signals dashboard
    display_signals_dashboard()

if __name__ == "__main__":
    main()