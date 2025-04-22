import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import (
    get_fingpt_news, analyze_fingpt_sentiment, get_fingpt_summary,
    get_fingpt_signals, analyze_ticker, get_historical_data,
    train_model
)

# Configure the page
st.set_page_config(
    page_title="AI Analysis | AI Trading Bot",
    page_icon="🤖",
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
    .news-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        min-width: 200px;
        border-radius: 5px;
        padding: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
    .neutral {
        color: #FFC107;
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

def get_sentiment_color(sentiment):
    """Get color based on sentiment"""
    if sentiment > 0.2:
        return "#4CAF50"  # green
    elif sentiment < -0.2:
        return "#F44336"  # red
    else:
        return "#FFC107"  # yellow/amber

def get_sentiment_label(sentiment):
    """Get label based on sentiment"""
    if sentiment > 0.2:
        return "Positive"
    elif sentiment < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Main content
st.markdown('<p class="main-header">AI Trading Analysis</p>', unsafe_allow_html=True)

tabs = st.tabs(["News & Sentiment", "Trading Signals", "AI Models"])

with tabs[0]:  # News & Sentiment
    st.subheader("Financial News Sentiment Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Symbol input
        symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
        
        # Date range
        days = st.slider("Days of News", min_value=1, max_value=30, value=7)
        
        # Number of news items
        num_news = st.slider("Number of News Items", min_value=5, max_value=50, value=10)
        
        # Get news button
        if st.button("Analyze News", type="primary"):
            with st.spinner("Analyzing news sentiment..."):
                st.session_state.news_analyzed = True
                # In a real app, this would call the get_fingpt_news function
                # For now, we'll use sample data
    
    with col2:
        # Sentiment summary metrics
        if 'news_analyzed' in st.session_state and st.session_state.news_analyzed:
            # Sample data - in a real app, this would come from the API
            sentiment_summary = {
                "overall": 0.42,
                "positive_count": 7,
                "negative_count": 2,
                "neutral_count": 1,
                "most_positive_date": "2023-04-18",
                "most_negative_date": "2023-04-20"
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_value = sentiment_summary["overall"]
                sentiment_color = get_sentiment_color(sentiment_value)
                sentiment_label = get_sentiment_label(sentiment_value)
                
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p class="metric-label">Overall Sentiment</p>
                    <p class="metric-value" style="color: {sentiment_color};">{sentiment_label}</p>
                    <p style="font-size: 1.2rem; color: {sentiment_color};">{sentiment_value:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p class="metric-label">Sentiment Distribution</p>
                    <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                        <div>
                            <p style="font-size: 1.5rem; color: #4CAF50;">{sentiment_summary["positive_count"]}</p>
                            <p>Positive</p>
                        </div>
                        <div>
                            <p style="font-size: 1.5rem; color: #FFC107;">{sentiment_summary["neutral_count"]}</p>
                            <p>Neutral</p>
                        </div>
                        <div>
                            <p style="font-size: 1.5rem; color: #F44336;">{sentiment_summary["negative_count"]}</p>
                            <p>Negative</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p class="metric-label">Key Dates</p>
                    <p><span style="color: #4CAF50;">Most Positive:</span> {sentiment_summary["most_positive_date"]}</p>
                    <p><span style="color: #F44336;">Most Negative:</span> {sentiment_summary["most_negative_date"]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sample news data
            news_data = [
                {
                    "date": "2023-04-21",
                    "title": f"{symbol} Reports Record Quarterly Revenue",
                    "source": "MarketWatch",
                    "url": "#",
                    "text": f"{symbol} Inc. reported record-breaking quarterly revenue, exceeding analyst expectations by 12%. The company's CEO attributed the strong performance to growing demand for their latest product line and expansion into emerging markets.",
                    "sentiment": 0.68
                },
                {
                    "date": "2023-04-20",
                    "title": f"Supply Chain Issues May Impact {symbol}'s Production",
                    "source": "Reuters",
                    "url": "#",
                    "text": f"Industry analysts warn that ongoing supply chain disruptions could affect {symbol}'s production capacity in the coming months. The company has acknowledged these challenges but states they have contingency plans in place.",
                    "sentiment": -0.35
                },
                {
                    "date": "2023-04-19",
                    "title": f"{symbol} Announces New Partnership with Tech Giant",
                    "source": "Bloomberg",
                    "url": "#",
                    "text": f"{symbol} has formed a strategic partnership with a leading tech company to develop next-generation products. The collaboration is expected to accelerate innovation and open new market opportunities.",
                    "sentiment": 0.52
                },
                {
                    "date": "2023-04-18",
                    "title": f"Institutional Investors Increase Stakes in {symbol}",
                    "source": "CNBC",
                    "url": "#",
                    "text": f"Several major institutional investors have increased their holdings in {symbol} during the past quarter, signaling growing confidence in the company's long-term prospects and management strategy.",
                    "sentiment": 0.45
                },
                {
                    "date": "2023-04-17",
                    "title": f"{symbol} Faces Regulatory Scrutiny in European Markets",
                    "source": "Financial Times",
                    "url": "#",
                    "text": f"Regulatory authorities in Europe are reviewing {symbol}'s business practices, particularly regarding data privacy and market competition. The company's legal team is cooperating with the investigation.",
                    "sentiment": -0.41
                }
            ]
            
            # Sentiment over time chart
            st.markdown("### Sentiment Trend")
            
            # Create sample data for sentiment trend
            dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days-1, -1, -1)]
            sentiments = [np.random.normal(sentiment_summary["overall"], 0.3) for _ in range(days)]
            
            # Clamp sentiments between -1 and 1
            sentiments = [max(min(s, 1.0), -1.0) for s in sentiments]
            
            # Create dataframe
            sentiment_df = pd.DataFrame({
                'Date': dates,
                'Sentiment': sentiments
            })
            
            # Create figure
            fig = go.Figure()
            
            # Add sentiment line
            fig.add_trace(go.Scatter(
                x=sentiment_df['Date'],
                y=sentiment_df['Sentiment'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='#1E88E5', width=2),
                marker=dict(
                    size=8,
                    color=[get_sentiment_color(s) for s in sentiment_df['Sentiment']],
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))
            
            # Add neutral line
            fig.add_shape(
                type="line",
                x0=sentiment_df['Date'].iloc[0],
                y0=0,
                x1=sentiment_df['Date'].iloc[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash"),
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                yaxis=dict(
                    range=[-1, 1],
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # News table
            st.markdown("### Recent News Articles")
            
            for article in news_data:
                sentiment_color = get_sentiment_color(article["sentiment"])
                sentiment_label = get_sentiment_label(article["sentiment"])
                
                st.markdown(f"""
                <div class="news-card">
                    <h3>{article["title"]}</h3>
                    <p style="color: #666; font-size: 0.8rem;">{article["date"]} • {article["source"]}</p>
                    <p>{article["text"]}</p>
                    <p style="text-align: right;"><span style="background-color: {sentiment_color}; color: white; padding: 3px 8px; border-radius: 10px;">{sentiment_label}: {article["sentiment"]:.2f}</span></p>
                </div>
                """, unsafe_allow_html=True)
            
            # News summary
            st.markdown("### AI-Generated Summary")
            
            summary = f"Over the past {days} days, news sentiment for {symbol} has been generally {get_sentiment_label(sentiment_summary['overall']).lower()}. The company reported strong quarterly earnings, exceeding analyst expectations, and announced a strategic partnership that was well-received by investors. However, there are some concerns about supply chain issues and regulatory scrutiny in European markets. Institutional investors have shown increased confidence by raising their stakes in the company."
            
            st.markdown(f"""
            <div class="card">
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Trading recommendation
            recommendation = "BUY" if sentiment_summary["overall"] > 0.2 else "SELL" if sentiment_summary["overall"] < -0.2 else "HOLD"
            recommendation_color = "#4CAF50" if recommendation == "BUY" else "#F44336" if recommendation == "SELL" else "#FFC107"
            
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <p class="metric-label">AI Trading Recommendation</p>
                <p class="metric-value" style="color: {recommendation_color};">{recommendation}</p>
                <p>Based on news sentiment analysis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Enter a stock symbol and click 'Analyze News' to see sentiment analysis.")

with tabs[1]:  # Trading Signals
    st.subheader("AI-Powered Trading Signals")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Symbol input
        signal_symbol = st.text_input("Enter Stock Symbol", "MSFT", key="signal_symbol").upper()
        
        # Time horizon
        time_horizon = st.selectbox(
            "Time Horizon",
            options=["Short-term (Days)", "Medium-term (Weeks)", "Long-term (Months)"],
            index=0
        )
        
        # Analysis type
        analysis_type = st.multiselect(
            "Analysis Components",
            options=["Technical Indicators", "Price Patterns", "News Sentiment", "Market Trends"],
            default=["Technical Indicators", "News Sentiment"]
        )
        
        # Get signals button
        if st.button("Generate Signals", type="primary"):
            with st.spinner("Generating AI trading signals..."):
                st.session_state.signals_generated = True
                # In a real app, this would call the get_fingpt_signals function
                # For now, we'll use sample data
    
    with col2:
        # Trading signals and explanation
        if 'signals_generated' in st.session_state and st.session_state.signals_generated:
            # Sample data - in a real app, this would come from the API
            signal_data = {
                "symbol": signal_symbol,
                "current_price": 386.50,
                "recommendation": "BUY",
                "confidence": 0.82,
                "target_price": 412.75,
                "stop_loss": 371.25,
                "time_horizon": time_horizon.split(" ")[0].lower(),
                "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                "technical_score": 0.75,
                "sentiment_score": 0.64,
                "market_trend_score": 0.68,
                "volume_analysis": 0.52
            }
            
            # Signal card
            recommendation_color = "#4CAF50" if signal_data["recommendation"] == "BUY" else "#F44336" if signal_data["recommendation"] == "SELL" else "#FFC107"
            
            st.markdown(f"""
            <div class="card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h2>{signal_data["symbol"]}</h2>
                        <p>Current Price: {format_currency(signal_data["current_price"])}</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="font-size: 2.5rem; color: {recommendation_color}; font-weight: bold; margin: 0;">{signal_data["recommendation"]}</p>
                        <p>Confidence: {signal_data["confidence"]*100:.0f}%</p>
                    </div>
                    <div style="text-align: right;">
                        <p>Target: <span style="color: #4CAF50;">{format_currency(signal_data["target_price"])}</span></p>
                        <p>Stop Loss: <span style="color: #F44336;">{format_currency(signal_data["stop_loss"])}</span></p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis components
            st.markdown("### Signal Analysis Components")
            
            # Create radar chart data
            categories = ['Technical Analysis', 'Sentiment Analysis', 'Volume Analysis', 'Market Trend', 'Volatility']
            
            values = [
                signal_data["technical_score"],
                signal_data["sentiment_score"],
                signal_data["volume_analysis"],
                signal_data["market_trend_score"],
                0.58  # Volatility score
            ]
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line=dict(color=recommendation_color),
                opacity=0.8
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                height=400,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.markdown("### Technical Indicators")
            
            # Sample technical indicators
            technical_indicators = [
                {"name": "Moving Average (50,200)", "value": "Bullish", "description": "The 50-day MA is above the 200-day MA, indicating an upward trend."},
                {"name": "RSI (14)", "value": "56.8", "description": "The RSI is in neutral territory but trending higher, suggesting increasing momentum."},
                {"name": "MACD", "value": "Positive", "description": "MACD is above the signal line, suggesting bullish momentum."},
                {"name": "Bollinger Bands", "value": "Upper Test", "description": "Price is testing the upper Bollinger Band, indicating strong upward pressure."},
                {"name": "Stochastic", "value": "68.2", "description": "Stochastic is in neutral territory with upward momentum."}
            ]
            
            # Display indicators
            col1, col2 = st.columns(2)
            
            with col1:
                for i in range(0, len(technical_indicators), 2):
                    indicator = technical_indicators[i]
                    value_color = "#4CAF50" if "bullish" in indicator["value"].lower() or "positive" in indicator["value"].lower() or "upper" in indicator["value"].lower() else "#F44336" if "bearish" in indicator["value"].lower() or "negative" in indicator["value"].lower() or "lower" in indicator["value"].lower() else "#000000"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <p style="margin-bottom: 5px;"><strong>{indicator["name"]}:</strong> <span style="color: {value_color};">{indicator["value"]}</span></p>
                        <p style="font-size: 0.9rem; color: #666;">{indicator["description"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                for i in range(1, len(technical_indicators), 2):
                    indicator = technical_indicators[i]
                    value_color = "#4CAF50" if "bullish" in indicator["value"].lower() or "positive" in indicator["value"].lower() or "upper" in indicator["value"].lower() else "#F44336" if "bearish" in indicator["value"].lower() or "negative" in indicator["value"].lower() or "lower" in indicator["value"].lower() else "#000000"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <p style="margin-bottom: 5px;"><strong>{indicator["name"]}:</strong> <span style="color: {value_color};">{indicator["value"]}</span></p>
                        <p style="font-size: 0.9rem; color: #666;">{indicator["description"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Price chart with signals
            st.markdown("### Price Chart with Signals")
            
            # Generate sample price data
            days = 60
            dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days-1, -1, -1)]
            
            # Start with the current price and work backwards with random changes
            current_price = signal_data["current_price"]
            prices = [current_price]
            
            for i in range(days-1):
                # Larger changes long ago, smaller changes recently
                volatility = 0.015 * (1 + (i / days))
                change = np.random.normal(0, volatility)
                # Add slight downward trend earlier, upward trend later
                if i > days/2:
                    change -= 0.002
                else:
                    change += 0.001
                prices.append(prices[-1] * (1 - change))
            
            prices = prices[::-1]  # Reverse to match dates
            
            # Create DataFrame
            price_df = pd.DataFrame({
                'Date': dates,
                'Price': prices
            })
            
            # Add some signals to the chart
            signals = [
                {"date": dates[int(days*0.2)], "type": "SELL", "price": prices[int(days*0.2)]},
                {"date": dates[int(days*0.5)], "type": "BUY", "price": prices[int(days*0.5)]},
                {"date": dates[int(days*0.8)], "type": "BUY", "price": prices[int(days*0.8)]}
            ]
            
            # Create figure
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=price_df['Date'],
                y=price_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Add signals
            buy_dates = [s["date"] for s in signals if s["type"] == "BUY"]
            buy_prices = [s["price"] for s in signals if s["type"] == "BUY"]
            
            sell_dates = [s["date"] for s in signals if s["type"] == "SELL"]
            sell_prices = [s["price"] for s in signals if s["type"] == "SELL"]
            
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    size=10,
                    color='#4CAF50',
                    symbol='triangle-up'
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    size=10,
                    color='#F44336',
                    symbol='triangle-down'
                )
            ))
            
            # Add the latest signal
            fig.add_trace(go.Scatter(
                x=[dates[-1]],
                y=[prices[-1]],
                mode='markers',
                name=f'Current: {signal_data["recommendation"]}',
                marker=dict(
                    size=12,
                    color=recommendation_color,
                    symbol='star',
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # AI explanation
            st.markdown("### AI Signal Explanation")
            
            explanation = f"""
            The AI trading model recommends a **{signal_data["recommendation"]}** position for {signal_data["symbol"]} with {signal_data["confidence"]*100:.0f}% confidence.
            
            **Technical Analysis:** The stock is showing strong bullish momentum with the 50-day moving average crossing above the 200-day moving average, creating a golden cross pattern. The RSI at 56.8 indicates growing momentum while still having room to run before becoming overbought. MACD is positive and trending higher, confirming the bullish momentum.
            
            **Sentiment Analysis:** News sentiment over the past week has been predominantly positive, with institutional investors increasing their positions. Recent product announcements have been well-received by analysts.
            
            **Risk Assessment:** Volatility is moderate, suggesting a favorable risk-reward ratio. The recommended stop-loss at {format_currency(signal_data["stop_loss"])} provides downside protection of {((signal_data["current_price"] - signal_data["stop_loss"]) / signal_data["current_price"] * 100):.1f}%, while the target price of {format_currency(signal_data["target_price"])} offers an upside potential of {((signal_data["target_price"] - signal_data["current_price"]) / signal_data["current_price"] * 100):.1f}%.
            
            **Time Horizon:** This signal is optimized for a {signal_data["time_horizon"]} time horizon, based on the selected parameters.
            """
            
            st.markdown(explanation)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Place Trade Based on Signal"):
                    st.success(f"{signal_data['recommendation']} order for {signal_data['symbol']} placed successfully!")
            
            with col2:
                if st.button("Save Signal to Watchlist"):
                    st.success(f"Signal for {signal_data['symbol']} saved to your watchlist")
        else:
            st.info(f"Enter a stock symbol and click 'Generate Signals' to see AI trading recommendations.")

with tabs[2]:  # AI Models
    st.subheader("Train & Deploy AI Trading Models")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            options=["DeepTradeBot LSTM", "FinGPT Sentiment", "FinRL Reinforcement Learning"],
            index=0
        )
        
        # Symbol input
        model_symbol = st.text_input("Training Symbol", "SPY", key="model_symbol").upper()
        
        # Training period
        col1, col2 = st.columns(2)
        
        with col1:
            training_start = st.date_input(
                "Training Start",
                value=datetime.now() - timedelta(days=365*2)
            )
        
        with col2:
            training_end = st.date_input(
                "Training End",
                value=datetime.now() - timedelta(days=30)
            )
        
        # Advanced parameters expander
        with st.expander("Advanced Parameters"):
            if model_type == "DeepTradeBot LSTM":
                st.number_input("Sequence Length", min_value=10, max_value=100, value=30)
                st.number_input("Hidden Units", min_value=32, max_value=512, value=128)
                st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
                st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
            elif model_type == "FinGPT Sentiment":
                st.slider("Sentiment Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
                st.number_input("News Lookback (Days)", min_value=1, max_value=30, value=7)
                st.checkbox("Include Social Media", value=True)
                st.checkbox("Include Earnings Reports", value=True)
            elif model_type == "FinRL Reinforcement Learning":
                st.selectbox("RL Algorithm", options=["PPO", "DQN", "A2C", "DDPG"], index=0)
                st.number_input("Training Episodes", min_value=10, max_value=1000, value=100)
                st.slider("Reward Risk Ratio", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
                st.checkbox("Use GPU for Training", value=True)
        
        # Train model button
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model on {model_symbol} data..."):
                st.session_state.model_trained = True
                # In a real app, this would call the train_model function
                # For now, we'll use sample data
    
    with col2:
        # Model training results
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            # Sample data - in a real app, this would come from the API
            training_results = {
                "model_id": "model_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                "model_type": model_type,
                "symbol": model_symbol,
                "training_start": training_start.strftime("%Y-%m-%d"),
                "training_end": training_end.strftime("%Y-%m-%d"),
                "accuracy": 0.68,
                "profit_factor": 1.85,
                "sharpe_ratio": 1.42,
                "max_drawdown": 12.5,
                "win_rate": 0.64,
                "training_time": "8 minutes 24 seconds",
                "validation_accuracy": 0.62
            }
            
            # Success message
            st.success(f"Model trained successfully! Model ID: {training_results['model_id']}")
            
            # Performance metrics
            st.markdown("### Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{training_results['accuracy']*100:.1f}%")
            
            with col2:
                st.metric("Win Rate", f"{training_results['win_rate']*100:.1f}%")
            
            with col3:
                st.metric("Profit Factor", f"{training_results['profit_factor']:.2f}")
            
            with col4:
                st.metric("Sharpe Ratio", f"{training_results['sharpe_ratio']:.2f}")
            
            # Training curves
            st.markdown("### Training Curves")
            
            # Generate sample training data
            epochs = 50
            training_loss = [np.random.uniform(0.8, 0.9) - (0.02 * i) + np.random.normal(0, 0.03) for i in range(epochs)]
            validation_loss = [loss + np.random.uniform(0.05, 0.15) for loss in training_loss]
            
            training_accuracy = [0.5 + (0.005 * i) + np.random.normal(0, 0.02) for i in range(epochs)]
            validation_accuracy = [acc - np.random.uniform(0.03, 0.08) for acc in training_accuracy]
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for loss
            fig.add_trace(go.Scatter(
                x=list(range(1, epochs + 1)),
                y=training_loss,
                mode='lines',
                name='Training Loss',
                line=dict(color='#F44336', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, epochs + 1)),
                y=validation_loss,
                mode='lines',
                name='Validation Loss',
                line=dict(color='#FF9800', width=2, dash='dash')
            ))
            
            # Add traces for accuracy on secondary y-axis
            fig.add_trace(go.Scatter(
                x=list(range(1, epochs + 1)),
                y=training_accuracy,
                mode='lines',
                name='Training Accuracy',
                line=dict(color='#4CAF50', width=2),
                yaxis="y2"
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, epochs + 1)),
                y=validation_accuracy,
                mode='lines',
                name='Validation Accuracy',
                line=dict(color='#2196F3', width=2, dash='dash'),
                yaxis="y2"
            ))
            
            fig.update_layout(
                xaxis_title="Epochs",
                yaxis=dict(
                    title="Loss",
                    side="left"
                ),
                yaxis2=dict(
                    title="Accuracy",
                    side="right",
                    overlaying="y",
                    range=[0, 1]
                ),
                height=300,
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
            
            # Prediction performance on test data
            st.markdown("### Test Performance")
            
            # Generate sample test data
            test_days = 30
            test_dates = [(training_end + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(test_days)]
            
            # Generate price data
            start_price = 100
            prices = [start_price]
            predictions = [start_price]
            
            for i in range(1, test_days):
                # Actual price changes with random walk
                change = np.random.normal(0.0005, 0.01)
                prices.append(prices[-1] * (1 + change))
                
                # Predictions follow the actual with some error
                prediction_error = np.random.normal(0, 0.02)
                if i < test_days - 1:
                    predictions.append(prices[-1] * (1 + change + prediction_error))
                else:
                    predictions.append(None)  # No prediction for the last day
            
            # Create dataframe
            test_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': prices,
                'Predicted': predictions
            })
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=test_df['Date'],
                y=test_df['Actual'],
                mode='lines',
                name='Actual Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=test_df['Date'],
                y=test_df['Predicted'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='#FFA000', width=2, dash='dash')
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.markdown("### Feature Importance")
            
            # Sample feature importance data
            feature_importance = {
                "Price_Momentum": 0.28,
                "Volume_Change": 0.18,
                "RSI_14": 0.15,
                "MACD": 0.12,
                "Sentiment_Score": 0.11,
                "Volatility": 0.08,
                "MA_Crossover": 0.05,
                "Volume_Profile": 0.03
            }
            
            # Create dataframe
            feature_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            }).sort_values(by='Importance', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                feature_df,
                y='Feature',
                x='Importance',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues',
                labels={'Importance': 'Relative Importance', 'Feature': 'Feature'}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Deploy Model"):
                    st.success(f"Model {training_results['model_id']} deployed successfully and is now active")
            
            with col2:
                if st.button("Download Model"):
                    st.info(f"Preparing model download...")
                    st.markdown(f"[Download Model File](https://example.com/models/{training_results['model_id']}.h5)")
            
            with col3:
                if st.button("Compare with Baseline"):
                    st.info("Comparing model performance with baseline strategies...")
        else:
            st.info(f"Select model parameters and click 'Train Model' to create a custom AI trading model.")
            
            # Display existing models
            st.markdown("### Your Existing Models")
            
            # Sample model list
            existing_models = [
                {"id": "model_20230412152233", "type": "DeepTradeBot LSTM", "symbol": "AAPL", "created": "2023-04-12", "accuracy": 0.72, "status": "Active"},
                {"id": "model_20230327091526", "type": "FinGPT Sentiment", "symbol": "MSFT", "created": "2023-03-27", "accuracy": 0.65, "status": "Active"},
                {"id": "model_20230215103045", "type": "FinRL Reinforcement Learning", "symbol": "SPY", "created": "2023-02-15", "accuracy": 0.68, "status": "Inactive"}
            ]
            
            # Convert to DataFrame for display
            models_df = pd.DataFrame({
                'Model ID': [model["id"] for model in existing_models],
                'Type': [model["type"] for model in existing_models],
                'Symbol': [model["symbol"] for model in existing_models],
                'Created': [model["created"] for model in existing_models],
                'Accuracy': [f"{model['accuracy']*100:.1f}%" for model in existing_models],
                'Status': [model["status"] for model in existing_models]
            })
            
            st.dataframe(models_df, use_container_width=True, hide_index=True)