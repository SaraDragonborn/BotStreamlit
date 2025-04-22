import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage with color"""
    color = "positive" if value >= 0 else "negative"
    return f'<span class="{color}">{value:+.2f}%</span>'

def plot_portfolio_composition(positions, account=None):
    """
    Create a pie chart of portfolio composition
    
    Args:
        positions: List of position objects
        account: Account information with cash value
    
    Returns:
        Plotly figure object
    """
    if not positions:
        return None
    
    # Extract position values
    position_values = []
    labels = []
    
    for position in positions:
        symbol = position.get('symbol', 'Unknown')
        market_value = float(position.get('market_value', 0))
        position_values.append(market_value)
        labels.append(symbol)
    
    # Add cash if we have account info
    if account:
        cash = float(account.get('cash', 0))
        if cash > 0:
            position_values.append(cash)
            labels.append('Cash')
    
    # Create figure
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=position_values,
        hole=0.4,
        marker_colors=['#1E88E5', '#42A5F5', '#64B5F6', '#90CAF9', '#BBDEFB', '#E3F2FD', '#B3E5FC', '#81D4FA', '#4FC3F7', '#29B6F6']
    )])
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        showlegend=True
    )
    
    return fig

def plot_equity_curve(initial_capital, start_date, end_date, include_benchmark=True, random_seed=None):
    """
    Create an equity curve with optional benchmark
    
    Args:
        initial_capital: Starting capital value
        start_date: Start date for the chart
        end_date: End date for the chart
        include_benchmark: Whether to include a benchmark line
        random_seed: Seed for random number generation (for consistent results)
    
    Returns:
        Plotly figure object and equity DataFrame
    """
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Set random seed for consistent results if provided
    np_random = np.random.RandomState(random_seed) if random_seed else np.random
    
    # Generate equity curve with random walk and positive drift
    equity = [initial_capital]
    for i in range(1, len(dates)):
        daily_return = np_random.normal(0.0007, 0.012)  # Small positive drift
        new_value = equity[-1] * (1 + daily_return)
        equity.append(new_value)
    
    # Create dataframe
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
    
    # Add benchmark if requested
    if include_benchmark:
        benchmark = [initial_capital]
        for i in range(1, len(dates)):
            daily_return = np_random.normal(0.0005, 0.01)  # Lower drift, lower volatility
            new_value = benchmark[-1] * (1 + daily_return)
            benchmark.append(new_value)
        
        equity_df['Benchmark'] = benchmark
        
        fig.add_trace(go.Scatter(
            x=equity_df['Date'],
            y=benchmark,
            mode='lines',
            name='Benchmark (SPY)',
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
    
    return fig, equity_df

def plot_drawdown(equity_values, dates):
    """
    Create a drawdown chart
    
    Args:
        equity_values: List of equity values
        dates: List of dates
    
    Returns:
        Plotly figure object and max drawdown value
    """
    # Calculate drawdown
    drawdown = []
    peak = equity_values[0]
    max_drawdown = 0
    
    for value in equity_values:
        if value > peak:
            peak = value
        
        current_drawdown = (peak - value) / peak * 100
        drawdown.append(current_drawdown)
        
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
    
    # Create dataframe
    drawdown_df = pd.DataFrame({
        'Date': dates,
        'Drawdown': drawdown
    })
    
    # Create figure
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
    
    return fig, max_drawdown

def plot_monthly_returns_heatmap(start_date, end_date, random_seed=None):
    """
    Create a monthly returns heatmap
    
    Args:
        start_date: Start date for the chart
        end_date: End date for the chart
        random_seed: Seed for random number generation (for consistent results)
    
    Returns:
        Plotly figure object and monthly returns DataFrame
    """
    # Set random seed for consistent results if provided
    np_random = np.random.RandomState(random_seed) if random_seed else np.random
    
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
    
    return fig, monthly_returns

def plot_radar_chart(categories, values, title=None, color="#1E88E5"):
    """
    Create a radar chart
    
    Args:
        categories: List of category names
        values: List of values for each category
        title: Optional chart title
        color: Color for the radar chart
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line=dict(color=color),
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
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        title=title
    )
    
    return fig

def plot_feature_importance(feature_importance, orientation='h'):
    """
    Create a feature importance chart
    
    Args:
        feature_importance: Dictionary of feature names and importance values
        orientation: Bar orientation ('h' for horizontal, 'v' for vertical)
    
    Returns:
        Plotly figure object
    """
    # Create dataframe
    feature_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    if orientation == 'h':
        feature_df = feature_df.sort_values(by='Importance', ascending=True)
        fig = px.bar(
            feature_df,
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature'}
        )
    else:
        feature_df = feature_df.sort_values(by='Importance', ascending=False)
        fig = px.bar(
            feature_df,
            x='Feature',
            y='Importance',
            color='Importance',
            color_continuous_scale='Blues',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature'}
        )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False
    )
    
    return fig

def plot_price_with_signals(symbol, dates, prices, signals=None, current_price=None, current_signal=None):
    """
    Create a price chart with buy/sell signals
    
    Args:
        symbol: Stock symbol
        dates: List of dates
        prices: List of prices
        signals: List of signal dictionaries with date, type, price
        current_price: Current price (optional)
        current_signal: Current signal type (optional)
    
    Returns:
        Plotly figure object
    """
    # Create DataFrame
    price_df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
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
    
    # Add signals if provided
    if signals:
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
    
    # Add current signal if provided
    if current_price and current_signal:
        signal_color = "#4CAF50" if current_signal == "BUY" else "#F44336" if current_signal == "SELL" else "#FFC107"
        
        fig.add_trace(go.Scatter(
            x=[dates[-1]],
            y=[current_price],
            mode='markers',
            name=f'Current: {current_signal}',
            marker=dict(
                size=12,
                color=signal_color,
                symbol='star',
                line=dict(width=1, color='DarkSlateGrey')
            )
        ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_training_progress(epochs, training_loss, validation_loss, training_accuracy, validation_accuracy):
    """
    Create a training progress chart with loss and accuracy
    
    Args:
        epochs: Number of epochs
        training_loss: List of training loss values
        validation_loss: List of validation loss values
        training_accuracy: List of training accuracy values
        validation_accuracy: List of validation accuracy values
    
    Returns:
        Plotly figure object
    """
    epoch_range = list(range(1, epochs + 1))
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for loss
    fig.add_trace(go.Scatter(
        x=epoch_range,
        y=training_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='#F44336', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epoch_range,
        y=validation_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='#FF9800', width=2, dash='dash')
    ))
    
    # Add traces for accuracy on secondary y-axis
    fig.add_trace(go.Scatter(
        x=epoch_range,
        y=training_accuracy,
        mode='lines',
        name='Training Accuracy',
        line=dict(color='#4CAF50', width=2),
        yaxis="y2"
    ))
    
    fig.add_trace(go.Scatter(
        x=epoch_range,
        y=validation_accuracy,
        mode='lines',
        name='Validation Accuracy',
        line=dict(color='#2196F3', width=2, dash='dash'),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Training Progress",
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
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_sentiment_trend(dates, sentiments):
    """
    Create a sentiment trend chart
    
    Args:
        dates: List of dates
        sentiments: List of sentiment values (-1 to 1)
    
    Returns:
        Plotly figure object
    """
    # Create dataframe
    sentiment_df = pd.DataFrame({
        'Date': dates,
        'Sentiment': sentiments
    })
    
    # Get colors based on sentiment
    def get_sentiment_color(sentiment):
        if sentiment > 0.2:
            return "#4CAF50"  # green
        elif sentiment < -0.2:
            return "#F44336"  # red
        else:
            return "#FFC107"  # yellow/amber
    
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
        title="Sentiment Trend",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(
            range=[-1, 1],
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        )
    )
    
    return fig