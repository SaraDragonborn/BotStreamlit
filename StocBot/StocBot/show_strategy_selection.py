"""
Script to demonstrate strategy selection based on market conditions.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import logging
from api.angel_api import AngelOneAPI
from strategies import MovingAverageCrossover, RSIReversal, MarketConditionAnalyzer
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategy_selection')

def analyze_market_conditions(api, index_symbol='NIFTY', days=30):
    """Analyze market conditions over a period of time."""
    # Get index data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    logger.info(f"Getting historical data for {index_symbol} from {start_date} to {end_date}")
    
    data = api.get_historical_data(
        symbol=index_symbol,
        timeframe='1day',  # Daily data for better trend visualization
        from_date=start_date,
        to_date=end_date
    )
    
    if data is None or data.empty:
        logger.error(f"Failed to get historical data for {index_symbol}")
        return None
    
    # Create market analyzer
    market_analyzer = MarketConditionAnalyzer()
    
    # Analyze data
    logger.info("Analyzing market conditions")
    
    # Calculate ADX for the entire period
    signals = data.copy()
    adx_period = market_analyzer.params['adx_period']
    trend_threshold = market_analyzer.params['trend_threshold']
    sideways_threshold = market_analyzer.params['sideways_threshold']
    
    # Calculate ADX using the ta library
    import ta
    adx = ta.trend.ADXIndicator(
        high=signals['high'],
        low=signals['low'],
        close=signals['close'],
        window=adx_period,
        fillna=True
    )
    
    signals['adx'] = adx.adx()
    signals['plus_di'] = adx.adx_pos()
    signals['minus_di'] = adx.adx_neg()
    
    # Determine market condition for each day
    signals['market_condition'] = 'neutral'
    signals.loc[signals['adx'] >= trend_threshold, 'market_condition'] = 'trending'
    signals.loc[signals['adx'] <= sideways_threshold, 'market_condition'] = 'sideways'
    
    # Determine recommended strategy for each day
    signals['strategy'] = signals['market_condition'].apply(
        lambda x: 'moving_average_crossover' if x == 'trending' else (
            'rsi_reversal' if x == 'sideways' else 'moving_average_crossover'
        )
    )
    
    return signals

def plot_market_conditions(signals, index_symbol='NIFTY'):
    """Plot market conditions and recommended strategies."""
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Price chart
    plt.subplot(3, 1, 1)
    plt.plot(signals.index, signals['close'], label='Close Price')
    plt.title(f'{index_symbol} Price Chart')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: ADX
    plt.subplot(3, 1, 2)
    plt.plot(signals.index, signals['adx'], label='ADX')
    plt.plot(signals.index, signals['plus_di'], label='+DI')
    plt.plot(signals.index, signals['minus_di'], label='-DI')
    plt.axhline(y=signals['adx'].mean(), color='r', linestyle='--', label='Mean ADX')
    plt.axhline(y=25, color='g', linestyle='--', label='Trend Threshold')
    plt.axhline(y=20, color='y', linestyle='--', label='Sideways Threshold')
    plt.title('ADX Indicator')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Market Conditions and Strategies
    plt.subplot(3, 1, 3)
    
    # Create color map for market conditions
    condition_colors = {
        'trending': 'green',
        'sideways': 'red',
        'neutral': 'blue'
    }
    
    # Map conditions to colors
    colors = [condition_colors[condition] for condition in signals['market_condition']]
    
    # Plot as bar chart
    plt.bar(signals.index, signals['adx'], color=colors)
    
    # Add annotations
    for i, (idx, row) in enumerate(signals.iterrows()):
        if i % 3 == 0:  # Only annotate every 3rd point to avoid clutter
            plt.annotate(
                row['market_condition'][0].upper(),  # Just the first letter
                (idx, row['adx'] + 1),
                ha='center'
            )
    
    plt.title('Market Conditions and Recommended Strategies')
    
    # Add a legend for conditions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Trending (MAC Strategy)'),
        Patch(facecolor='red', label='Sideways (RSI Strategy)'),
        Patch(facecolor='blue', label='Neutral (Default Strategy)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'data/market_conditions_{index_symbol}.png')
    logger.info(f"Plot saved to data/market_conditions_{index_symbol}.png")
    
    # Show the plot
    plt.show()

def show_sample_signals(api, symbol='RELIANCE', days=30):
    """Show sample signals for different strategies."""
    # Get stock data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
    
    data = api.get_historical_data(
        symbol=symbol,
        timeframe='1day',  # Daily data for better signal visualization
        from_date=start_date,
        to_date=end_date
    )
    
    if data is None or data.empty:
        logger.error(f"Failed to get historical data for {symbol}")
        return
    
    # Create strategies
    mac_strategy = MovingAverageCrossover()
    rsi_strategy = RSIReversal()
    
    # Generate signals
    mac_signals = mac_strategy.generate_signals(data)
    rsi_signals = rsi_strategy.generate_signals(data)
    
    # Plot signals
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Price chart with MAC signals
    plt.subplot(3, 1, 1)
    plt.plot(mac_signals.index, mac_signals['close'], label='Close Price')
    plt.plot(mac_signals.index, mac_signals['short_ma'], label=f'{mac_strategy.params["short_window"]} EMA')
    plt.plot(mac_signals.index, mac_signals['long_ma'], label=f'{mac_strategy.params["long_window"]} EMA')
    
    # Plot buy signals
    buy_signals = mac_signals[mac_signals['signal'] == 1]
    sell_signals = mac_signals[mac_signals['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{symbol} - Moving Average Crossover Strategy')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(rsi_signals.index, rsi_signals['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold')
    plt.title('RSI Indicator')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Price chart with RSI signals
    plt.subplot(3, 1, 3)
    plt.plot(rsi_signals.index, rsi_signals['close'], label='Close Price')
    
    # Plot buy signals
    buy_signals = rsi_signals[rsi_signals['signal'] == 1]
    sell_signals = rsi_signals[rsi_signals['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{symbol} - RSI Reversal Strategy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'data/strategy_signals_{symbol}.png')
    logger.info(f"Plot saved to data/strategy_signals_{symbol}.png")
    
    # Show the plot
    plt.show()
    
    # Calculate statistics
    mac_buy_signals = len(mac_signals[mac_signals['signal'] == 1])
    mac_sell_signals = len(mac_signals[mac_signals['signal'] == -1])
    rsi_buy_signals = len(rsi_signals[rsi_signals['signal'] == 1])
    rsi_sell_signals = len(rsi_signals[rsi_signals['signal'] == -1])
    
    print(f"\nStrategy Signals for {symbol} ({days} days):")
    print(f"  Moving Average Crossover: {mac_buy_signals} buy, {mac_sell_signals} sell")
    print(f"  RSI Reversal: {rsi_buy_signals} buy, {rsi_sell_signals} sell")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate strategy selection')
    parser.add_argument('--index', type=str, default='NIFTY', help='Index symbol for market condition analysis')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Stock symbol for signal demonstration')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    
    args = parser.parse_args()
    
    # Check if API credentials are set
    if not config.get('ANGEL_ONE_API_KEY'):
        logger.error("API credentials not set. Please set them in .env file.")
        sys.exit(1)
    
    # Initialize API
    api = AngelOneAPI()
    
    # Authenticate with API
    if not api.authenticate():
        logger.error("Failed to authenticate with API")
        sys.exit(1)
    
    # Analyze market conditions
    signals = analyze_market_conditions(api, args.index, args.days)
    if signals is not None:
        # Count conditions
        condition_counts = signals['market_condition'].value_counts()
        
        print(f"\nMarket Conditions for {args.index} (last {args.days} days):")
        for condition, count in condition_counts.items():
            print(f"  {condition.capitalize()}: {count} days ({count/len(signals)*100:.1f}%)")
        
        # Plot market conditions
        plot_market_conditions(signals, args.index)
        
        # Show sample signals
        show_sample_signals(api, args.symbol, args.days)

if __name__ == '__main__':
    main()