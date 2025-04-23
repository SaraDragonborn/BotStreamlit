"""
Performance Tracker Module
=======================================
Tracks and analyzes trading performance.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from utils.logger import get_performance_logger
from config import get_config

config = get_config()
logger = get_performance_logger()

class PerformanceTracker:
    """
    Performance Tracker component.
    
    Tracks and analyzes trading performance metrics:
    - P&L tracking
    - Trade statistics
    - Drawdown analysis
    - Strategy performance
    """
    
    def __init__(self, data_directory: str = 'data/performance'):
        """
        Initialize the Performance Tracker.
        
        Parameters:
        -----------
        data_directory : str, default='data/performance'
            Directory for performance data storage
        """
        self.data_directory = data_directory
        
        # Initialize state
        self.trades = []
        self.daily_performance = {}
        self.equity_curve = []
        self.strategy_performance = {}
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Load existing data if available
        self._load_performance_data()
        
        logger.info("Performance Tracker initialized")
    
    def _load_performance_data(self) -> None:
        """Load performance data from files."""
        try:
            # Load trades
            trades_file = os.path.join(self.data_directory, 'trades.json')
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self.trades = json.load(f)
                logger.info(f"Loaded {len(self.trades)} trades from file")
            
            # Load daily performance
            daily_file = os.path.join(self.data_directory, 'daily_performance.json')
            if os.path.exists(daily_file):
                with open(daily_file, 'r') as f:
                    self.daily_performance = json.load(f)
                logger.info(f"Loaded daily performance data")
            
            # Load equity curve
            equity_file = os.path.join(self.data_directory, 'equity_curve.json')
            if os.path.exists(equity_file):
                with open(equity_file, 'r') as f:
                    self.equity_curve = json.load(f)
                logger.info(f"Loaded equity curve data")
            
            # Load strategy performance
            strategy_file = os.path.join(self.data_directory, 'strategy_performance.json')
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r') as f:
                    self.strategy_performance = json.load(f)
                logger.info(f"Loaded strategy performance data")
            
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
    
    def _save_performance_data(self) -> None:
        """Save performance data to files."""
        try:
            # Save trades
            trades_file = os.path.join(self.data_directory, 'trades.json')
            with open(trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
            
            # Save daily performance
            daily_file = os.path.join(self.data_directory, 'daily_performance.json')
            with open(daily_file, 'w') as f:
                json.dump(self.daily_performance, f, indent=2)
            
            # Save equity curve
            equity_file = os.path.join(self.data_directory, 'equity_curve.json')
            with open(equity_file, 'w') as f:
                json.dump(self.equity_curve, f, indent=2)
            
            # Save strategy performance
            strategy_file = os.path.join(self.data_directory, 'strategy_performance.json')
            with open(strategy_file, 'w') as f:
                json.dump(self.strategy_performance, f, indent=2)
            
            logger.info("Performance data saved")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
    
    def add_trade(self, trade: Dict) -> None:
        """
        Add a trade to the performance tracker.
        
        Parameters:
        -----------
        trade : Dict
            Trade information
            - symbol: symbol traded
            - entry_date: entry date/time
            - exit_date: exit date/time (or None if open)
            - entry_price: entry price
            - exit_price: exit price (or None if open)
            - shares: number of shares
            - direction: 'long' or 'short'
            - pnl: profit/loss amount
            - pnl_pct: profit/loss percentage
            - strategy: strategy used
            - market: market traded ('US', 'India', etc.)
        """
        try:
            # Add trade ID if not present
            if 'id' not in trade:
                trade['id'] = len(self.trades) + 1
            
            # Add timestamp if not present
            if 'timestamp' not in trade:
                trade['timestamp'] = datetime.datetime.now().isoformat()
            
            # Calculate P&L if not provided
            if 'pnl' not in trade and trade.get('exit_price') is not None:
                direction_multiplier = 1 if trade.get('direction', 'long').lower() == 'long' else -1
                shares = trade.get('shares', 0)
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                
                trade['pnl'] = (exit_price - entry_price) * shares * direction_multiplier
            
            if 'pnl_pct' not in trade and trade.get('exit_price') is not None:
                direction_multiplier = 1 if trade.get('direction', 'long').lower() == 'long' else -1
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                
                if entry_price > 0:
                    trade['pnl_pct'] = (exit_price / entry_price - 1) * direction_multiplier
            
            # Add to trades list
            self.trades.append(trade)
            
            # Update daily performance
            if trade.get('exit_date'):
                exit_date = trade.get('exit_date')
                if isinstance(exit_date, str):
                    if 'T' in exit_date:
                        exit_date = exit_date.split('T')[0]
                    else:
                        exit_date = exit_date.split(' ')[0]
                
                if exit_date not in self.daily_performance:
                    self.daily_performance[exit_date] = {
                        'pnl': 0,
                        'trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0
                    }
                
                self.daily_performance[exit_date]['pnl'] += trade.get('pnl', 0)
                self.daily_performance[exit_date]['trades'] += 1
                
                if trade.get('pnl', 0) > 0:
                    self.daily_performance[exit_date]['winning_trades'] += 1
                elif trade.get('pnl', 0) < 0:
                    self.daily_performance[exit_date]['losing_trades'] += 1
            
            # Update strategy performance
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'pnl': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0
                }
            
            self.strategy_performance[strategy]['trades'] += 1
            self.strategy_performance[strategy]['pnl'] += trade.get('pnl', 0)
            
            if trade.get('pnl', 0) > 0:
                self.strategy_performance[strategy]['winning_trades'] += 1
            elif trade.get('pnl', 0) < 0:
                self.strategy_performance[strategy]['losing_trades'] += 1
            
            # Calculate strategy metrics
            winning_trades = self.strategy_performance[strategy]['winning_trades']
            losing_trades = self.strategy_performance[strategy]['losing_trades']
            total_trades = self.strategy_performance[strategy]['trades']
            
            if total_trades > 0:
                self.strategy_performance[strategy]['win_rate'] = winning_trades / total_trades
            
            # Calculate average win/loss
            wins = [t.get('pnl', 0) for t in self.trades 
                   if t.get('strategy') == strategy and t.get('pnl', 0) > 0]
            losses = [t.get('pnl', 0) for t in self.trades 
                     if t.get('strategy') == strategy and t.get('pnl', 0) < 0]
            
            if wins:
                self.strategy_performance[strategy]['avg_win'] = sum(wins) / len(wins)
            
            if losses:
                self.strategy_performance[strategy]['avg_loss'] = sum(losses) / len(losses)
            
            # Calculate profit factor
            total_wins = sum(wins)
            total_losses = sum(abs(l) for l in losses)
            
            if total_losses > 0:
                self.strategy_performance[strategy]['profit_factor'] = total_wins / total_losses
            else:
                self.strategy_performance[strategy]['profit_factor'] = float('inf')
            
            # Save data
            self._save_performance_data()
            
            logger.info(f"Trade added: {trade.get('symbol')}, P&L: {trade.get('pnl', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
    
    def update_equity(self, equity: float, timestamp: Optional[str] = None) -> None:
        """
        Update the equity curve.
        
        Parameters:
        -----------
        equity : float
            Current equity value
        timestamp : str, optional
            Timestamp (if None, use current time)
        """
        try:
            if timestamp is None:
                timestamp = datetime.datetime.now().isoformat()
            
            # Add to equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity
            })
            
            # Save data
            self._save_performance_data()
            
            logger.debug(f"Equity updated: {equity:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating equity: {str(e)}")
    
    def get_performance_summary(self) -> Dict:
        """
        Get a summary of trading performance.
        
        Returns:
        --------
        Dict
            Performance summary
        """
        try:
            # Calculate overall metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trades if t.get('pnl', 0) < 0])
            break_even_trades = total_trades - winning_trades - losing_trades
            
            total_pnl = sum(t.get('pnl', 0) for t in self.trades)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average win/loss
            wins = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]
            losses = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]
            
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            # Calculate profit factor
            total_wins = sum(wins)
            total_losses = sum(abs(l) for l in losses)
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Calculate drawdown
            drawdown = 0
            max_drawdown = 0
            peak_equity = 0
            
            if self.equity_curve:
                for point in self.equity_curve:
                    equity = point.get('equity', 0)
                    
                    if equity > peak_equity:
                        peak_equity = equity
                    
                    dd = 1 - (equity / peak_equity) if peak_equity > 0 else 0
                    max_drawdown = max(max_drawdown, dd)
            
            # Get most recent equity
            current_equity = self.equity_curve[-1]['equity'] if self.equity_curve else 0
            
            # Calculate sharpe ratio if we have daily performance
            sharpe_ratio = 0
            
            if self.daily_performance:
                daily_returns = []
                sorted_dates = sorted(self.daily_performance.keys())
                
                for date in sorted_dates:
                    pnl = self.daily_performance[date]['pnl']
                    daily_returns.append(pnl)
                
                if daily_returns:
                    mean_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    
                    if std_return > 0:
                        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
            
            # Compile summary
            summary = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'break_even_trades': break_even_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'current_equity': current_equity,
                'strategy_performance': self.strategy_performance
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def get_daily_performance(self) -> pd.DataFrame:
        """
        Get daily performance as a DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            Daily performance data
        """
        try:
            if not self.daily_performance:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            
            for date, metrics in self.daily_performance.items():
                data.append({
                    'date': date,
                    'pnl': metrics['pnl'],
                    'trades': metrics['trades'],
                    'winning_trades': metrics['winning_trades'],
                    'losing_trades': metrics['losing_trades']
                })
            
            df = pd.DataFrame(data)
            
            # Sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate additional metrics
            if len(df) > 0:
                df['win_rate'] = df['winning_trades'] / df['trades']
                df['cumulative_pnl'] = df['pnl'].cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting daily performance: {str(e)}")
            return pd.DataFrame()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as a DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            Equity curve data
        """
        try:
            if not self.equity_curve:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(self.equity_curve)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting equity curve: {str(e)}")
            return pd.DataFrame()
    
    def get_trades(self, 
                 symbol: Optional[str] = None, 
                 strategy: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get trades as a DataFrame with optional filtering.
        
        Parameters:
        -----------
        symbol : str, optional
            Filter by symbol
        strategy : str, optional
            Filter by strategy
        start_date : str, optional
            Filter by start date
        end_date : str, optional
            Filter by end date
            
        Returns:
        --------
        pandas.DataFrame
            Trades data
        """
        try:
            if not self.trades:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(self.trades)
            
            # Apply filters
            if symbol:
                df = df[df['symbol'] == symbol]
            
            if strategy:
                df = df[df['strategy'] == strategy]
            
            # Convert dates for filtering
            if 'entry_date' in df.columns:
                df['entry_date_dt'] = pd.to_datetime(df['entry_date'])
            
            if 'exit_date' in df.columns:
                df['exit_date_dt'] = pd.to_datetime(df['exit_date'])
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                if 'entry_date_dt' in df.columns:
                    df = df[df['entry_date_dt'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                if 'exit_date_dt' in df.columns:
                    df = df[df['exit_date_dt'] <= end_dt]
            
            # Drop temporary columns
            if 'entry_date_dt' in df.columns:
                df = df.drop('entry_date_dt', axis=1)
                
            if 'exit_date_dt' in df.columns:
                df = df.drop('exit_date_dt', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            return pd.DataFrame()
    
    def plot_equity_curve(self, filename: Optional[str] = None) -> None:
        """
        Plot equity curve.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save plot to
        """
        try:
            df = self.get_equity_curve()
            
            if df.empty:
                logger.warning("No equity data to plot")
                return
            
            # Set up plot
            plt.figure(figsize=(12, 6))
            
            # Plot equity curve
            plt.plot(df['timestamp'], df['equity'], label='Equity')
            
            # Add labels and title
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            
            # Rotate date labels
            plt.gcf().autofmt_xdate()
            
            # Add performance stats
            summary = self.get_performance_summary()
            stats_text = (
                f"Total P&L: ${summary.get('total_pnl', 0):.2f}\n"
                f"Win Rate: {summary.get('win_rate_pct', 0):.2f}%\n"
                f"Profit Factor: {summary.get('profit_factor', 0):.2f}\n"
                f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%\n"
                f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}"
            )
            plt.figtext(0.15, 0.15, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save or show plot
            if filename:
                plt.savefig(filename)
                logger.info(f"Plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
    
    def plot_pnl_distribution(self, filename: Optional[str] = None) -> None:
        """
        Plot P&L distribution.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save plot to
        """
        try:
            df = self.get_trades()
            
            if df.empty or 'pnl' not in df.columns:
                logger.warning("No trade data with P&L to plot")
                return
            
            # Set up plot
            plt.figure(figsize=(10, 6))
            
            # Plot P&L distribution
            plt.hist(df['pnl'], bins=20, alpha=0.7, color='blue')
            
            # Add labels and title
            plt.title('P&L Distribution')
            plt.xlabel('P&L ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Add vertical line at 0
            plt.axvline(x=0, color='r', linestyle='--')
            
            # Add performance stats
            summary = self.get_performance_summary()
            stats_text = (
                f"Total Trades: {summary.get('total_trades', 0)}\n"
                f"Win Rate: {summary.get('win_rate_pct', 0):.2f}%\n"
                f"Avg Win: ${summary.get('avg_win', 0):.2f}\n"
                f"Avg Loss: ${summary.get('avg_loss', 0):.2f}"
            )
            plt.figtext(0.15, 0.15, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save or show plot
            if filename:
                plt.savefig(filename)
                logger.info(f"Plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting P&L distribution: {str(e)}")
    
    def plot_strategy_comparison(self, filename: Optional[str] = None) -> None:
        """
        Plot strategy comparison.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save plot to
        """
        try:
            if not self.strategy_performance:
                logger.warning("No strategy performance data to plot")
                return
            
            # Extract data
            strategies = list(self.strategy_performance.keys())
            win_rates = [self.strategy_performance[s]['win_rate'] * 100 for s in strategies]
            profit_factors = [self.strategy_performance[s]['profit_factor'] for s in strategies]
            pnls = [self.strategy_performance[s]['pnl'] for s in strategies]
            
            # Cap profit factors for better visualization
            profit_factors = [min(pf, 5) for pf in profit_factors]
            
            # Set up plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot win rates
            x = np.arange(len(strategies))
            width = 0.3
            
            ax1.bar(x - width/2, win_rates, width, label='Win Rate (%)', color='blue', alpha=0.7)
            ax1.set_ylabel('Win Rate (%)')
            ax1.set_ylim(0, 100)
            
            # Plot profit factors on second y-axis
            ax2 = ax1.twinx()
            ax2.bar(x + width/2, profit_factors, width, label='Profit Factor', color='green', alpha=0.7)
            ax2.set_ylabel('Profit Factor')
            ax2.set_ylim(0, 5)
            
            # Add labels and title
            plt.title('Strategy Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies, rotation=45, ha='right')
            
            # Add legend
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Add P&L annotations
            for i, pnl in enumerate(pnls):
                ax1.annotate(f"${pnl:.2f}", 
                           xy=(i, win_rates[i] + 5),
                           ha='center',
                           va='bottom')
            
            plt.tight_layout()
            
            # Save or show plot
            if filename:
                plt.savefig(filename)
                logger.info(f"Plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting strategy comparison: {str(e)}")
    
    def export_performance_report(self, filename: str) -> None:
        """
        Export performance report to file.
        
        Parameters:
        -----------
        filename : str
            Filename to save report to
        """
        try:
            # Get summary and data
            summary = self.get_performance_summary()
            daily_df = self.get_daily_performance()
            equity_df = self.get_equity_curve()
            
            # Create report dictionary
            report = {
                'summary': summary,
                'daily_performance': daily_df.to_dict(orient='records') if not daily_df.empty else [],
                'equity_curve': equity_df.to_dict(orient='records') if not equity_df.empty else [],
                'trades': self.trades,
                'strategy_performance': self.strategy_performance,
                'generated_at': datetime.datetime.now().isoformat()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {str(e)}")

# Create global instance
performance_tracker = PerformanceTracker()