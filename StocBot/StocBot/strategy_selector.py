"""
Strategy Selector (Brain)
=======================================
Selects the optimal trading strategy based on market conditions and historical performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json
import datetime
from strategies.strategy_base import StrategyBase
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from strategies.trend_strategy import TrendFollowingStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.breakout_strategy import BreakoutStrategy
from utils.logger import strategy_logger, get_strategy_logger

logger = get_strategy_logger()

class StrategySelector:
    """
    Strategy Selector (Brain) component.
    
    Dynamically selects the optimal trading strategy based on:
    1. Current market conditions
    2. Historical strategy performance
    3. Asset characteristics
    4. Market regime detection
    """
    
    def __init__(self, 
                selection_mode: str = 'performance',
                lookback_periods: int = 30,
                performance_metric: str = 'sharpe_ratio',
                weights: Optional[Dict[str, float]] = None):
        """
        Initialize the Strategy Selector.
        
        Parameters:
        -----------
        selection_mode : str, default='performance'
            Strategy selection mode: 'performance', 'rotation', 'ensemble', 'adaptive'
        lookback_periods : int, default=30
            Number of periods to look back for performance evaluation
        performance_metric : str, default='sharpe_ratio'
            Metric to use for performance evaluation: 'sharpe_ratio', 'total_return', 'profit_factor'
        weights : Dict[str, float], optional
            Custom weights for different criteria if using 'adaptive' mode
        """
        self.selection_mode = selection_mode
        self.lookback_periods = lookback_periods
        self.performance_metric = performance_metric
        
        # Initialize weights for adaptive mode
        self.weights = weights or {
            'recent_performance': 0.5,
            'market_regime': 0.3,
            'volatility': 0.2
        }
        
        # Load all available strategies
        self.strategies = self._load_strategies()
        
        # Performance tracking
        self.performance_history = {}
        
        # Cache for market regime detection
        self.current_regime = 'unknown'
        
        logger.info(f"Strategy Selector initialized with mode: {selection_mode}")
    
    def _load_strategies(self) -> Dict[str, StrategyBase]:
        """
        Load all available trading strategies.
        
        Returns:
        --------
        Dict[str, StrategyBase]
            Dictionary of strategy instances
        """
        strategies = {
            'moving_average_crossover': MovingAverageCrossover(),
            'rsi': RSIStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy()
        }
        
        logger.info(f"Loaded {len(strategies)} trading strategies")
        return strategies
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Recent price data
            
        Returns:
        --------
        str
            Market regime: 'trending', 'ranging', 'volatile', 'unknown'
        """
        try:
            # Need at least 30 periods of data
            if len(data) < 30:
                return 'unknown'
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate autocorrelation (trend indicator)
            autocorr = returns.autocorr(lag=1)
            
            # Calculate directional movement (up days vs down days)
            up_days = (returns > 0).sum() / len(returns)
            
            # Determine market regime
            if volatility > 0.3:  # High volatility
                regime = 'volatile'
            elif abs(autocorr) > 0.2:  # Strong autocorrelation indicates trend
                regime = 'trending'
            else:  # Low autocorrelation and volatility indicate range
                regime = 'ranging'
            
            logger.info(f"Market regime detected: {regime} (volatility: {volatility:.4f}, autocorr: {autocorr:.4f})")
            self.current_regime = regime
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'
    
    def _evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Evaluate all strategies on recent data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Recent price data
            
        Returns:
        --------
        Dict[str, Dict]
            Dictionary of strategy evaluation results
        """
        evaluation_results = {}
        
        for name, strategy in self.strategies.items():
            try:
                # Run backtest for evaluation
                result = strategy.backtest(data)
                
                # Store result
                evaluation_results[name] = {
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'total_return': result.get('total_return', 0),
                    'profit_factor': result.get('profit_factor', 1),
                    'win_rate': result.get('win_rate', 0),
                    'max_drawdown': result.get('max_drawdown', 0)
                }
                
                # Update performance history
                if name not in self.performance_history:
                    self.performance_history[name] = []
                
                self.performance_history[name].append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metrics': evaluation_results[name]
                })
                
                # Only keep recent history
                if len(self.performance_history[name]) > 50:  # Keep last 50 evaluations
                    self.performance_history[name] = self.performance_history[name][-50:]
                
            except Exception as e:
                logger.error(f"Error evaluating strategy {name}: {str(e)}")
                evaluation_results[name] = {
                    'sharpe_ratio': 0,
                    'total_return': 0,
                    'profit_factor': 1,
                    'win_rate': 0,
                    'max_drawdown': 0
                }
        
        return evaluation_results
    
    def _select_best_strategy(self, 
                            evaluation_results: Dict[str, Dict], 
                            market_regime: str) -> str:
        """
        Select the best strategy based on evaluation results and market regime.
        
        Parameters:
        -----------
        evaluation_results : Dict[str, Dict]
            Dictionary of strategy evaluation results
        market_regime : str
            Current market regime
            
        Returns:
        --------
        str
            Name of the selected strategy
        """
        if self.selection_mode == 'performance':
            # Select based on single performance metric
            metric = self.performance_metric
            best_strategy = max(evaluation_results.items(), 
                               key=lambda x: x[1].get(metric, 0))[0]
        
        elif self.selection_mode == 'rotation':
            # Rotation based on market regime
            regime_strategy_map = {
                'trending': ['trend_following', 'moving_average_crossover'],
                'ranging': ['mean_reversion', 'rsi'],
                'volatile': ['breakout', 'rsi'],
                'unknown': list(self.strategies.keys())
            }
            
            # Get candidate strategies for current regime
            candidates = regime_strategy_map.get(market_regime, list(self.strategies.keys()))
            
            # Select best performing from candidates
            metric = self.performance_metric
            best_strategy = max(
                [(name, evaluation_results[name].get(metric, 0)) 
                 for name in candidates], 
                key=lambda x: x[1]
            )[0]
        
        elif self.selection_mode == 'ensemble':
            # For ensemble, all strategies are used so just return the best one for logging
            metric = self.performance_metric
            best_strategy = max(evaluation_results.items(), 
                               key=lambda x: x[1].get(metric, 0))[0]
        
        elif self.selection_mode == 'adaptive':
            # Adaptive weighting based on market conditions
            scores = {}
            
            # Regime-based strategy preferences
            regime_scores = {
                'trending': {
                    'trend_following': 1.0,
                    'moving_average_crossover': 0.8,
                    'breakout': 0.7,
                    'rsi': 0.5,
                    'mean_reversion': 0.3
                },
                'ranging': {
                    'mean_reversion': 1.0,
                    'rsi': 0.9,
                    'breakout': 0.6,
                    'moving_average_crossover': 0.4,
                    'trend_following': 0.3
                },
                'volatile': {
                    'breakout': 1.0,
                    'rsi': 0.8,
                    'mean_reversion': 0.7,
                    'trend_following': 0.5,
                    'moving_average_crossover': 0.4
                },
                'unknown': {
                    'moving_average_crossover': 0.8,
                    'rsi': 0.8,
                    'trend_following': 0.8,
                    'mean_reversion': 0.8,
                    'breakout': 0.8
                }
            }
            
            # Normalize performance metrics
            normalized_metrics = {}
            for metric in ['sharpe_ratio', 'total_return', 'profit_factor']:
                values = [result.get(metric, 0) for result in evaluation_results.values()]
                max_value = max(values) if values and max(values) > 0 else 1
                
                normalized_metrics[metric] = {
                    name: result.get(metric, 0) / max_value if max_value > 0 else 0
                    for name, result in evaluation_results.items()
                }
            
            # Calculate final scores
            for name in self.strategies.keys():
                # Performance component
                perf_score = (
                    normalized_metrics['sharpe_ratio'][name] * 0.5 +
                    normalized_metrics['total_return'][name] * 0.3 +
                    normalized_metrics['profit_factor'][name] * 0.2
                )
                
                # Regime component
                regime_score = regime_scores.get(market_regime, regime_scores['unknown']).get(name, 0.5)
                
                # Calculate final score using weights
                scores[name] = (
                    self.weights['recent_performance'] * perf_score +
                    self.weights['market_regime'] * regime_score
                )
            
            # Select strategy with highest score
            best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        
        else:
            # Default to best sharpe ratio
            best_strategy = max(evaluation_results.items(), 
                               key=lambda x: x[1].get('sharpe_ratio', 0))[0]
        
        return best_strategy
    
    def select_strategy(self, data: pd.DataFrame) -> Tuple[str, StrategyBase]:
        """
        Select the optimal trading strategy for the given data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Recent price data
            
        Returns:
        --------
        Tuple[str, StrategyBase]
            Name and instance of the selected strategy
        """
        # Detect market regime
        market_regime = self._detect_market_regime(data)
        
        # Evaluate all strategies
        evaluation_results = self._evaluate_strategies(data)
        
        # Select the best strategy
        best_strategy_name = self._select_best_strategy(evaluation_results, market_regime)
        best_strategy = self.strategies[best_strategy_name]
        
        logger.info(f"Selected strategy: {best_strategy_name} (Mode: {self.selection_mode}, Regime: {market_regime})")
        
        performance_metrics = evaluation_results[best_strategy_name]
        logger.info(f"Strategy metrics: Sharpe: {performance_metrics['sharpe_ratio']:.2f}, " + 
                   f"Return: {performance_metrics['total_return']:.2%}, " +
                   f"PF: {performance_metrics['profit_factor']:.2f}")
        
        return best_strategy_name, best_strategy
    
    def get_ensemble_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using an ensemble of strategies.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Recent price data
            
        Returns:
        --------
        pandas.DataFrame
            Data with ensemble signal column added
        """
        if self.selection_mode != 'ensemble':
            logger.warning("Ensemble signals requested but mode is not 'ensemble'")
        
        # Generate signals for all strategies
        signals_df = data.copy()
        signals_df['ensemble_signal'] = 0
        
        # Evaluate all strategies
        evaluation_results = self._evaluate_strategies(data)
        
        # Calculate weights based on performance
        weights = {}
        metric = self.performance_metric
        total_metric = sum(max(0.1, result.get(metric, 0)) for result in evaluation_results.values())
        
        for name, result in evaluation_results.items():
            metric_value = max(0.1, result.get(metric, 0))  # Ensure positive weight
            weights[name] = metric_value / total_metric if total_metric > 0 else 1.0 / len(evaluation_results)
        
        # Generate and combine signals
        for name, strategy in self.strategies.items():
            strategy_signals = strategy.generate_signals(data)
            weight = weights[name]
            
            # Add weighted signals to ensemble
            signals_df['ensemble_signal'] += strategy_signals['signal'] * weight
        
        # Threshold for ensemble signals
        signals_df['signal'] = 0
        signals_df.loc[signals_df['ensemble_signal'] > 0.3, 'signal'] = 1
        signals_df.loc[signals_df['ensemble_signal'] < -0.3, 'signal'] = -1
        
        # Log the ensemble weights
        logger.info(f"Ensemble weights: {weights}")
        
        # Log the number of signals
        buy_signals = (signals_df['signal'] == 1).sum()
        sell_signals = (signals_df['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals from ensemble")
        
        return signals_df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the optimal strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Recent price data
            
        Returns:
        --------
        pandas.DataFrame
            Data with signal column added
        """
        # Handle ensemble mode separately
        if self.selection_mode == 'ensemble':
            return self.get_ensemble_signals(data)
        
        # Select strategy and generate signals
        strategy_name, strategy = self.select_strategy(data)
        signals = strategy.generate_signals(data)
        
        return signals
    
    def save_performance_history(self, filepath: str = 'data/strategy_performance.json') -> None:
        """
        Save performance history to file.
        
        Parameters:
        -----------
        filepath : str, default='data/strategy_performance.json'
            Path to save the performance history
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            logger.info(f"Performance history saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")
    
    def load_performance_history(self, filepath: str = 'data/strategy_performance.json') -> bool:
        """
        Load performance history from file.
        
        Parameters:
        -----------
        filepath : str, default='data/strategy_performance.json'
            Path to load the performance history from
            
        Returns:
        --------
        bool
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.performance_history = json.load(f)
                
                logger.info(f"Performance history loaded from {filepath}")
                return True
            else:
                logger.warning(f"Performance history file not found: {filepath}")
                return False
            
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
            return False