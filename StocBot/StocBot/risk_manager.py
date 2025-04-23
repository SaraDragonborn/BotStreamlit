"""
Risk Manager
=======================================
Manages and controls trading risks based on portfolio, market conditions, and strategy performance.
"""

import os
import json
import datetime
import math
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from utils.logger import get_trade_logger

logger = get_trade_logger()

class RiskManager:
    """
    Risk Manager component.
    
    Responsible for:
    1. Position sizing
    2. Risk allocation
    3. Stop loss and take profit placement
    4. Portfolio-level risk controls
    5. Drawdown management
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,  # Max 2% portfolio risk per day
                 max_position_risk: float = 0.01,  # Max 1% risk per position
                 max_correlation_allowed: float = 0.7,  # Max correlation between positions
                 max_sector_allocation: float = 0.25,  # Max 25% in one sector
                 target_positions: int = 10,  # Target number of positions
                 use_atr_for_stops: bool = True,  # Use ATR for stop loss calculation
                 atr_stop_multiplier: float = 2.0,  # ATR multiplier for stops
                 profit_risk_ratio: float = 2.0,  # Target risk:reward ratio
                 max_drawdown_exit: float = 0.2,  # Exit all if drawdown exceeds 20%
                 data_directory: str = 'data/risk'):
        """
        Initialize the Risk Manager.
        
        Parameters:
        -----------
        max_portfolio_risk : float, default=0.02
            Maximum daily portfolio risk as decimal (0.02 = 2%)
        max_position_risk : float, default=0.01
            Maximum risk per position as decimal (0.01 = 1%)
        max_correlation_allowed : float, default=0.7
            Maximum correlation allowed between positions
        max_sector_allocation : float, default=0.25
            Maximum allocation to a single sector as decimal (0.25 = 25%)
        target_positions : int, default=10
            Target number of positions in portfolio
        use_atr_for_stops : bool, default=True
            Whether to use ATR for stop loss calculation
        atr_stop_multiplier : float, default=2.0
            Multiplier for ATR when calculating stop loss
        profit_risk_ratio : float, default=2.0
            Target profit-to-risk ratio (2.0 = 2:1 risk:reward)
        max_drawdown_exit : float, default=0.2
            Maximum drawdown allowed before exiting all positions (0.2 = 20%)
        data_directory : str, default='data/risk'
            Directory for risk data storage
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation_allowed = max_correlation_allowed
        self.max_sector_allocation = max_sector_allocation
        self.target_positions = target_positions
        self.use_atr_for_stops = use_atr_for_stops
        self.atr_stop_multiplier = atr_stop_multiplier
        self.profit_risk_ratio = profit_risk_ratio
        self.max_drawdown_exit = max_drawdown_exit
        self.data_directory = data_directory
        
        # Initialize state
        self.portfolio_value = 0.0
        self.buying_power = 0.0
        self.current_positions = {}  # {symbol: {size, entry_price, ...}}
        self.position_risks = {}     # {symbol: risk_amount}
        self.sector_allocations = {} # {sector: allocation}
        self.drawdown_history = []
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        
        # Ensure data directory exists
        os.makedirs(self.data_directory, exist_ok=True)
        
        logger.info(f"Risk Manager initialized with max position risk: {max_position_risk:.1%}, "
                   f"target positions: {target_positions}")
    
    def update_portfolio_state(self, 
                              portfolio_value: float,
                              buying_power: float,
                              positions: Dict[str, Dict],
                              sector_data: Optional[Dict[str, str]] = None):
        """
        Update the current portfolio state.
        
        Parameters:
        -----------
        portfolio_value : float
            Current portfolio value
        buying_power : float
            Current buying power (cash available)
        positions : Dict[str, Dict]
            Current positions {symbol: {size, entry_price, current_price, ...}}
        sector_data : Dict[str, str], optional
            Sector information for symbols {symbol: sector}
        """
        self.portfolio_value = portfolio_value
        self.buying_power = buying_power
        self.current_positions = positions
        
        # Update peak equity for drawdown calculation
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = 1 - (portfolio_value / self.peak_equity)
            self.drawdown_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'peak_equity': self.peak_equity,
                'drawdown': self.current_drawdown
            })
        
        # Calculate sector allocations if sector data provided
        if sector_data:
            self.sector_allocations = {}
            for symbol, position in positions.items():
                sector = sector_data.get(symbol, 'Unknown')
                position_value = position.get('market_value', 0)
                
                if sector not in self.sector_allocations:
                    self.sector_allocations[sector] = 0
                
                self.sector_allocations[sector] += position_value
            
            # Convert to allocation percentages
            if portfolio_value > 0:
                self.sector_allocations = {
                    sector: value / portfolio_value
                    for sector, value in self.sector_allocations.items()
                }
        
        # Log current portfolio state
        logger.info(f"Portfolio updated: Value=${portfolio_value:.2f}, "
                   f"Buying power=${buying_power:.2f}, "
                   f"Positions={len(positions)}, "
                   f"Drawdown={self.current_drawdown:.2%}")
    
    def calculate_position_size(self, 
                               symbol: str,
                               entry_price: float,
                               stop_price: float,
                               strategy_weight: float = 1.0) -> Tuple[int, Dict]:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Parameters:
        -----------
        symbol : str
            Symbol to trade
        entry_price : float
            Expected entry price
        stop_price : float
            Stop loss price
        strategy_weight : float, default=1.0
            Weight of the strategy signal (0.0 to 1.0)
            
        Returns:
        --------
        Tuple[int, Dict]
            (Position size in shares/contracts, Position info dict)
        """
        # Validate inputs
        if entry_price <= 0:
            logger.error(f"Invalid entry price: {entry_price}")
            return 0, {'error': 'Invalid entry price'}
        
        if stop_price <= 0:
            logger.error(f"Invalid stop price: {stop_price}")
            return 0, {'error': 'Invalid stop price'}
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            logger.error(f"Risk per share is zero or negative: {risk_per_share}")
            return 0, {'error': 'Invalid risk per share'}
        
        # Determine dollar risk based on portfolio and position risk limits
        portfolio_risk_amount = self.portfolio_value * self.max_portfolio_risk
        position_risk_amount = self.portfolio_value * self.max_position_risk
        
        # Adjust for number of positions
        target_risk_per_position = portfolio_risk_amount / self.target_positions
        risk_amount = min(position_risk_amount, target_risk_per_position)
        
        # Adjust for strategy weight
        risk_amount = risk_amount * strategy_weight
        
        # Calculate shares based on risk
        shares = math.floor(risk_amount / risk_per_share)
        
        # Calculate total position value
        position_value = shares * entry_price
        
        # Ensure we have enough buying power
        if position_value > self.buying_power:
            # Adjust shares to match available buying power
            shares = math.floor(self.buying_power / entry_price)
            logger.warning(f"Position size reduced due to buying power constraints: {shares} shares")
        
        # Check if this is a meaningful position
        if shares <= 0:
            logger.warning(f"Position size calculation resulted in zero shares for {symbol}")
            return 0, {'error': 'Position size too small'}
        
        # Create position info
        position_info = {
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'take_profit_price': entry_price + (risk_per_share * self.profit_risk_ratio),
            'shares': shares,
            'risk_amount': shares * risk_per_share,
            'position_value': shares * entry_price,
            'portfolio_risk_pct': (shares * risk_per_share) / self.portfolio_value if self.portfolio_value > 0 else 0
        }
        
        # Log position sizing info
        logger.info(f"Calculated position for {symbol}: {shares} shares, "
                   f"risk=${position_info['risk_amount']:.2f} ({position_info['portfolio_risk_pct']:.2%})")
        
        # Store position risk info
        self.position_risks[symbol] = position_info['risk_amount']
        
        return shares, position_info
    
    def calculate_stop_loss(self, 
                           data: pd.DataFrame, 
                           entry_price: float, 
                           direction: str = 'long',
                           atr_periods: int = 14) -> float:
        """
        Calculate the stop loss price based on ATR or fixed percentage.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Recent price data
        entry_price : float
            Expected entry price
        direction : str, default='long'
            Trade direction ('long' or 'short')
        atr_periods : int, default=14
            Periods for ATR calculation
            
        Returns:
        --------
        float
            Stop loss price
        """
        if self.use_atr_for_stops and len(data) >= atr_periods:
            # Calculate ATR
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=atr_periods).mean().iloc[-1]
            
            # Calculate stop distance
            stop_distance = atr * self.atr_stop_multiplier
            
            # Calculate stop price
            if direction.lower() == 'long':
                stop_price = entry_price - stop_distance
            else:  # short
                stop_price = entry_price + stop_distance
        else:
            # Use fixed percentage if ATR not available
            stop_percentage = 0.05  # 5% default
            
            if direction.lower() == 'long':
                stop_price = entry_price * (1 - stop_percentage)
            else:  # short
                stop_price = entry_price * (1 + stop_percentage)
        
        return stop_price
    
    def check_correlation_risk(self, new_symbol: str, price_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Check if adding a new position would increase correlation risk.
        
        Parameters:
        -----------
        new_symbol : str
            Symbol being considered for a new position
        price_data : Dict[str, pd.DataFrame]
            Price data for current portfolio symbols and new symbol
            
        Returns:
        --------
        bool
            True if correlation risk is acceptable, False otherwise
        """
        # If we don't have portfolio positions or price data, correlation is fine
        if not self.current_positions or new_symbol not in price_data:
            return True
        
        # Calculate correlations
        correlations = {}
        new_returns = price_data[new_symbol]['close'].pct_change().dropna()
        
        for symbol in self.current_positions:
            if symbol in price_data:
                current_returns = price_data[symbol]['close'].pct_change().dropna()
                
                # Ensure we have matching dates by reindexing
                if len(new_returns) > 0 and len(current_returns) > 0:
                    # Get common index
                    common_index = new_returns.index.intersection(current_returns.index)
                    
                    if len(common_index) >= 20:  # Need sufficient data points
                        corr = new_returns[common_index].corr(current_returns[common_index])
                        correlations[symbol] = corr
        
        # Check if any correlations exceed threshold
        high_correlations = [symbol for symbol, corr in correlations.items() 
                            if abs(corr) > self.max_correlation_allowed]
        
        if high_correlations:
            logger.warning(f"High correlation for {new_symbol} with: {', '.join(high_correlations)}")
            return False
        
        return True
    
    def check_sector_risk(self, new_symbol: str, sector: str) -> bool:
        """
        Check if adding a new position would exceed sector allocation limits.
        
        Parameters:
        -----------
        new_symbol : str
            Symbol being considered for a new position
        sector : str
            Sector of the new symbol
            
        Returns:
        --------
        bool
            True if sector risk is acceptable, False otherwise
        """
        current_sector_allocation = self.sector_allocations.get(sector, 0)
        
        if current_sector_allocation >= self.max_sector_allocation:
            logger.warning(f"Sector allocation limit reached for {sector}: {current_sector_allocation:.2%}")
            return False
        
        return True
    
    def check_drawdown_limit(self) -> bool:
        """
        Check if current drawdown exceeds limit.
        
        Returns:
        --------
        bool
            True if drawdown is within limits, False if exceeded
        """
        if self.current_drawdown >= self.max_drawdown_exit:
            logger.warning(f"Maximum drawdown exceeded: {self.current_drawdown:.2%}")
            return False
        
        return True
    
    def get_portfolio_risk_exposure(self) -> Dict:
        """
        Get the current portfolio risk exposure.
        
        Returns:
        --------
        Dict
            Portfolio risk metrics
        """
        total_position_risk = sum(self.position_risks.values())
        portfolio_risk_pct = total_position_risk / self.portfolio_value if self.portfolio_value > 0 else 0
        
        risk_metrics = {
            'total_risk_amount': total_position_risk,
            'portfolio_risk_pct': portfolio_risk_pct,
            'position_count': len(self.current_positions),
            'sector_allocations': self.sector_allocations,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity
        }
        
        return risk_metrics
    
    def adjust_position_for_volatility(self, symbol: str, volatility_percentile: float, shares: int) -> int:
        """
        Adjust position size based on relative volatility.
        
        Parameters:
        -----------
        symbol : str
            Symbol to trade
        volatility_percentile : float
            Percentile of current volatility (0-1 range)
        shares : int
            Original position size in shares
            
        Returns:
        --------
        int
            Adjusted position size
        """
        # Inverse relationship - higher volatility = smaller position
        volatility_factor = 1 - (volatility_percentile * 0.5)  # Scale to 0.5-1.0 range
        
        # Apply adjustment
        adjusted_shares = math.floor(shares * volatility_factor)
        
        # Ensure minimum position size
        adjusted_shares = max(1, adjusted_shares)
        
        if adjusted_shares < shares:
            logger.info(f"Position size for {symbol} adjusted for volatility: {shares} -> {adjusted_shares}")
        
        return adjusted_shares
    
    def should_exit_all_positions(self) -> bool:
        """
        Determine if all positions should be exited based on risk thresholds.
        
        Returns:
        --------
        bool
            True if all positions should be exited, False otherwise
        """
        # Check drawdown limit
        if not self.check_drawdown_limit():
            return True
        
        # Add other risk checks as needed
        
        return False
    
    def save_risk_state(self) -> None:
        """
        Save current risk state to file.
        """
        filepath = os.path.join(self.data_directory, 'risk_state.json')
        
        try:
            state = {
                'timestamp': datetime.datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'buying_power': self.buying_power,
                'current_drawdown': self.current_drawdown,
                'peak_equity': self.peak_equity,
                'position_count': len(self.current_positions),
                'sector_allocations': self.sector_allocations,
                'position_risks': self.position_risks
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Risk state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving risk state: {str(e)}")
    
    def load_risk_state(self) -> bool:
        """
        Load risk state from file.
        
        Returns:
        --------
        bool
            True if loaded successfully, False otherwise
        """
        filepath = os.path.join(self.data_directory, 'risk_state.json')
        
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.portfolio_value = state.get('portfolio_value', 0)
                self.buying_power = state.get('buying_power', 0)
                self.current_drawdown = state.get('current_drawdown', 0)
                self.peak_equity = state.get('peak_equity', 0)
                self.sector_allocations = state.get('sector_allocations', {})
                self.position_risks = state.get('position_risks', {})
                
                logger.info(f"Risk state loaded from {filepath}")
                return True
            else:
                logger.warning(f"Risk state file not found: {filepath}")
                return False
            
        except Exception as e:
            logger.error(f"Error loading risk state: {str(e)}")
            return False