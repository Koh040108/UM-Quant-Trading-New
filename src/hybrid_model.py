"""
Hybrid model combining HMM and XGBoost for trading signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hmm_model import MarketHMM
from src.xgboost_model import XGBoostPredictor

class HybridTradingModel:
    """
    Hybrid trading model that combines HMM and XGBoost predictions.
    """
    
    def __init__(self, n_states=5, n_lags=2, random_state=42):
        """
        Initialize the hybrid model.
        
        Args:
            n_states (int): Number of HMM states
            n_lags (int): Number of lag features for XGBoost
            random_state (int): Random seed for reproducibility
        """
        self.hmm_model = MarketHMM(n_states=n_states)
        self.xgb_model = XGBoostPredictor(n_lags=n_lags, random_state=random_state)
        
    def fit(self, df, price_col='price'):
        """
        Fit both models to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
        """
        print("Training HMM model...")
        self.hmm_model.fit(df)
        
        print("Training XGBoost model...")
        self.xgb_model.fit(df, price_col=price_col)
        
    def predict(self, df, price_col='price', threshold=0.0):
        """
        Generate predictions from both models.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
            threshold (float): HMM threshold
            
        Returns:
            pd.DataFrame: DataFrame with predictions and signals
        """
        # Add HMM states to the data
        print("Generating HMM states...")
        hmm_data = self.hmm_model.add_states_to_df(df)
        
        # Generate HMM signals
        hmm_signals = self.hmm_model.generate_trading_signals(
            hmm_data, threshold=threshold, price_col=price_col
        )
        
        # Generate XGBoost predictions and signals
        print("Generating XGBoost predictions...")
        xgb_data = self.xgb_model.predict(df, price_col=price_col)
        xgb_signals = self.xgb_model.generate_trading_signals(xgb_data)
        
        # Combine the results
        print("Combining signals...")
        combined = self._combine_signals(hmm_signals, xgb_signals)
        
        return combined
    
    def _combine_signals(self, hmm_df, xgb_df):
        """
        Combine signals from both models.
        
        Args:
            hmm_df (pd.DataFrame): DataFrame with HMM signals
            xgb_df (pd.DataFrame): DataFrame with XGBoost signals
            
        Returns:
            pd.DataFrame: DataFrame with combined signals
        """
        # Make sure the indices match
        hmm_df = hmm_df.loc[xgb_df.index]
        
        # Create result DataFrame
        result = hmm_df.copy()
        
        # Add XGBoost columns
        for col in ['xgb_pred', 'savgol', 'kalman', 'gradient', 'second_deriv', 'xgb_signal']:
            if col in xgb_df.columns:
                result[col] = xgb_df[col]
        
        # Generate combined signal
        # Signal rules:
        # 1. If both models agree, use that signal
        # 2. If HMM is neutral but XGBoost has a signal, use XGBoost
        # 3. If HMM has a signal but XGBoost is neutral, use HMM with reduced confidence
        # 4. If models disagree, use the one with higher confidence metrics
        
        result['combined_signal'] = 0
        
        # Both agree
        mask_agree_buy = (result['signal'] == 1) & (result['xgb_signal'] == 1)
        mask_agree_sell = (result['signal'] == -1) & (result['xgb_signal'] == -1)
        result.loc[mask_agree_buy, 'combined_signal'] = 1
        result.loc[mask_agree_sell, 'combined_signal'] = -1
        
        # HMM neutral, XGBoost has signal
        mask_hmm_neutral_xgb_buy = (result['signal'] == 0) & (result['xgb_signal'] == 1)
        mask_hmm_neutral_xgb_sell = (result['signal'] == 0) & (result['xgb_signal'] == -1)
        result.loc[mask_hmm_neutral_xgb_buy, 'combined_signal'] = 1
        result.loc[mask_hmm_neutral_xgb_sell, 'combined_signal'] = -1
        
        # XGBoost neutral, HMM has signal (use HMM)
        mask_xgb_neutral_hmm_buy = (result['xgb_signal'] == 0) & (result['signal'] == 1)
        mask_xgb_neutral_hmm_sell = (result['xgb_signal'] == 0) & (result['signal'] == -1)
        result.loc[mask_xgb_neutral_hmm_buy, 'combined_signal'] = 1
        result.loc[mask_xgb_neutral_hmm_sell, 'combined_signal'] = -1
        
        # Disagreement (prioritize XGBoost's derivative-based signals as they're more responsive)
        mask_disagree_hmm_buy_xgb_sell = (result['signal'] == 1) & (result['xgb_signal'] == -1)
        mask_disagree_hmm_sell_xgb_buy = (result['signal'] == -1) & (result['xgb_signal'] == 1)
        
        # Use gradient strength to decide
        strong_gradient_mask = abs(result['gradient']) > 0.05
        result.loc[mask_disagree_hmm_buy_xgb_sell & strong_gradient_mask, 'combined_signal'] = -1
        result.loc[mask_disagree_hmm_buy_xgb_sell & ~strong_gradient_mask, 'combined_signal'] = 0
        result.loc[mask_disagree_hmm_sell_xgb_buy & strong_gradient_mask, 'combined_signal'] = 1
        result.loc[mask_disagree_hmm_sell_xgb_buy & ~strong_gradient_mask, 'combined_signal'] = 0
        
        return result
    
    def backtest_strategy(self, df, price_col='price', fee=0.001, allow_shorts=True):
        """
        Backtest the hybrid trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            price_col (str): Name of the price column
            fee (float): Trading fee as a percentage
            allow_shorts (bool): Whether to allow short selling
            
        Returns:
            tuple: (results DataFrame, performance metrics dict)
        """
        # Use the HMM model's backtest function with our combined signal
        results = df.copy()
        
        # Replace the original signal with our combined signal
        results['signal'] = results['combined_signal']
        
        # Ensure price column exists
        if price_col not in results.columns:
            if 'close' in results.columns:
                price_col = 'close'
                print(f"Warning: Using 'close' instead of '{price_col}' for backtesting")
            elif 'price' in results.columns:
                price_col = 'price'
                print(f"Warning: Using 'price' instead of '{price_col}' for backtesting")
            elif 'value' in results.columns:
                price_col = 'value'
                print(f"Warning: Using 'value' instead of '{price_col}' for backtesting")
            else:
                raise ValueError(f"Could not find price column for backtesting. Available columns: {results.columns.tolist()}")
        
        # Calculate returns for backtesting
        if 'returns' not in results.columns:
            results['returns'] = results[price_col].pct_change()
            results['returns'].fillna(0, inplace=True)
        
        # Pre-process to ensure all required columns exist
        if 'strategy_position' not in results.columns:
            results['strategy_position'] = 0
            # Calculate positions based on signals
            position = 0
            positions = []
            
            for i, row in results.iterrows():
                signal = row['signal']
                
                # Skip the first row to avoid NaN returns
                if i == results.index[0]:
                    positions.append(0)
                    continue
                    
                # Determine position change based on signal
                if signal == 1 and position <= 0:  # Buy signal when not long
                    position = 1
                elif signal == -1 and position >= 0 and allow_shorts:  # Sell signal when not short
                    position = -1
                elif signal == 0 and position != 0:  # Neutral signal when in a position
                    position = 0
                
                positions.append(position)
            
            # Replace the first position with the second position to handle initialization
            if len(positions) > 1:
                positions[0] = positions[1]
                
            results['strategy_position'] = positions
        
        # Calculate strategy returns with fees if not already calculated
        if 'strategy_returns' not in results.columns:
            results['strategy_returns'] = 0
            
            for i in range(1, len(results)):
                # Check if we made a trade
                if results['strategy_position'].iloc[i] != results['strategy_position'].iloc[i-1]:
                    # Apply fee
                    position_change = results['strategy_position'].iloc[i] - results['strategy_position'].iloc[i-1]
                    # Absolute value of position change determines the fee
                    fee_amount = fee * abs(position_change)
                    results.loc[results.index[i], 'strategy_returns'] = results['returns'].iloc[i] * results['strategy_position'].iloc[i] - fee_amount
                else:
                    # No fee if no position change
                    results.loc[results.index[i], 'strategy_returns'] = results['returns'].iloc[i] * results['strategy_position'].iloc[i]
        
        # Calculate cumulative returns if not already calculated
        if 'strategy_cumulative' not in results.columns:
            results['strategy_cumulative'] = (1 + results['strategy_returns']).cumprod() - 1
        
        if 'buy_hold_returns' not in results.columns:
            results['buy_hold_returns'] = results['returns']
        
        if 'buy_hold_cumulative' not in results.columns:
            results['buy_hold_cumulative'] = (1 + results['buy_hold_returns']).cumprod() - 1
        
        # Calculate portfolio value
        if 'portfolio_value' not in results.columns:
            initial_value = 1.0  # Starting with $1
            results['portfolio_value'] = initial_value * (1 + results['strategy_cumulative'])
        
        # Run the backtest
        results, performance = self.hmm_model.backtest_strategy(
            results, price_col=price_col, fee=fee, allow_shorts=allow_shorts
        )
        
        return results, performance
    
    def plot_signals(self, df, price_col='price'):
        """
        Plot the hybrid model signals.
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            price_col (str): Name of the price column
        """
        plt.figure(figsize=(14, 12))
        
        # Plot price and smoothed predictions
        plt.subplot(4, 1, 1)
        plt.plot(df[price_col], label='Price')
        if 'kalman' in df.columns:
            plt.plot(df['kalman'], label='Kalman Smoothed')
        plt.legend()
        plt.title('Price and Smoothed Prediction')
        
        # Plot HMM states
        plt.subplot(4, 1, 2)
        plt.plot(df[price_col], alpha=0.3, label='Price')
        plt.scatter(df.index, df[price_col], c=df['hmm_state'], s=10, cmap='viridis', label='HMM States')
        plt.legend()
        plt.title('HMM States')
        
        # Plot derivatives
        plt.subplot(4, 1, 3)
        if 'gradient' in df.columns and 'second_deriv' in df.columns:
            plt.plot(df['gradient'], label='First Derivative')
            plt.plot(df['second_deriv'], label='Second Derivative')
            plt.axhline(y=0.02, color='g', linestyle='--', label='Buy Threshold')
            plt.axhline(y=-0.03, color='r', linestyle='--', label='Sell Threshold')
            plt.legend()
            plt.title('Derivatives')
        
        # Plot signals
        plt.subplot(4, 1, 4)
        plt.plot(df[price_col], alpha=0.3, label='Price')
        
        # HMM signals
        hmm_buy_mask = df['signal'] == 1
        hmm_sell_mask = df['signal'] == -1
        
        # XGBoost signals
        xgb_buy_mask = df['xgb_signal'] == 1
        xgb_sell_mask = df['xgb_signal'] == -1
        
        # Combined signals
        combined_buy_mask = df['combined_signal'] == 1
        combined_sell_mask = df['combined_signal'] == -1
        
        # Plot HMM signals
        plt.scatter(df[hmm_buy_mask].index, df.loc[hmm_buy_mask, price_col], 
                   color='g', marker='^', s=30, alpha=0.5, label='HMM Buy')
        plt.scatter(df[hmm_sell_mask].index, df.loc[hmm_sell_mask, price_col], 
                   color='r', marker='v', s=30, alpha=0.5, label='HMM Sell')
        
        # Plot XGBoost signals
        plt.scatter(df[xgb_buy_mask].index, df.loc[xgb_buy_mask, price_col], 
                   color='c', marker='^', s=30, alpha=0.5, label='XGB Buy')
        plt.scatter(df[xgb_sell_mask].index, df.loc[xgb_sell_mask, price_col], 
                   color='m', marker='v', s=30, alpha=0.5, label='XGB Sell')
        
        # Plot combined signals (larger markers)
        plt.scatter(df[combined_buy_mask].index, df.loc[combined_buy_mask, price_col], 
                   color='g', marker='^', s=100, label='Combined Buy')
        plt.scatter(df[combined_sell_mask].index, df.loc[combined_sell_mask, price_col], 
                   color='r', marker='v', s=100, label='Combined Sell')
        
        plt.legend()
        plt.title('Trading Signals')
        
        plt.tight_layout()
        plt.show() 