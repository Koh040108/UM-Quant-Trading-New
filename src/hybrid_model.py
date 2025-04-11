"""
Hybrid model combining HMM, XGBoost and LSTM for trading signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hmm_model import MarketHMM
from src.xgboost_model import XGBoostPredictor
from src.lstm_model import LSTMPredictor

class HybridTradingModel:
    """
    Hybrid trading model that combines HMM, XGBoost, and LSTM predictions.
    """
    
    def __init__(self, n_states=5, n_lags=2, window_size=30, use_lstm=True, random_state=42):
        """
        Initialize the hybrid model.
        
        Args:
            n_states (int): Number of HMM states
            n_lags (int): Number of lag features for XGBoost
            window_size (int): Size of the lookback window for LSTM
            use_lstm (bool): Whether to use LSTM model
            random_state (int): Random seed for reproducibility
        """
        self.hmm_model = MarketHMM(n_states=n_states)
        self.xgb_model = XGBoostPredictor(n_lags=n_lags, random_state=random_state)
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.lstm_model = LSTMPredictor(window_size=window_size)
        else:
            self.lstm_model = None
        
    def fit(self, df, price_col='price'):
        """
        Fit all models to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
        """
        print("Training HMM model...")
        self.hmm_model.fit(df)
        
        print("Training XGBoost model...")
        self.xgb_model.fit(df, price_col=price_col)
        
        if self.use_lstm:
            print("Training LSTM model...")
            self.lstm_model.fit(df, price_col=price_col)
        
    def predict(self, df, price_col='price', threshold=0.0):
        """
        Generate predictions from all models.
        
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
        
        if self.use_lstm:
            # Generate LSTM predictions and signals
            print("Generating LSTM predictions...")
            lstm_data = self.lstm_model.predict(df, price_col=price_col)
            lstm_signals = self.lstm_model.generate_trading_signals(lstm_data)
            
            # Combine all three models
            print("Combining signals from all three models...")
            combined = self._combine_all_signals(hmm_signals, xgb_signals, lstm_signals)
        else:
            # Combine just HMM and XGBoost
            print("Combining HMM and XGBoost signals...")
            combined = self._combine_signals(hmm_signals, xgb_signals)
        
        return combined
    
    def _combine_signals(self, hmm_df, xgb_df):
        """
        Combine signals from HMM and XGBoost models.
        
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
    
    def _combine_all_signals(self, hmm_df, xgb_df, lstm_df):
        """
        Combine signals from all three models (HMM, XGBoost, and LSTM).
        
        Args:
            hmm_df (pd.DataFrame): DataFrame with HMM signals
            xgb_df (pd.DataFrame): DataFrame with XGBoost signals
            lstm_df (pd.DataFrame): DataFrame with LSTM signals
            
        Returns:
            pd.DataFrame: DataFrame with combined signals
        """
        # First combine HMM and XGBoost
        result = self._combine_signals(hmm_df, xgb_df)
        
        # Check that we have data to work with
        if result.empty:
            print("Warning: Empty result after combining HMM and XGBoost signals")
            return result
            
        # Find common indices for all three datasets
        common_idx = result.index.intersection(lstm_df.index)
        
        if len(common_idx) == 0:
            print("Warning: No common indices between combined signals and LSTM signals")
            return result
            
        # Make sure we have aligned data
        result = result.loc[common_idx].copy()
        lstm_aligned = lstm_df.loc[common_idx].copy()
        
        # Add LSTM columns
        for col in ['lstm_pred', 'lstm_signal']:
            if col in lstm_aligned.columns:
                result[col] = lstm_aligned[col]
        
        # Check if we have all required signal columns
        required_columns = ['signal', 'xgb_signal', 'lstm_signal']
        for col in required_columns:
            if col not in result.columns:
                print(f"Warning: Missing required column '{col}' in combined signals")
                if col == 'signal':
                    result['signal'] = 0
                elif col == 'xgb_signal':
                    result['xgb_signal'] = 0
                elif col == 'lstm_signal':
                    result['lstm_signal'] = 0
        
        # Create a temporary matrix of signals for voting
        signal_matrix = pd.DataFrame({
            'hmm': result['signal'],
            'xgb': result['xgb_signal'],
            'lstm': result['lstm_signal']
        })
        
        # Count votes for each direction
        signal_matrix['buy_votes'] = (signal_matrix > 0).sum(axis=1)
        signal_matrix['sell_votes'] = (signal_matrix < 0).sum(axis=1)
        signal_matrix['neutral_votes'] = (signal_matrix == 0).sum(axis=1)
        
        # Create ensemble signal based on voting
        result['ensemble_signal'] = 0
        
        # Unanimous agreement
        mask_unanimous_buy = signal_matrix['buy_votes'] == 3
        mask_unanimous_sell = signal_matrix['sell_votes'] == 3
        result.loc[mask_unanimous_buy, 'ensemble_signal'] = 1
        result.loc[mask_unanimous_sell, 'ensemble_signal'] = -1
        
        # Majority agreement (2 out of 3)
        mask_majority_buy = (signal_matrix['buy_votes'] == 2) & (signal_matrix['sell_votes'] < 2)
        mask_majority_sell = (signal_matrix['sell_votes'] == 2) & (signal_matrix['buy_votes'] < 2)
        result.loc[mask_majority_buy, 'ensemble_signal'] = 1
        result.loc[mask_majority_sell, 'ensemble_signal'] = -1
        
        # When there's complete disagreement or majority is neutral, rely on HMM for market regime
        mask_disagree = (signal_matrix['buy_votes'] == 1) & (signal_matrix['sell_votes'] == 1) & (signal_matrix['neutral_votes'] == 1)
        mask_neutral_majority = signal_matrix['neutral_votes'] >= 2
        
        # For disagreement, use HMM (market regime) as tiebreaker
        result.loc[mask_disagree, 'ensemble_signal'] = result.loc[mask_disagree, 'signal']
        
        # For neutral majority, stay neutral (0)
        result.loc[mask_neutral_majority, 'ensemble_signal'] = 0
        
        # Use the ensemble signal as the final combined signal
        result['combined_signal'] = result['ensemble_signal']
        
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
        # Check if the dataframe is empty
        if df.empty:
            print("Error: Empty DataFrame provided for backtesting")
            # Return empty results and default performance metrics
            empty_perf = {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Trades': 0,
                'Buy Trades': 0,
                'Sell Trades': 0,
                'Trading Frequency': 0.0,
                'Long Win Rate': 0.0,
                'Short Win Rate': 0.0,
                'Buy Hold Return': 0.0,
                'Buy Hold Annualized Return': 0.0,
                'Buy Hold Volatility': 0.0,
                'Buy Hold Sharpe': 0.0
            }
            return pd.DataFrame(), empty_perf
        
        # Use the HMM model's backtest function with our combined signal
        results = df.copy()
        
        # Replace the original signal with our combined signal
        if 'combined_signal' in results.columns:
            results['signal'] = results['combined_signal']
        elif 'ensemble_signal' in results.columns:
            results['signal'] = results['ensemble_signal']
        
        # Check if we have a signal column
        if 'signal' not in results.columns:
            print("Error: No signal column found in results DataFrame")
            # Try to create one from other available signals
            if 'hmm_signal' in results.columns:
                results['signal'] = results['hmm_signal']
            elif 'xgb_signal' in results.columns:
                results['signal'] = results['xgb_signal']
            elif 'lstm_signal' in results.columns:
                results['signal'] = results['lstm_signal']
            else:
                # Default to neutral signal
                results['signal'] = 0
        
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
            
            # Check that positions has the right length before setting it
            if len(positions) <= len(results):
                if len(positions) < len(results):
                    # Fill the missing positions with the last value or 0
                    missing = len(results) - len(positions)
                    last_position = positions[-1] if positions else 0
                    positions.extend([last_position] * missing)
                
                results['strategy_position'] = positions
            else:
                # Something went wrong with position calculation
                print("Warning: Position calculation issue - using neutral positions")
                results['strategy_position'] = 0
        
        # Run the backtest
        try:
            return self.hmm_model.backtest_strategy(
                results, price_col=price_col, fee=fee, allow_shorts=allow_shorts
            )
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            # Create emergency fallback performance metrics
            empty_perf = {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Trades': 0,
                'Buy Trades': 0,
                'Sell Trades': 0,
                'Trading Frequency': 0.0,
                'Long Win Rate': 0.0,
                'Short Win Rate': 0.0,
                'Buy Hold Return': 0.0,
                'Buy Hold Annualized Return': 0.0,
                'Buy Hold Volatility': 0.0,
                'Buy Hold Sharpe': 0.0
            }
            return results, empty_perf
    
    def plot_signals(self, df, price_col='price'):
        """
        Plot the hybrid model signals.
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            price_col (str): Name of the price column
        """
        plt.figure(figsize=(14, 15))
        
        # Plot price and smoothed predictions
        plt.subplot(5, 1, 1)
        plt.plot(df[price_col], label='Price')
        if 'kalman' in df.columns:
            plt.plot(df['kalman'], label='Kalman Smoothed')
        plt.legend()
        plt.title('Price and Smoothed Prediction')
        
        # Plot HMM states
        plt.subplot(5, 1, 2)
        plt.plot(df[price_col], alpha=0.3, label='Price')
        plt.scatter(df.index, df[price_col], c=df['hmm_state'], s=10, cmap='viridis', label='HMM States')
        plt.legend()
        plt.title('HMM States')
        
        # Plot derivatives
        plt.subplot(5, 1, 3)
        if 'gradient' in df.columns and 'second_deriv' in df.columns:
            plt.plot(df['gradient'], label='First Derivative')
            plt.plot(df['second_deriv'], label='Second Derivative')
            plt.axhline(y=0.02, color='g', linestyle='--', label='Buy Threshold')
            plt.axhline(y=-0.03, color='r', linestyle='--', label='Sell Threshold')
            plt.legend()
            plt.title('Derivatives')
        
        # Plot LSTM predictions if available
        if 'lstm_pred' in df.columns:
            plt.subplot(5, 1, 4)
            plt.plot(df[price_col], alpha=0.3, label='Price')
            plt.plot(df['lstm_pred'], label='LSTM Prediction', color='purple')
            plt.legend()
            plt.title('LSTM Predictions')
        
        # Plot all signals
        subplot_position = 5 if 'lstm_pred' in df.columns else 4
        plt.subplot(5, 1, subplot_position)
        plt.plot(df[price_col], alpha=0.3, label='Price')
        
        # HMM signals
        hmm_buy_mask = df['signal'] == 1
        hmm_sell_mask = df['signal'] == -1
        
        # XGBoost signals
        xgb_buy_mask = df['xgb_signal'] == 1
        xgb_sell_mask = df['xgb_signal'] == -1
        
        # LSTM signals
        if 'lstm_signal' in df.columns:
            lstm_buy_mask = df['lstm_signal'] == 1
            lstm_sell_mask = df['lstm_signal'] == -1
        
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
        
        # Plot LSTM signals if available
        if 'lstm_signal' in df.columns:
            plt.scatter(df[lstm_buy_mask].index, df.loc[lstm_buy_mask, price_col], 
                       color='b', marker='^', s=30, alpha=0.5, label='LSTM Buy')
            plt.scatter(df[lstm_sell_mask].index, df.loc[lstm_sell_mask, price_col], 
                       color='y', marker='v', s=30, alpha=0.5, label='LSTM Sell')
        
        # Plot combined signals (larger markers)
        plt.scatter(df[combined_buy_mask].index, df.loc[combined_buy_mask, price_col], 
                   color='g', marker='^', s=100, label='Combined Buy')
        plt.scatter(df[combined_sell_mask].index, df.loc[combined_sell_mask, price_col], 
                   color='r', marker='v', s=100, label='Combined Sell')
        
        plt.legend()
        plt.title('Trading Signals')
        
        plt.tight_layout()
        plt.show() 