"""
Hybrid model combining HMM, XGBoost and LSTM for trading signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import joblib
from datetime import datetime

from src.hmm_model import MarketHMM
from src.xgboost_model import XGBoostPredictor
from src.lstm_model import LSTMPredictor
from src.config import MODELS_DIR

class HybridTradingModel:
    """
    Hybrid trading model that combines HMM for market regime detection, 
    LSTM for sequential forecasting, and XGBoost for feature-based classification.
    
    Architecture:
    1. HMM detects underlying market regimes (bullish, bearish, sideways)
    2. LSTM forecasts price movements using temporal patterns
    3. XGBoost provides classification signals based on engineered features
    4. Ensemble mechanism combines outputs for final trading decisions
    """
    
    def __init__(self, n_states=5, n_lags=2, window_size=30, use_lstm=True, random_state=42):
        """
        Initialize the hybrid model.
        
        Args:
            n_states (int): Number of HMM states for regime detection
            n_lags (int): Number of lag features for XGBoost
            window_size (int): Size of the lookback window for LSTM
            use_lstm (bool): Whether to use LSTM model
            random_state (int): Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_lags = n_lags
        self.window_size = window_size
        self.use_lstm = use_lstm
        self.random_state = random_state
        
        # Initialize HMM for market regime detection
        self.hmm_model = MarketHMM(n_states=n_states, random_state=random_state)
        
        # Initialize XGBoost for feature-based classification
        self.xgb_model = XGBoostPredictor(n_lags=n_lags, random_state=random_state)
        
        # Initialize LSTM for sequential forecasting if enabled
        if use_lstm:
            try:
                # Try importing torch to check if it's available
                import torch
                self.lstm_model = LSTMPredictor(window_size=window_size)
                self.lstm_available = True
            except ImportError:
                print("PyTorch not available. LSTM will not be used.")
                self.lstm_model = None
                self.lstm_available = False
        else:
            self.lstm_model = None
            self.lstm_available = False
        
        # Model weights for ensemble (will be calibrated during training)
        self.model_weights = {
            'hmm': 0.30,
            'xgb': 0.40,
            'lstm': 0.30 if (use_lstm and self.lstm_available) else 0.0
        }
        
        # If LSTM is not available, redistribute weights
        if not self.lstm_available and use_lstm:
            total = self.model_weights['hmm'] + self.model_weights['xgb']
            self.model_weights['hmm'] = self.model_weights['hmm'] / total
            self.model_weights['xgb'] = self.model_weights['xgb'] / total
        
        # Tracking variables
        self.is_trained = False
        
    def fit(self, df, price_col='close', no_tuning=False):
        """
        Fit the hybrid model using all the individual models.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Column name for price data
            no_tuning (bool): Skip parameter tuning for the models
            
        Returns:
            HybridTradingModel: The fitted model
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure we have the price column, prioritizing 'close'
        if price_col not in df_copy.columns:
            price_alternatives = ['close', 'price', 'adjclose', 'adjusted_close']
            for alt in price_alternatives:
                if alt in df_copy.columns:
                    print(f"Price column '{price_col}' not found. Using '{alt}' instead.")
                    price_col = alt
                    break
        
        # Store the detected price column for consistency
        self.detected_price_col = price_col
        
        # Step 1: Train HMM for market regime detection
        print("\nStep 1: Training HMM model for market regime detection...")
        try:
            self.hmm_model.fit(df_copy)
        except Exception as e:
            print(f"Error training HMM model: {str(e)}")
            print("Continuing with other models...")
        
        # Step 2: Add regime information to data
        print("\nStep 2: Adding market regimes to data...")
        try:
            df_with_regimes = self.hmm_model.add_states_to_df(df_copy)
        except Exception as e:
            print(f"Error adding regimes to data: {str(e)}")
            df_with_regimes = df_copy
        
        # Step 3: Train XGBoost model with enhanced features
        print("\nStep 3: Training XGBoost model...")
        try:
            self.xgb_model.fit(df_with_regimes, price_col=price_col, no_tuning=no_tuning)
        except Exception as e:
            print(f"Error training XGBoost model: {str(e)}")
        
        # Step 4: Train LSTM model if available
        if self.use_lstm:
            print("\nStep 4: Training LSTM model...")
            try:
                import torch  # Check if PyTorch is available
                self.lstm_model.fit(df_with_regimes, price_col=price_col)
                print("LSTM model trained successfully")
                self.lstm_available = True
            except (ImportError, Exception) as e:
                self.lstm_available = False
                print(f"Error training LSTM model: {str(e)}")
                print("LSTM will not be used in the hybrid model")
        
        # Step 5: Calibrate model weights
        print("\nStep 5: Calibrating model weights...")
        self._calibrate_weights(df_with_regimes, price_col)
        
        self.is_trained = True
        print("\n✅ Hybrid model training complete!")
        print(f"Model weights: HMM: {self.model_weights['hmm']:.2f}, XGBoost: {self.model_weights['xgb']:.2f}, LSTM: {self.model_weights['lstm']:.2f}")
        
        return self
        
    def _calibrate_weights(self, df, price_col='close'):
        """
        Calibrate the weights of each model based on their individual performance.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
        """
        # Use detected price column if available
        if hasattr(self, 'detected_price_col'):
            price_col = self.detected_price_col
            
        # Get individual model predictions for a validation period
        val_size = min(int(len(df) * 0.3), 100)  # Use last 30% up to max 100 days
        val_df = df.iloc[-val_size:]
        
        try:
            # Get HMM predictions
            try:
                hmm_data = self.hmm_model.add_states_to_df(val_df)
                hmm_signals = self.hmm_model.generate_trading_signals(hmm_data, price_col=price_col)
                hmm_weight = 0.2  # Lower HMM weight since it's less responsive
            except Exception as e:
                print(f"Error getting HMM signals for calibration: {str(e)}")
                hmm_weight = 0.1  # Reduce weight further if there was an error
            
            # Get XGBoost predictions
            try:
                xgb_data = self.xgb_model.predict(val_df, price_col=price_col)
                xgb_weight = 0.4  # Keep XGBoost weight
            except Exception as e:
                print(f"Error getting XGBoost predictions for calibration: {str(e)}")
                xgb_weight = 0.3  # Reduce weight if there was an error
            
            # Initialize LSTM weight
            lstm_weight = 0.0  # Default to 0 if LSTM not available
            
            # Adjust weights if LSTM is available
            if self.use_lstm and self.lstm_available:
                try:
                    # Try to get LSTM predictions
                    lstm_data = self.lstm_model.predict(val_df, price_col=price_col)
                    if 'lstm_pred' in lstm_data.columns and not lstm_data['lstm_pred'].isna().all():
                        lstm_weight = 0.4  # Increase LSTM weight for better signal quality
                    else:
                        lstm_weight = 0.0
                except Exception as e:
                    print(f"Error getting LSTM predictions for calibration: {str(e)}")
                    lstm_weight = 0.0
                
            # Normalize weights to sum to 1
            total_weight = hmm_weight + xgb_weight + lstm_weight
            self.model_weights = {
                'hmm': hmm_weight / total_weight,
                'xgb': xgb_weight / total_weight,
                'lstm': lstm_weight / total_weight
            }
        except Exception as e:
            print(f"Error calibrating weights: {str(e)}")
            # Fallback to default weights
            self.model_weights = {
                'hmm': 0.20,
                'xgb': 0.40, 
                'lstm': 0.40 if (self.use_lstm and self.lstm_available) else 0.0
            }
            
            # Renormalize if LSTM is not available
            if not (self.use_lstm and self.lstm_available):
                total = self.model_weights['hmm'] + self.model_weights['xgb']
                self.model_weights['hmm'] /= total
                self.model_weights['xgb'] /= total
    
    def predict(self, df, price_col='close', threshold=0.0):
        """
        Generate predictions from all models and combine them using calibrated weights.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
            threshold (float): HMM threshold
            
        Returns:
            pd.DataFrame: DataFrame with predictions and signals
        """
        if not self.is_trained:
            print("Warning: Model has not been trained yet. Results may be unreliable.")
        
        # Ensure we're working with a clean copy of the data
        original_df = df.copy()
        
        # Use detected price column if available
        if hasattr(self, 'detected_price_col'):
            if price_col != self.detected_price_col:
                print(f"Note: Using price column '{self.detected_price_col}' from training instead of '{price_col}'")
                price_col = self.detected_price_col
        # Validate price column exists
        elif price_col not in original_df.columns:
            price_alternatives = ['close', 'price', 'adjclose', 'adjusted_close']
            for alt in price_alternatives:
                if alt in original_df.columns:
                    print(f"Price column '{price_col}' not found. Using '{alt}' instead.")
                    price_col = alt
                    break
        
        # Step 1: HMM for Market Regime Detection
        print("\nStep 1: Detecting market regimes with HMM...")
        try:
            hmm_data = self.hmm_model.add_states_to_df(original_df)
            
            # Check if hmm_state column is available
            if 'hmm_state' in hmm_data.columns:
                try:
                    hmm_signals = self.hmm_model.generate_trading_signals(
                        hmm_data, threshold=threshold, price_col=price_col
                    )
                    hmm_success = True
                except Exception as e:
                    print(f"Error generating HMM trading signals: {str(e)}")
                    hmm_signals = hmm_data.copy()
                    hmm_signals['signal'] = 0
                    hmm_success = True  # Still consider it a success since we have hmm_state
            else:
                print("HMM add_states_to_df didn't produce hmm_state column")
                hmm_signals = original_df.copy()
                hmm_signals['signal'] = 0
                hmm_signals['hmm_state'] = 0
                hmm_data = hmm_signals.copy()
                hmm_success = False
        except Exception as e:
            print(f"Error in HMM prediction: {str(e)}")
            print("Falling back to empty DataFrame with default structure")
            hmm_signals = original_df.copy()
            hmm_signals['signal'] = 0
            hmm_signals['hmm_state'] = 0
            hmm_data = hmm_signals.copy()
            hmm_success = False
        
        # Align index for merging later
        if 'date' in hmm_signals.columns:
            hmm_signals = hmm_signals.set_index('date')
        
        # Step 2: XGBoost for Feature-Based Classification with Enhanced Feature Consistency
        print("\nStep 2: Generating XGBoost predictions...")
        try:
            # Prepare a clean input dataframe for XGBoost with HMM states if available
            xgb_input = original_df.copy()
            
            # Add regime information if available from HMM
            if hmm_success and 'hmm_state' in hmm_data.columns:
                xgb_input['hmm_state'] = hmm_data['hmm_state']
                
                # Simple one-hot encoding for regime states
                for state in range(self.n_states):
                    xgb_input[f'regime_{state}'] = (xgb_input['hmm_state'] == state).astype(int)
            else:
                # Add default hmm_state column to avoid missing feature errors
                print("Adding default hmm_state column for XGBoost")
                xgb_input['hmm_state'] = 0
            
            # Generate XGBoost predictions
            xgb_data = self.xgb_model.predict(xgb_input, price_col=price_col)
            
            # Check if we got valid predictions
            if 'xgb_pred' in xgb_data.columns and not xgb_data['xgb_pred'].isna().all():
                print("XGBoost predictions successful - incorporating into hybrid model")
                
                # Generate signals from XGBoost predictions
                # We use the Kalman-filtered + derivative signal (similar to the standalone version)
                xgb_data['xgb_signal'] = 0  # Default to no signal
                
                # Long signal: positive gradient and acceleration (second derivative)
                xgb_data.loc[(xgb_data['gradient'] > 0) & (xgb_data['second_deriv'] > 0), 'xgb_signal'] = 1
                
                # Short signal: negative gradient and acceleration (second derivative)
                xgb_data.loc[(xgb_data['gradient'] < 0) & (xgb_data['second_deriv'] < 0), 'xgb_signal'] = -1
                
                # Align index for merging later
                if 'date' in xgb_data.columns:
                    xgb_data = xgb_data.set_index('date')
                
                xgb_success = True
            else:
                print("XGBoost predictions failed or returned empty results - using HMM only")
                xgb_data = None
                xgb_success = False
        except Exception as e:
            print(f"Error in XGBoost prediction: {str(e)}")
            print("Falling back to HMM only")
            xgb_data = None
            xgb_success = False
        
        # Step 3: LSTM for Sequential Predictions (if available)
        lstm_signals = None
        lstm_success = False

        if self.use_lstm and self.lstm_available:
            try:
                print("\nStep 3: Generating LSTM predictions...")
                
                # Create a copy to avoid modifying the original
                lstm_df = original_df.copy()
                
                # Make sure hmm_state exists - LSTM looks for this column
                if 'hmm_state' not in lstm_df.columns and hmm_success:
                    print("Adding hmm_state from HMM results to LSTM input")
                    common_idx = lstm_df.index.intersection(hmm_data.index)
                    lstm_df['hmm_state'] = 0  # Default value
                    if 'hmm_state' in hmm_data.columns:
                        lstm_df.loc[common_idx, 'hmm_state'] = hmm_data.loc[common_idx, 'hmm_state']
                elif 'hmm_state' not in lstm_df.columns:
                    # If HMM wasn't successful, add dummy hmm_state column
                    print("Adding default hmm_state column for LSTM")
                    lstm_df['hmm_state'] = 0
                
                # LSTM model might have feature mismatch issues - let's handle them
                try:
                    lstm_data = self.lstm_model.predict(lstm_df, price_col=price_col)
                except Exception as e:
                    if "feature mismatch" in str(e).lower() or "expects" in str(e).lower() and "features" in str(e).lower():
                        print(f"LSTM feature mismatch error: {str(e)}")
                        print("Attempting to fix feature mismatch...")
                        
                        # If original data doesn't have 'close' but we're using 'close' as price column
                        if 'close' not in lstm_df.columns and price_col != 'close':
                            lstm_df['close'] = lstm_df[price_col]
                        
                        # Try again with the modified DataFrame
                        lstm_data = self.lstm_model.predict(lstm_df, price_col=price_col)
                    else:
                        # Re-raise if not a feature mismatch error
                        raise e
                
                if 'lstm_pred' in lstm_data.columns and not lstm_data['lstm_pred'].isna().all():
                    print("LSTM predictions successful - incorporating into hybrid model")
                    lstm_signals = self.lstm_model.generate_trading_signals(lstm_data)
                    
                    # Align index for merging later
                    if 'date' in lstm_signals.columns:
                        lstm_signals = lstm_signals.set_index('date')
                    
                    lstm_success = True
                else:
                    print("LSTM predictions failed or returned empty results - using HMM and XGBoost only")
                    lstm_signals = None
            except Exception as e:
                print(f"Error generating LSTM predictions: {str(e)}")
                print("Falling back to HMM and XGBoost only")
                lstm_signals = None
        
        # Adjust weights based on model success
        adjusted_weights = self.model_weights.copy()
        
        # Zero out weights for failed models and renormalize
        if not hmm_success:
            adjusted_weights['hmm'] = 0
        if not xgb_success:
            adjusted_weights['xgb'] = 0
        if not lstm_success:
            adjusted_weights['lstm'] = 0
            
        # Ensure at least one model has weight
        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            print("All models failed. Using equal weights for any available signals.")
            if hmm_success:
                adjusted_weights['hmm'] = 1
            if xgb_success:
                adjusted_weights['xgb'] = 1
            if lstm_success:
                adjusted_weights['lstm'] = 1
                
            # Re-calculate total weight
            total_weight = sum(adjusted_weights.values())
        
        # Normalize weights
        for key in adjusted_weights:
            if total_weight > 0:
                adjusted_weights[key] /= total_weight
        
        print(f"\nAdjusted model weights: HMM: {adjusted_weights['hmm']:.2f}, XGBoost: {adjusted_weights['xgb']:.2f}, LSTM: {adjusted_weights['lstm']:.2f}")
        
        # Step 4: Combine all signals
        try:
            print("\nStep 4: Combining signals for final prediction...")
            
            # Initialize the combined dataframe
            # Start with the model that succeeded
            combined = None
            if hmm_success:
                combined = hmm_signals.copy()
            elif xgb_success:
                combined = xgb_data.copy()
            elif lstm_success:
                combined = lstm_signals.copy()
            else:
                # If all models failed, use original dataframe
                combined = pd.DataFrame(index=original_df.index)
            
            # If we still don't have a dataframe, create one
            if combined is None or combined.empty:
                combined = pd.DataFrame(index=original_df.index)
            
            # Add HMM signals if available
            if hmm_success and combined is not None:
                common_idx = combined.index.intersection(hmm_signals.index)
                if len(common_idx) > 0:
                    # Copy HMM signals
                    for col in ['hmm_state', 'signal']:
                        if col in hmm_signals.columns and col not in combined.columns:
                            combined[col] = np.nan
                            combined.loc[common_idx, col] = hmm_signals.loc[common_idx, col]
            
            # Add XGBoost signals if available
            if xgb_success and combined is not None:
                common_idx = combined.index.intersection(xgb_data.index)
                if len(common_idx) > 0:
                    for col in ['xgb_pred', 'savgol', 'kalman', 'gradient', 'second_deriv', 'xgb_signal']:
                        if col in xgb_data.columns and col not in combined.columns:
                            combined[col] = np.nan
                            combined.loc[common_idx, col] = xgb_data.loc[common_idx, col]
            
            # Add LSTM signals if available
            if lstm_success and combined is not None:
                common_idx = combined.index.intersection(lstm_signals.index)
                if len(common_idx) > 0:
                    for col in ['lstm_pred', 'lstm_signal']:
                        if col in lstm_signals.columns and col not in combined.columns:
                            combined[col] = np.nan
                            combined.loc[common_idx, col] = lstm_signals.loc[common_idx, col]
            
            # Initialize combined signal
            combined['combined_signal'] = 0.0
            
            # Add weighted contribution from each model
            if hmm_success and 'signal' in combined.columns:
                hmm_signal = combined['signal'].fillna(0)
                combined['combined_signal'] += hmm_signal * adjusted_weights['hmm']
            
            if xgb_success and 'xgb_signal' in combined.columns:
                xgb_signal = combined['xgb_signal'].fillna(0)
                combined['combined_signal'] += xgb_signal * adjusted_weights['xgb']
            
            if lstm_success and 'lstm_signal' in combined.columns:
                lstm_signal = combined['lstm_signal'].fillna(0)
                combined['combined_signal'] += lstm_signal * adjusted_weights['lstm']
            
            # Convert to discrete signals (-1, 0, 1)
            # Strong signal (above 0.5 or below -0.5)
            combined.loc[combined['combined_signal'] >= 0.5, 'combined_signal'] = 1
            combined.loc[combined['combined_signal'] <= -0.5, 'combined_signal'] = -1
            
            # Weaker signals (between thresholds)
            combined.loc[(combined['combined_signal'] > 0.2) & (combined['combined_signal'] < 0.5), 'combined_signal'] = 1
            combined.loc[(combined['combined_signal'] < -0.2) & (combined['combined_signal'] > -0.5), 'combined_signal'] = -1
            
            # Neutral zone
            combined.loc[(combined['combined_signal'] >= -0.2) & (combined['combined_signal'] <= 0.2), 'combined_signal'] = 0
            
            # Add model weights to result for analysis
            combined['hmm_weight'] = adjusted_weights['hmm']
            combined['xgb_weight'] = adjusted_weights['xgb']
            combined['lstm_weight'] = adjusted_weights['lstm']
            
            # Ensure 'signal' column exists for compatibility with backtest_strategy
            if 'signal' not in combined.columns:
                combined['signal'] = combined['combined_signal']
            
            # Print signal statistics
            total_rows = len(combined)
            buy_signals = (combined['combined_signal'] == 1).sum()
            sell_signals = (combined['combined_signal'] == -1).sum()
            neutral_signals = (combined['combined_signal'] == 0).sum()
            
            print(f"\nSignal Statistics:")
            print(f"Buy Signals: {buy_signals} ({buy_signals/total_rows*100:.2f}% of data)")
            print(f"Sell Signals: {sell_signals} ({sell_signals/total_rows*100:.2f}% of data)")
            print(f"Neutral Signals: {neutral_signals} ({neutral_signals/total_rows*100:.2f}% of data)")
            
            return combined
            
        except Exception as e:
            print(f"Error combining signals: {str(e)}")
            print("Returning basic signal DataFrame as fallback")
            result = pd.DataFrame(index=original_df.index)
            result['signal'] = 0
            result['combined_signal'] = 0
            result['hmm_weight'] = adjusted_weights['hmm']
            result['xgb_weight'] = adjusted_weights['xgb']
            result['lstm_weight'] = adjusted_weights['lstm']
            return result
        
    def backtest_strategy(self, df, price_col='close', fee=0.001, allow_shorts=True):
        """
        Backtest the hybrid strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            price_col (str): Column name for price
            fee (float): Trading fee as percentage
            allow_shorts (bool): Whether to allow short positions
            
        Returns:
            tuple: (results, performance) where results is a DataFrame with backtest results
                  and performance is a dict with performance metrics
        """
        # Check if the required columns are present
        for col in ['combined_signal']:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        results = df.copy()
        
        # Use the detected price column if available
        if hasattr(self, 'detected_price_col'):
            if price_col != self.detected_price_col:
                print(f"Note: Using price column '{self.detected_price_col}' from training instead of '{price_col}'")
                price_col = self.detected_price_col
                
        # Use the 'close' column if the specified price_col is not available
        if price_col not in results.columns:
            price_alternatives = ['close', 'price', 'value', 'adjclose', 'adjusted_close']
            for alt in price_alternatives:
                if alt in results.columns:
                    print(f"Price column '{price_col}' not found. Using '{alt}' instead.")
                    price_col = alt
                    break
            else:
                raise ValueError(f"Price column '{price_col}' not found in DataFrame")
        
        # Calculate price changes
        results['price_change'] = results[price_col].pct_change()
        
        # Initialize position column
        results['position'] = 0
        
        # Generate positions from signals
        # 1 for long, -1 for short, 0 for no position
        # Match position to signal directly with a 1-day lag to avoid lookahead bias
        results['position'] = results['combined_signal'].shift(1)
        
        # Fill initial NaN position with 0 (no position)
        results['position'] = results['position'].fillna(0)
        
        # If shorts are not allowed, replace -1 with 0
        if not allow_shorts:
            results.loc[results['position'] == -1, 'position'] = 0
        
        # Identify trade entries and exits
        results['trade_entry'] = results['position'].diff() != 0
        
        # Calculate strategy returns
        results['strategy_return'] = results['position'] * results['price_change']
        
        # Apply fees on trade entries
        results.loc[results['trade_entry'], 'strategy_return'] -= fee
        
        # Calculate cumulative returns
        results['cum_return'] = (1 + results['strategy_return']).cumprod() - 1
        results['cum_price_return'] = (1 + results['price_change']).cumprod() - 1
        
        # Calculate drawdowns
        results['peak'] = results['cum_return'].cummax()
        results['drawdown'] = (results['cum_return'] - results['peak']) / (1 + results['peak'])
        
        # Calculate buy & hold drawdowns
        results['bh_peak'] = results['cum_price_return'].cummax()
        results['bh_drawdown'] = (results['cum_price_return'] - results['bh_peak']) / (1 + results['bh_peak'])
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(results)
        
        return results, performance
        
    def _calculate_performance_metrics(self, results):
        """
        Calculate performance metrics for backtest results.
        
        Args:
            results (pd.DataFrame): DataFrame with backtest results
            
        Returns:
            dict: Dictionary with performance metrics
        """
        # Trading days per year for annualization
        trading_days = 365
        
        # Extract returns
        strategy_returns = results['strategy_return'].dropna()
        price_returns = results['price_change'].dropna()
        
        # Return performance metrics
        if len(strategy_returns) > 0:
            # Calculate returns
            total_return = results['cum_return'].iloc[-1]
            buy_hold_return = results['cum_price_return'].iloc[-1]
            
            # Calculate annualized returns
            years = len(results) / trading_days
            annual_return = (1 + total_return) ** (1 / years) - 1
            buy_hold_annual = (1 + buy_hold_return) ** (1 / years) - 1
            
            # Calculate volatility
            daily_vol = strategy_returns.std()
            annual_vol = daily_vol * np.sqrt(trading_days)
            
            buy_hold_daily_vol = price_returns.std()
            buy_hold_annual_vol = buy_hold_daily_vol * np.sqrt(trading_days)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.01  # Assuming 1% risk-free rate
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            buy_hold_sharpe = (buy_hold_annual - risk_free_rate) / buy_hold_annual_vol if buy_hold_annual_vol > 0 else 0
            
            # Calculate max drawdown
            max_drawdown = results['drawdown'].min()
            bh_max_drawdown = results['bh_drawdown'].min()
            
            # Calculate win rate
            strategy_returns_no_fees = results['position'] * results['price_change']
            wins = (strategy_returns_no_fees > 0).sum()
            losses = (strategy_returns_no_fees < 0).sum()
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            # Calculate trading frequency
            trades = results['trade_entry'].sum()
            trade_frequency = trades / len(results)
            
            # Calculate profit factor
            gross_profit = strategy_returns_no_fees[strategy_returns_no_fees > 0].sum()
            gross_loss = abs(strategy_returns_no_fees[strategy_returns_no_fees < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate average trade return
            avg_trade_return = strategy_returns.mean() * 100  # as percentage
            
            return {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Profit Factor': profit_factor,
                'Trading Frequency': trade_frequency,
                'Number of Trades': trades,
                'Average Trade': avg_trade_return,
                'Buy Hold Return': buy_hold_return,
                'Buy Hold Annual': buy_hold_annual,
                'Buy Hold Volatility': buy_hold_annual_vol,
                'Buy Hold Sharpe': buy_hold_sharpe,
                'Buy Hold Max Drawdown': bh_max_drawdown
            }
        else:
            # Return zeros if no strategy returns
            return {
                'Total Return': 0.0,
                'Annual Return': 0.0,
                'Annual Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Profit Factor': 0.0,
                'Trading Frequency': 0.0,
                'Number of Trades': 0,
                'Average Trade': 0.0,
                'Buy Hold Return': 0.0,
                'Buy Hold Annual': 0.0,
                'Buy Hold Volatility': 0.0,
                'Buy Hold Sharpe': 0.0,
                'Buy Hold Max Drawdown': 0.0
            }
    
    def plot_signals(self, df, price_col='price'):
        """
        Plot the hybrid model signals and performance.
        
        Args:
            df (pd.DataFrame): DataFrame with backtest results
            price_col (str): Column name for price
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if 'combined_signal' not in df.columns:
            raise ValueError("DataFrame must contain 'combined_signal' column")
        
        # Use 'close' column if the specified price_col is not available
        if price_col not in df.columns:
            if 'close' in df.columns:
                price_col = 'close'
            elif 'price' in df.columns:
                price_col = 'price'
            elif 'value' in df.columns:
                price_col = 'value'
            else:
                raise ValueError(f"Price column '{price_col}' not found in DataFrame")
        
        # Set the style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Plot price in the first subplot
        ax1.plot(df.index, df[price_col], label=f'{price_col.capitalize()}', color='blue')
        ax1.set_title(f'{price_col.capitalize()} with Signals', fontsize=14)
        ax1.set_ylabel(f'{price_col.capitalize()}')
        
        # Plot buy signals
        buy_signals = df[df['combined_signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals[price_col], marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = df[df['combined_signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals[price_col], marker='v', color='red', s=100, label='Sell Signal')
        
        # Add HMM states if available
        if 'hmm_state' in df.columns:
            # Color states based on their return profile
            state_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            # Create spans for each state
            last_state = None
            span_start = None
            
            for i, (idx, row) in enumerate(df.iterrows()):
                state = row['hmm_state']
                
                if state != last_state:
                    if last_state is not None and span_start is not None:
                        # End the previous span
                        if i > 0:
                            span_end = idx
                            color = state_colors[int(last_state) % len(state_colors)]
                            ax1.axvspan(span_start, span_end, alpha=0.2, color=color)
                    
                    # Start a new span
                    span_start = idx
                    last_state = state
            
            # Add the final span
            if span_start is not None and last_state is not None:
                color = state_colors[int(last_state) % len(state_colors)]
                ax1.axvspan(span_start, df.index[-1], alpha=0.2, color=color)
        
        ax1.legend()
        
        # Plot signals from each model in the second subplot
        ax2.set_title('Model Signals', fontsize=14)
        
        if 'signal' in df.columns:
            ax2.plot(df.index, df['signal'], label='HMM Signal', color='purple', alpha=0.7)
        
        if 'xgb_signal' in df.columns:
            ax2.plot(df.index, df['xgb_signal'], label='XGBoost Signal', color='orange', alpha=0.7)
        
        if 'lstm_signal' in df.columns:
            ax2.plot(df.index, df['lstm_signal'], label='LSTM Signal', color='green', alpha=0.7)
        
        ax2.plot(df.index, df['combined_signal'], label='Combined Signal', color='black', linewidth=2)
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Signal')
        ax2.legend()
        
        # Plot returns in the third subplot
        if 'cum_return' in df.columns and 'cum_price_return' in df.columns:
            ax3.set_title('Cumulative Returns', fontsize=14)
            ax3.plot(df.index, df['cum_return'], label='Strategy', color='green')
            ax3.plot(df.index, df['cum_price_return'], label='Buy & Hold', color='blue', linestyle='--')
            
            # Plot drawdowns in red
            if 'drawdown' in df.columns:
                ax3_twin = ax3.twinx()
                ax3_twin.fill_between(df.index, 0, df['drawdown'], alpha=0.3, color='red', label='Drawdown')
                ax3_twin.set_ylabel('Drawdown', color='red')
                ax3_twin.tick_params(axis='y', colors='red')
                ax3_twin.set_ylim(-1, 0)
        
        ax3.set_ylabel('Cumulative Return')
        ax3.legend(loc='upper left')
        
        # Format the figure
        plt.tight_layout()
        
        # Return the figure
        return fig

    def save_model(self, model_dir=None):
        """
        Save the hybrid model components.
        
        Args:
            model_dir (str, optional): Directory to save the model
            
        Returns:
            str: Path to the saved model directory
        """
        if model_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = os.path.join(MODELS_DIR, f"hybrid_model_{timestamp}")
        
        # Create directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save HMM model
        hmm_path = os.path.join(model_dir, "hmm_model.pkl")
        joblib.dump(self.hmm_model, hmm_path)
        
        # Save XGBoost model
        xgb_path = os.path.join(model_dir, "xgb_model.pkl")
        joblib.dump(self.xgb_model, xgb_path)
        
        # Save LSTM model if available
        if self.use_lstm and self.lstm_available:
            lstm_path = os.path.join(model_dir, "lstm_model.pt")
            self.lstm_model.save_model(lstm_path)
        
        # Save model configuration and weights
        config = {
            'n_states': self.n_states,
            'n_lags': self.n_lags,
            'window_size': self.window_size,
            'use_lstm': self.use_lstm,
            'lstm_available': self.lstm_available,
            'random_state': self.random_state,
            'model_weights': self.model_weights,
            'is_trained': self.is_trained
        }
        
        config_path = os.path.join(model_dir, "config.pkl")
        joblib.dump(config, config_path)
        
        print(f"Hybrid model saved to {model_dir}")
        return model_dir
    
    def load_model(self, model_dir):
        """
        Load the hybrid model components.
        
        Args:
            model_dir (str): Directory where the model is saved
            
        Returns:
            self: The loaded hybrid model
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        # Load configuration
        config_path = os.path.join(model_dir, "config.pkl")
        config = joblib.load(config_path)
        
        # Update model attributes
        self.n_states = config['n_states']
        self.n_lags = config['n_lags']
        self.window_size = config['window_size']
        self.use_lstm = config['use_lstm']
        self.lstm_available = config['lstm_available']
        self.random_state = config['random_state']
        self.model_weights = config['model_weights']
        self.is_trained = config['is_trained']
        
        # Load HMM model
        hmm_path = os.path.join(model_dir, "hmm_model.pkl")
        self.hmm_model = joblib.load(hmm_path)
        
        # Load XGBoost model
        xgb_path = os.path.join(model_dir, "xgb_model.pkl")
        self.xgb_model = joblib.load(xgb_path)
        
        # Load LSTM model if available
        if self.use_lstm and self.lstm_available:
            lstm_path = os.path.join(model_dir, "lstm_model.pt")
            if os.path.exists(lstm_path):
                # Need to create a new LSTM model with the right parameters
                self.lstm_model = LSTMPredictor(window_size=self.window_size)
                try:
                    # Estimate input dimension from model file size or use a default
                    input_dim = 10  # Default guess
                    self.lstm_model.load_model(lstm_path, input_dim)
                except Exception as e:
                    print(f"Error loading LSTM model: {str(e)}")
                    self.lstm_available = False
            else:
                print(f"LSTM model file not found at {lstm_path}")
                self.lstm_available = False
        
        print(f"Hybrid model loaded from {model_dir}")
        return self 