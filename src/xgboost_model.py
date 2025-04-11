"""
XGBoost model for price prediction and smoothing with Kalman filters.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import xgboost as xgb
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

class XGBoostPredictor:
    """
    XGBoost-based price predictor with Kalman filtering for smoother signals.
    """
    
    def __init__(self, n_lags=2, random_state=42):
        """
        Initialize the XGBoost predictor.
        
        Args:
            n_lags (int): Number of lag features to use (Markov property)
            random_state (int): Random seed for reproducibility
        """
        self.n_lags = n_lags
        self.random_state = random_state
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        )
        self.kf = None
        self.scaler_params = {}
        
    def _create_lag_features(self, df, price_col='price'):
        """
        Create lag features for the price column.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Create lag features
        for i in range(1, self.n_lags + 1):
            result[f'lag_{i}'] = result[price_col].shift(i)
        
        # Drop rows with NaN values (due to lag features)
        result = result.dropna()
        
        return result
    
    def _scale_features(self, X, is_train=True):
        """
        Scale features using standardization.
        
        Args:
            X (pd.DataFrame): Features to scale
            is_train (bool): Whether this is training data
            
        Returns:
            pd.DataFrame: Scaled features
        """
        # Create a copy
        X_scaled = X.copy()
        
        # Scale each column
        for col in X_scaled.columns:
            if is_train:
                # Calculate mean and std for training data
                mean = X_scaled[col].mean()
                std = X_scaled[col].std()
                self.scaler_params[col] = {'mean': mean, 'std': std}
            else:
                # Use mean and std from training data
                mean = self.scaler_params[col]['mean']
                std = self.scaler_params[col]['std']
            
            # Apply scaling
            X_scaled[col] = (X_scaled[col] - mean) / std
        
        return X_scaled
    
    def fit(self, df, price_col='price'):
        """
        Fit the XGBoost model to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
        """
        # Create lag features
        data_with_lags = self._create_lag_features(df, price_col)
        
        # Define target and features
        X = data_with_lags[[f'lag_{i}' for i in range(1, self.n_lags + 1)]]
        y = data_with_lags[price_col]
        
        # Scale features
        X_scaled = self._scale_features(X, is_train=True)
        
        # Fit the XGBoost model
        self.model.fit(X_scaled, y)
        
        print(f"XGBoost model trained with {len(X_scaled)} samples")
        
    def predict(self, df, price_col='price', apply_smoothing=True):
        """
        Generate predictions and apply smoothing.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
            apply_smoothing (bool): Whether to apply Kalman filtering
            
        Returns:
            pd.DataFrame: DataFrame with predictions and signals
        """
        # Create lag features
        data_with_lags = self._create_lag_features(df, price_col)
        
        # Define features
        X = data_with_lags[[f'lag_{i}' for i in range(1, self.n_lags + 1)]]
        
        # Scale features
        X_scaled = self._scale_features(X, is_train=False)
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        
        # Create result DataFrame
        result = data_with_lags.copy()
        result['xgb_pred'] = predictions
        
        # Apply Savgol filter for initial smoothing
        result['savgol'] = savgol_filter(
            result['xgb_pred'].values, 
            window_length=5, 
            polyorder=2
        )
        
        if apply_smoothing:
            # Apply Kalman filter for further smoothing
            self._apply_kalman_filter(result)
            
            # Calculate derivatives for signal generation
            self._calculate_derivatives(result)
        
        return result
    
    def _apply_kalman_filter(self, df):
        """
        Apply Kalman filter to the savgol-smoothed predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with savgol-smoothed predictions
        """
        # Define observation matrix (identity matrix)
        observation_matrix = np.identity(1)
        
        # Estimate initial state
        initial_state_mean = np.mean(df['savgol'])
        initial_state_covariance = np.cov(df['savgol'])
        
        # Define transition matrix (linear relationship between states)
        transition_matrix = np.array([[1]])
        
        # Define noise covariances
        process_noise_covariance = np.array([[1e-5]])
        observation_noise_covariance = np.array([[1e-3]])
        
        # Create KalmanFilter object
        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_noise_covariance,
            transition_covariance=process_noise_covariance
        )
        
        # Apply Kalman filter
        filtered_state_means, _ = self.kf.filter(df['savgol'].values.reshape(-1, 1))
        df['kalman'] = filtered_state_means.flatten()
    
    def _calculate_derivatives(self, df):
        """
        Calculate derivatives for signal generation.
        
        Args:
            df (pd.DataFrame): DataFrame with Kalman-filtered data
        """
        # First derivative (gradient)
        df['gradient'] = df['kalman'] - df['kalman'].shift(1)
        
        # Second derivative
        df['second_deriv'] = df['gradient'] - df['gradient'].shift(1)
        
        # Fill NaN values
        df.fillna(0, inplace=True)
    
    def generate_trading_signals(self, df, trading_hours=(10, 16)):
        """
        Generate trading signals based on derivatives.
        
        Args:
            df (pd.DataFrame): DataFrame with derivative data
            trading_hours (tuple): Hours of the day to trade (start, end)
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Create a copy
        result = df.copy()
        
        # Add hour of day
        if 'date' in result.columns:
            result['hour'] = result['date'].dt.hour
        else:
            result['hour'] = 12  # Default to middle of day if no date column
        
        # Determine if within trading hours
        result['trading'] = np.where(
            (result['hour'] >= trading_hours[0]) & (result['hour'] < trading_hours[1]), 
            1, 0
        )
        
        # Generate buy/sell signals based on derivatives
        # Long trades
        result['buy_signal_long'] = np.where(
            result['trading'] == 1,
            np.where(
                (result['second_deriv'] < -0.03) & (result['gradient'] > 0.02),
                1, 0
            ),
            0
        )
        
        result['sell_signal_long'] = np.where(
            result['trading'] == 1,
            np.where(
                (result['second_deriv'] > -0.01) & (result['gradient'] < 0),
                -1, 0
            ),
            0
        )
        
        # Short trades
        result['buy_signal_short'] = np.where(
            result['trading'] == 1,
            np.where(
                (result['second_deriv'] > 0.03) & (result['gradient'] < -0.04),
                1, 0
            ),
            0
        )
        
        result['sell_signal_short'] = np.where(
            result['trading'] == 1,
            np.where(
                (result['second_deriv'] < 0.01) & (result['gradient'] > -0.03),
                -1, 0
            ),
            0
        )
        
        # Calculate holding positions
        result['holding_long'] = np.where(
            result['buy_signal_long'] == 1, 1,
            np.where(result['sell_signal_long'] == -1, 0, np.nan)
        )
        result['holding_long'].fillna(method='ffill', inplace=True)
        result['prev_holding_long'] = result['holding_long'].shift(1)
        
        result['holding_short'] = np.where(
            result['buy_signal_short'] == 1, 1,
            np.where(result['sell_signal_short'] == -1, 0, np.nan)
        )
        result['holding_short'].fillna(method='ffill', inplace=True)
        result['prev_holding_short'] = result['holding_short'].shift(1)
        
        # Calculate position changes
        result['change_long'] = np.where(
            (result['holding_long'] == 1) & (result['prev_holding_long'] == 0), 1,
            np.where((result['holding_long'] == 0) & (result['prev_holding_long'] == 1), -1, 0)
        )
        
        result['change_short'] = np.where(
            (result['holding_short'] == 1) & (result['prev_holding_short'] == 0), 1,
            np.where((result['holding_short'] == 0) & (result['prev_holding_short'] == 1), -1, 0)
        )
        
        # Generate final signal
        result['xgb_signal'] = np.where(
            result['change_long'] == 1, 1,
            np.where(result['change_short'] == 1, -1, 0)
        )
        
        return result
        
    def plot_predictions(self, df, price_col='price'):
        """
        Plot the predictions and signals.
        
        Args:
            df (pd.DataFrame): DataFrame with predictions and signals
            price_col (str): Name of the price column
        """
        plt.figure(figsize=(14, 10))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(df[price_col], label='Actual Price')
        plt.plot(df['xgb_pred'], label='XGBoost Prediction')
        plt.plot(df['savgol'], label='Savgol Filter')
        plt.plot(df['kalman'], label='Kalman Filter')
        plt.legend()
        plt.title('Price and Predictions')
        
        # Plot derivatives
        plt.subplot(3, 1, 2)
        plt.plot(df['gradient'], label='Gradient')
        plt.plot(df['second_deriv'], label='Second Derivative')
        plt.axhline(y=0.02, color='g', linestyle='--', label='Gradient Buy Threshold')
        plt.axhline(y=-0.03, color='r', linestyle='--', label='Second Deriv Buy Threshold')
        plt.legend()
        plt.title('Derivatives')
        
        # Plot signals
        plt.subplot(3, 1, 3)
        buy_long_mask = df['change_long'] == 1
        sell_long_mask = df['change_long'] == -1
        buy_short_mask = df['change_short'] == 1
        sell_short_mask = df['change_short'] == -1
        
        plt.plot(df[price_col], label='Price', alpha=0.5)
        plt.scatter(df[buy_long_mask].index, df.loc[buy_long_mask, price_col], 
                   color='g', marker='^', s=100, label='Buy Long')
        plt.scatter(df[sell_long_mask].index, df.loc[sell_long_mask, price_col], 
                   color='r', marker='v', s=100, label='Sell Long')
        plt.scatter(df[buy_short_mask].index, df.loc[buy_short_mask, price_col], 
                   color='m', marker='^', s=100, label='Buy Short')
        plt.scatter(df[sell_short_mask].index, df.loc[sell_short_mask, price_col], 
                   color='k', marker='v', s=100, label='Sell Short')
        plt.legend()
        plt.title('Trading Signals')
        
        plt.tight_layout()
        plt.show() 