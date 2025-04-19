"""
XGBoost model for price prediction and smoothing with Kalman filters.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import xgboost as xgb
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier

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
        
        # Ensure scaler_params is initialized
        if not hasattr(self, 'scaler_params') or self.scaler_params is None:
            self.scaler_params = {}
        
        # Scale each column
        for col in X_scaled.columns:
            if is_train:
                # Calculate mean and std for training data
                mean = X_scaled[col].mean()
                std = max(X_scaled[col].std(), 1e-8)  # Avoid division by zero
                self.scaler_params[col] = {'mean': mean, 'std': std}
            else:
                # Use mean and std from training data if available
                if col in self.scaler_params:
                    mean = self.scaler_params[col]['mean']
                    std = self.scaler_params[col]['std']
                else:
                    # If column not in scaler_params, use values from this data
                    print(f"Warning: Column {col} not found in scaler_params. Using current data stats.")
                    mean = X_scaled[col].mean()
                    std = max(X_scaled[col].std(), 1e-8)
                    self.scaler_params[col] = {'mean': mean, 'std': std}
            
            # Apply scaling
            X_scaled[col] = (X_scaled[col] - mean) / std
        
        return X_scaled
    
    def fit(self, df, price_col='close', target_type='regression', cv_folds=3, no_tuning=False):
        """
        Fit the XGBoost model to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Column name for price data
            target_type (str): 'regression' or 'classification'
            cv_folds (int): Number of cross-validation folds for parameter tuning
            no_tuning (bool): Whether to skip parameter tuning
        """
        # Prepare the data
        X, y = self._prepare_data(df, price_col)
        
        # Save feature columns for later use
        self.feature_cols = list(X.columns)
        
        # Initialize scaler_params before using it
        self.scaler_params = {}
        for col in X.columns:
            self.scaler_params[col] = {
                'mean': X[col].mean(),
                'std': max(X[col].std(), 1e-8)  # Avoid division by zero
            }
        
        # If no_tuning is True, skip parameter tuning and use default parameters
        if no_tuning:
            print("Skipping parameter tuning, using default XGBoost parameters")
            # Default model with conservative parameters
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.1,
                random_state=self.random_state
            )
        else:
            # Simplified parameter grid for faster tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
            }
            
            # Define evaluation metric based on target type
            eval_metric = 'rmse' if target_type == 'regression' else 'auc'
            
            # Set up GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                xgb.XGBRegressor(
                    objective='reg:squarederror' if target_type == 'regression' else 'binary:logistic',
                    random_state=self.random_state
                ),
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error' if target_type == 'regression' else 'roc_auc',
                n_jobs=-1,  # Use all available cores
                verbose=1   # More output to track progress
            )
            
            # Fit the grid search to find best parameters
            print("Performing XGBoost parameter tuning with cross-validation (reduced grid)...")
            try:
                grid_search.fit(X, y)
                
                # Get best parameters
                best_params = grid_search.best_params_
                print(f"Best parameters: {best_params}")
                print(f"Best CV score: {-grid_search.best_score_ if target_type == 'regression' else grid_search.best_score_:.6f}")
                
                # Train final model with best parameters
                self.model = xgb.XGBRegressor(
                    **best_params,
                    objective='reg:squarederror' if target_type == 'regression' else 'binary:logistic',
                    random_state=self.random_state
                )
                
            except Exception as e:
                print(f"Error during parameter tuning: {str(e)}")
                print("Falling back to default parameters")
                # Default model with conservative parameters
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=3,
                    gamma=0.1,
                    random_state=self.random_state
                )
        
        # Train the final model
        print(f"Training final XGBoost model with {len(X)} samples...")
        try:
            # Check if eval_metric is supported in the current XGBoost version
            self.model.fit(X, y)
        except TypeError as e:
            if 'unexpected keyword argument' in str(e) and 'eval_metric' in str(e):
                print("Note: eval_metric parameter not supported in this XGBoost version")
                # Try again without eval_metric
                self.model.fit(X, y)
            else:
                # Re-raise if it's a different TypeError
                raise
        
        # Feature importances
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 important features:")
        print(feature_importance.head(10))
        
        # Store feature statistics for signal scaling
        self.feature_stats = {}
        for col in X.columns:
            self.feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max()
            }
        
        print(f"XGBoost model trained with {len(X)} samples")
        return self

    def _prepare_data(self, df, price_col='close'):
        """
        Prepare data for XGBoost model.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Column name for price
            
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Check if dataframe is empty
        if data.empty:
            print("Warning: Empty DataFrame provided to _prepare_data")
            # Return empty DataFrame with expected columns
            empty_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(10)])
            empty_y = pd.Series(dtype='float64')
            return empty_X, empty_y
            
        # Check if the price column exists
        if price_col not in data.columns:
            # Try common price column names, prioritizing 'close'
            available_price_cols = [col for col in data.columns if col.lower() in ['close', 'price', 'open', 'adj_close', 'adjusted_close']]
            if not available_price_cols:
                # Try pattern matching
                available_price_cols = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
            
            if available_price_cols:
                price_col = available_price_cols[0]
                print(f"Price column '{price_col}' not found, using '{available_price_cols[0]}' instead")
            else:
                print(f"Warning: Price column '{price_col}' not found and no alternative available")
                print(f"Available columns: {data.columns.tolist()}")
                # Return empty DataFrame with expected columns
                empty_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(10)])
                empty_y = pd.Series(dtype='float64')
                return empty_X, empty_y
        
        # Store the detected price column for use in other methods
        self.detected_price_col = price_col
        
        # Calculate target (next period's return)
        data['target'] = data[price_col].pct_change(1).shift(-1)
        
        # Check if we have enough data after calculating target
        if data['target'].count() < 10:  # Minimum data requirement
            print(f"Warning: Not enough valid data points after calculating target. Found {data['target'].count()} non-NaN values.")
            empty_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(10)])
            empty_y = pd.Series(dtype='float64')
            return empty_X, empty_y
        
        # Add lagged features
        for lag in range(1, self.n_lags + 1):
            for col in [price_col, 'volume'] + [c for c in data.columns if 'rsi' in c or 'macd' in c or 'roc' in c]:
                if col in data.columns:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Linear regression slope of price
        try:
            for window in [5, 10, 20]:
                slopes = []
                for i in range(len(data)):
                    if i >= window:
                        price_window = data[price_col].iloc[i-window:i].values
                        if not np.isnan(price_window).any() and len(price_window) == window:
                            x = np.arange(window)
                            try:
                                slope, _, _, _, _ = linregress(x, price_window)
                                slopes.append(slope)
                            except Exception:
                                slopes.append(np.nan)
                        else:
                            slopes.append(np.nan)
                    else:
                        slopes.append(np.nan)
                data[f'price_slope_{window}'] = slopes
        except Exception as e:
            print(f"Warning: Error calculating price slopes: {str(e)}")
            
        # Calculate price acceleration (change in momentum)
        try:
            data['momentum_5d'] = data[price_col].diff(5)
            data['momentum_10d'] = data[price_col].diff(10)
            data['acceleration'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
        except Exception as e:
            print(f"Warning: Error calculating momentum features: {str(e)}")
        
        # Add custom regime feature based on volatility and trend
        try:
            if 'volatility_7d' in data.columns and 'volatility_30d' in data.columns:
                # Ensure no division by zero
                valid_idx = data['volatility_30d'] > 0
                vol_ratio = data.loc[valid_idx, 'volatility_7d'] / data.loc[valid_idx, 'volatility_30d']
                data.loc[valid_idx, 'volatility_regime'] = np.where(vol_ratio > 1.2, 1, 
                                                        np.where(vol_ratio < 0.8, -1, 0))
        except Exception as e:
            print(f"Warning: Error calculating volatility regime: {str(e)}")
        
        # Drop rows with NaN targets but keep as many feature rows as possible
        data_with_target = data.dropna(subset=['target'])
        
        # Check if we have any data left
        if len(data_with_target) == 0:
            print("Warning: No data left after dropping NaN targets")
            empty_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(10)])
            empty_y = pd.Series(dtype='float64')
            return empty_X, empty_y
        
        # Exclude non-feature columns
        exclude_cols = ['date', 'timestamp', 'time', 'target', price_col]
        feature_cols = [col for col in data_with_target.columns if col not in exclude_cols]
        
        # Select features and target
        X_raw = data_with_target[feature_cols]
        y = data_with_target['target']
        
        # Fill remaining NaN values in features with column means or zeros
        X = X_raw.copy()
        for col in X.columns:
            if X[col].isna().any():
                col_mean = X[col].mean()
                if pd.isna(col_mean):  # If column mean is NaN, fill with zeros
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_mean)
        
        print(f"XGBoost features: {len(feature_cols)} columns, {len(X)} rows")
        return X, y
        
    def predict(self, df, price_col='close', apply_smoothing=True):
        """
        Generate predictions and apply smoothing.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Name of the price column
            apply_smoothing (bool): Whether to apply Kalman filtering
            
        Returns:
            pd.DataFrame: DataFrame with predictions and signals
        """
        try:
            # Check if the model has been trained
            if not hasattr(self, 'model') or not hasattr(self, 'feature_cols'):
                print("Model not trained yet. Please call fit() before predict().")
                # Return the original DataFrame with default columns
                result = df.copy()
                self._add_default_columns(result)
                return result
            
            # Check if input data is empty
            if df is None or df.empty:
                print("Warning: Empty DataFrame provided to predict method")
                # Return empty DataFrame with default columns
                result = pd.DataFrame()
                self._add_default_columns(result)
                return result
            
            # Check if the price column was detected during training and use it consistently
            if hasattr(self, 'detected_price_col') and price_col != self.detected_price_col:
                print(f"Note: Using price column '{self.detected_price_col}' from training instead of '{price_col}'")
                price_col = self.detected_price_col
            
            # Prepare data using the same method as in fit
            X_test, _ = self._prepare_data(df, price_col)
            
            # Check if we got any features 
            if X_test.empty:
                print("Warning: No valid features extracted for prediction")
                # Return the original DataFrame with default columns
                result = df.copy()
                self._add_default_columns(result)
                return result
            
            # Ensure feature consistency between training and prediction
            missing_features = set(self.feature_cols) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(self.feature_cols)
            
            if missing_features:
                print(f"Warning: Missing {len(missing_features)} features in prediction data")
                if len(missing_features) > 10:
                    print(f"First 10 missing features: {list(missing_features)[:10]}")
                else:
                    print(f"Missing features: {missing_features}")
                    
                # If 'price' is missing but we're using 'close', add a mapping feature
                if 'price' in missing_features and price_col == 'close' and price_col in df.columns:
                    print(f"Creating 'price' feature from '{price_col}' column")
                    # Add a copy of the close price as 'price' to X_test
                    data_copy = df.copy()
                    data_copy['price'] = data_copy[price_col]
                    # Rerun _prepare_data to regenerate features
                    X_test, _ = self._prepare_data(data_copy, price_col='price')
                else:
                    # Add missing features with zeros or appropriate values
                    for feature in missing_features:
                        X_test[feature] = 0
            
            if extra_features:
                print(f"Warning: {len(extra_features)} extra features in prediction data that weren't in training")
                # Only keep features that were in training
                X_test = X_test[self.feature_cols]
            
            # Ensure order of features matches training
            X_test = X_test[self.feature_cols]
            
            # Scale features
            X_scaled = self._scale_features(X_test, is_train=False)
            
            # Final check for NaN or infinite values
            if np.isnan(X_scaled.values).any() or np.isinf(X_scaled.values).any():
                print("Warning: NaN or infinite values in scaled features after preprocessing")
                # Handle NaN or inf values
                X_scaled = X_scaled.fillna(0)
                X_scaled = X_scaled.replace([np.inf, -np.inf], 0)
            
            # Generate predictions
            predictions = self.model.predict(X_scaled)
            
            # Create result DataFrame
            result = df.copy()
            result['xgb_pred'] = np.nan  # Initialize with NaN
            
            # Map predictions back to original DataFrame indices
            # Handle the case where the _prepare_data function might drop rows
            try:
                predictions_df = pd.DataFrame({'xgb_pred': predictions}, index=X_test.index)
                
                # Handle index mismatch
                if not predictions_df.index.equals(result.index):
                    print(f"Warning: Index mismatch between predictions and original data ({len(predictions_df)} vs {len(result)})")
                    
                    # Ensure predictions_df has all indices from result
                    missing_indices = result.index.difference(predictions_df.index)
                    if not missing_indices.empty:
                        for idx in missing_indices:
                            predictions_df.loc[idx] = np.nan
                
                # Align and update the result DataFrame
                common_indices = result.index.intersection(predictions_df.index)
                result.loc[common_indices, 'xgb_pred'] = predictions_df.loc[common_indices, 'xgb_pred']
            except Exception as e:
                print(f"Warning: Error mapping predictions to original DataFrame: {str(e)}")
            
            # Apply Savitzky-Golay filter for smoothing
            valid_indices = ~np.isnan(result['xgb_pred'])
            
            if valid_indices.sum() >= 7:  # Need at least 7 points for filter with window size 5
                try:
                    valid_values = result.loc[valid_indices, 'xgb_pred'].values
                    # Apply filter on valid values
                    smoothed = savgol_filter(valid_values, 5, 2)
                    # Put smoothed values back
                    result.loc[valid_indices, 'savgol'] = smoothed
                except Exception as e:
                    print(f"Error applying Savitzky-Golay filter: {str(e)}")
                    result['savgol'] = result['xgb_pred']
            else:
                # Not enough valid values, just copy predictions
                print(f"Not enough valid predictions ({valid_indices.sum()}) for Savitzky-Golay filter, using raw predictions")
                result['savgol'] = result['xgb_pred']
            
            if apply_smoothing:
                # Apply Kalman filter for further smoothing
                try:
                    self._apply_kalman_filter(result)
                except Exception as e:
                    print(f"Error applying Kalman filter: {str(e)}")
                    result['kalman'] = result['savgol']
                
                # Calculate derivatives for signal generation
                try:
                    self._calculate_derivatives(result)
                except Exception as e:
                    print(f"Error calculating derivatives: {str(e)}")
                    result['gradient'] = 0
                    result['second_deriv'] = 0
            else:
                # Just add empty columns for compatibility
                result['kalman'] = result['savgol']
                result['gradient'] = 0
                result['second_deriv'] = 0
            
            return result
        
        except Exception as e:
            print(f"Error in XGBoost prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original DataFrame with default columns
            result = df.copy()
            self._add_default_columns(result)
            return result
    
    def _add_default_columns(self, df):
        """Add default columns to the DataFrame for consistent API"""
        for col in ['xgb_pred', 'savgol', 'kalman', 'gradient', 'second_deriv']:
            if col not in df.columns:
                df[col] = 0
        return df
    
    def _apply_kalman_filter(self, df):
        """Apply Kalman filter to smooth signals.
        
        Args:
            df (pd.DataFrame): DataFrame with signals to smooth
            
        Returns:
            pd.DataFrame: DataFrame with smoothed signals
        """
        if df is None or df.empty:
            print("Warning: Empty DataFrame provided to Kalman filter")
            return df
            
        if 'savgol' not in df.columns:
            print("No savgol column to apply Kalman filter")
            # Add kalman columns to maintain API compatibility
            df['kalman'] = np.nan
            df['kalman_smoothed'] = np.nan
            return df
        
        # Setup Kalman filter parameters
        try:
            # Define observation matrix (identity matrix)
            observation_matrix = np.identity(1)
            
            # Estimate initial state - get only valid values
            valid_savgols = df['savgol'].dropna().values
            
            # Check if we have enough valid data for filtering
            if len(valid_savgols) < 2:
                print("Not enough valid savgol values for Kalman filter (need at least 2)")
                # Add fallback values by simply copying savgol to kalman columns
                df['kalman'] = df['savgol']
                df['kalman_smoothed'] = df['savgol']
                return df
                
            # Safety check for NaN or infinite values
            if np.isnan(valid_savgols).any() or np.isinf(valid_savgols).any():
                print("Warning: Invalid values found in savgol data")
                # Replace invalid values with the mean of valid values
                valid_values_only = valid_savgols[~np.isnan(valid_savgols) & ~np.isinf(valid_savgols)]
                if len(valid_values_only) > 0:
                    valid_savgols = np.where(np.isnan(valid_savgols) | np.isinf(valid_savgols), 
                                            np.mean(valid_values_only), valid_savgols)
                else:
                    print("No valid values found in savgol data, skipping Kalman filter")
                    df['kalman'] = np.nan
                    df['kalman_smoothed'] = np.nan
                    return df
                
            initial_state_mean = np.mean(valid_savgols)
            
            # Check for very small variance - avoid numerical issues
            var_value = np.var(valid_savgols)
            if var_value < 1e-10:
                print("Warning: Near-zero variance in savgol data, adding noise")
                # Add small noise to avoid numerical instability
                var_value = 1e-5
                
            initial_state_covariance = var_value
            
            # Define transition matrix (linear relationship between states)
            transition_matrix = np.array([[1.0]])
            
            # Define noise covariances - use conservative values for stability
            process_noise_covariance = np.array([[max(var_value * 0.01, 1e-5)]])  # Small process noise
            observation_noise_covariance = np.array([[max(var_value * 0.1, 1e-4)]])  # Larger observation noise
            
            # Create KalmanFilter object
            kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
                observation_covariance=observation_noise_covariance,
                transition_covariance=process_noise_covariance
            )
            
            # Get mask of valid savgol values
            valid_mask = ~df['savgol'].isna()
            
            # If we don't have any valid values, return with NaN
            if not valid_mask.any():
                print("No valid savgol data for Kalman filtering")
                df['kalman'] = np.nan
                df['kalman_smoothed'] = np.nan
                return df
            
            # Apply Kalman filter only on valid indices
            # Fill missing values with initial_state_mean for Kalman input
            observations = df['savgol'].fillna(initial_state_mean).values.reshape(-1, 1)
            
            # Apply filter
            smoothed_state_means, _ = kf.filter(observations)
            
            # Initialize columns with NaN
            df['kalman_smoothed'] = np.nan
            df['kalman'] = np.nan
            
            # Add smoothed values to DataFrame (only update valid indices to preserve NaN pattern)
            df.loc[valid_mask, 'kalman_smoothed'] = smoothed_state_means.flatten()[valid_mask]
            df.loc[valid_mask, 'kalman'] = smoothed_state_means.flatten()[valid_mask]
            
            return df
            
        except Exception as e:
            print(f"Error applying Kalman filter: {str(e)}")
            # Add fallback values
            df['kalman'] = df['savgol']
            df['kalman_smoothed'] = df['savgol']
            return df
    
    def _calculate_derivatives(self, df):
        """Calculate derivatives of the Kalman smoothed predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with Kalman-filtered predictions
            
        Returns:
            pd.DataFrame: DataFrame with derivatives added
        """
        if df is None or df.empty:
            print("Warning: Empty DataFrame provided to calculate derivatives")
            return df
            
        if 'kalman_smoothed' not in df.columns and 'kalman' not in df.columns:
            # Try to apply Kalman filter first
            if 'savgol' in df.columns:
                try:
                    df = self._apply_kalman_filter(df)
                except Exception as e:
                    print(f"Error applying Kalman filter: {str(e)}")
                    # Add empty derivative columns
                    df['gradient'] = 0
                    df['second_deriv'] = 0
                    return df
            else:
                print("No Kalman-smoothed data to calculate derivatives")
                # Add empty derivative columns
                df['gradient'] = 0
                df['second_deriv'] = 0
                return df
        
        try:
            # Determine which column to use for derivatives
            source_col = 'kalman_smoothed' if 'kalman_smoothed' in df.columns else 'kalman'
            
            # Check if we have enough valid values
            valid_values = df[source_col].dropna()
            if len(valid_values) < 3:  # Need at least 3 points for meaningful derivatives
                print(f"Not enough valid {source_col} values to calculate derivatives (need at least 3)")
                df['gradient'] = 0
                df['second_deriv'] = 0
                return df
            
            # Initialize derivative columns
            df['gradient'] = np.nan
            df['second_deriv'] = np.nan
            
            # Get mask of valid values
            valid_mask = ~df[source_col].isna()
            
            # Calculate first derivative (gradient)
            # Only calculate for rows with valid values
            df.loc[valid_mask, 'gradient'] = df.loc[valid_mask, source_col].diff()
            
            # Calculate second derivative (acceleration)
            df.loc[valid_mask, 'second_deriv'] = df.loc[valid_mask, 'gradient'].diff()
            
            # Safety check - replace any NaNs or infinities with zeros
            df['gradient'] = df['gradient'].fillna(0)
            df['second_deriv'] = df['second_deriv'].fillna(0)
            
            # Replace any infinite values
            df.loc[np.isinf(df['gradient']), 'gradient'] = 0
            df.loc[np.isinf(df['second_deriv']), 'second_deriv'] = 0
            
            return df
            
        except Exception as e:
            print(f"Error calculating derivatives: {str(e)}")
            df['gradient'] = 0
            df['second_deriv'] = 0
            return df
    
    def generate_trading_signals(self, df, buy_threshold=0.0014, sell_threshold=-0.0013, min_trade_freq=0.03):
        """Generate trading signals based on XGBoost predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with predictions
            buy_threshold (float): Threshold for buy signals
            sell_threshold (float): Threshold for sell signals
            min_trade_freq (float): Minimum trading frequency
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check if we have prediction columns
        if 'xgb_probability' not in result.columns:
            if 'xgb_prediction' not in result.columns:
                print("No XGBoost predictions found. Running predict first...")
                result = self.predict(df)
        
        # Apply Kalman filtering and calculate derivatives
        try:
            result = self._apply_kalman_filter(result)
            result = self._calculate_derivatives(result)
            use_gradient = 'gradient' in result.columns and result['gradient'].notna().any()
        except Exception as e:
            print(f"Error in signal processing: {str(e)}")
            use_gradient = False
            
        # Generate signals based on probability threshold or gradient
        result['xgb_signal'] = 0  # Neutral by default
        
        if use_gradient:
            print("Using gradient-based signals")
            # Calculate actual signal counts for initial thresholds
            actual_buy_signals = (result['gradient'] >= buy_threshold).sum()
            actual_sell_signals = (result['gradient'] <= sell_threshold).sum()
            
            # Calculate minimum required signals
            total_rows = len(result.dropna(subset=['gradient']))
            min_signal_rows = int(total_rows * min_trade_freq)
            
            # Adjust thresholds if needed to meet minimum trading frequency
            if (actual_buy_signals + actual_sell_signals) < min_signal_rows:
                print(f"Initial XGBoost signal count too low: {actual_buy_signals} buys, {actual_sell_signals} sells")
                print(f"Minimum required: {min_signal_rows} signals. Adjusting thresholds...")
                
                # Calculate percentile-based thresholds
                valid_gradients = result['gradient'].dropna()
                if len(valid_gradients) > 0:
                    buy_threshold = np.percentile(valid_gradients, 93)  # Top 7%
                    sell_threshold = np.percentile(valid_gradients, 7)  # Bottom 7%
                    print(f"Adjusted thresholds - Buy: {buy_threshold:.4f}, Sell: {sell_threshold:.4f}")
            
            # Generate signals based on adjusted gradient thresholds
            result.loc[result['gradient'] >= buy_threshold, 'xgb_signal'] = 1
            result.loc[result['gradient'] <= sell_threshold, 'xgb_signal'] = -1
            
        else:
            print("Using probability-based signals")
            # Calculate actual signal counts for initial thresholds
            prob_buy_threshold = 0.5 + buy_threshold
            prob_sell_threshold = 0.5 - sell_threshold
            
            if 'xgb_probability' in result.columns:
                actual_buy_signals = (result['xgb_probability'] >= prob_buy_threshold).sum()
                actual_sell_signals = (result['xgb_probability'] <= prob_sell_threshold).sum()
            else:
                # Fall back to prediction if probability not available
                actual_buy_signals = (result['xgb_prediction'] == 1).sum()
                actual_sell_signals = (result['xgb_prediction'] == 0).sum()
                
            # Calculate minimum required signals
            total_rows = len(result.dropna(subset=['xgb_prediction']) if 'xgb_prediction' in result.columns 
                             else result)
            min_signal_rows = int(total_rows * min_trade_freq)
            
            # Adjust thresholds if needed to meet minimum trading frequency
            if (actual_buy_signals + actual_sell_signals) < min_signal_rows:
                print(f"Initial XGBoost signal count too low: {actual_buy_signals} buys, {actual_sell_signals} sells")
                print(f"Minimum required: {min_signal_rows} signals. Adjusting thresholds...")
                
                # Gradually reduce thresholds until we have enough signals
                adjusted_buy_threshold = buy_threshold
                adjusted_sell_threshold = sell_threshold
                
                while (actual_buy_signals + actual_sell_signals) < min_signal_rows and adjusted_buy_threshold > 0:
                    adjusted_buy_threshold -= 0.001
                    adjusted_sell_threshold -= 0.001
                    
                    prob_buy_threshold = 0.5 + adjusted_buy_threshold
                    prob_sell_threshold = 0.5 - adjusted_sell_threshold
                    
                    if 'xgb_probability' in result.columns:
                        actual_buy_signals = (result['xgb_probability'] >= prob_buy_threshold).sum()
                        actual_sell_signals = (result['xgb_probability'] <= prob_sell_threshold).sum()
                    else:
                        # If thresholds reach 0, we'll use all predictions
                        actual_buy_signals = (result['xgb_prediction'] == 1).sum()
                        actual_sell_signals = (result['xgb_prediction'] == 0).sum()
                
                print(f"Adjusted XGBoost signal thresholds - Buy: prob > {prob_buy_threshold:.4f}, "
                      f"Sell: prob < {prob_sell_threshold:.4f}")
                
                buy_threshold = adjusted_buy_threshold
                sell_threshold = adjusted_sell_threshold
            
            # Generate signals based on final thresholds
            if 'xgb_probability' in result.columns:
                result.loc[result['xgb_probability'] >= 0.5 + buy_threshold, 'xgb_signal'] = 1
                result.loc[result['xgb_probability'] <= 0.5 - sell_threshold, 'xgb_signal'] = -1
            else:
                # Fall back to prediction if probability not available
                result.loc[result['xgb_prediction'] == 1, 'xgb_signal'] = 1
                result.loc[result['xgb_prediction'] == 0, 'xgb_signal'] = -1
        
        # For compatibility with other models
        result['signal'] = result['xgb_signal']
        
        # Print signal statistics
        buy_signals = (result['xgb_signal'] == 1).sum()
        sell_signals = (result['xgb_signal'] == -1).sum()
        total_rows = len(result)
        
        print(f"XGBoost model generated {buy_signals} buy signals and {sell_signals} sell signals")
        print(f"Signal frequency: {(buy_signals + sell_signals) / total_rows:.4f} ({buy_signals + sell_signals} / {total_rows})")
        
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