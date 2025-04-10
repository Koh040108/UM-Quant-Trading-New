"""
Feature engineering module for processing crypto data and extracting meaningful features.
"""

import os
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import DATA_DIR, LOOKBACK_WINDOW


class FeatureEngineer:
    """Class for creating features from raw crypto data"""
    
    def __init__(self, normalize=True, scaler_type="standard"):
        """
        Initialize the feature engineer.
        
        Args:
            normalize (bool): Whether to normalize features
            scaler_type (str): Type of scaler to use ('standard' or 'minmax')
        """
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.scaler = None
        
        if self.normalize:
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe."""
        result = df.copy()
        
        # Check if we have OHLCV data
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # Full OHLCV data available - add all technical indicators
            # Add simple moving averages
            result['sma_7'] = df['close'].rolling(window=7).mean()
            result['sma_14'] = df['close'].rolling(window=14).mean()
            result['sma_30'] = df['close'].rolling(window=30).mean()
            
            # Add moving average convergence divergence (MACD)
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            result['macd'] = ema_12 - ema_26
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # Add relative strength index (RSI)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # Add Bollinger Bands
            result['bb_middle'] = df['close'].rolling(window=20).mean()
            result['bb_upper'] = result['bb_middle'] + 2 * df['close'].rolling(window=20).std()
            result['bb_lower'] = result['bb_middle'] - 2 * df['close'].rolling(window=20).std()
            
            # Add true range and average true range (ATR)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            result['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result['atr'] = result['tr'].rolling(window=14).mean()
            
        elif 'price' in df.columns or 'close' in df.columns:
            # Simplified data with only price - add basic indicators
            price_col = 'price' if 'price' in df.columns else 'close'
            
            # Add simple moving averages
            result['sma_7'] = df[price_col].rolling(window=7).mean()
            result['sma_14'] = df[price_col].rolling(window=14).mean()
            result['sma_30'] = df[price_col].rolling(window=30).mean()
            
            # Add MACD
            ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
            ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
            result['macd'] = ema_12 - ema_26
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # Add RSI
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # Add Bollinger Bands
            result['bb_middle'] = df[price_col].rolling(window=20).mean()
            result['bb_upper'] = result['bb_middle'] + 2 * df[price_col].rolling(window=20).std()
            result['bb_lower'] = result['bb_middle'] - 2 * df[price_col].rolling(window=20).std()
            
            print(f"Using simplified price data with {price_col} column.")
        else:
            raise ValueError("DataFrame must contain OHLCV columns or at least a 'price' column")
        
        return result
    
    def add_custom_features(self, df, lookback_window=LOOKBACK_WINDOW):
        """
        Add custom features to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            lookback_window (int): Lookback window for feature calculation
            
        Returns:
            pd.DataFrame: DataFrame with added custom features
        """
        result = df.copy()
        
        # Ensure we have a 'close' column
        if 'close' not in result.columns:
            return result
        
        # Price momentum features
        result['price_change_1d'] = result['close'].pct_change(1)
        result['price_change_3d'] = result['close'].pct_change(3)
        result['price_change_7d'] = result['close'].pct_change(7)
        
        # Rolling statistics
        result['close_7d_mean'] = result['close'].rolling(window=7).mean()
        result['close_7d_std'] = result['close'].rolling(window=7).std()
        result['close_7d_max'] = result['close'].rolling(window=7).max()
        result['close_7d_min'] = result['close'].rolling(window=7).min()
        
        # Volatility features
        result['volatility_7d'] = result['close'].pct_change().rolling(window=7).std()
        result['volatility_14d'] = result['close'].pct_change().rolling(window=14).std()
        result['volatility_30d'] = result['close'].pct_change().rolling(window=30).std()
        
        # High-Low Range
        if all(col in result.columns for col in ['high', 'low']):
            result['daily_range'] = (result['high'] - result['low']) / result['close']
            result['daily_range_7d_mean'] = result['daily_range'].rolling(window=7).mean()
        
        # Volume features
        if 'volume' in result.columns:
            result['volume_change_1d'] = result['volume'].pct_change(1)
            result['volume_change_7d'] = result['volume'].pct_change(7)
            result['volume_7d_mean'] = result['volume'].rolling(window=7).mean()
            result['volume_close_ratio'] = result['volume'] / result['close']
        
        # Lagged features for time series modeling
        for lag in range(1, lookback_window + 1):
            result[f'close_lag_{lag}'] = result['close'].shift(lag)
            
            if 'volume' in result.columns:
                result[f'volume_lag_{lag}'] = result['volume'].shift(lag)
        
        # Fill NaN values created by the calculations
        result.fillna(method='bfill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def add_on_chain_features(self, price_df, onchain_dfs):
        """
        Merge on-chain data features with price data.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            onchain_dfs (dict): Dictionary of DataFrames with on-chain data
            
        Returns:
            pd.DataFrame: Merged DataFrame with price and on-chain features
        """
        result = price_df.copy()
        
        # Ensure we have a date column for merging
        if 'date' not in result.columns and 'timestamp' in result.columns:
            result['date'] = pd.to_datetime(result['timestamp'], unit='ms')
        
        # Merge each on-chain DataFrame
        for source, df in onchain_dfs.items():
            if df.empty:
                continue
                
            # Ensure the on-chain data has a date column
            if 'date' not in df.columns and 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Rename columns to avoid conflicts
            rename_cols = {}
            for col in df.columns:
                if col not in ['date', 'timestamp']:
                    rename_cols[col] = f"{source}_{col}"
            
            df_renamed = df.rename(columns=rename_cols)
            
            # Merge on date
            result = pd.merge(result, df_renamed, on='date', how='left')
        
        # Fill NaN values created by the merge
        result.fillna(method='ffill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def normalize_features(self, df, fit=True):
        """
        Normalize features using the selected scaler.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            fit (bool): Whether to fit the scaler
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        if not self.normalize or self.scaler is None:
            return df
        
        # Make a copy of the DataFrame
        result = df.copy()
        
        # Get numeric columns, excluding dates and non-numeric columns
        exclude_cols = ['date', 'timestamp']
        numeric_cols = [col for col in result.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col])]
        
        if not numeric_cols:
            return result
        
        # Handle null or infinite values before scaling
        result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
        result[numeric_cols] = result[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        if fit:
            result[numeric_cols] = self.scaler.fit_transform(result[numeric_cols])
        else:
            result[numeric_cols] = self.scaler.transform(result[numeric_cols])
        
        return result
    
    def handle_outliers(self, df, method='winsorize', threshold=3.0):
        """
        Detect and handle outliers in the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            method (str): Method to handle outliers ('winsorize', 'clip', or 'remove')
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        result = df.copy()
        
        # Exclude non-numeric columns
        exclude_cols = ['date', 'timestamp']
        numeric_cols = [col for col in result.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col])]
        
        for col in numeric_cols:
            # Calculate z-scores for the column
            mean = result[col].mean()
            std = result[col].std()
            
            if std == 0:  # Skip columns with zero standard deviation
                continue
                
            z_scores = (result[col] - mean) / std
            
            if method == 'winsorize':
                # Winsorize: cap values at threshold
                result[col] = result[col].clip(
                    lower=mean - threshold * std,
                    upper=mean + threshold * std
                )
            elif method == 'clip':
                # Replace values beyond threshold with NaN and then fill
                mask = abs(z_scores) > threshold
                result.loc[mask, col] = np.nan
                result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
            elif method == 'remove':
                # Set rows with outliers to NaN (more appropriate for single-column outliers)
                mask = abs(z_scores) > threshold
                result.loc[mask, col] = np.nan
        
        # Fill any remaining NaN values
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return result
    
    def create_windows(self, df, window_size=LOOKBACK_WINDOW, target_col='close', horizon=1):
        """
        Create time windows for sequence modeling.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            window_size (int): Size of the lookback window
            target_col (str): Column to use as prediction target
            horizon (int): Forecast horizon
            
        Returns:
            tuple: (X, y) where X is a 3D array of windows and y is target values
        """
        # Columns to exclude from features
        exclude_cols = ['date', 'timestamp', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Extract numpy arrays
        data = df[feature_cols].values
        target = df[target_col].values
        
        X, y = [], []
        
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:(i + window_size)])
            y.append(target[i + window_size + horizon - 1])
        
        return np.array(X), np.array(y)
    
    def process_data(self, price_df, onchain_dfs=None, add_indicators=True, add_custom=True):
        """
        Process data by adding features and normalizing.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            onchain_dfs (dict, optional): Dictionary of DataFrames with on-chain data
            add_indicators (bool): Whether to add technical indicators
            add_custom (bool): Whether to add custom features
            
        Returns:
            pd.DataFrame: Processed DataFrame with all features
        """
        result = price_df.copy()
        
        # Add technical indicators
        if add_indicators:
            result = self.add_technical_indicators(result)
        
        # Add custom features
        if add_custom:
            result = self.add_custom_features(result)
        
        # Add on-chain features if provided
        if onchain_dfs is not None:
            result = self.add_on_chain_features(result, onchain_dfs)
        
        # Normalize features
        if self.normalize:
            result = self.normalize_features(result)
        
        return result
    
    def create_training_data(self, df, window_size=LOOKBACK_WINDOW, target_col='close', horizon=1, 
                             binary_target=False, threshold=0):
        """
        Create training data for machine learning models.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            window_size (int): Size of the lookback window
            target_col (str): Column to use as prediction target
            horizon (int): Forecast horizon
            binary_target (bool): Whether to create binary classification target
            threshold (float): Threshold for binary classification
            
        Returns:
            tuple: (X, y) for training
        """
        # Process the data
        processed_df = self.process_data(df)
        
        # Create target variable
        if binary_target:
            # Create binary target for price direction
            price_change = processed_df[target_col].pct_change(horizon).shift(-horizon)
            processed_df['target'] = (price_change > threshold).astype(int)
            target_col = 'target'
        
        # Create windows
        X, y = self.create_windows(processed_df, window_size, target_col, horizon)
        
        return X, y
    
    def load_and_process_data(self, crypto, start_date=None, end_date=None):
        """
        Load and process all available data for a specific cryptocurrency.
        
        Args:
            crypto (str): Cryptocurrency symbol (e.g., 'BTC')
            start_date (str): Optional start date filter in YYYY-MM-DD format
            end_date (str): Optional end date filter in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Processed DataFrame with all features
        """
        try:
            # Check for CCXT/market data first (default format)
            price_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{crypto}_ccxt_market") and f.endswith('.csv')]
            
            if price_files:
                price_file = os.path.join(DATA_DIR, price_files[0])
                price_df = pd.read_csv(price_file)
                print(f"Using price data from {price_file}")
            else:
                # Try to find any price data
                glassnode_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{crypto}_glassnode_market_price") and f.endswith('.csv')]
                if glassnode_files:
                    price_file = os.path.join(DATA_DIR, glassnode_files[0])
                    price_df = pd.read_csv(price_file)
                    
                    # Rename value to price for consistency if needed
                    if 'value' in price_df.columns and 'price' not in price_df.columns:
                        price_df = price_df.rename(columns={'value': 'price'})
                        
                    print(f"Using Glassnode price data from {price_file}")
                else:
                    print(f"No price data found for {crypto}")
                    return pd.DataFrame()
            
            # Ensure date column is in datetime format
            if 'date' in price_df.columns:
                price_df['date'] = pd.to_datetime(price_df['date'])
            elif 'timestamp' in price_df.columns:
                price_df['date'] = pd.to_datetime(price_df['timestamp'], unit='ms')
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                price_df = price_df[price_df['date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                price_df = price_df[price_df['date'] <= end_date]
            
            if price_df.empty:
                print(f"No data found for {crypto} in the specified date range")
                return pd.DataFrame()
            
            # Look for on-chain data files
            print(f"Looking for on-chain data for {crypto}...")
            onchain_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{crypto}_") and f.endswith('.csv') and not f.startswith(f"{crypto}_ccxt")]
            
            onchain_dfs = {}
            for file in onchain_files:
                # Extract source and feature from filename
                parts = file.split('_', 2)
                if len(parts) < 3:
                    continue
                
                source = parts[1]
                feature = parts[2].replace('.csv', '')
                
                try:
                    file_path = os.path.join(DATA_DIR, file)
                    df = pd.read_csv(file_path)
                    
                    # Ensure date column is in datetime format
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    elif 'timestamp' in df.columns:
                        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Apply date filters
                    if start_date:
                        df = df[df['date'] >= start_date]
                    if end_date:
                        df = df[df['date'] <= end_date]
                    
                    # Skip empty dataframes
                    if df.empty:
                        continue
                    
                    key = f"{source}_{feature}"
                    onchain_dfs[key] = df
                    print(f"  Loaded {key} with {len(df)} rows")
                except Exception as e:
                    print(f"  Error loading {file}: {str(e)}")
            
            # Process the data
            print(f"Processing {len(onchain_dfs)} on-chain datasets for {crypto}")
            processed_df = self.process_data(price_df, onchain_dfs)
            
            # Handle outliers to improve model stability
            processed_df = self.handle_outliers(processed_df, method='winsorize', threshold=3.0)
            
            # Sort by date for time-series analysis
            if 'date' in processed_df.columns:
                processed_df = processed_df.sort_values('date')
            
            return processed_df
            
        except Exception as e:
            print(f"Error in load_and_process_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    feature_engineer = FeatureEngineer(normalize=True)
    
    # Assuming price data is available
    try:
        processed_data = feature_engineer.load_and_process_data("BTC")
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Columns: {processed_data.columns.tolist()[:10]} ... (and {len(processed_data.columns) - 10} more)")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("This example assumes you have already fetched and saved data using the data_fetcher module.") 