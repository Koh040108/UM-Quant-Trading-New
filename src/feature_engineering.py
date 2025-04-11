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

from src.config import DATA_DIR, LOOKBACK_WINDOW


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
        
        # Determine which price column to use
        price_col = None
        if 'close' in result.columns:
            price_col = 'close'
        elif 'price' in result.columns:
            price_col = 'price'
        else:
            print("Warning: No price or close column found. Cannot create price-based features.")
            return result
        
        # Price momentum features
        result['price_change_1d'] = result[price_col].pct_change(1)
        result['price_change_3d'] = result[price_col].pct_change(3)
        result['price_change_7d'] = result[price_col].pct_change(7)
        
        # Add Rate of Change (ROC) for different time periods
        result['roc_5'] = self.calculate_roc(result[price_col], window=5)
        result['roc_10'] = self.calculate_roc(result[price_col], window=10)
        result['roc_20'] = self.calculate_roc(result[price_col], window=20)
        
        # Rolling statistics
        result[f'{price_col}_7d_mean'] = result[price_col].rolling(window=7).mean()
        result[f'{price_col}_7d_std'] = result[price_col].rolling(window=7).std()
        result[f'{price_col}_7d_max'] = result[price_col].rolling(window=7).max()
        result[f'{price_col}_7d_min'] = result[price_col].rolling(window=7).min()
        
        # Volatility features
        result['volatility_7d'] = result[price_col].pct_change().rolling(window=7).std()
        result['volatility_14d'] = result[price_col].pct_change().rolling(window=14).std()
        result['volatility_30d'] = result[price_col].pct_change().rolling(window=30).std()
        
        # High-Low Range
        if all(col in result.columns for col in ['high', 'low']):
            result['daily_range'] = (result['high'] - result['low']) / result[price_col]
            result['daily_range_7d_mean'] = result['daily_range'].rolling(window=7).mean()
        
        # Volume features
        if 'volume' in result.columns:
            result['volume_change_1d'] = result['volume'].pct_change(1)
            result['volume_change_7d'] = result['volume'].pct_change(7)
            result['volume_7d_mean'] = result['volume'].rolling(window=7).mean()
            result['volume_close_ratio'] = result['volume'] / result[price_col]
        
        # Lagged features for time series modeling
        for lag in range(1, lookback_window + 1):
            result[f'{price_col}_lag_{lag}'] = result[price_col].shift(lag)
            
            if 'volume' in result.columns:
                result[f'volume_lag_{lag}'] = result['volume'].shift(lag)
        
        # Fill NaN values created by the calculations
        result.fillna(method='ffill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def calculate_roc(self, series, window=10):
        """Calculate Rate of Change (ROC) for a given series.
        
        Args:
            series (pd.Series): Price or other value series
            window (int): Lookback period for ROC calculation
            
        Returns:
            pd.Series: Rate of Change values
        """
        # Calculate percentage change over the specified window
        roc = series.pct_change(periods=window) * 100
        return roc
    
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
        elif 'date' not in result.columns and 'time' in result.columns:
            result['date'] = pd.to_datetime(result['time'])
        
        # Ensure date is in datetime format
        if 'date' in result.columns and not pd.api.types.is_datetime64_any_dtype(result['date']):
            result['date'] = pd.to_datetime(result['date'])
        
        # Merge each on-chain DataFrame
        for source, df in onchain_dfs.items():
            if df.empty:
                continue
            
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Determine which column to use for date joining
            date_column = None
            
            # Check for date columns in order of preference
            if 'date' in df_copy.columns:
                date_column = 'date'
            elif 'datetime' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['datetime'])
                date_column = 'date'
            elif 'timestamp' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
                date_column = 'date'
            elif 'start_time' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['start_time'], unit='ms')
                date_column = 'date'
            elif 'time' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['time'])
                date_column = 'date'
            else:
                print(f"Skipping {source}: No date/timestamp/time/datetime column found")
                continue
            
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            
            # Print a sample of dates to debug
            print(f"  {source} date samples: {df_copy[date_column].iloc[:3].tolist()}")
            print(f"  Result date samples: {result['date'].iloc[:3].tolist()}")
            
            # Rename columns to avoid conflicts
            rename_cols = {}
            for col in df_copy.columns:
                if col not in ['date']:
                    rename_cols[col] = f"{source}_{col}"
            
            df_renamed = df_copy.rename(columns=rename_cols)
            
            try:
                # Merge on date
                result = pd.merge(result, df_renamed, on='date', how='left')
                print(f"  Successfully merged {source} data")
            except Exception as e:
                print(f"  Error merging {source} data: {str(e)}")
        
        # Fill NaN values created by the merge
        numeric_cols = [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col])]
        if numeric_cols:
            # Handle null or infinite values
            result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
            result[numeric_cols] = result[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
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
        Complete data processing pipeline.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            onchain_dfs (dict): Dictionary of DataFrames with on-chain data
            add_indicators (bool): Whether to add technical indicators
            add_custom (bool): Whether to add custom features
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        result = price_df.copy()
        
        # Step 1: Add technical indicators
        if add_indicators:
            result = self.add_technical_indicators(result)
        
        # Step 2: Add custom features
        if add_custom:
            result = self.add_custom_features(result)
            
        # Step 3: Add on-chain features if provided
        if onchain_dfs is not None and len(onchain_dfs) > 0:
            result = self.add_on_chain_features(result, onchain_dfs)
        
        # Step 4: Handle outliers
        result = self.handle_outliers(result)
        
        # Step 5: Normalize features if requested
        if self.normalize:
            # Make sure to drop non-numeric columns before normalizing
            date_cols = [col for col in result.columns if 'date' in col.lower() or 'time' in col.lower()]
            numeric_cols = [col for col in result.columns if col not in date_cols and pd.api.types.is_numeric_dtype(result[col])]
            
            if numeric_cols:
                # Replace inf values and handle NaN before normalization
                result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
                result[numeric_cols] = result[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Store non-numeric columns
                non_numeric = result[date_cols].copy()
                
                # Normalize numeric columns
                normalized = self.normalize_features(result[numeric_cols])
                
                # Combine back with non-numeric columns
                result = pd.concat([non_numeric, normalized], axis=1)
        
        # Print a message before returning so we can confirm the key columns are present
        if 'price_change_1d' in result.columns and 'volatility_7d' in result.columns:
            print("Successfully created required features for HMM model.")
        else:
            print("Warning: Required features for HMM model not created.")
            print(f"Available columns: {result.columns.tolist()}")
        
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
        Load and process data for a cryptocurrency.
        
        Args:
            crypto (str): Cryptocurrency symbol (e.g., 'BTC')
            start_date (str): Start date for filtering
            end_date (str): End date for filtering
            
        Returns:
            pd.DataFrame: Processed DataFrame with all features
        """
        try:
            # Find price data files for this crypto
            price_files = [f for f in os.listdir(DATA_DIR) if f.startswith(crypto) and ('market_data' in f or 'price' in f) and f.endswith('.csv')]
            
            if not price_files:
                print(f"No price data files found for {crypto}. Please make sure data is downloaded.")
                return pd.DataFrame()
            
            # Use the most recent or most appropriate file
            price_file = sorted(price_files)[-1]  # Sort alphabetically and take the last one
            price_path = os.path.join(DATA_DIR, price_file)
            print(f"Using price data from {price_path}")
            
            # Load price data
            price_df = pd.read_csv(price_path)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in price_df.columns and 'date' not in price_df.columns:
                price_df['date'] = pd.to_datetime(price_df['timestamp'], unit='ms')
            elif 'time' in price_df.columns and 'date' not in price_df.columns:
                price_df['date'] = pd.to_datetime(price_df['time'])
            
            # Ensure date column is always in datetime format
            if 'date' in price_df.columns and not pd.api.types.is_datetime64_any_dtype(price_df['date']):
                price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Filter by date if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                price_df = price_df[price_df['date'] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                price_df = price_df[price_df['date'] <= end_date]
            
            # Ensure OHLCV columns or at least a price column
            if 'close' not in price_df.columns and 'price' not in price_df.columns:
                if all(col in price_df.columns for col in ['open', 'high', 'low']):
                    # If we have most OHLC columns but no close, we can use the last available price
                    price_df['close'] = price_df['open'].shift(-1)  # Use next open as this period's close
                    price_df.fillna(method='ffill', inplace=True)
                elif 'value' in price_df.columns:
                    # Some datasets use 'value' for price
                    price_df['price'] = price_df['value']
                else:
                    print(f"Warning: No price data found in {price_file}. Please check the data format.")
            
            # Look for on-chain data
            print(f"Looking for on-chain data for {crypto}...")
            onchain_dfs = {}
            onchain_files = [f for f in os.listdir(DATA_DIR) if (f.startswith(crypto) or ('_' + crypto.lower() + '_' in f)) and f != price_file and f.endswith('.csv')]
            
            for file in onchain_files:
                try:
                    file_path = os.path.join(DATA_DIR, file)
                    # Extract dataset name from filename
                    dataset_name = file.replace(crypto + '_', '').replace('.csv', '')
                    
                    # Load data
                    df = pd.read_csv(file_path)
                    
                    # Convert date columns to datetime
                    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'])
                    elif 'timestamp' in df.columns and 'date' not in df.columns:
                        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    elif 'time' in df.columns and 'date' not in df.columns:
                        df['date'] = pd.to_datetime(df['time'])
                    
                    # Only include non-empty dataframes
                    if not df.empty:
                        onchain_dfs[dataset_name] = df
                        print(f"  Loaded {dataset_name} with {len(df)} rows")
                except Exception as e:
                    print(f"  Error loading {file}: {str(e)}")
            
            print(f"Processing {len(onchain_dfs)} on-chain datasets for {crypto}")
            
            # Process all data
            processed_data = self.process_data(price_df, onchain_dfs, add_indicators=True, add_custom=True)
            
            return processed_data
        except Exception as e:
            print(f"Error processing data: {str(e)}")
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