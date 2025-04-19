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
        
        # Ensure 'close' is available - this is our primary price column
        if 'close' not in result.columns:
            if 'price' in result.columns:
                print(f"Copying 'price' column to 'close' for indicator calculation")
                result['close'] = result['price']
            elif 'price_usd_close' in result.columns:
                print(f"Copying 'price_usd_close' column to 'close' for indicator calculation")
                result['close'] = result['price_usd_close']
            elif 'value' in result.columns:
                print(f"Copying 'value' column to 'close' for indicator calculation")
                result['close'] = result['value']
            else:
                print(f"Warning: No price column found to use as 'close'. Technical indicators may not be calculated correctly.")
                return result
        else:
            print(f"Using existing 'close' column for indicator calculation")
        
        # Check if we have OHLCV data
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            print(f"Full OHLCV data available - adding complete set of technical indicators")
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
        else:
            print(f"Simplified price data - adding basic technical indicators using close column")
            # Simplified data with only close price - add basic indicators
            # Add simple moving averages
            result['sma_7'] = result['close'].rolling(window=7).mean()
            result['sma_14'] = result['close'].rolling(window=14).mean()
            result['sma_30'] = result['close'].rolling(window=30).mean()
            
            # Add MACD
            ema_12 = result['close'].ewm(span=12, adjust=False).mean()
            ema_26 = result['close'].ewm(span=26, adjust=False).mean()
            result['macd'] = ema_12 - ema_26
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # Add RSI
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # Add Bollinger Bands
            result['bb_middle'] = result['close'].rolling(window=20).mean()
            result['bb_upper'] = result['bb_middle'] + 2 * result['close'].rolling(window=20).std()
            result['bb_lower'] = result['bb_middle'] - 2 * result['close'].rolling(window=20).std()
        
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
        
        # Ensure close is available as our primary price column
        if 'close' not in result.columns:
            if 'price' in result.columns:
                print(f"Copying 'price' column to 'close' for feature calculation")
                result['close'] = result['price']
            elif 'price_usd_close' in result.columns:
                print(f"Copying 'price_usd_close' column to 'close' for feature calculation")
                result['close'] = result['price_usd_close']
            elif 'value' in result.columns:
                print(f"Copying 'value' column to 'close' for feature calculation")
                result['close'] = result['value']
            else:
                print("Warning: No price column found. Cannot create price-based features.")
                return result
        else:
            print(f"Using existing 'close' column for custom feature calculation")
        
        # Now we always use 'close' as the price column
        price_col = 'close'
        
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
        
        # Safely handle NaN values
        if result is not None:
            result = result.ffill()
            result = result.fillna(0)
        else:
            print("Warning: Result DataFrame is None, cannot fill NaN values")
        
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
        Merge on-chain data features with price data using start_time directly.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            onchain_dfs (dict): Dictionary of DataFrames with on-chain data
            
        Returns:
            pd.DataFrame: Merged DataFrame with price and on-chain features
        """
        result = price_df.copy()
        
        # Ensure we have a price column using 'close'
        if 'close' in result.columns:
            print("Using 'close' as the primary price column")
            if 'price' not in result.columns:
                result['price'] = result['close']
        elif 'price_usd_close' in result.columns:
            print("Using 'price_usd_close' as the primary price column and renaming to 'close'")
            result['close'] = result['price_usd_close']
            result['price'] = result['price_usd_close']
        
        # Identify which column to use as the datetime index
        index_col = None
        if 'start_time' in result.columns:
            index_col = 'start_time'
            print(f"Using start_time as the primary index column")
        elif 'timestamp' in result.columns:
            index_col = 'timestamp'
            print(f"Using timestamp as the primary index column")
        elif 'date' in result.columns:
            index_col = 'date'
            print(f"Using date as the primary index column")
        elif 'time' in result.columns:
            index_col = 'time'
            print(f"Using time as the primary index column")
        else:
            print("ERROR: No suitable date/time column found in price data")
            return result
        
        # Ensure the index column is in datetime format
        if index_col in result.columns:
            try:
                if index_col in ['start_time', 'timestamp']:
                    # Convert epoch milliseconds to datetime
                    result[index_col] = pd.to_datetime(result[index_col], unit='ms')
                else:
                    # Convert string to datetime
                    result[index_col] = pd.to_datetime(result[index_col])
                
                print(f"Converted {index_col} to datetime: {result[index_col].dtype}")
            except Exception as e:
                print(f"Error converting {index_col} to datetime: {str(e)}")
                # Don't return early, try to proceed
        
        # Print price data info for debugging
        if 'close' in result.columns:
            print(f"Price range (close): {result['close'].min()} to {result['close'].max()}")
        
        # Set the index for merging
        result = result.set_index(index_col)
        print(f"Set index on {index_col} with {len(result)} rows")
        
        # Process each on-chain DataFrame
        for source, df in onchain_dfs.items():
            if df is None or df.empty:
                print(f"Skipping empty {source} dataset")
                continue
            
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            print(f"Processing {source} dataset with {len(df_copy)} rows")
            
            # Match index column from price data in on-chain data
            if index_col in df_copy.columns:
                onchain_index_col = index_col
                print(f"Found matching {index_col} column in {source} data")
            else:
                # Look for alternative time columns in the on-chain data
                potential_time_cols = ['start_time', 'timestamp', 'date', 'time', 'datetime']
                onchain_index_col = None
                
                for col in potential_time_cols:
                    if col in df_copy.columns:
                        onchain_index_col = col
                        print(f"Using {onchain_index_col} from {source} data for alignment")
                        break
                
                if onchain_index_col is None:
                    print(f"ERROR: No suitable time column found in {source} data")
                    continue
            
            # Convert on-chain index to datetime
            try:
                if onchain_index_col in ['start_time', 'timestamp']:
                    # Convert epoch milliseconds to datetime
                    df_copy[onchain_index_col] = pd.to_datetime(df_copy[onchain_index_col], unit='ms')
                else:
                    # Convert string to datetime
                    df_copy[onchain_index_col] = pd.to_datetime(df_copy[onchain_index_col])
                
                print(f"Converted {source}.{onchain_index_col} to datetime")
            except Exception as e:
                print(f"Error converting {source}.{onchain_index_col} to datetime: {str(e)}")
                continue
            
            # Check if we successfully converted to datetime
            if not pd.api.types.is_datetime64_any_dtype(df_copy[onchain_index_col]):
                print(f"ERROR: Failed to convert {source}.{onchain_index_col} to datetime")
                continue
            
            # Print date ranges for debugging
            print(f"Price data {index_col} range: {result.index.min()} to {result.index.max()}")
            print(f"{source} data {onchain_index_col} range: {df_copy[onchain_index_col].min()} to {df_copy[onchain_index_col].max()}")
            
            # Check date overlap
            onchain_min_date = df_copy[onchain_index_col].min()
            onchain_max_date = df_copy[onchain_index_col].max()
            price_min_date = result.index.min()
            price_max_date = result.index.max()
            
            if onchain_min_date > price_max_date or onchain_max_date < price_min_date:
                print(f"WARNING: No date overlap between price data and {source} data")
                continue
            
            # Identify numeric features only to avoid issues with dates and strings
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                print(f"No numeric columns found in {source} data")
                continue
            
            # Rename numeric columns to avoid conflicts
            renamed_cols = {col: f"{source}_{col}" for col in numeric_cols}
            
            # Create a clean dataframe with just the index column and renamed numeric columns
            clean_df = pd.DataFrame()
            clean_df[onchain_index_col] = df_copy[onchain_index_col]
            
            for old_col, new_col in renamed_cols.items():
                clean_df[new_col] = df_copy[old_col]
            
            # Set index for joining
            clean_df = clean_df.set_index(onchain_index_col)
            
            # Check if indices need alignment and perform reindexing
            if not clean_df.index.equals(result.index):
                print(f"Reindexing {source} data to match price data")
                
                # Reindex to align with price data
                try:
                    # Sort by index to ensure proper alignment
                    clean_df = clean_df.sort_index()
                    result = result.sort_index()
                    
                    # Use merge_asof for non-exact timestamp alignment
                    # Convert both to DataFrame with reset index
                    result_reset = result.reset_index()
                    clean_df_reset = clean_df.reset_index()
                    
                    # Merge using merge_asof which matches on nearest timestamps
                    merged = pd.merge_asof(
                        result_reset, 
                        clean_df_reset,
                        left_on=index_col,
                        right_on=onchain_index_col,
                        direction='nearest'
                    )
                    
                    # Set index back to the original index column
                    merged = merged.set_index(index_col)
                    
                    # Update result
                    result = merged
                    print(f"Successfully merged {source} data using merge_asof")
                    
                except Exception as e:
                    print(f"Error during merge_asof for {source}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                # If indices match exactly, perform a simple join
                print(f"Direct joining {source} data (exact index match)")
                result = result.join(clean_df, how='left')
        
        # Reset index to convert back to a column
        result = result.reset_index()
        
        # Fill missing values in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Handle null or infinite values
            result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
            result[numeric_cols] = result[numeric_cols].ffill().bfill().fillna(0)
        
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
        
        # Identify non-numeric and date-like columns to exclude
        exclude_cols = []
        
        # Add all columns with date-like names or that contain date-like strings
        for col in result.columns:
            # Check column name for date indicators
            if any(date_indicator in col.lower() for date_indicator in ['date', 'time', 'timestamp', 'hour', 'day', 'year', 'month']):
                exclude_cols.append(col)
                continue
            
            # Check if column has string data that might contain dates
            if result[col].dtype == 'object':
                # Sample the first non-null value to check if it's a date-like string
                sample = result[col].dropna().head(1)
                if len(sample) > 0:
                    sample_val = sample.iloc[0]
                    if isinstance(sample_val, str) and any(date_char in sample_val for date_char in [':', '-', '/', 'T']):
                        exclude_cols.append(col)
        
        # Get only numeric columns that aren't in the exclude list
        numeric_cols = [col for col in result.columns 
                      if col not in exclude_cols 
                      and pd.api.types.is_numeric_dtype(result[col])]
        
        if not numeric_cols:
            print("Warning: No numeric columns to normalize")
            return result
        
        # Print what we're normalizing
        print(f"Normalizing {len(numeric_cols)} numeric features (excluded {len(exclude_cols)} non-numeric/date columns)")
        
        # Handle null or infinite values before scaling
        result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
        result[numeric_cols] = result[numeric_cols].ffill().bfill().fillna(0)
        
        # Extract numeric data for scaling
        numeric_data = result[numeric_cols].values
        
        # Scale features
        try:
            if fit:
                scaled_data = self.scaler.fit_transform(numeric_data)
            else:
                scaled_data = self.scaler.transform(numeric_data)
            
            # Put scaled data back into DataFrame
            result[numeric_cols] = scaled_data
        except Exception as e:
            print(f"Error during normalization: {e}")
            print(f"First few rows of numeric data: {numeric_data[:3]}")
            # Return unscaled data if scaling fails
            return result
        
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
                result[col] = result[col].ffill().bfill()
            elif method == 'remove':
                # Set rows with outliers to NaN (more appropriate for single-column outliers)
                mask = abs(z_scores) > threshold
                result.loc[mask, col] = np.nan
        
        # Fill any remaining NaN values
        result = result.ffill().bfill().fillna(0)
        
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
        Process data by merging price and on-chain data, adding technical indicators.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            onchain_dfs (dict): Dictionary of DataFrames with on-chain data
            add_indicators (bool): Whether to add technical indicators
            add_custom (bool): Whether to add custom features
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            if price_df is None or price_df.empty:
                print("Error: Empty or None price data provided")
                return None

            # Ensure we have a date column in datetime format
            if 'date' not in price_df.columns:
                if 'timestamp' in price_df.columns:
                    price_df['date'] = pd.to_datetime(price_df['timestamp'], unit='ms')
                elif 'time' in price_df.columns:
                    price_df['date'] = pd.to_datetime(price_df['time'])
                elif 'start_time' in price_df.columns:
                    price_df['date'] = pd.to_datetime(price_df['start_time'], unit='ms')
                else:
                    print("Warning: No date column found in price data")
                    return None
            
            # Make sure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(price_df['date']):
                price_df['date'] = pd.to_datetime(price_df['date'])
                
            # Check the time interval of price data
            if len(price_df) > 1:
                time_diff = (price_df['date'].iloc[1] - price_df['date'].iloc[0]).total_seconds() / 3600
                if time_diff >= 23 and time_diff <= 25:  # Daily data (approximately 24 hours)
                    print(f"Detected daily price data with {time_diff:.1f} hour interval.")
                    print("Converting daily data to hourly to match on-chain data...")
                    # Convert daily data to hourly by forward-filling
                    price_df = self._convert_daily_to_hourly(price_df)
                else:
                    print(f"Price data interval: {time_diff:.1f} hours")
                
            # Process the data
            result = price_df.copy()
            
            # Add technical indicators if requested
            if add_indicators:
                print("Adding technical indicators...")
                result = self.add_technical_indicators(result)
            
            # Add custom features if requested
            if add_custom:
                print("Adding custom features...")
                result = self.add_custom_features(result)
            
            # Merge on-chain data if available
            if onchain_dfs and len(onchain_dfs) > 0:
                print(f"Merging {len(onchain_dfs)} on-chain datasets...")
                result = self.add_on_chain_features(result, onchain_dfs)
            
            # Handle null or infinite values before normalization
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
            result[numeric_cols] = result[numeric_cols].ffill().bfill().fillna(0)
            
            # Normalize features if requested
            if self.normalize:
                print("Normalizing features...")
                result = self.normalize_features(result)
            
            return result
            
        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_daily_to_hourly(self, daily_df):
        """
        Convert daily price data to hourly by forward filling.
        
        Args:
            daily_df (pd.DataFrame): DataFrame with daily price data
            
        Returns:
            pd.DataFrame: DataFrame with hourly price data
        """
        # Sort by date
        daily_df = daily_df.sort_values('date')
        
        # Get the date range
        start_date = daily_df['date'].min()
        end_date = daily_df['date'].max() + pd.Timedelta(days=1)
        
        # Create an hourly date range (using 'h' instead of deprecated 'H')
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Create a new DataFrame with the hourly range
        hourly_df = pd.DataFrame({'date': hourly_range})
        
        # Merge with the daily data
        merged_df = pd.merge_asof(hourly_df, daily_df, on='date', direction='backward')
        
        # Forward fill to handle gaps
        merged_df = merged_df.ffill()
        
        print(f"Converted {len(daily_df)} daily records to {len(merged_df)} hourly records")
        
        return merged_df
    
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
    
    def feature_selection(self, df, top_n=30, target_col='price_change_1d'):
        """
        Select most important features using correlation with target.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            top_n (int): Number of top features to keep
            target_col (str): Target column for correlation calculation
            
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        # Safety check
        if df is None or df.empty:
            print("Warning: Empty DataFrame provided to feature_selection. Returning original.")
            return df
        
        if target_col not in df.columns:
            print(f"Warning: Target column '{target_col}' not found. Using price changes if available.")
            # Try to find a suitable target column
            potential_targets = ['price_change_1d', 'returns', 'price_change']
            for col in potential_targets:
                if col in df.columns:
                    target_col = col
                    print(f"Using '{target_col}' as target column")
                    break
            else:
                print("No suitable target column found. Skipping feature selection.")
                return df

        # Calculate correlations with target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Always keep date and price columns
        always_keep = ['date', 'timestamp', 'time', 'close', 'price', 
                       'price_usd_close', 'open', 'high', 'low', 'volume']
        always_keep = [col for col in always_keep if col in df.columns]
        
        # Get top correlated features
        top_features = correlations.iloc[:top_n+1].index.tolist()  # +1 to include target
        
        # Combine always_keep with top_features
        selected_features = list(set(always_keep + top_features))
        
        # Print feature selection summary
        print(f"\nFeature Selection Summary:")
        print(f"Original features: {df.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Reduction: {df.shape[1] - len(selected_features)} features removed")
        
        # Select features
        result = df[selected_features]
        
        # Print top 10 features by correlation
        print("\nTop 10 features by correlation with target:")
        for i, (feature, corr) in enumerate(correlations.iloc[:11].items()):
            if feature != target_col:  # Skip the target itself
                print(f"{i}. {feature}: {corr:.4f}")
        
        return result
    
    def load_and_process_data(self, crypto, start_date=None, end_date=None):
        """
        Load and process data for a specific cryptocurrency.
        
        Args:
            crypto (str): Cryptocurrency symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Processed DataFrame with date index
        """
        # Load raw data files
        onchain_dfs = {}
        price_df = None
        
        # Find all data files for this crypto
        all_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
        crypto_files = [f for f in all_files if f.startswith(crypto)]
        
        if not crypto_files:
            print(f"No data files found for {crypto}")
            return pd.DataFrame()
        
        for file in crypto_files:
            file_path = os.path.join(DATA_DIR, file)
            
            # Determine file type
            if 'ohlcv' in file.lower() or 'price' in file.lower():
                price_df = pd.read_csv(file_path)
                print(f"Loaded price data from {file}: {len(price_df)} rows")
            else:
                # Assume any non-price file is on-chain data
                df = pd.read_csv(file_path)
                source = file.split('_')[-1].split('.')[0]  # Extract data source from filename
                onchain_dfs[source] = df
                print(f"Loaded {source} data from {file}: {len(df)} rows")
        
        if price_df is None:
            print("No price data found. Cannot proceed.")
            return pd.DataFrame()
        
        # Process the data
        processed_data = self.process_data(
            price_df, 
            onchain_dfs=onchain_dfs,
            add_indicators=True,
            add_custom=True
        )
        
        # Apply feature selection to reduce dimensionality
        processed_data = self.feature_selection(processed_data, top_n=30)
        
        # Filter by date if provided
        if start_date:
            processed_data = processed_data[processed_data['date'] >= start_date]
        if end_date:
            processed_data = processed_data[processed_data['date'] <= end_date]
        
        # Sort by date
        processed_data = processed_data.sort_values('date')
        
        # Apply outlier detection and handling
        processed_data = self.handle_outliers(processed_data, method='winsorize', threshold=3.0)
        
        return processed_data


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