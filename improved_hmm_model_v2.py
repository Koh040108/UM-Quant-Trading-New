"""
Improved HMM model v2 with enhanced features focusing on:
1. Moving Averages (OHLC)
2. Price Volatility
3. Whale Activity Spikes
4. Exchange Net Flow Rate Changes
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from scipy import stats

# Add the src directory to the path
sys.path.append('.')

from src.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, 
    TRADING_FEE, MIN_TRADE_FREQUENCY, MIN_SHARPE_RATIO, MAX_DRAWDOWN_LIMIT
)
from src.feature_engineering import FeatureEngineer
from src.hmm_model import MarketHMM
from src.visualization import create_performance_dashboard

def load_and_prepare_data(crypto='BTC', interval='4h'):
    """
    Load and prepare data with enhanced features focusing on:
    1. Moving Averages (OHLC)
    2. Price Volatility
    3. Whale Activity Spikes
    4. Exchange Net Flow Rate Changes
    
    Args:
        crypto (str): Cryptocurrency symbol
        interval (str): Data interval
        
    Returns:
        pd.DataFrame: Prepared DataFrame with enhanced features
    """
    # Find appropriate data file
    market_data_file = os.path.join(DATA_DIR, f"{crypto}_market_data_{interval}.csv")
    ccxt_file = os.path.join(DATA_DIR, f"{crypto}_ccxt_market_data_{interval}.csv")
    
    if os.path.exists(market_data_file):
        data_file = market_data_file
    elif os.path.exists(ccxt_file):
        data_file = ccxt_file
    else:
        print("No suitable data file found")
        return None
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Ensure date is in datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Make sure we have the required columns
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']
    
    # If we have open/high/low data, use them; otherwise, estimate from close
    if 'open' not in df.columns:
        print("No OHLC data available, estimating from close price")
        df['open'] = df['close'].shift(1).fillna(df['close'])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(df)))
    
    # Add on-chain data with special focus on whale activity and exchange flows
    onchain_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{crypto}_") and (
        "cryptoquant" in f or "glassnode" in f or "coinglass" in f)]
    
    # Create dictionaries to store specific metrics
    whale_metrics = {}
    exchange_flow_metrics = {}
    other_metrics = {}
    
    if onchain_files:
        print(f"Found {len(onchain_files)} on-chain data files, incorporating into features")
        for file in onchain_files:
            try:
                file_path = os.path.join(DATA_DIR, file)
                onchain_df = pd.read_csv(file_path)
                
                # Extract feature name
                feature_name = file.replace(f"{crypto}_cryptoquant_", "").replace(".csv", "").replace("-", "_")
                
                # Ensure date column exists and is in datetime format
                if 'date' in onchain_df.columns:
                    onchain_df['date'] = pd.to_datetime(onchain_df['date'])
                elif 'timestamp' in onchain_df.columns:
                    onchain_df['date'] = pd.to_datetime(onchain_df['timestamp'], unit='ms')
                
                # Skip if no usable date column
                if 'date' not in onchain_df.columns:
                    continue
                
                # Extract value column (could be 'value', 'v', or a specialized column name)
                value_col = None
                for col in ['value', 'v', 'data', 'count', 'mean', 'sum']:
                    if col in onchain_df.columns:
                        value_col = col
                        break
                
                if value_col is None and len(onchain_df.columns) >= 2:
                    # If no standard value column, use the second column
                    value_col = onchain_df.columns[1]
                
                if value_col is None:
                    continue
                
                # Create a clean dataframe with date and value only
                clean_df = onchain_df[['date', value_col]].rename(columns={value_col: feature_name})
                
                # Categorize metrics
                if "whale" in feature_name or "large" in feature_name or "transfer" in feature_name:
                    whale_metrics[feature_name] = clean_df
                    print(f"Added whale metric: {feature_name}")
                elif "exchange" in feature_name or "inflow" in feature_name or "outflow" in feature_name:
                    exchange_flow_metrics[feature_name] = clean_df
                    print(f"Added exchange flow metric: {feature_name}")
                else:
                    other_metrics[feature_name] = clean_df
                    print(f"Added other metric: {feature_name}")
                
            except Exception as e:
                print(f"Error incorporating {file}: {str(e)}")
    
    # Process all on-chain metrics and merge into main dataframe
    for category, metrics_dict in [
        ("Whale", whale_metrics), 
        ("Exchange Flow", exchange_flow_metrics), 
        ("Other", other_metrics)
    ]:
        if metrics_dict:
            print(f"Processing {category} metrics...")
            for feature_name, metric_df in metrics_dict.items():
                # Ensure date is a column, not an index
                if 'date' in metric_df.index.names:
                    metric_df = metric_df.reset_index()
                
                # Ensure df also has date as a column
                if 'date' in df.index.names:
                    df = df.reset_index()
                
                # Print the actual metric_df columns for debugging
                print(f"  - {feature_name} columns: {metric_df.columns.tolist()}")
                
                # If there's no 'date' column, check if we need to create one
                if 'date' not in metric_df.columns:
                    # Try to find a timestamp column
                    timestamp_cols = [col for col in metric_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if timestamp_cols:
                        # Use the first timestamp column found
                        metric_df['date'] = pd.to_datetime(metric_df[timestamp_cols[0]])
                        print(f"  - Created date column from {timestamp_cols[0]}")
                    else:
                        # If we can't find a timestamp column, skip this metric
                        print(f"  - Skipping {feature_name}: No date column found")
                        continue
                
                # Ensure date columns are datetime
                if 'date' in metric_df.columns:
                    metric_df['date'] = pd.to_datetime(metric_df['date'])
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Sort both dataframes by date
                    df = df.sort_values('date')
                    metric_df = metric_df.sort_values('date')
                    
                    # Rename feature column if it has the same name as the dataframe (likely index)
                    value_cols = [col for col in metric_df.columns if col != 'date']
                    if not value_cols:
                        print(f"  - Skipping {feature_name}: No value columns found")
                        continue
                    
                    # Use only date and the value column for merging
                    metric_df = metric_df[['date', value_cols[0]]].rename(columns={value_cols[0]: feature_name})
                    
                    # Print debug info
                    print(f"  - {feature_name} shape: {metric_df.shape}")
                    print(f"  - {feature_name} columns after processing: {metric_df.columns.tolist()}")
                    print(f"  - {feature_name} date sample: {metric_df['date'].head(3)}")
                    
                    try:
                        # Merge with main dataframe using asof merge (nearest match)
                        df = pd.merge_asof(df, metric_df, on='date', direction='nearest')
                        print(f"  - Successfully merged {feature_name}")
                    except Exception as e:
                        print(f"  - Error merging {feature_name}: {str(e)}")
                else:
                    print(f"  - Skipping {feature_name}: Date column processing failed")
    
    # 1. Enhanced Moving Averages (OHLC)
    print("Adding enhanced moving average features...")
    # Typical Price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Moving Averages on different price points
    ma_periods = [5, 10, 20, 50]
    for period in ma_periods:
        df[f'close_ma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'typical_ma_{period}'] = df['typical_price'].rolling(window=period).mean()
        
        # OHLC Moving Average Relationships
        df[f'high_low_ratio_ma_{period}'] = df['high'].rolling(window=period).mean() / df['low'].rolling(window=period).mean()
        df[f'open_close_ratio_ma_{period}'] = df['open'].rolling(window=period).mean() / df['close'].rolling(window=period).mean()
    
    # Cross-over signals
    for fast_period, slow_period in [(5, 20), (10, 50), (20, 50)]:
        df[f'ma_cross_{fast_period}_{slow_period}'] = (
            df[f'close_ma_{fast_period}'] > df[f'close_ma_{slow_period}']
        ).astype(int)
        df[f'ma_cross_change_{fast_period}_{slow_period}'] = df[f'ma_cross_{fast_period}_{slow_period}'].diff().fillna(0)
    
    # 2. Enhanced Price Volatility Features
    print("Adding enhanced volatility features...")
    volatility_periods = [7, 14, 30]
    for period in volatility_periods:
        # Standard volatility (rolling std of returns)
        df[f'volatility_{period}d'] = df['close'].pct_change().rolling(window=period).std().fillna(0)
        
        # Range-based volatility
        df[f'range_volatility_{period}d'] = (
            (df['high'] - df['low']) / df['close']
        ).rolling(window=period).mean().fillna(0)
        
        # Relative volatility (current vs historical)
        if period > 7:
            df[f'volatility_ratio_{period}d_7d'] = (
                df[f'volatility_{period}d'] / df['volatility_7d']
            ).fillna(1)
        
        # Volatility trends
        df[f'volatility_trend_{period}d'] = (
            df[f'volatility_{period}d'] > df[f'volatility_{period}d'].shift(period//2)
        ).astype(int)
    
    # 3. Enhanced Whale Activity Features
    print("Adding enhanced whale activity features...")
    # Check if any whale metrics are available
    if any("whale" in col for col in df.columns):
        whale_cols = [col for col in df.columns if "whale" in col or "transfer" in col]
        
        for col in whale_cols:
            # Z-score to detect unusual whale activity
            df[f'{col}_zscore'] = (
                df[col] - df[col].rolling(window=30).mean()
            ) / df[col].rolling(window=30).std().fillna(0.0001)
            
            # Detect spikes in whale activity
            df[f'{col}_spike'] = (df[f'{col}_zscore'] > 2).astype(int)
            
            # Cumulative whale activity
            df[f'{col}_cumsum_7d'] = df[col].rolling(window=7).sum()
    else:
        print("No whale metrics available in the dataset")
    
    # 4. Enhanced Exchange Flow Features
    print("Adding enhanced exchange flow features...")
    # Check if inflow/outflow metrics are available
    inflow_cols = [col for col in df.columns if "inflow" in col]
    outflow_cols = [col for col in df.columns if "outflow" in col]
    
    if inflow_cols and outflow_cols:
        # Assuming we have at least one inflow and one outflow metric
        main_inflow = inflow_cols[0]
        main_outflow = outflow_cols[0]
        
        # Net flow metrics
        df['net_flow'] = df[main_inflow] - df[main_outflow]
        df['net_flow_ratio'] = df[main_inflow] / df[main_outflow].replace(0, 0.0001)
        
        # Flow momentum
        df['net_flow_ma_7d'] = df['net_flow'].rolling(window=7).mean()
        df['net_flow_ma_14d'] = df['net_flow'].rolling(window=14).mean()
        
        # Flow trend
        df['net_flow_trend'] = (df['net_flow_ma_7d'] > df['net_flow_ma_14d']).astype(int)
        
        # Z-score to detect unusual flow patterns
        df['net_flow_zscore'] = (
            df['net_flow'] - df['net_flow'].rolling(window=30).mean()
        ) / df['net_flow'].rolling(window=30).std().fillna(0.0001)
        
        # Detect significant flow changes
        df['net_flow_significant'] = (abs(df['net_flow_zscore']) > 2).astype(int)
    else:
        print("No exchange flow metrics available in the dataset")
    
    # 5. Additional Technical Indicators
    print("Adding technical indicators...")
    # RSI
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 0.0001)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # RSI conditions
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_cross_change'] = df['macd_cross'].diff().fillna(0)
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # BB conditions
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(window=50).mean() * 0.8).astype(int)
    df['bb_upper_touch'] = (df['high'] > df['bb_upper']).astype(int)
    df['bb_lower_touch'] = (df['low'] < df['bb_lower']).astype(int)
    
    # 6. Combined Regime Features
    print("Adding combined regime features...")
    # Volatility regime
    df['low_vol_regime'] = (
        (df['volatility_7d'] < df['volatility_30d'] * 0.7) & 
        (df['bb_width'] < df['bb_width'].rolling(window=30).mean() * 0.8)
    ).astype(int)
    
    df['high_vol_regime'] = (
        (df['volatility_7d'] > df['volatility_30d'] * 1.3) & 
        (df['bb_width'] > df['bb_width'].rolling(window=30).mean() * 1.2)
    ).astype(int)
    
    # Trend regime based on multiple indicators
    df['strong_uptrend'] = (
        (df['ma_cross_5_20'] == 1) & 
        (df['ma_cross_10_50'] == 1) & 
        (df['macd_cross'] == 1) &
        (df['rsi_14'] > 50)
    ).astype(int)
    
    df['strong_downtrend'] = (
        (df['ma_cross_5_20'] == 0) & 
        (df['ma_cross_10_50'] == 0) & 
        (df['macd_cross'] == 0) &
        (df['rsi_14'] < 50)
    ).astype(int)
    
    # Combined whale/exchange flow signals
    # Only create if we have the metrics
    if any("whale" in col for col in df.columns) and any("flow" in col for col in df.columns):
        whale_spike_cols = [col for col in df.columns if "whale" in col and "spike" in col]
        if whale_spike_cols and 'net_flow_significant' in df.columns:
            df['whale_flow_signal'] = (
                df[whale_spike_cols[0]] & 
                df['net_flow_significant']
            ).astype(int)
    
    # Fill any NaN values
    df = df.ffill().bfill().fillna(0)
    
    print(f"Data prepared with {len(df)} rows and {len(df.columns)} features")
    return df

def optimize_hmm_parameters(train_data, valid_data, parameter_grid):
    """
    Optimize HMM parameters using validation set.
    
    Args:
        train_data (pd.DataFrame): Training data
        valid_data (pd.DataFrame): Validation data
        parameter_grid (dict): Dictionary of parameter ranges to try
        
    Returns:
        tuple: (best_model, best_params, best_performance)
    """
    print(f"Optimizing HMM parameters with {len(parameter_grid)} combinations")
    
    best_sharpe = -float('inf')
    best_model = None
    best_params = None
    best_performance = None
    best_drawdown = -float('inf')
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(normalize=True)
    
    # Process training data
    processed_train = feature_engineer.process_data(train_data)
    
    # Process validation data
    processed_valid = feature_engineer.process_data(valid_data)
    
    # Try each parameter combination
    for i, params in enumerate(parameter_grid):
        print(f"Testing parameters {i+1}/{len(parameter_grid)}: {params}")
        
        try:
            # Create and train model
            model = MarketHMM(
                n_states=params['n_states'],
                n_iter=params['n_iter'],
                random_state=42
            )
            model.fit(processed_train)
            
            # Generate signals on validation set
            with_states = model.add_states_to_df(processed_valid)
            signals = model.generate_trading_signals(with_states, threshold=params['threshold'])
            
            # Skip if we don't have enough signals
            if len(signals) < 10:
                print(f"  Not enough signals, skipping")
                continue
                
            # Backtest
            results, performance = model.backtest_strategy(signals, fee=TRADING_FEE)
            
            # Fix NaN values
            for key in performance:
                if pd.isna(performance[key]):
                    performance[key] = 0.0
            
            # Get key metrics
            sharpe_ratio = performance.get('Sharpe Ratio', 0.0)
            max_drawdown = performance.get('Max Drawdown', -100.0)
            trading_freq = performance.get('Trading Frequency', 0.0)
            
            print(f"  Results: Sharpe={sharpe_ratio:.2f}, Drawdown={max_drawdown:.2f}%, Trading Freq={trading_freq:.4f}")
            
            # Check if this combination is better
            # First priority: meet drawdown requirement
            if max_drawdown >= MAX_DRAWDOWN_LIMIT:
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_model = model
                    best_params = params
                    best_performance = performance
                    best_drawdown = max_drawdown
                    print(f"  New best parameters found!")
            # If we haven't found any parameters that meet drawdown requirement, 
            # keep track of the best drawdown seen
            elif best_drawdown < MAX_DRAWDOWN_LIMIT and max_drawdown > best_drawdown:
                best_sharpe = sharpe_ratio
                best_model = model
                best_params = params
                best_performance = performance
                best_drawdown = max_drawdown
                print(f"  New best drawdown parameters found!")
                
        except Exception as e:
            print(f"  Error with parameters {params}: {str(e)}")
    
    if best_model is None:
        print("No suitable parameters found. Using default parameters.")
        best_model = MarketHMM(n_states=5, n_iter=1000, random_state=42)
        best_model.fit(processed_train)
        best_params = {"n_states": 5, "n_iter": 1000, "threshold": 0.0}
        
    return best_model, best_params, best_performance

def generate_advanced_signals(df, hmm_model, threshold=0.0, risk_factor=0.5):
    """
    Generate trading signals with advanced signal rules and risk management.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        hmm_model (MarketHMM): Trained HMM model
        threshold (float): Return threshold for signal generation
        risk_factor (float): Factor to scale positions based on volatility (0.0-1.0)
        
    Returns:
        pd.DataFrame: DataFrame with trading signals and risk-adjusted positions
    """
    # Add states to dataframe
    with_states = hmm_model.add_states_to_df(df)
    
    # Generate basic signals from HMM model
    signals = hmm_model.generate_trading_signals(with_states, threshold=threshold)
    
    # Add signal quality scores based on technical and on-chain indicators
    # Higher score = stronger signal
    signals['technical_score'] = 0
    
    # Add points for aligned technical indicators
    if 'macd_cross' in signals.columns:
        signals.loc[signals['position'] > 0, 'technical_score'] += signals['macd_cross'] * 1  # +1 if MACD is positive
        signals.loc[signals['position'] < 0, 'technical_score'] += (1 - signals['macd_cross']) * 1  # +1 if MACD is negative
    
    if 'rsi_14' in signals.columns:
        # Add RSI confirmation for long/short
        signals.loc[(signals['position'] > 0) & (signals['rsi_14'] > 50), 'technical_score'] += 1
        signals.loc[(signals['position'] < 0) & (signals['rsi_14'] < 50), 'technical_score'] += 1
        
        # But avoid extremes (overbought/oversold)
        signals.loc[(signals['position'] > 0) & (signals['rsi_14'] > 75), 'technical_score'] -= 1
        signals.loc[(signals['position'] < 0) & (signals['rsi_14'] < 25), 'technical_score'] -= 1
    
    # Add MA crossover confirmation
    if 'ma_cross_5_20' in signals.columns:
        signals.loc[(signals['position'] > 0) & (signals['ma_cross_5_20'] == 1), 'technical_score'] += 1
        signals.loc[(signals['position'] < 0) & (signals['ma_cross_5_20'] == 0), 'technical_score'] += 1
    
    # Add on-chain indicators if available
    signals['onchain_score'] = 0
    
    # Whale activity
    whale_cols = [col for col in signals.columns if "whale" in col and "spike" in col]
    if whale_cols:
        for col in whale_cols[:2]:  # Use up to 2 whale metrics
            signals.loc[signals['position'] > 0, 'onchain_score'] += signals[col] * 1
    
    # Exchange flows
    if 'net_flow_trend' in signals.columns:
        signals.loc[(signals['position'] > 0) & (signals['net_flow_trend'] == 1), 'onchain_score'] += 1
        signals.loc[(signals['position'] < 0) & (signals['net_flow_trend'] == 0), 'onchain_score'] += 1
    
    # Combine scores and scale positions
    signals['signal_quality'] = signals['technical_score'] + signals['onchain_score']
    max_quality = max(1, signals['signal_quality'].max())  # Avoid division by zero
    signals['signal_quality_factor'] = (signals['signal_quality'] / max_quality).clip(0.1, 1.0)
    
    # Apply signal quality to position sizing
    signals['quality_position'] = signals['position'] * signals['signal_quality_factor']
    
    # Add risk management based on volatility
    # Lower position size during high volatility periods
    if 'volatility_7d' in signals.columns:
        vol_max = signals['volatility_7d'].max() if signals['volatility_7d'].max() > 0 else 0.01
        signals['volatility_factor'] = 1.0 - (signals['volatility_7d'] / vol_max * risk_factor)
        signals['volatility_factor'] = signals['volatility_factor'].clip(0.1, 1.0)
    else:
        signals['volatility_factor'] = 1.0
    
    # Apply volatility scaling to positions
    signals['risk_position'] = signals['quality_position'] * signals['volatility_factor']
    
    # Add stop-loss logic
    # If drawdown from peak exceeds threshold, reduce position
    signals['price_peak'] = signals['close'].expanding().max()
    signals['drawdown'] = (signals['close'] - signals['price_peak']) / signals['price_peak']
    
    # Adaptive stop loss based on volatility
    if 'volatility_30d' in signals.columns:
        # Use 30-day volatility to set adaptive stop-loss levels
        # Higher volatility = wider stops
        vol_factor = signals['volatility_30d'] / signals['volatility_30d'].mean()
        signals['stop_threshold'] = -0.05 * vol_factor.clip(0.5, 2.0)
        signals['severe_threshold'] = -0.1 * vol_factor.clip(0.5, 2.0)
    else:
        signals['stop_threshold'] = -0.05  # 5% drawdown
        signals['severe_threshold'] = -0.1  # 10% drawdown
    
    # Apply stop-loss rules
    signals['risk_position_with_stops'] = signals['risk_position'].copy()
    
    # Reduce position when in drawdown
    for i in range(1, len(signals)):
        if signals['drawdown'].iloc[i] < signals['stop_threshold'].iloc[i]:
            signals.loc[signals.index[i], 'risk_position_with_stops'] *= 0.5
        
        # Completely exit when drawdown is severe
        if signals['drawdown'].iloc[i] < signals['severe_threshold'].iloc[i]:
            signals.loc[signals.index[i], 'risk_position_with_stops'] = 0
    
    # Use the risk-adjusted position with stops for final trading
    signals['position'] = signals['risk_position_with_stops']
    
    # Ensure smooth transitions between positions (limit position changes)
    for i in range(1, len(signals)):
        curr_pos = signals['position'].iloc[i]
        prev_pos = signals['position'].iloc[i-1]
        
        # If trying to increase position size too quickly, ramp up gradually
        if abs(curr_pos - prev_pos) > 0.5:
            signals.loc[signals.index[i], 'position'] = prev_pos + 0.5 * np.sign(curr_pos - prev_pos)
    
    return signals

def run_improved_model_v2(crypto='BTC', interval='4h'):
    """
    Run improved HMM model v2 with better utilization of:
    1. Moving Averages (OHLC)
    2. Price Volatility
    3. Whale Activity Spikes
    4. Exchange Net Flow Rate Changes
    """
    # Ensure directories exist
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load and prepare data with enhanced features
    df = load_and_prepare_data(crypto, interval)
    
    if df is None or len(df) < 100:
        print("Insufficient data for modeling")
        return None, None
    
    # Split data: 60% train, 20% validation, 20% test
    train_size = int(len(df) * 0.6)
    valid_size = int(len(df) * 0.2)
    
    train_data = df.iloc[:train_size].copy()
    valid_data = df.iloc[train_size:train_size+valid_size].copy()
    test_data = df.iloc[train_size+valid_size:].copy()
    
    print(f"Data split:")
    print(f"  Train: {len(train_data)} rows ({train_data['date'].min()} to {train_data['date'].max()})")
    print(f"  Validation: {len(valid_data)} rows ({valid_data['date'].min()} to {valid_data['date'].max()})")
    print(f"  Test: {len(test_data)} rows ({test_data['date'].min()} to {test_data['date'].max()})")
    
    # Enhanced parameter grid with more state options and thresholds
    parameter_grid = ParameterGrid({
        'n_states': [3, 5, 7, 9, 11],  # Added more state options
        'n_iter': [1000],
        'threshold': [-0.005, -0.002, 0.0, 0.002, 0.005, 0.01]  # More threshold options
    })
    
    # Optimize parameters
    best_model, best_params, valid_performance = optimize_hmm_parameters(
        train_data, valid_data, parameter_grid
    )
    
    # Now use the best model on the test set
    feature_engineer = FeatureEngineer(normalize=True)
    processed_test = feature_engineer.process_data(test_data)
    
    print(f"Best parameters: {best_params}")
    print("Testing on holdout test set with advanced signal generation...")
    
    # Generate signals with advanced methods
    signals = generate_advanced_signals(
        processed_test, 
        best_model, 
        threshold=best_params['threshold'],
        risk_factor=0.6  # Slightly increased risk factor for more aggressive volatility scaling
    )
    
    # Backtest the strategy
    results, performance = best_model.backtest_strategy(signals, fee=TRADING_FEE)
    
    # Fix any NaN values
    for key in performance:
        if pd.isna(performance[key]):
            performance[key] = 0.0
    
    # Save performance metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    perf_df = pd.DataFrame(list(performance.items()), columns=['Metric', 'Value'])
    perf_file = os.path.join(RESULTS_DIR, f"{crypto}_improved_v2_performance_{timestamp}.csv")
    perf_df.to_csv(perf_file, index=False)
    print(f"Performance metrics saved to {perf_file}")
    
    # Generate visualizations
    print("Generating performance dashboard...")
    dashboard_dir = create_performance_dashboard(
        results=results,
        performance=performance,
        crypto=crypto
    )
    print(f"Dashboard created at {dashboard_dir}")
    
    # Print performance
    print("\nPerformance Metrics on Test Data:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate if criteria are met
    sharpe_ratio = performance.get('Sharpe Ratio', 0)
    max_drawdown = performance.get('Max Drawdown', 0)
    trading_frequency = performance.get('Trading Frequency', 0)
    
    print("\nHackathon Criteria Evaluation:")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f} (Target ≥ {MIN_SHARPE_RATIO})")
    print(f"Max Drawdown: {max_drawdown:.4f} (Target ≥ {MAX_DRAWDOWN_LIMIT})")
    print(f"Trading Frequency: {trading_frequency:.4f} (Target ≥ {MIN_TRADE_FREQUENCY})")
    
    meets_criteria = (
        sharpe_ratio >= MIN_SHARPE_RATIO and
        max_drawdown >= MAX_DRAWDOWN_LIMIT and
        trading_frequency >= MIN_TRADE_FREQUENCY
    )
    
    if meets_criteria:
        print("\n✅ Strategy meets all UMHackathon performance criteria!")
    else:
        print("\n❌ Strategy does not meet all UMHackathon performance criteria.")
    
    # Save the best model
    model_filename = f"{crypto}_optimized_hmm_model_v2_{best_params['n_states']}_states_{timestamp}.pkl"
    best_model.save_model(filename=model_filename)
    print(f"Optimized model saved as {model_filename}")
    
    return results, performance

if __name__ == "__main__":
    print("Running improved HMM model v2 with enhanced features focusing on:")
    print("1. Moving Averages (OHLC)")
    print("2. Price Volatility")
    print("3. Whale Activity Spikes")
    print("4. Exchange Net Flow Rate Changes")
    run_improved_model_v2(crypto='BTC', interval='4h') 