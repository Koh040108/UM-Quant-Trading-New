"""
Improved HMM model with enhanced features, optimal parameter selection, and risk management.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

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
    Load and prepare data with enhanced features for improved performance.
    
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
    
    # Add on-chain data if available
    onchain_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{crypto}_") and "cryptoquant" in f]
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
                
                # Extract the most useful columns
                if 'value' in onchain_df.columns:
                    onchain_df = onchain_df[['date', 'value']].rename(columns={'value': feature_name})
                    # Merge with main dataframe
                    df = pd.merge_asof(df, onchain_df, on='date', direction='nearest')
                    print(f"Added on-chain feature: {feature_name}")
            except Exception as e:
                print(f"Error incorporating {file}: {str(e)}")
    
    # Create enhanced features
    
    # 1. Basic price and volume features
    df['price_change_1d'] = df['close'].pct_change(1).fillna(0)
    df['price_change_3d'] = df['close'].pct_change(3).fillna(0)
    df['price_change_7d'] = df['close'].pct_change(7).fillna(0)
    
    # 2. Volatility features at multiple timeframes
    df['volatility_7d'] = df['close'].pct_change().rolling(window=7).std().fillna(0)
    df['volatility_14d'] = df['close'].pct_change().rolling(window=14).std().fillna(0)
    df['volatility_30d'] = df['close'].pct_change().rolling(window=30).std().fillna(0)
    
    # 3. Trend features
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_ratio_7_20'] = (df['sma_7'] / df['sma_20']).fillna(1.0)
    df['sma_ratio_20_50'] = (df['sma_20'] / df['sma_50']).fillna(1.0)
    
    # 4. Volume-based features
    if 'volume' in df.columns:
        df['volume_change_1d'] = df['volume'].pct_change(1).fillna(0)
        df['volume_change_7d'] = df['volume'].pct_change(7).fillna(0)
        df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_ratio_1d_7d'] = (df['volume'] / df['volume_sma_7']).fillna(1.0)
    
    # 5. Momentum indicators
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 6. Advanced volatility features
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 7. Relative performance features
    df['close_ratio_to_max_30d'] = df['close'] / df['close'].rolling(window=30).max()
    df['close_ratio_to_min_30d'] = df['close'] / df['close'].rolling(window=30).min()
    
    # 8. Regime shift indicators
    df['trend_change'] = (np.sign(df['price_change_7d']) != np.sign(df['price_change_7d'].shift(1))).astype(int)
    df['volatility_change'] = (df['volatility_7d'] > df['volatility_7d'].shift(7) * 1.2).astype(int)
    
    # Fill any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
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

def generate_risk_managed_signals(df, hmm_model, threshold=0.0, risk_factor=0.5):
    """
    Generate trading signals with improved risk management.
    
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
    
    # Generate basic signals
    signals = hmm_model.generate_trading_signals(with_states, threshold=threshold)
    
    # Add risk management - scale positions based on volatility
    # Lower position size during high volatility periods
    signals['volatility_factor'] = 1.0 - (signals['volatility_7d'] / signals['volatility_7d'].max() * risk_factor)
    signals['volatility_factor'] = signals['volatility_factor'].clip(0.1, 1.0)
    
    # Apply risk scaling to positions
    signals['risk_position'] = signals['position'] * signals['volatility_factor']
    
    # Add stop-loss logic
    # If drawdown from peak exceeds threshold, reduce position
    signals['price_peak'] = signals['close'].cummax()
    signals['drawdown'] = (signals['close'] - signals['price_peak']) / signals['price_peak']
    
    # Reduce position when in drawdown
    stop_threshold = -0.05  # 5% drawdown
    signals.loc[signals['drawdown'] < stop_threshold, 'risk_position'] *= 0.5
    
    # Completely exit when drawdown is severe
    severe_threshold = -0.1  # 10% drawdown
    signals.loc[signals['drawdown'] < severe_threshold, 'risk_position'] = 0
    
    # Use the risk-adjusted position for trading
    signals['position'] = signals['risk_position']
    
    return signals

def run_improved_model(crypto='BTC', interval='4h'):
    """
    Run improved HMM model with parameter optimization and enhanced risk management.
    """
    # Ensure directories exist
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load and prepare data
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
    
    # Define parameter grid for optimization
    parameter_grid = ParameterGrid({
        'n_states': [3, 5, 7, 9],
        'n_iter': [1000],
        'threshold': [-0.002, 0.0, 0.002, 0.005, 0.01]
    })
    
    # Optimize parameters
    best_model, best_params, valid_performance = optimize_hmm_parameters(
        train_data, valid_data, parameter_grid
    )
    
    # Now use the best model on the test set
    feature_engineer = FeatureEngineer(normalize=True)
    processed_test = feature_engineer.process_data(test_data)
    
    print(f"Best parameters: {best_params}")
    print("Testing on holdout test set with risk management...")
    
    # Generate signals with risk management
    signals = generate_risk_managed_signals(
        processed_test, 
        best_model, 
        threshold=best_params['threshold'],
        risk_factor=0.5  # Adjust this for different risk profiles
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
    perf_file = os.path.join(RESULTS_DIR, f"{crypto}_improved_performance_{timestamp}.csv")
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
    model_filename = f"{crypto}_optimized_hmm_model_{best_params['n_states']}_states_{timestamp}.pkl"
    best_model.save_model(filename=model_filename)
    print(f"Optimized model saved as {model_filename}")
    
    return results, performance

if __name__ == "__main__":
    print("Running improved HMM model with parameter optimization and risk management")
    run_improved_model(crypto='BTC', interval='4h') 