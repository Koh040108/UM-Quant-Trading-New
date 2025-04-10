"""
This script runs the model with existing data without attempting to fetch from API
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append('src')

from src.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, CRYPTOCURRENCIES, 
    DATA_INTERVALS, DEFAULT_INTERVAL, HMM_STATES,
    TRADING_FEE, MIN_TRADE_FREQUENCY, MIN_SHARPE_RATIO, MAX_DRAWDOWN_LIMIT
)
from src.feature_engineering import FeatureEngineer
from src.hmm_model import MarketHMM

def ensure_price_data_exists(crypto='BTC', interval='4h', start_date='2024-01-01', end_date='2024-03-14'):
    """Ensure price data exists, create if not."""
    ccxt_filename = f"{crypto}_ccxt_market_data_{interval}.csv"
    ccxt_path = os.path.join(DATA_DIR, ccxt_filename)
    
    if not os.path.exists(ccxt_path):
        print(f"No price data found for {crypto}. Creating synthetic price data for basic functionality.")
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range based on interval
        if interval == '1h':
            date_range = pd.date_range(start=start, end=end, freq='1H')
        elif interval == '4h':
            date_range = pd.date_range(start=start, end=end, freq='4H')
        else:  # Default to daily
            date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Generate simple random walk prices
        np.random.seed(42)
        n_steps = len(date_range)
        price_changes = np.random.normal(0.0001, 0.01, n_steps)
        prices = 100 * (1 + np.cumsum(price_changes))
        volumes = prices * np.random.lognormal(0, 0.5, n_steps)
        
        # Create and save dataframe
        df = pd.DataFrame({
            'date': date_range,
            'price': prices,
            'volume': volumes
        })
        
        # Create directory if needed
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        df.to_csv(ccxt_path, index=False)
        print(f"Created basic price data with {len(df)} records: {ccxt_path}")
        
        # Also save in glassnode format
        glassnode_df = df[['date', 'price']].rename(columns={'price': 'value'})
        glassnode_path = os.path.join(DATA_DIR, f"{crypto}_glassnode_market_price_usd_close.csv")
        glassnode_df.to_csv(glassnode_path, index=False)
        print(f"Created glassnode price data: {glassnode_path}")
        
    else:
        print(f"Using existing price data: {ccxt_path}")

def run_hmm_model(crypto='BTC', interval='4h', states=7, threshold=0.015, 
                  start_date='2024-01-01', end_date='2024-03-14'):
    """Run the HMM model with existing data."""
    # Ensure data directories exist
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Ensure price data exists
    ensure_price_data_exists(crypto, interval, start_date, end_date)
    
    # Load and process data
    print(f"Processing data for {crypto} from {start_date} to {end_date}")
    feature_engineer = FeatureEngineer(normalize=True)
    try:
        processed_data = feature_engineer.load_and_process_data(
            crypto=crypto,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Date range: {processed_data['date'].min()} to {processed_data['date'].max()}")
        
        # Train model
        print(f"Training HMM model with {states} states")
        hmm_model = MarketHMM(n_states=states)
        hmm_model.fit(processed_data)
        
        # Run backtest
        print("Running backtest...")
        
        # Add states to the data
        with_states = hmm_model.add_states_to_df(processed_data)
        
        # Generate trading signals
        signals = hmm_model.generate_trading_signals(with_states, threshold=threshold)
        
        # Backtest the strategy
        results, performance = hmm_model.backtest_strategy(signals, fee=TRADING_FEE)
        
        # Save performance metrics
        perf_df = pd.DataFrame(list(performance.items()), columns=['Metric', 'Value'])
        perf_file = os.path.join(RESULTS_DIR, f"{crypto}_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        perf_df.to_csv(perf_file, index=False)
        print(f"Performance metrics saved to {perf_file}")
        
        # Print performance
        print("\nPerformance Metrics:")
        for metric, value in performance.items():
            print(f"{metric}: {value:.4f}")
        
        # Evaluate if criteria are met
        sharpe_ratio = performance['Sharpe Ratio']
        max_drawdown = performance['Max Drawdown']
        trading_frequency = performance['Trading Frequency']
        
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
            
        return results, performance
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("Please check if data files exist in the data directory.")
        return None, None

if __name__ == "__main__":
    # Run the model
    run_hmm_model(
        crypto='BTC', 
        interval='4h', 
        states=7,
        threshold=0.015,
        start_date='2024-01-01',
        end_date='2024-03-14'
    ) 