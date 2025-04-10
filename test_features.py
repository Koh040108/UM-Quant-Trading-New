"""
Simple test script to check what features are needed for the HMM model.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append('.')

from src.feature_engineering import FeatureEngineer
from src.hmm_model import MarketHMM
from src.config import DATA_DIR

def main():
    # Load sample data
    print("Loading data...")
    market_data_file = os.path.join(DATA_DIR, "BTC_market_data_4h.csv")
    
    if not os.path.exists(market_data_file):
        print(f"Cannot find data file: {market_data_file}")
        return
    
    # Load data
    df = pd.read_csv(market_data_file)
    
    # Ensure date column is in datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Print original columns
    print(f"Original columns: {df.columns.tolist()}")
    
    # Create close column from price
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']
        print("Created 'close' column from 'price'")
    
    # Add required features manually
    print("Adding required features...")
    df['price_change_1d'] = df['close'].pct_change(1).fillna(0)
    df['volatility_7d'] = df['close'].pct_change().rolling(window=7).std().fillna(0)
    
    # Print features
    print(f"Added features. New columns: {df.columns.tolist()}")
    
    # Try to fit HMM model
    try:
        print("\nTrying to fit HMM model...")
        hmm_model = MarketHMM(n_states=7)
        hmm_model.fit(df)
        print("Success! HMM model fitted correctly.")
        
        # Test state prediction
        states = hmm_model.predict_states(df)
        unique_states = np.unique(states)
        print(f"Predicted {len(unique_states)} unique states: {unique_states}")
        
        # Test adding states to dataframe
        df_with_states = hmm_model.add_states_to_df(df)
        print(f"Added states to dataframe. New columns: {df_with_states.columns.tolist()}")
        
        # Generate trading signals
        signals = hmm_model.generate_trading_signals(df_with_states, threshold=0.015)
        print(f"Generated trading signals. Columns: {signals.columns.tolist()}")
        
        # Count number of buy/sell signals
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
    except Exception as e:
        print(f"Error fitting HMM model: {str(e)}")
    
if __name__ == "__main__":
    main() 