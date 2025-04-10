"""
Main script to run the complete crypto trading strategy pipeline.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, CRYPTOCURRENCIES, 
    DATA_INTERVALS, DEFAULT_INTERVAL, HMM_STATES,
    TRADING_FEE, MIN_TRADE_FREQUENCY, MIN_SHARPE_RATIO, MAX_DRAWDOWN_LIMIT
)
from cybotrade_fetcher import fetch_all_data
from feature_engineering import FeatureEngineer
from hmm_model import MarketHMM
from visualization import create_performance_dashboard

# Print API key status
api_key = os.getenv("CYBOTRADE_API_KEY")
print(f"\nAPI Key Status:")
print(f"Loaded: {api_key is not None}")
print(f"Length: {len(api_key) if api_key else 0}")
print(f"First 5 chars: {api_key[:5] if api_key and len(api_key) >= 5 else 'None'}")
print(f"Last 5 chars: {api_key[-5:] if api_key and len(api_key) >= 5 else 'None'}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Trading Strategy with HMM')
    
    parser.add_argument('--crypto', type=str, default='BTC', choices=CRYPTOCURRENCIES,
                        help='Cryptocurrency to analyze')
    parser.add_argument('--interval', type=str, default=DEFAULT_INTERVAL, choices=list(DATA_INTERVALS.keys()),
                        help='Data interval')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--refresh_data', action='store_true',
                        help='Fetch new data even if existing files are present')
    parser.add_argument('--cybotrade_api_key', type=str, default=None,
                        help='API key for Cybotrade (if not set in environment)')
    parser.add_argument('--states', type=int, default=HMM_STATES,
                        help='Number of hidden states for HMM')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Return threshold for profitable states')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained model to load')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip plotting results')
    
    return parser.parse_args()


def setup_directories():
    """Ensure required directories exist."""
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def prepare_data(args):
    """Prepare data for modeling."""
    crypto = args.crypto
    interval = args.interval
    start_date = args.start_date
    end_date = args.end_date
    
    # Set default dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year of data
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Preparing data for {crypto} from {start_date} to {end_date}")
    
    # Check if we should fetch new data
    if args.refresh_data:
        print("Refreshing data using Cybotrade data source...")
        fetch_all_data(
            cryptos=[crypto],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            api_key=args.cybotrade_api_key
        )
    
    # Check if data files exist
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith(crypto) and f.endswith('.csv')]
    
    if not data_files:
        print("No data files found. Please run with --refresh_data flag to fetch data.")
        return None
    
    # Load and process data
    feature_engineer = FeatureEngineer(normalize=True)
    try:
        processed_data = feature_engineer.load_and_process_data(
            crypto=crypto,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Processed data shape: {processed_data.shape}")
        if not processed_data.empty:
            print(f"Date range: {processed_data['date'].min()} to {processed_data['date'].max()}")
            print(f"Features: {', '.join(processed_data.columns)}")
            
            # Display feature statistics
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            print("\nFeature Statistics:")
            stats_df = processed_data[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
            print(stats_df.round(4))
            
            return processed_data
        else:
            print("Error: Processed data is empty. Check your data files.")
            return None
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None


def train_model(data, args):
    """Train the HMM model."""
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        hmm_model = MarketHMM()
        hmm_model.load_model(args.load_model)
    else:
        print(f"Training new HMM model with {args.states} states")
        hmm_model = MarketHMM(n_states=args.states)
        hmm_model.fit(data)
        
        if args.save_model:
            model_path = hmm_model.save_model()
            print(f"Model saved to {model_path}")
    
    return hmm_model


def run_backtest(hmm_model, data, args):
    """Run backtest with the trained model."""
    print("Running backtest...")
    
    # Add states to the data
    with_states = hmm_model.add_states_to_df(data)
    
    # Generate trading signals
    signals = hmm_model.generate_trading_signals(with_states, threshold=args.threshold)
    
    # Backtest the strategy
    results, performance = hmm_model.backtest_strategy(signals, fee=TRADING_FEE)
    
    # Print performance
    print_performance(performance)
    
    # Generate visualizations unless skipped
    if not args.skip_plots:
        print("\nGenerating performance visualizations...")
        dashboard_dir = create_performance_dashboard(
            results=results,
            performance=performance,
            crypto=args.crypto
        )
        print(f"Performance dashboard created in {dashboard_dir}")
    
    return results, performance


def evaluate_performance(performance):
    """Evaluate if the strategy meets performance criteria."""
    sharpe_ratio = performance['Sharpe Ratio']
    max_drawdown = performance['Max Drawdown']
    trading_frequency = performance['Trading Frequency']
    
    print("\nPerformance Evaluation:")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f} (Target ≥ {MIN_SHARPE_RATIO})")
    print(f"Max Drawdown: {max_drawdown:.4f} (Target ≥ {MAX_DRAWDOWN_LIMIT})")
    print(f"Trading Frequency: {trading_frequency:.4f} (Target ≥ {MIN_TRADE_FREQUENCY})")
    
    meets_criteria = (
        sharpe_ratio >= MIN_SHARPE_RATIO and
        max_drawdown >= MAX_DRAWDOWN_LIMIT and
        trading_frequency >= MIN_TRADE_FREQUENCY
    )
    
    if meets_criteria:
        print("\n✅ Strategy meets all performance criteria!")
    else:
        print("\n❌ Strategy does not meet all performance criteria.")
        
        if sharpe_ratio < MIN_SHARPE_RATIO:
            print(f"  - Sharpe Ratio is below target ({sharpe_ratio:.4f} < {MIN_SHARPE_RATIO})")
        if max_drawdown < MAX_DRAWDOWN_LIMIT:
            print(f"  - Max Drawdown is worse than target ({max_drawdown:.4f} < {MAX_DRAWDOWN_LIMIT})")
        if trading_frequency < MIN_TRADE_FREQUENCY:
            print(f"  - Trading Frequency is below target ({trading_frequency:.4f} < {MIN_TRADE_FREQUENCY})")
    
    # Compare to buy-and-hold
    print("\nComparison to Buy & Hold:")
    print(f"Strategy Total Return: {performance['Total Return']:.4f}")
    print(f"Buy & Hold Total Return: {performance['Buy Hold Return']:.4f}")
    print(f"Strategy Sharpe: {performance['Sharpe Ratio']:.4f}")
    print(f"Buy & Hold Sharpe: {performance['Buy Hold Sharpe']:.4f}")
    
    return meets_criteria


def print_performance(performance):
    """Print detailed performance metrics."""
    print("\nDetailed Performance Metrics:")
    
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    setup_directories()
    
    # Prepare data
    data = prepare_data(args)
    
    if data is None or data.empty:
        print("Error: No data available for processing. Please check data sources.")
        return 1
    
    # Train model
    hmm_model = train_model(data, args)
    
    if hmm_model is None:
        print("Error: Model training failed. Please check your data and parameters.")
        return 1
    
    # Run backtest
    results, performance = run_backtest(hmm_model, data, args)
    
    # Evaluate performance against criteria
    success = evaluate_performance(performance)
    
    # Save results
    if not args.skip_plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(RESULTS_DIR, f"{args.crypto}_results_{timestamp}.csv")
        results.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 