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

from src.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, CRYPTOCURRENCIES, 
    DATA_INTERVALS, DEFAULT_INTERVAL, HMM_STATES,
    TRADING_FEE, MIN_TRADE_FREQUENCY, MIN_SHARPE_RATIO, MAX_DRAWDOWN_LIMIT
)
from src.cybotrade_fetcher import fetch_all_data
from src.feature_engineering import FeatureEngineer
from src.hmm_model import MarketHMM
from src.visualization import create_performance_dashboard
from src.xgboost_model import XGBoostPredictor
from src.hybrid_model import HybridTradingModel
from src.lstm_model import LSTMPredictor

# Print API key status
api_key = os.getenv("CYBOTRADE_API_KEY")
print(f"\nAPI Key Status:")
print(f"Loaded: {api_key is not None}")
print(f"Length: {len(api_key) if api_key else 0}")
print(f"First 5 chars: {api_key[:5] if api_key and len(api_key) >= 5 else 'None'}")
print(f"Last 5 chars: {api_key[-5:] if api_key and len(api_key) >= 5 else 'None'}")
print(f"\nNote: By default, all existing data will be deleted and fresh data will be fetched.")
print(f"To use existing data instead, run with the --no_refresh flag.\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Trading Strategy with HMM')
    
    parser.add_argument('--crypto', type=str, default='BTC', choices=CRYPTOCURRENCIES,
                        help='Cryptocurrency to analyze')
    parser.add_argument('--interval', type=str, default='1h', choices=list(DATA_INTERVALS.keys()),
                        help='Data interval')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--cybotrade_api_key', type=str, default=None,
                        help='API key for Cybotrade (if not set in environment)')
    parser.add_argument('--states', type=int, default=5,
                        help='Number of hidden states for HMM')
    parser.add_argument('--threshold', type=float, default=0.0002,
                        help='Return threshold for profitable states')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained model to load')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip plotting results')
    parser.add_argument('--no_refresh', action='store_true',
                        help='Do not refresh data (use existing files if available)')
    parser.add_argument('--no_shorts', action='store_true',
                        help='Disable short selling (only allow long positions)')
    parser.add_argument('--use_regimes', action='store_true',
                        help='Use market regime detection as a trading filter')
    parser.add_argument('--regime_states', type=int, default=2,
                        help='Number of market regimes to detect (default: 2)')
    parser.add_argument('--model', type=str, default='hybrid', choices=['hmm', 'xgboost', 'lstm', 'hybrid'],
                        help='Model to use for prediction')
    parser.add_argument('--n_lags', type=int, default=2,
                        help='Number of lag features for XGBoost (default: 2)')
    parser.add_argument('--window_size', type=int, default=30,
                        help='Window size for LSTM model (default: 30)')
    parser.add_argument('--use_lstm', action='store_true',
                        help='Include LSTM model in hybrid approach')
    # Performance metric thresholds
    parser.add_argument('--min_sharpe', type=float, default=MIN_SHARPE_RATIO,
                        help=f'Minimum Sharpe ratio target (default: {MIN_SHARPE_RATIO})')
    parser.add_argument('--max_drawdown', type=float, default=MAX_DRAWDOWN_LIMIT,
                        help=f'Maximum drawdown limit as negative percentage (default: {MAX_DRAWDOWN_LIMIT})')
    parser.add_argument('--min_trade_freq', type=float, default=MIN_TRADE_FREQUENCY,
                        help=f'Minimum trading frequency target (default: {MIN_TRADE_FREQUENCY})')
    
    return parser.parse_args()


def setup_directories():
    """Ensure required directories exist."""
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def clear_data_directory(crypto=None):
    """Delete old data files to ensure fresh data fetching.
    
    Args:
        crypto (str, optional): If provided, only delete files for this crypto.
    """
    if not os.path.exists(DATA_DIR):
        return
        
    files_to_delete = []
    
    if crypto:
        # Only delete files for the specified crypto
        files_to_delete = [f for f in os.listdir(DATA_DIR) if f.startswith(crypto) and f.endswith('.csv')]
    else:
        # Delete all CSV files in the data directory
        files_to_delete = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if files_to_delete:
        print(f"Deleting {len(files_to_delete)} old data files...")
        for file in files_to_delete:
            file_path = os.path.join(DATA_DIR, file)
            try:
                os.remove(file_path)
                print(f"  Deleted: {file}")
            except Exception as e:
                print(f"  Error deleting {file}: {str(e)}")


def prepare_data(args):
    """Prepare data for modeling."""
    crypto = args.crypto
    interval = args.interval
    start_date = args.start_date
    end_date = args.end_date
    
    # Set default dates if not provided
    if start_date is None:
        # Always get 3 years of data
        end_date_obj = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        start_date = (end_date_obj - timedelta(days=365*3)).strftime('%Y-%m-%d')
        print(f"Setting start date to exactly 3 years before end date: {start_date}")
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Preparing data for {crypto} from {start_date} to {end_date}")
    
    # Check if we should use existing data or fetch fresh data
    existing_data_files = [f for f in os.listdir(DATA_DIR) if f.startswith(crypto) and f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    
    if args.no_refresh and existing_data_files:
        print(f"Using existing data files ({len(existing_data_files)} found)...")
    else:
        # Clean old data files and fetch new data
        clear_data_directory(crypto)
        print(f"Fetching fresh data using Cybotrade data source...")
        fetch_all_data(
            cryptos=[crypto],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            api_key=args.cybotrade_api_key
        )
    
    # Check if data files exist
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith(crypto) and f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    
    if not data_files:
        print("No data files found. Please check your API key and connection.")
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
    # Split data into training and testing sets (2 years training, 1 year testing)
    data = data.sort_values('date')
    
    # Calculate date ranges based on available data
    data_min_date = data['date'].min()
    data_max_date = data['date'].max()
    actual_days = (data_max_date - data_min_date).days
    
    print(f"Total data span: {actual_days} days ({actual_days/365:.2f} years)")
    
    # If we have at least 2 years of data, use 2 years for training
    # Otherwise use 2/3 of the available data
    if actual_days >= 365*2:
        train_end_date = data_min_date + pd.Timedelta(days=365*2)
        print(f"Using exactly 2 years for training, 1 year for testing")
    else:
        train_end_date = data_min_date + pd.Timedelta(days=actual_days * 2/3)
        print(f"Using {actual_days * 2/3:.0f} days for training (2/3 of available data)")
    
    train_data = data[data['date'] <= train_end_date]
    test_data = data[data['date'] > train_end_date]
    
    print(f"Training data: {len(train_data)} rows from {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"Testing data: {len(test_data)} rows from {test_data['date'].min()} to {test_data['date'].max()}")
    
    # Create and train the selected model
    model = None
    
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        model = MarketHMM()
        model.load_model(args.load_model)
    else:
        if args.model == 'hmm':
            print(f"Training new HMM model with {args.states} states")
            model = MarketHMM(n_states=args.states)
            model.fit(train_data)
        elif args.model == 'xgboost':
            print(f"Training new XGBoost model with {args.n_lags} lags")
            model = XGBoostPredictor(n_lags=args.n_lags)
            model.fit(train_data)
        elif args.model == 'lstm':
            print(f"Training new LSTM model with window size {args.window_size}")
            model = LSTMPredictor(window_size=args.window_size)
            model.fit(train_data)
        elif args.model == 'hybrid':
            print(f"Training new Hybrid model with {args.states} HMM states, {args.n_lags} XGBoost lags, and LSTM with window size {args.window_size}")
            model = HybridTradingModel(
                n_states=args.states, 
                n_lags=args.n_lags,
                window_size=args.window_size,
                use_lstm=args.use_lstm
            )
            model.fit(train_data)
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        # Save model if requested
        if args.save_model:
            if args.model == 'hmm':
                model_path = model.save_model()
                print(f"HMM model saved to {model_path}")
            elif args.model == 'xgboost' and hasattr(model, 'save_model'):
                model_path = model.save_model()
                print(f"XGBoost model saved to {model_path}")
            elif args.model == 'lstm' and hasattr(model, 'save_model'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(MODELS_DIR, f"lstm_model_{timestamp}.pt")
                model.save_model(model_path)
                print(f"LSTM model saved to {model_path}")
            elif args.model == 'hybrid' and hasattr(model, 'save_model'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(MODELS_DIR, f"hybrid_model_{timestamp}")
                model.save_model(model_path)
                print(f"Hybrid model saved to {model_path}")
            else:
                print(f"Model saving not implemented for {args.model} model")
    
    return model, test_data


def run_backtest(model, data, args):
    """Run backtest with the trained model."""
    print("Running backtest...")
    
    # Determine which price column to use
    price_cols = ['price_usd_close', 'close', 'price', 'value']
    price_col = None
    
    # Find the first available price column in the data
    for col in price_cols:
        if col in data.columns:
            price_col = col
            print(f"Using '{price_col}' as the price column")
            break
    
    if price_col is None:
        # If no standard price column is found, try to find any column with 'price' in the name
        price_candidates = [col for col in data.columns if 'price' in col.lower()]
        if price_candidates:
            price_col = price_candidates[0]
            print(f"Using '{price_col}' as the price column")
        else:
            # If all else fails, use the first numeric column as a last resort
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                print(f"WARNING: No price column found. Using '{price_col}' as a proxy.")
            else:
                print("ERROR: No suitable price column found in the data.")
                return None, None
    
    results = None
    performance = None
    
    # Generate signals based on model type
    if args.model == 'hmm':
        # Add states to the data
        with_states = model.add_states_to_df(data)
        
        # Generate trading signals, using regime detection if specified
        signals = model.generate_trading_signals(
            with_states, 
            threshold=args.threshold, 
            price_col=price_col,
            use_regimes=args.use_regimes
        )
        
        # Backtest the strategy
        allow_shorts = not args.no_shorts
        results, performance = model.backtest_strategy(
            signals, 
            price_col=price_col, 
            fee=TRADING_FEE,
            allow_shorts=allow_shorts
        )
    elif args.model == 'xgboost':
        # Generate predictions and signals
        predictions = model.predict(data, price_col=price_col)
        signals = model.generate_trading_signals(predictions)
        
        # Use the HMM backtesting logic
        dummy_hmm = MarketHMM()
        results, performance = dummy_hmm.backtest_strategy(
            signals,
            price_col=price_col,
            fee=TRADING_FEE,
            allow_shorts=not args.no_shorts
        )
    elif args.model == 'lstm':
        # Generate predictions and signals
        predictions = model.predict(data, price_col=price_col)
        signals = model.generate_trading_signals(predictions)
        
        # Use the HMM backtesting logic
        dummy_hmm = MarketHMM()
        results, performance = dummy_hmm.backtest_strategy(
            signals,
            price_col=price_col,
            fee=TRADING_FEE,
            allow_shorts=not args.no_shorts
        )
    elif args.model == 'hybrid':
        # Generate combined predictions
        combined = model.predict(data, price_col=price_col, threshold=args.threshold)
        
        # Backtest with the hybrid model
        results, performance = model.backtest_strategy(
            combined,
            price_col=price_col,
            fee=TRADING_FEE,
            allow_shorts=not args.no_shorts
        )
    
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
        
        # Generate model-specific visualizations
        if args.model == 'xgboost':
            model.plot_predictions(results, price_col=price_col)
        elif args.model == 'lstm':
            model.plot_predictions(results, price_col=price_col)
        elif args.model == 'hybrid':
            model.plot_signals(results, price_col=price_col)
    
    return results, performance


def evaluate_performance(performance, args):
    """Evaluate if the strategy meets performance criteria."""
    sharpe_ratio = performance['Sharpe Ratio']
    max_drawdown = performance['Max Drawdown']
    trading_frequency = performance['Trading Frequency']
    
    # Use command line args for threshold values if provided
    min_sharpe_target = args.min_sharpe
    max_drawdown_limit = args.max_drawdown
    min_trade_freq_target = args.min_trade_freq
    
    print("\nPerformance Evaluation:")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f} (Target ≥ {min_sharpe_target})")
    print(f"Max Drawdown: {max_drawdown:.4f} (Target > {max_drawdown_limit})")
    print(f"Trading Frequency: {trading_frequency:.4f} (Target ≥ {min_trade_freq_target})")
    
    meets_criteria = (
        sharpe_ratio >= min_sharpe_target and
        max_drawdown > max_drawdown_limit and
        trading_frequency >= min_trade_freq_target
    )
    
    if meets_criteria:
        print("\n✅ Strategy meets all performance criteria!")
    else:
        print("\n❌ Strategy does not meet all performance criteria.")
        
        if sharpe_ratio < min_sharpe_target:
            print(f"  - Sharpe Ratio is below target ({sharpe_ratio:.4f} < {min_sharpe_target})")
        if max_drawdown <= max_drawdown_limit:
            print(f"  - Max Drawdown is worse than target ({max_drawdown:.4f} ≤ {max_drawdown_limit})")
        if trading_frequency < min_trade_freq_target:
            print(f"  - Trading Frequency is below target ({trading_frequency:.4f} < {min_trade_freq_target})")
    
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
    
    # Train model and get test data
    model, test_data = train_model(data, args)
    
    if model is None:
        print("Error: Model training failed. Please check your data and parameters.")
        return 1
    
    # Run backtest on test data
    results, performance = run_backtest(model, test_data, args)
    
    # Evaluate performance against criteria
    success = evaluate_performance(performance, args)
    
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