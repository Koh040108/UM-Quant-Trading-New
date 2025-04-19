import argparse
import os
from datetime import datetime
import pandas as pd
from src.config import CRYPTOCURRENCIES, DATA_INTERVALS, MODELS_DIR, MIN_SHARPE_RATIO, MAX_DRAWDOWN_LIMIT, MIN_TRADE_FREQUENCY
from src.hmm_model import MarketHMM
from src.xgboost_model import XGBoostPredictor
from src.lstm_model import LSTMPredictor
from src.hybrid_model import HybridTradingModel

# Add the --lstm_backend flag to the argument parser
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
    parser.add_argument('--no_lstm', action='store_true',
                        help='Exclude LSTM model from hybrid approach')
    parser.add_argument('--lstm_backend', type=str, default='auto', choices=['auto', 'tensorflow', 'pytorch'],
                        help='Backend to use for LSTM model (auto will try tensorflow first, then pytorch)')
    # Performance metric thresholds
    parser.add_argument('--min_sharpe', type=float, default=MIN_SHARPE_RATIO,
                        help=f'Minimum Sharpe ratio target (default: {MIN_SHARPE_RATIO})')
    parser.add_argument('--max_drawdown', type=float, default=MAX_DRAWDOWN_LIMIT,
                        help=f'Maximum drawdown limit as negative percentage (default: {MAX_DRAWDOWN_LIMIT})')
    parser.add_argument('--min_trade_freq', type=float, default=MIN_TRADE_FREQUENCY,
                        help=f'Minimum trading frequency target (default: {MIN_TRADE_FREQUENCY})')
    
    return parser.parse_args()

# Update the train_model function to use the specified LSTM backend
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
            device = None
            # Force specific backend if requested
            if args.lstm_backend == 'tensorflow':
                try:
                    import tensorflow as tf
                    print("Using TensorFlow backend as requested")
                except ImportError:
                    print("Warning: TensorFlow not available. Falling back to PyTorch.")
                    args.lstm_backend = 'pytorch'
            
            if args.lstm_backend == 'pytorch':
                try:
                    import torch
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"Using PyTorch backend as requested on device: {device}")
                except ImportError:
                    print("Warning: PyTorch not available. Cannot continue.")
                    return None, None
            
            model = LSTMPredictor(window_size=args.window_size, device=device)
            model.fit(train_data)
        elif args.model == 'hybrid':
            print(f"Training new Hybrid model with {args.states} HMM states, {args.n_lags} XGBoost lags, and LSTM with window size {args.window_size}")
            model = HybridTradingModel(
                n_states=args.states, 
                n_lags=args.n_lags,
                window_size=args.window_size,
                use_lstm=not args.no_lstm
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

def main():
    """Main function to run the program."""
    # Parse arguments
    args = parse_args()
    
    # Here, you would typically load data
    # For now, we'll just print the arguments
    print("Running with arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    print("\nNote: This is a simplified version of main.py for testing.")
    print("To test if our fix for the price_col parameter works, we need to implement more functionality.")

if __name__ == "__main__":
    main() 