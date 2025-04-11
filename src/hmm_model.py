"""
Hidden Markov Model for detecting market regimes and generating trading signals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

from src.config import MODELS_DIR, HMM_STATES, TRADING_FEE


class MarketHMM:
    """
    Hidden Markov Model for identifying crypto market states and generating trading signals.
    """
    
    def __init__(self, n_states=HMM_STATES, n_iter=1000, random_state=42):
        """
        Initialize the HMM model.
        
        Args:
            n_states (int): Number of hidden states
            n_iter (int): Maximum number of iterations for EM algorithm
            random_state (int): Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        # Create models directory if it doesn't exist
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
    
    def _prepare_features(self, df, feature_columns=None):
        """
        Prepare features for the HMM model.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            feature_columns (list, optional): List of feature columns to use
            
        Returns:
            np.array: Prepared features
        """
        if feature_columns is None:
            # Check for ROC features first
            roc_columns = [col for col in df.columns if col.startswith('roc_')]
            if roc_columns:
                # If ROC features are available, use them
                feature_columns = roc_columns
                print(f"Using ROC features: {feature_columns}")
            else:
                # Default to using price changes and volatility
                feature_columns = ['price_change_1d', 'volatility_7d']
            
            # Check if columns exist in the DataFrame
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if not available_columns:
                # Fallback to using close or price column
                price_col = None
                if 'close' in df.columns:
                    price_col = 'close'
                elif 'price' in df.columns:
                    price_col = 'price'
                
                if price_col:
                    # Calculate returns
                    returns = df[price_col].pct_change().fillna(0)
                    # Calculate volatility
                    volatility = returns.rolling(window=7).std().fillna(0)
                    # Calculate ROC
                    roc = df[price_col].pct_change(periods=10).fillna(0) * 100
                    
                    # Create a new DataFrame with these features
                    tmp_df = pd.DataFrame({
                        'returns': returns,
                        'volatility': volatility,
                        'roc': roc
                    })
                    
                    # Scale the features
                    X = self.scaler.fit_transform(tmp_df)
                    return X
                else:
                    # Print available columns to help debug
                    print(f"Available columns: {df.columns.tolist()}")
                    raise ValueError("Required columns not found in DataFrame. Need either 'price_change_1d' and 'volatility_7d', or 'close'/'price' column.")
            
            # Use available columns
            feature_columns = available_columns
        
        # Extract features
        X = df[feature_columns].values
        
        # Scale the features
        X = self.scaler.fit_transform(X)
        
        return X
    
    def fit(self, df, feature_columns=None):
        """
        Fit the HMM model to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            feature_columns (list, optional): List of feature columns to use
            
        Returns:
            self: Fitted model
        """
        # Prepare features
        X = self._prepare_features(df, feature_columns)
        
        # Initialize the HMM model
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        # Fit the model
        self.model.fit(X)
        
        return self
    
    def predict_states(self, df, feature_columns=None):
        """
        Predict hidden states for the given data.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            feature_columns (list, optional): List of feature columns to use
            
        Returns:
            np.array: Predicted states
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Prepare features
        X = self._prepare_features(df, feature_columns)
        
        # Predict states
        states = self.model.predict(X)
        
        return states
    
    def decode_states(self, df, feature_columns=None):
        """
        Decode the most likely sequence of hidden states.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            feature_columns (list, optional): List of feature columns to use
            
        Returns:
            tuple: (logprob, states) where logprob is the log probability
                  and states is the sequence of states
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Prepare features
        X = self._prepare_features(df, feature_columns)
        
        # Decode the sequence
        logprob, states = self.model.decode(X)
        
        return logprob, states
    
    def add_states_to_df(self, df, feature_columns=None):
        """
        Add predicted states to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            feature_columns (list, optional): List of feature columns to use
            
        Returns:
            pd.DataFrame: DataFrame with added state column
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Predict states
        states = self.predict_states(result, feature_columns)
        
        # Add states to DataFrame
        result['hmm_state'] = states
        
        return result
    
    def analyze_states(self, df, price_col='close'):
        """
        Analyze the characteristics of each state.
        
        Args:
            df (pd.DataFrame): DataFrame with features and hmm_state column
            price_col (str): Column name for price
            
        Returns:
            pd.DataFrame: DataFrame with state characteristics
        """
        if 'hmm_state' not in df.columns:
            raise ValueError("DataFrame does not contain hmm_state column")
        
        # Use price column if close is not available
        if price_col not in df.columns:
            if 'price' in df.columns:
                print(f"Using 'price' column instead of '{price_col}' for state analysis")
                price_col = 'price'
            elif 'value' in df.columns:
                print(f"Using 'value' column instead of '{price_col}' for state analysis")
                price_col = 'value'
            else:
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Required price column '{price_col}' not found in DataFrame")
        
        # Calculate returns
        df['returns'] = df[price_col].pct_change()
        
        # Group by state
        state_analysis = df.groupby('hmm_state').agg({
            'returns': ['mean', 'std', 'count'],
            price_col: ['mean', 'min', 'max']
        })
        
        # Add annualized metrics (assuming daily data)
        trading_days = 365
        state_analysis[('returns', 'annualized_mean')] = state_analysis[('returns', 'mean')] * trading_days
        state_analysis[('returns', 'annualized_std')] = state_analysis[('returns', 'std')] * np.sqrt(trading_days)
        state_analysis[('returns', 'sharpe_ratio')] = state_analysis[('returns', 'annualized_mean')] / state_analysis[('returns', 'annualized_std')]
        
        # Calculate state transition probabilities
        transitions = np.zeros((self.n_states, self.n_states))
        for i in range(len(df) - 1):
            from_state = df['hmm_state'].iloc[i]
            to_state = df['hmm_state'].iloc[i + 1]
            transitions[from_state, to_state] += 1
        
        # Normalize to get probabilities
        for i in range(self.n_states):
            if transitions[i].sum() > 0:
                transitions[i] = transitions[i] / transitions[i].sum()
        
        # Calculate mean duration of each state
        state_durations = []
        for state in range(self.n_states):
            duration = 0
            count = 0
            current_duration = 0
            
            for s in df['hmm_state']:
                if s == state:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        duration += current_duration
                        count += 1
                        current_duration = 0
            
            # Don't forget the last run
            if current_duration > 0:
                duration += current_duration
                count += 1
            
            mean_duration = duration / count if count > 0 else 0
            state_durations.append(mean_duration)
        
        return state_analysis, transitions, state_durations
    
    def generate_trading_signals(self, df, threshold=0.0, price_col='close'):
        """
        Generate trading signals based on HMM states.
        
        Args:
            df (pd.DataFrame): DataFrame with features and hmm_state column
            threshold (float): Return threshold for profitable states
            price_col (str): Column name for price
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        if 'hmm_state' not in df.columns:
            raise ValueError("DataFrame does not contain hmm_state column")
        
        # Use price column if close is not available
        if price_col not in df.columns:
            if 'price' in df.columns:
                print(f"Using 'price' column instead of '{price_col}' for signal generation")
                price_col = 'price'
            elif 'value' in df.columns:
                print(f"Using 'value' column instead of '{price_col}' for signal generation")
                price_col = 'value'
            else:
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Required price column '{price_col}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate returns
        result['returns'] = result[price_col].pct_change()
        
        # Compute state characteristics based on historical returns
        state_returns = {}
        state_sharpe = {}
        state_volatility = {}
        
        for state in range(self.n_states):
            state_data = result[result['hmm_state'] == state]
            mean_return = state_data['returns'].mean()
            std_return = state_data['returns'].std()
            
            state_returns[state] = mean_return
            state_volatility[state] = std_return
            
            # Calculate Sharpe ratio (using 0 as risk-free rate for simplicity)
            if std_return > 0:
                sharpe = mean_return / std_return
            else:
                sharpe = 0
                
            state_sharpe[state] = sharpe
        
        # Dynamically identify states based on characteristics
        bullish_state = max(state_sharpe, key=state_sharpe.get)
        bearish_state = min(state_sharpe, key=state_sharpe.get)
        
        # Any state with sharpe below threshold is neutral
        neutral_states = [state for state, sharpe in state_sharpe.items() 
                         if abs(sharpe) < threshold and state != bullish_state and state != bearish_state]
        
        print(f"\nState Classification:")
        print(f"Bullish State: {bullish_state} (Sharpe: {state_sharpe[bullish_state]:.4f}, Return: {state_returns[bullish_state]:.4f})")
        print(f"Bearish State: {bearish_state} (Sharpe: {state_sharpe[bearish_state]:.4f}, Return: {state_returns[bearish_state]:.4f})")
        print(f"Neutral States: {neutral_states}")
        
        # Define a function to map states to signals
        def get_signal(state):
            if state == bullish_state and state_returns[state] > threshold:
                return 1  # Buy signal
            elif state == bearish_state and state_returns[state] < -threshold:
                return -1  # Sell signal
            else:
                return 0  # Neutral
        
        # Apply the function to create signals
        result['signal'] = result['hmm_state'].apply(get_signal)
        
        # Calculate signal statistics
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        neutral_signals = (result['signal'] == 0).sum()
        total_signals = len(result)
        
        print(f"\nSignal Statistics:")
        print(f"Buy Signals: {buy_signals} ({buy_signals/total_signals:.2%} of data)")
        print(f"Sell Signals: {sell_signals} ({sell_signals/total_signals:.2%} of data)")
        print(f"Neutral Signals: {neutral_signals} ({neutral_signals/total_signals:.2%} of data)")
        
        return result
    
    def backtest_strategy(self, df, price_col='close', fee=TRADING_FEE, allow_shorts=True):
        """
        Backtest the HMM trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with signal column
            price_col (str): Column name for price
            fee (float): Trading fee as a percentage
            allow_shorts (bool): Whether to allow short selling
            
        Returns:
            tuple: (results_df, performance_metrics)
        """
        if 'signal' not in df.columns:
            raise ValueError("DataFrame does not contain signal column")
        
        # Use price column if close is not available
        if price_col not in df.columns:
            if 'price' in df.columns:
                print(f"Using 'price' column instead of '{price_col}' for backtesting")
                price_col = 'price'
            elif 'value' in df.columns:
                print(f"Using 'value' column instead of '{price_col}' for backtesting")
                price_col = 'value'
            else:
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Required price column '{price_col}' not found in DataFrame")
        
        # Make a copy of the dataframe to avoid modifying the original
        results = df.copy()
        
        # Get price data
        price = results[price_col]
        
        # Calculate returns
        returns = price.pct_change().fillna(0)
        
        # Initialize strategy columns
        results['returns'] = returns
        results['strategy_position'] = 0
        results['strategy_returns'] = 0
        results['buy_hold_returns'] = returns
        results['strategy_cumulative'] = 0
        results['buy_hold_cumulative'] = 0
        
        # Get position changes based on signals
        position = 0
        positions = []
        trades = 0
        buy_trades = 0
        sell_trades = 0
        
        for i, row in results.iterrows():
            signal = row['signal']
            
            # Skip the first row to avoid NaN returns
            if i == results.index[0]:
                positions.append(0)
                continue
                
            # Determine position change based on signal
            if signal == 1 and position <= 0:  # Buy signal when not long
                prev_position = position
                position = 1
                trades += 1
                buy_trades += 1
                if prev_position < 0:  # If we were short, count closing the short as a trade
                    trades += 1
            elif signal == -1 and position >= 0 and allow_shorts:  # Sell signal when not short
                prev_position = position
                position = -1
                trades += 1
                sell_trades += 1
                if prev_position > 0:  # If we were long, count closing the long as a trade
                    trades += 1
            elif signal == 0 and position != 0:  # Neutral signal when in a position
                position = 0
                trades += 1
            
            positions.append(position)
        
        # Replace the first position with the second position to handle initialization
        if len(positions) > 1:
            positions[0] = positions[1]
            
        results['strategy_position'] = positions
        
        # Calculate strategy returns with fees
        for i in range(1, len(results)):
            # Check if we made a trade
            if results['strategy_position'].iloc[i] != results['strategy_position'].iloc[i-1]:
                # Apply fee
                position_change = results['strategy_position'].iloc[i] - results['strategy_position'].iloc[i-1]
                # Absolute value of position change determines the fee
                fee_amount = fee * abs(position_change)
                results.loc[results.index[i], 'strategy_returns'] = returns.iloc[i] * results['strategy_position'].iloc[i] - fee_amount
            else:
                # No fee if no position change
                results.loc[results.index[i], 'strategy_returns'] = returns.iloc[i] * results['strategy_position'].iloc[i]
        
        # Calculate cumulative returns
        results['strategy_cumulative'] = (1 + results['strategy_returns']).cumprod() - 1
        results['buy_hold_cumulative'] = (1 + results['buy_hold_returns']).cumprod() - 1
        
        # Calculate performance metrics
        total_days = len(results)
        trading_days_per_year = 365
        years = total_days / trading_days_per_year
        
        # Calculate key metrics
        total_return = results['strategy_cumulative'].iloc[-1]
        buy_hold_return = results['buy_hold_cumulative'].iloc[-1]
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        annualized_buy_hold_return = (1 + buy_hold_return) ** (1 / years) - 1
        
        # Calculate volatility
        daily_std = results['strategy_returns'].std()
        annualized_std = daily_std * (trading_days_per_year ** 0.5)
        
        buy_hold_daily_std = results['buy_hold_returns'].std()
        buy_hold_annualized_std = buy_hold_daily_std * (trading_days_per_year ** 0.5)
        
        # Calculate Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0
        buy_hold_sharpe = annualized_buy_hold_return / buy_hold_annualized_std if buy_hold_annualized_std > 0 else 0
        
        # Calculate drawdown
        peak = results['strategy_cumulative'].cummax()
        drawdown = (results['strategy_cumulative'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        strategy_wins = (results['strategy_returns'] > 0).sum()
        win_rate = strategy_wins / total_days if total_days > 0 else 0
        
        # Calculate trade frequency
        trade_frequency = trades / total_days
        
        # Long vs short performance
        long_returns = results[results['strategy_position'] > 0]['strategy_returns']
        short_returns = results[results['strategy_position'] < 0]['strategy_returns']
        neutral_returns = results[results['strategy_position'] == 0]['strategy_returns']
        
        long_win_rate = (long_returns > 0).sum() / len(long_returns) if len(long_returns) > 0 else 0
        short_win_rate = (short_returns > 0).sum() / len(short_returns) if len(short_returns) > 0 else 0
        
        # Compile performance metrics
        performance = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': annualized_std,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Trades': trades,
            'Buy Trades': buy_trades,
            'Sell Trades': sell_trades,
            'Trading Frequency': trade_frequency,
            'Long Win Rate': long_win_rate,
            'Short Win Rate': short_win_rate,
            'Buy Hold Return': buy_hold_return,
            'Buy Hold Annualized Return': annualized_buy_hold_return,
            'Buy Hold Volatility': buy_hold_annualized_std,
            'Buy Hold Sharpe': buy_hold_sharpe
        }
        
        return results, performance
    
    def plot_states_and_returns(self, df, price_col='close'):
        """
        Plot the hidden states along with price and returns.
        
        Args:
            df (pd.DataFrame): DataFrame with features, hmm_state, and returns
            price_col (str): Column name for price
        """
        if 'hmm_state' not in df.columns:
            raise ValueError("DataFrame does not contain hmm_state column")
        
        # Use price column if close is not available
        if price_col not in df.columns:
            if 'price' in df.columns:
                print(f"Using 'price' column instead of '{price_col}' for plotting")
                price_col = 'price'
            elif 'value' in df.columns:
                print(f"Using 'value' column instead of '{price_col}' for plotting")
                price_col = 'value'
            else:
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Required price column '{price_col}' not found in DataFrame")
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot price
        ax1.plot(df['date'], df[price_col], label=price_col.capitalize())
        ax1.set_title(f'{price_col.capitalize()} Price and Hidden States')
        ax1.set_ylabel(f'{price_col.capitalize()} Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot states
        scatter = ax2.scatter(df['date'], df[price_col], c=df['hmm_state'], cmap='viridis', 
                             label='Hidden States', s=30, alpha=0.6)
        ax2.set_ylabel(f'{price_col.capitalize()} Price')
        legend1 = ax2.legend(*scatter.legend_elements(), title="States")
        ax2.add_artist(legend1)
        ax2.grid(True)
        
        # Plot returns
        ax3.plot(df['date'], df['returns'] * 100, label='Daily Returns %', color='blue')
        ax3.set_title('Daily Returns')
        ax3.set_ylabel('Returns (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename=None):
        """
        Save the trained HMM model to disk.
        
        Args:
            filename (str, optional): Filename to save the model
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if filename is None:
            filename = f"hmm_model_{self.n_states}_states.pkl"
        
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Save the model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_states': self.n_states
        }, filepath)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """
        Load a trained HMM model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            self: Loaded model
        """
        # Load the model
        saved_model = joblib.load(filepath)
        
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.n_states = saved_model['n_states']
        
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Example usage
    from feature_engineering import FeatureEngineer
    
    try:
        # Load and process data
        feature_engineer = FeatureEngineer(normalize=True)
        
        # Assuming data is available
        processed_data = feature_engineer.load_and_process_data("BTC")
        
        # Create the HMM model
        hmm_model = MarketHMM(n_states=5)
        
        # Fit the model
        hmm_model.fit(processed_data)
        
        # Add states to DataFrame
        with_states = hmm_model.add_states_to_df(processed_data)
        
        # Generate trading signals
        signals = hmm_model.generate_trading_signals(with_states)
        
        # Backtest the strategy
        results, performance = hmm_model.backtest_strategy(signals)
        
        # Print performance metrics
        for metric, value in performance.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot states and returns
        hmm_model.plot_states_and_returns(results)
        
        # Save the model
        hmm_model.save_model()
        
    except Exception as e:
        print(f"Error running HMM model: {str(e)}")
        print("This example assumes you have already processed data using the feature_engineering module.") 