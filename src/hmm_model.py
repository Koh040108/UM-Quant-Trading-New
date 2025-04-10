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
            # Default to using price changes and volatility
            feature_columns = ['price_change_1d', 'volatility_7d']
            
            # Check if columns exist in the DataFrame
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if not available_columns:
                # Fallback to using close price
                if 'close' in df.columns:
                    # Calculate returns
                    returns = df['close'].pct_change().fillna(0)
                    # Calculate volatility
                    volatility = returns.rolling(window=7).std().fillna(0)
                    
                    # Create a new DataFrame with these features
                    tmp_df = pd.DataFrame({
                        'returns': returns,
                        'volatility': volatility
                    })
                    
                    # Scale the features
                    X = self.scaler.fit_transform(tmp_df)
                    return X
                else:
                    raise ValueError("Required columns not found in DataFrame")
            
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
        Generate trading signals based on the HMM states.
        
        Args:
            df (pd.DataFrame): DataFrame with features and hmm_state column
            threshold (float): Return threshold for selecting profitable states
            price_col (str): Column name for price
            
        Returns:
            pd.DataFrame: DataFrame with added trading signals
        """
        if 'hmm_state' not in df.columns:
            df = self.add_states_to_df(df)
        
        # Calculate returns
        df['returns'] = df[price_col].pct_change()
        
        # Analyze state characteristics
        state_analysis, _, _ = self.analyze_states(df, price_col)
        
        # Identify profitable states (positive expected return)
        profitable_states = state_analysis[('returns', 'mean')].where(
            state_analysis[('returns', 'mean')] > threshold
        ).dropna().index.tolist()
        
        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        df['signal'] = 0
        df['signal'] = df['hmm_state'].apply(lambda x: 1 if x in profitable_states else -1)
        
        # Generate positions (1 for long, -1 for short, 0 for flat)
        df['position'] = df['signal'].shift(1)
        df['position'].fillna(0, inplace=True)
        df['position'] = df['position'].astype(int)
        
        return df
    
    def backtest_strategy(self, df, price_col='close', fee=TRADING_FEE):
        """
        Backtest the HMM-based trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with features, signals, and positions
            price_col (str): Column name for price
            fee (float): Trading fee as a decimal (e.g., 0.001 for 0.1%)
            
        Returns:
            tuple: (returns_df, performance_metrics)
        """
        if 'position' not in df.columns:
            raise ValueError("DataFrame does not contain position column")
        
        # Calculate price returns
        df['price_returns'] = df[price_col].pct_change()
        
        # Calculate strategy returns (without fees)
        df['strategy_returns_no_fee'] = df['price_returns'] * df['position']
        
        # Calculate trading costs
        df['trade'] = df['position'].diff().fillna(0) != 0
        df['fee_cost'] = df['trade'].astype(int) * fee
        
        # Calculate strategy returns (with fees)
        df['strategy_returns'] = df['strategy_returns_no_fee'] - df['fee_cost']
        
        # Calculate cumulative returns
        df['cum_price_returns'] = (1 + df['price_returns']).cumprod() - 1
        df['cum_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1
        
        # Calculate drawdowns
        df['price_peak'] = df['cum_price_returns'].cummax()
        df['strategy_peak'] = df['cum_strategy_returns'].cummax()
        df['price_drawdown'] = (df['cum_price_returns'] - df['price_peak']) / (1 + df['price_peak'])
        df['strategy_drawdown'] = (df['cum_strategy_returns'] - df['strategy_peak']) / (1 + df['strategy_peak'])
        
        # Calculate performance metrics
        total_trades = df['trade'].sum()
        win_trades = ((df['strategy_returns'] > 0) & df['trade']).sum()
        loss_trades = ((df['strategy_returns'] < 0) & df['trade']).sum()
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Trading frequency
        trading_frequency = total_trades / len(df)
        
        # Annualized metrics (assuming daily data)
        trading_days = 252
        days = len(df)
        annual_factor = trading_days / days
        
        total_return = df['cum_strategy_returns'].iloc[-1]
        annualized_return = (1 + total_return) ** annual_factor - 1
        
        # Risk metrics
        volatility = df['strategy_returns'].std() * np.sqrt(trading_days)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = df['strategy_drawdown'].min()
        
        # Create a buy-hold comparison
        buy_hold_return = df['cum_price_returns'].iloc[-1]
        buy_hold_annualized = (1 + buy_hold_return) ** annual_factor - 1
        buy_hold_volatility = df['price_returns'].std() * np.sqrt(trading_days)
        buy_hold_sharpe = buy_hold_annualized / buy_hold_volatility if buy_hold_volatility > 0 else 0
        buy_hold_max_drawdown = df['price_drawdown'].min()
        
        # Compile performance metrics
        performance = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Trading Frequency': trading_frequency,
            'Buy Hold Return': buy_hold_return,
            'Buy Hold Annualized': buy_hold_annualized,
            'Buy Hold Sharpe': buy_hold_sharpe,
            'Buy Hold Max Drawdown': buy_hold_max_drawdown
        }
        
        return df, performance
    
    def plot_states_and_returns(self, df, price_col='close'):
        """
        Plot the hidden states along with price and returns.
        
        Args:
            df (pd.DataFrame): DataFrame with features, hmm_state, and returns
            price_col (str): Column name for price
        """
        if 'hmm_state' not in df.columns:
            raise ValueError("DataFrame does not contain hmm_state column")
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot price
        ax1.plot(df.index, df[price_col], label='Price')
        ax1.set_title('Price and HMM States')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Plot hidden states
        ax2.plot(df.index, df['hmm_state'], label='Hidden State', marker='o', markersize=3, linestyle='-')
        ax2.set_ylabel('State')
        ax2.grid(True)
        
        # Highlight different regions with different colors
        states = df['hmm_state'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(states)))
        
        for i, state in enumerate(states):
            mask = df['hmm_state'] == state
            ax2.fill_between(df.index, 0, 1, where=mask, transform=ax2.get_xaxis_transform(), 
                             color=colors[i], alpha=0.3, label=f'State {state}')
        
        ax2.legend(loc='upper right')
        
        # Plot returns or cumulative returns
        if 'cum_strategy_returns' in df.columns:
            ax3.plot(df.index, df['cum_price_returns'], label='Buy & Hold', color='blue')
            ax3.plot(df.index, df['cum_strategy_returns'], label='HMM Strategy', color='green')
            ax3.set_ylabel('Cumulative Returns')
            ax3.set_title('Strategy Performance')
        else:
            ax3.plot(df.index, df['returns'], label='Returns', color='blue')
            ax3.set_ylabel('Returns')
            ax3.set_title('Daily Returns')
        
        ax3.grid(True)
        ax3.legend(loc='upper left')
        
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