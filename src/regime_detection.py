"""
Market regime detection using Hidden Markov Models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from hmmlearn.hmm import GaussianHMM
import pickle
import os
from datetime import datetime

from src.config import MODELS_DIR


class MarketRegimeDetector:
    """
    Detects market regimes using Hidden Markov Models.
    """
    
    def __init__(self, n_regimes=2, random_state=42, n_iter=1000):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes (int): Number of regimes/states to detect
            random_state (int): Random seed for reproducibility
            n_iter (int): Maximum number of iterations for fitting
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.n_iter = n_iter
        self.model = None
        
        # Create models directory if it doesn't exist
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
    
    def train(self, returns, filename=None):
        """
        Train the regime detection model on historical returns.
        
        Args:
            returns (pd.Series): Series of asset returns
            filename (str, optional): Filename to save the trained model
            
        Returns:
            self: Trained model
        """
        # Prepare data for HMM - reshaping to column vector
        X = np.column_stack([returns.values])
        
        # Create and fit the HMM model
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        
        # Save model if filename is provided
        if filename:
            model_path = os.path.join(MODELS_DIR, filename)
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {model_path}")
        
        return self
    
    def predict_regimes(self, returns):
        """
        Predict regimes for the given returns data.
        
        Args:
            returns (pd.Series): Series of asset returns
            
        Returns:
            np.array: Array of regime predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = np.column_stack([returns.values])
        regimes = self.model.predict(X)
        
        return regimes
    
    def get_current_regime(self, returns, lookback=30):
        """
        Get the current market regime using recent returns.
        
        Args:
            returns (pd.Series): Series of asset returns
            lookback (int): Number of periods to use for regime detection
            
        Returns:
            int: Current regime (0-indexed)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Use the most recent returns data
        recent_returns = returns.iloc[-lookback:]
        X = np.column_stack([recent_returns.values])
        
        # Get the most recent regime
        regimes = self.model.predict(X)
        current_regime = regimes[-1]
        
        return current_regime
    
    def analyze_regimes(self, returns, prices=None):
        """
        Analyze the characteristics of each regime.
        
        Args:
            returns (pd.Series): Series of asset returns
            prices (pd.Series, optional): Series of asset prices
            
        Returns:
            tuple: (regimes, stats) where regimes is an array of predicted regimes 
                  and stats is a DataFrame with regime statistics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = np.column_stack([returns.values])
        regimes = self.model.predict(X)
        
        # Create a DataFrame with returns and regimes
        df = pd.DataFrame({
            'returns': returns.values,
            'regime': regimes
        }, index=returns.index)
        
        # Add prices if provided
        if prices is not None:
            df['price'] = prices
        
        # Calculate statistics for each regime
        stats = []
        for i in range(self.n_regimes):
            regime_data = df[df['regime'] == i]
            
            # Calculate key metrics
            mean_return = regime_data['returns'].mean()
            std_return = regime_data['returns'].std()
            sharpe = mean_return / std_return if std_return > 0 else 0
            count = len(regime_data)
            pct = count / len(df) * 100
            
            # Get price changes if prices are available
            price_change = None
            if prices is not None:
                # Calculate total price change during this regime
                if len(regime_data) > 0:
                    first_price = regime_data['price'].iloc[0]
                    last_price = regime_data['price'].iloc[-1]
                    price_change = (last_price / first_price - 1) * 100
            
            stats.append({
                'regime': i,
                'count': count,
                'percentage': pct,
                'mean_return': mean_return * 100,  # Convert to percentage
                'std_return': std_return * 100,    # Convert to percentage
                'sharpe_ratio': sharpe,
                'price_change': price_change       # In percentage
            })
        
        return regimes, pd.DataFrame(stats).set_index('regime')
    
    def plot_regimes(self, returns, prices, regimes=None):
        """
        Plot the asset prices colored by regime.
        
        Args:
            returns (pd.Series): Series of asset returns
            prices (pd.Series): Series of asset prices
            regimes (np.array, optional): Pre-computed regimes
        """
        if regimes is None:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
                
            X = np.column_stack([returns.values])
            regimes = self.model.predict(X)
        
        fig, axs = plt.subplots(
            self.n_regimes, 
            figsize=(15, 10),
            sharex=True, sharey=True
        )
        
        colors = cm.rainbow(
            np.linspace(0, 1, self.n_regimes)
        )
        
        for i, (ax, color) in enumerate(zip(axs, colors)):
            mask = regimes == i
            ax.plot_date(
                prices.index[mask],
                prices.values[mask],
                ".", linestyle='none',
                c=color
            )
            ax.set_title(f"Market Regime #{i}")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename):
        """
        Save the trained model to disk.
        
        Args:
            filename (str): Filename to save the model
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_path = os.path.join(MODELS_DIR, filename)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            self: Loaded model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Example usage
    from src.feature_engineering import FeatureEngineer
    
    try:
        # Load and process data
        feature_engineer = FeatureEngineer(normalize=True)
        processed_data = feature_engineer.load_and_process_data("BTC")
        
        # Extract returns
        if 'price' in processed_data.columns:
            price_col = 'price'
        elif 'close' in processed_data.columns:
            price_col = 'close'
        else:
            raise ValueError("No price column found")
            
        returns = processed_data[price_col].pct_change().dropna()
        prices = processed_data[price_col].loc[returns.index]
        
        # Create and train regime detector
        regime_detector = MarketRegimeDetector(n_regimes=2)
        regime_detector.train(returns)
        
        # Analyze regimes
        regimes, stats = regime_detector.analyze_regimes(returns, prices)
        print("Regime Statistics:")
        print(stats)
        
        # Plot regimes
        regime_detector.plot_regimes(returns, prices, regimes)
        
        # Save model
        regime_detector.save_model("btc_regime_detector.pkl")
    
    except Exception as e:
        print(f"Error: {str(e)}") 