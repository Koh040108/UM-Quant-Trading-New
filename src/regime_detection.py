"""
Market regime detection using statistical methods to identify market states.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

from src.config import MODELS_DIR

class MarketRegimeDetector:
    """
    Detects market regimes (e.g., bull, bear, sideways) using machine learning methods.
    
    This class uses volatility and return metrics to classify market states.
    """
    
    def __init__(self, n_regimes=2, method='gmm', features=None, window_size=30):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes (int): Number of market regimes to detect
            method (str): Method for regime detection ('gmm' or 'kmeans')
            features (list): List of features to use for regime detection
            window_size (int): Window size for calculating features
        """
        self.n_regimes = n_regimes
        self.method = method.lower()
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        
        # Default features for regime detection if none specified
        self.features = features or ['volatility_14d', 'roc_5', 'roc_20']
        
    def _prepare_features(self, df, price_col='price'):
        """
        Prepare features for regime detection.
        
        Args:
            df (pd.DataFrame): DataFrame with price and other features
            price_col (str): Name of the price column
            
        Returns:
            pd.DataFrame: DataFrame with features for regime detection
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Calculate features if they don't exist
        if 'volatility_14d' not in data.columns and price_col in data.columns:
            data['volatility_14d'] = data[price_col].pct_change().rolling(window=14).std()
        
        if 'roc_5' not in data.columns and price_col in data.columns:
            data['roc_5'] = data[price_col].pct_change(5)
        
        if 'roc_20' not in data.columns and price_col in data.columns:
            data['roc_20'] = data[price_col].pct_change(20)
            
        # Only use the specified features for regime detection
        features_to_use = [f for f in self.features if f in data.columns]
        
        if not features_to_use:
            raise ValueError(f"None of the specified features {self.features} found in the data.")
        
        # Drop rows with missing values
        feature_data = data[features_to_use].dropna()
        
        return feature_data
    
    def fit(self, df, price_col='price'):
        """
        Fit the regime detection model.
        
        Args:
            df (pd.DataFrame): DataFrame with price and features
            price_col (str): Name of the price column
            
        Returns:
            MarketRegimeDetector: Self for method chaining
        """
        # Prepare features
        feature_data = self._prepare_features(df, price_col)
        
        if feature_data.empty:
            raise ValueError("No valid data after preparation.")
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Fit model
        if self.method == 'gmm':
            self.model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        else:  # kmeans
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42)
        
        self.model.fit(scaled_data)
        
        return self
    
    def predict(self, df, price_col='price'):
        """
        Predict regimes for the data.
        
        Args:
            df (pd.DataFrame): DataFrame with price and features
            price_col (str): Name of the price column
            
        Returns:
            pd.DataFrame: Original DataFrame with regime column added
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Prepare features
        feature_data = self._prepare_features(df, price_col)
        
        if feature_data.empty:
            # If we can't calculate features, return original data with NaN regimes
            result['regime'] = np.nan
            return result
        
        # Scale features
        scaled_data = self.scaler.transform(feature_data)
        
        # Predict regimes
        regimes = self.model.predict(scaled_data)
        
        # Create a Series with regimes and the same index as feature_data
        regime_series = pd.Series(regimes, index=feature_data.index)
        
        # Add regime column to result, handling any index mismatches
        result['regime'] = np.nan
        result.loc[regime_series.index, 'regime'] = regime_series
        
        # Forward fill regimes for any timestamps without predictions
        result['regime'] = result['regime'].fillna(method='ffill')
        
        return result
    
    def label_regimes(self, df, price_col='price'):
        """
        Label regimes as bullish, bearish, or neutral based on returns.
        
        Args:
            df (pd.DataFrame): DataFrame with regimes and price
            price_col (str): Name of the price column
            
        Returns:
            dict: Dictionary mapping regime ids to labels
            dict: Dictionary with regime statistics
        """
        if 'regime' not in df.columns:
            raise ValueError("DataFrame does not contain regime column. Run predict() first.")
        
        # Calculate future returns (shifted backwards since we want to know regime -> future return)
        df = df.copy()
        df['future_return'] = df[price_col].pct_change().shift(-1)
        
        # Calculate stats for each regime
        regime_stats = {}
        regime_labels = {}
        
        for regime in sorted(df['regime'].unique()):
            if np.isnan(regime):
                continue
                
            regime_df = df[df['regime'] == regime]
            
            # Skip regimes with too little data
            if len(regime_df) < 5:
                continue
            
            mean_return = regime_df['future_return'].mean()
            std_return = regime_df['future_return'].std()
            
            # Calculate Sharpe ratio (using 0 as risk-free rate for simplicity)
            sharpe = mean_return / std_return if std_return > 0 else 0
            
            # Store stats
            regime_stats[regime] = {
                'mean_return': mean_return,
                'std_return': std_return,
                'sharpe': sharpe,
                'count': len(regime_df)
            }
            
            # Label regimes
            if mean_return > 0 and sharpe > 0.1:
                regime_labels[regime] = 'bullish'
            elif mean_return < 0 and sharpe < -0.1:
                regime_labels[regime] = 'bearish'
            else:
                regime_labels[regime] = 'neutral'
        
        return regime_labels, regime_stats
    
    def plot_regimes(self, df, price_col='price', save_path=None):
        """
        Plot price chart with regime overlay.
        
        Args:
            df (pd.DataFrame): DataFrame with price and regime data
            price_col (str): Name of the price column
            save_path (str): Path to save the figure
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if 'regime' not in df.columns:
            raise ValueError("DataFrame does not contain regime column. Run predict() first.")
        
        # Label regimes
        regime_labels, regime_stats = self.label_regimes(df, price_col)
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax1.plot(df['date'], df[price_col], 'k-', linewidth=1, label='Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        
        # Create twin axis for regimes
        ax2 = ax1.twinx()
        ax2.set_ylabel('Regime')
        
        # Plot regimes with colors
        regimes = [r for r in sorted(df['regime'].unique()) if not np.isnan(r)]
        
        for regime in regimes:
            mask = df['regime'] == regime
            color = 'green' if regime_labels.get(regime) == 'bullish' else 'red' if regime_labels.get(regime) == 'bearish' else 'gray'
            label = f"{regime_labels.get(regime, 'unknown')} ({regime})"
            
            # Plot as a step function
            ax2.fill_between(
                df['date'], 
                0, 
                1, 
                where=mask, 
                alpha=0.3, 
                color=color, 
                step='post',
                label=label
            )
        
        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # Set title with regime info
        title = "Market Regimes\n"
        for regime, label in regime_labels.items():
            stats = regime_stats[regime]
            title += f"Regime {regime} ({label}): Return={stats['mean_return']:.4f}, Sharpe={stats['sharpe']:.2f}, Count={stats['count']}\n"
        
        plt.title(title)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_model(self, filepath=None):
        """
        Save the regime detection model.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            str: Path where the model was saved
        """
        import joblib
        
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if filepath is None:
            # Create a default filepath
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, f"regime_detector_{self.n_regimes}_{self.method}.joblib")
        
        # Save the model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'method': self.method,
            'features': self.features,
            'window_size': self.window_size
        }, filepath)
        
        return filepath
    
    def load_model(self, filepath):
        """
        Load a saved regime detection model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            MarketRegimeDetector: Self for method chaining
        """
        import joblib
        
        # Load the model
        saved_model = joblib.load(filepath)
        
        # Load components
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.n_regimes = saved_model['n_regimes']
        self.method = saved_model['method']
        self.features = saved_model['features']
        self.window_size = saved_model['window_size']
        
        return self 