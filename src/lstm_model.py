"""
LSTM model for sequence-based price prediction and trading signal generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    """
    LSTM neural network model for time series prediction.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, output_dim=1, dropout=0.35):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in LSTM layers
            num_layers (int): Number of LSTM layers
            output_dim (int): Output dimension (1 for regression, 3 for buy/hold/sell classification)
            dropout (float): Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        out = self.sigmoid(x)
        
        return out


class LSTMPredictor:
    """
    LSTM-based predictor for trading signals.
    """
    
    def __init__(self, window_size=30, hidden_dim=64, num_layers=4, output_dim=1, dropout=0.35, learning_rate=0.001, weight_decay=1e-5, device=None):
        """
        Initialize the LSTM predictor.
        
        Args:
            window_size (int): Size of the lookback window for LSTM
            hidden_dim (int): Number of hidden units in LSTM layers
            num_layers (int): Number of LSTM layers
            output_dim (int): Output dimension (1 for regression, 3 for buy/hold/sell classification)
            dropout (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for regularization
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler() if output_dim == 1 else None
        
    def _prepare_data(self, df, price_col='price', target_col=None, feature_cols=None):
        """
        Prepare data for LSTM training/prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with price and features
            price_col (str): Name of the price column
            target_col (str, optional): Name of the target column, defaults to price_col
            feature_cols (list, optional): List of feature columns to use
            
        Returns:
            tuple: (X, y) where X is input sequences and y is target values
        """
        # Create a copy of the DataFrame
        data = df.copy()
        
        # Set target column
        if target_col is None:
            target_col = price_col
        
        # Select features
        if feature_cols is None:
            # Remove date and other non-numeric columns
            exclude_cols = ['date', 'timestamp']
            feature_cols = [col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])]
        
        # Ensure target column is included in features for sequence prediction
        if target_col not in feature_cols:
            feature_cols.append(target_col)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.window_size):
            # Get window of data
            window = data[feature_cols].iloc[i:i+self.window_size].values
            
            # Get target (next price or signal)
            target = data[target_col].iloc[i+self.window_size]
            
            X.append(window)
            y.append([target])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def _scale_data(self, X, y=None, is_train=True):
        """
        Scale the data using MinMaxScaler.
        
        Args:
            X (np.array): Input sequences
            y (np.array, optional): Target values
            is_train (bool): Whether this is training data
            
        Returns:
            tuple: (X_scaled, y_scaled) scaled data
        """
        # Get number of samples and features
        n_samples, n_timesteps, n_features = X.shape
        
        # Reshape X to 2D for scaling
        X_reshaped = X.reshape(-1, n_features)
        
        # Scale features
        if is_train:
            X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.feature_scaler.transform(X_reshaped)
        
        # Reshape back to 3D
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale target values if regression
        if y is not None and self.output_dim == 1 and is_train:
            y_scaled = self.target_scaler.fit_transform(y)
        elif y is not None and self.output_dim == 1:
            y_scaled = self.target_scaler.transform(y)
        else:
            y_scaled = y
            
        return X_scaled, y_scaled
    
    def fit(self, df, price_col='price', target_col=None, feature_cols=None, epochs=200, batch_size=32, validation_split=0.2, patience=20):
        """
        Fit the LSTM model to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Name of the price column
            target_col (str, optional): Name of the target column, defaults to price_col
            feature_cols (list, optional): List of feature columns to use
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            patience (int): Patience for early stopping
            
        Returns:
            self: Fitted model
        """
        # Prepare data
        X, y = self._prepare_data(df, price_col, target_col, feature_cols)
        
        # Set input dimension based on data
        input_dim = X.shape[2]
        
        # Initialize model
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Scale data
        X_scaled, y_scaled = self._scale_data(X, y, is_train=True)
        
        # Split into training and validation sets
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss() if self.output_dim == 1 else nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Early stopping
        best_val_loss = float('inf')
        no_improve_epochs = 0
        best_model_state = None
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print("Training LSTM model...")
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_model_state = self.model.state_dict().copy()
            else:
                no_improve_epochs += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check early stopping
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return self
    
    def predict(self, df, price_col='price', feature_cols=None, signal_threshold=0.5):
        """
        Generate predictions from the trained LSTM model.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Name of the price column
            feature_cols (list, optional): List of feature columns to use
            signal_threshold (float): Threshold for generating buy/sell signals
            
        Returns:
            pd.DataFrame: DataFrame with predictions and signals
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare data
        X, _ = self._prepare_data(df, price_col, None, feature_cols)
        
        # Scale features
        X_scaled, _ = self._scale_data(X, is_train=False)
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions if regression
        if self.output_dim == 1 and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)
        
        # Create result DataFrame
        # Add window_size to align predictions with original data
        result = df.iloc[self.window_size:].copy().reset_index(drop=True)
        
        # Add prediction column
        result['lstm_pred'] = predictions
        
        # Generate signals based on prediction values
        result['lstm_signal'] = 0
        result.loc[result['lstm_pred'] >= signal_threshold, 'lstm_signal'] = 1
        result.loc[result['lstm_pred'] < signal_threshold, 'lstm_signal'] = -1
        
        return result
    
    def generate_trading_signals(self, df):
        """
        Generate trading signals from LSTM predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with LSTM predictions
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Signals are already in lstm_signal column
        # Rename lstm_signal to signal for compatibility with backtesting
        if 'lstm_signal' in result.columns:
            result['signal'] = result['lstm_signal']
            
        return result
    
    def plot_predictions(self, df, price_col='price'):
        """
        Plot LSTM predictions and signals against actual prices.
        
        Args:
            df (pd.DataFrame): DataFrame with predictions and signals
            price_col (str): Name of the price column
        """
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Price and Predictions
        plt.subplot(2, 1, 1)
        plt.plot(df[price_col], label='Actual Price', color='blue')
        plt.title('LSTM Model Predictions')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Price and Trading Signals
        plt.subplot(2, 1, 2)
        plt.plot(df[price_col], label='Price', color='blue', alpha=0.5)
        
        # Plot buy signals
        buy_signals = df[df['lstm_signal'] == 1]
        plt.scatter(buy_signals.index, buy_signals[price_col], 
                   color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = df[df['lstm_signal'] == -1]
        plt.scatter(sell_signals.index, sell_signals[price_col], 
                   color='red', marker='v', s=100, label='Sell Signal')
        
        plt.title('LSTM Trading Signals')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'window_size': self.window_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout
        }, filepath)
        
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath, input_dim):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to load the model from
            input_dim (int): Number of input features
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model parameters
        self.window_size = checkpoint['window_size']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.output_dim = checkpoint['output_dim']
        self.dropout = checkpoint['dropout']
        
        # Initialize model
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scalers
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        
        print(f"Model loaded from {filepath}") 