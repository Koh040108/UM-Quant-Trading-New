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
    LSTM neural network model for time series prediction with attention mechanism.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=1, dropout=0.3):
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
        
        # Batch normalization for input
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM for better sequence understanding
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with skip connections
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Skip connection layers
        self.skip_connection = nn.Linear(hidden_dim * 2, hidden_dim // 2)  # *2 for bidirectional
        
        # Output layer
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
        # Batch norm on features
        batch_size, seq_len, features = x.size()
        x_reshaped = x.reshape(-1, features)
        x_normalized = self.batch_norm(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, features)
        
        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: (batch, seq_len, hidden_dim*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # shape: (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # shape: (batch, hidden_dim*2)
        
        # Skip connection for residual learning
        skip = self.skip_connection(context_vector)
        
        # Fully connected layers
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Add skip connection
        x = x + skip
        
        # Output layer
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
        
    def _prepare_data(self, df, price_col='close', target_col=None, feature_cols=None):
        """
        Prepare data for LSTM model in a memory-efficient way.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Column name for price data
            target_col (str): Target column for prediction (defaults to price_col)
            feature_cols (list): List of feature columns to use (if None, use all numeric columns)
            
        Returns:
            tuple: (X, y) where X is a 3D array of windows and y is target values
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have price_col - prioritize 'close' if available
        if price_col not in df.columns:
            price_alternatives = ['close', 'price', 'adjclose', 'adjusted_close']
            for alt in price_alternatives:
                if alt in df.columns:
                    print(f"Price column '{price_col}' not found. Using '{alt}' instead.")
                    price_col = alt
                    break
            
            # If still not found, check case-insensitive match
            if price_col not in df.columns:
                for col in df.columns:
                    if col.lower() in ['close', 'price', 'adjclose', 'adjusted_close']:
                        print(f"Using '{col}' as price column (case-insensitive match)")
                        price_col = col
                        break
        
        # Store the detected price column for consistency
        self.detected_price_col = price_col
        
        # Set target to be the same as price_col if not specified
        if target_col is None:
            target_col = price_col
        
        # If no features specified, use all numeric columns except target and date-related columns
        if feature_cols is None:
            # Exclude date-related columns and the target
            exclude_cols = [col for col in df.columns if any(
                date_str in col.lower() for date_str in ['date', 'time', 'timestamp'])]
            
            # Also exclude the target column if different from price_col
            if target_col != price_col:
                exclude_cols.append(target_col)
            
            # Get numeric columns (excluding exclusions)
            feature_cols = [col for col in df.select_dtypes(include=['float64', 'int']).columns 
                          if col not in exclude_cols]
        
        # Verify we have data to work with
        if len(feature_cols) == 0:
            print("No feature columns selected. Using price_col as the only feature.")
            feature_cols = [price_col]
        
        # Save feature columns for later use in prediction
        self.feature_cols = feature_cols.copy()
        
        # Use target_pct_change to predict percentage change
        df['target_pct_change'] = df[target_col].pct_change(1).shift(-1)
        
        # Check if we have enough data for the window size
        if len(df) <= self.window_size:
            print(f"Warning: Not enough data for window size {self.window_size}. Using smaller window.")
            self.window_size = max(5, len(df) // 2)
        
        # Prepare feature and target data
        # Ensure all data is numeric and finite
        data = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        targets = df['target_pct_change'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # For memory efficiency, use float32 instead of float64
        data = data.astype(np.float32)
        targets = targets.astype(np.float32)
        
        print(f"Preparing LSTM data with {len(feature_cols)} features and window size {self.window_size}")
        print(f"Memory usage: {data.memory_usage().sum() / 1e6:.2f} MB for features")

        # Create sequences using a generator approach to save memory
        n_samples = len(data) - self.window_size
        
        # Only create a subset of data if the dataset is large
        MAX_SAMPLES = 10000  # Maximum samples to use for training
        if n_samples > MAX_SAMPLES:
            print(f"Dataset is large ({n_samples} samples). Using {MAX_SAMPLES} samples to conserve memory.")
            # Choose evenly spaced indices
            indices = np.linspace(0, n_samples-1, MAX_SAMPLES, dtype=int)
        else:
            indices = np.arange(n_samples)
        
        # Create batch generator instead of loading all data at once
        def batch_generator(batch_size=128):
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = []
                batch_y = []
                
                for idx in batch_indices:
                    # Create window
                    window = data.iloc[idx:idx+self.window_size].values
                    target = targets.iloc[idx+self.window_size-1]
                    
                    batch_X.append(window)
                    batch_y.append(target)
                
                yield np.array(batch_X), np.array(batch_y)
        
        # Create small sample for validation and testing
        sample_X = []
        sample_y = []
        
        # Just use the first 1000 samples for validation
        for idx in indices[:min(1000, len(indices))]:
            window = data.iloc[idx:idx+self.window_size].values
            target = targets.iloc[idx+self.window_size-1]
            
            sample_X.append(window)
            sample_y.append(target)
        
        X_sample = np.array(sample_X)
        y_sample = np.array(sample_y)
        
        return batch_generator, X_sample, y_sample, feature_cols
    
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
            # Reshape y to 2D array for scaling
            y = y.reshape(-1, 1)
            y_scaled = self.target_scaler.fit_transform(y)
        elif y is not None and self.output_dim == 1:
            # Reshape y to 2D array for scaling
            y = y.reshape(-1, 1)
            y_scaled = self.target_scaler.transform(y)
        else:
            y_scaled = y
            
        return X_scaled, y_scaled
    
    def _build_model(self, input_shape):
        """
        Build and compile the LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (window_size, n_features)
        """
        # Use PyTorch backend only
        self.backend = 'pytorch'
        print("Using PyTorch backend for LSTM model")
        
        # PyTorch implementation
        print(f"Building PyTorch LSTM model with input shape {input_shape}")
        input_dim = input_shape[1]  # Number of features
        
        # Initialize model
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"PyTorch LSTM model created with {input_dim} input features")
        
        return self.model

    def fit(self, df, price_col='close', target_col=None, epochs=150, patience=15, batch_size=128, no_tuning=False):
        """
        Fit the LSTM model to the data.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Column name for price data
            target_col (str): Target column for prediction (defaults to price_col)
            epochs (int): Maximum number of training epochs
            patience (int): Patience for early stopping
            batch_size (int): Batch size for training
            no_tuning (bool): If True, use default hyperparameters without tuning
            
        Returns:
            self: Fitted model
        """
        # If no_tuning is True, use simpler default parameters
        if no_tuning:
            print("Using simplified LSTM configuration (no tuning)")
            # Reduce complexity for faster training
            self.hidden_dim = 32
            self.num_layers = 2
            self.dropout = 0.2
            epochs = 50
            patience = 5
            batch_size = 256
        
        # Prepare data for PyTorch
        train_loader, val_loader, input_dim = self._prepare_pytorch_data(df, price_col, target_col)
        
        if train_loader is None or val_loader is None:
            print("Error preparing data for LSTM. Check your data format and try again.")
            return self
        
        print(f"Training LSTM with window size {self.window_size}")
        
        try:
            # Get input shape for model building
            # Instead of using indexing, iterate through the loader to get the first batch
            first_batch = next(iter(train_loader))
            X_batch, y_batch = first_batch
            _, _, n_features = X_batch.shape
            input_shape = (self.window_size, n_features)
            
            # Build model if needed
            if self.model is None:
                self._build_model(input_shape)
            
            # Scale data
            X_scaled, y_scaled = self._scale_data(X_batch.numpy(), y_batch.numpy(), is_train=True)
            
            # Reshape scaled data back to 3D
            X_scaled = X_scaled.reshape(-1, self.window_size, n_features)
            
            # Split into train and validation
            validation_split = 0.2
            split_idx = int(len(X_scaled) * (1 - validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
            
            # Create sample weights that emphasize more recent data
            # Newer samples (closer to split_idx) get higher weights
            sample_weights = np.linspace(0.5, 1.0, len(X_train))
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Convert sample weights to PyTorch tensor
            weights_tensor = torch.FloatTensor(sample_weights).to(self.device)
            
            # Create a weighted random sampler
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(X_train),
                replacement=True
            )
            
            # Create dataloaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Learning rate scheduler - Reduce LR on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=patience//3,
                verbose=True
            )
            
            # Initialize early stopping variables
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model = None
            
            print(f"Starting PyTorch LSTM training: {epochs} max epochs, batch size {batch_size}, patience {patience}")
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(X_batch)
                    
                    # Calculate loss
                    loss = criterion(outputs, y_batch)
                    
                    # Backward pass and optimization
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Update learning rate scheduler
                scheduler.step(avg_val_loss)
                
                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1 or avg_val_loss < best_val_loss:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model = self.model.state_dict()
                    print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.6f}")
                        break
            
            # Load the best model
            if best_model is not None:
                self.model.load_state_dict(best_model)
                print(f"PyTorch LSTM training complete. Final best validation loss: {best_val_loss:.6f}")
            
            # Save feature columns for prediction
            if 'feature_cols' not in locals():
                self.feature_cols = [col for col in df.select_dtypes(include=['float64', 'int']).columns 
                                  if not any(date_str in col.lower() for date_str in ['date', 'time', 'timestamp'])]
            
        except Exception as e:
            print(f"Error during PyTorch training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        return self

    def _prepare_pytorch_data(self, df, price_col='close', target_col=None, feature_cols=None):
        """
        Prepare data for PyTorch LSTM model in a memory-efficient way.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Column name for price data
            target_col (str): Target column for prediction (defaults to price_col)
            feature_cols (list): List of feature columns to use (if None, use all numeric columns)
            
        Returns:
            tuple: (train_loader, val_loader, input_dim) where train_loader and val_loader are DataLoader objects
                  and input_dim is the number of input features
        """
        try:
            # Add enhanced features for LSTM
            print("Adding enhanced features for LSTM...")
            data = self._add_enhanced_features(df.copy(), price_col)
            
            # Use target_pct_change to predict percentage change
            if 'target' not in data.columns:
                data['target'] = data[price_col].pct_change(1).shift(-1)
                data['target'].fillna(0, inplace=True)  # Fill NaN values
            
            # Drop rows with NaN values
            data = data.dropna()
            
            # If no features specified, use all numeric columns except target and date-related columns
            if feature_cols is None:
                # Exclude date-related columns and the target
                exclude_cols = ['date', 'timestamp', 'time', 'target'] + [col for col in data.columns if 'date' in col.lower()]
                
                # Get numeric columns (excluding exclusions)
                feature_cols = [col for col in data.select_dtypes(include=['float64', 'int64', 'float32']).columns 
                              if col not in exclude_cols and col != price_col]
            
            print(f"Using {len(feature_cols)} features including enhanced trend and volatility indicators")
            
            # Create sequences
            X = data[feature_cols].values
            y = data['target'].values
            
            # Scale the data
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = y  # No need to scale binary targets
                        
            # Create sequences
            seq_X, seq_y = [], []
            for i in range(len(X_scaled) - self.window_size):
                seq_X.append(X_scaled[i:i+self.window_size])
                seq_y.append(y_scaled[i+self.window_size-1])
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(np.array(seq_X))
            y_tensor = torch.FloatTensor(np.array(seq_y))
            
            # Create TensorDataset
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Split into training and validation sets
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            # Return train and validation loaders along with input dimension
            input_dim = X_scaled.shape[1]  # Number of features
            
            # Save feature columns for later use in prediction
            self.feature_cols = feature_cols
            
            return train_loader, val_loader, input_dim
            
        except Exception as e:
            print(f"Error preparing PyTorch data: {str(e)}")
            # Return empty DataLoader and input_dim as fallback
            input_dim = 10  # Default fallback
            empty_dataset = TensorDataset(
                torch.FloatTensor(np.zeros((1, self.window_size, input_dim))),
                torch.FloatTensor(np.zeros(1))
            )
            empty_loader = DataLoader(empty_dataset, batch_size=1)
            return empty_loader, empty_loader, input_dim

    def _add_enhanced_features(self, df, price_col='close'):
        """
        Add enhanced features specifically for LSTM model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            price_col (str): Price column name
            
        Returns:
            pd.DataFrame: DataFrame with enhanced features
        """
        data = df.copy()
        
        # Technical indicators
        if price_col in data.columns:
            # RSI with multiple windows
            if 'rsi' not in data.columns:
                windows = [7, 14, 21]
                for window in windows:
                    delta = data[price_col].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=window).mean()
                    avg_loss = loss.rolling(window=window).mean()
                    rs = avg_gain / avg_loss
                    data[f'rsi_custom_{window}'] = 100 - (100 / (1 + rs))
                    
            # Custom momentum features
            if 'momentum_5d' not in data.columns:
                # Price momentum
                for window in [5, 10, 15]:
                    data[f'momentum_{window}d'] = data[price_col].pct_change(window)
                
                # Volatility calculations
                for window in [5, 10, 20]:
                    data[f'volatility_{window}d'] = data[price_col].pct_change().rolling(window).std()
                
                # Directional indicators
                data['direction'] = np.where(data[price_col].diff() > 0, 1, -1)
                data['direction_ma_5'] = data['direction'].rolling(5).mean()
                data['direction_ma_10'] = data['direction'].rolling(10).mean()
                
                # Trend strength (ADX-like)
                data['trend_strength'] = abs(data['direction_ma_10']) * data['volatility_10d']
                
                # Relative strength to market (if available)
                if 'market_price' in data.columns:
                    data['rel_strength'] = data[price_col].pct_change() / data['market_price'].pct_change()
                    data['rel_strength_ma_5'] = data['rel_strength'].rolling(5).mean()
        
        # Fill missing values
        # Use more sophisticated methods depending on the feature type
        for col in data.columns:
            if 'rsi' in col:
                data[col].fillna(50, inplace=True)  # Default to neutral
            elif 'direction' in col:
                data[col].fillna(0, inplace=True)  # Default to neutral
            elif 'volatility' in col or 'momentum' in col:
                data[col].fillna(0, inplace=True)  # Default to no volatility/momentum
                
        # Forward-fill any remaining NaNs, then backward-fill if needed
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        
        return data

    def generate_trading_signals(self, df, signal_threshold=0.01, percentile_based=True, min_trading_freq=0.03):
        """
        Generate trading signals from LSTM predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with LSTM predictions
            signal_threshold (float): Threshold for generating buy/sell signals when not using percentiles
            percentile_based (bool): Whether to use percentile-based thresholds instead of absolute values
            min_trading_freq (float): Minimum trading frequency to enforce (as percentage of data rows)
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Check if lstm_pred column exists
        if 'lstm_pred' not in result.columns:
            print("Error: No LSTM predictions found. Cannot generate signals.")
            result['signal'] = 0
            return result
        
        # Generate signals based on prediction values
        result['signal'] = 0
        
        if percentile_based:
            # Calculate percentiles based on prediction distribution
            top_percentile = max(100 - min_trading_freq * 50, 95)  # More selective: top 5% for buy signals
            bottom_percentile = min(min_trading_freq * 50, 5)      # More selective: bottom 5% for sell signals
            
            buy_threshold = np.percentile(result['lstm_pred'].dropna(), top_percentile)
            sell_threshold = np.percentile(result['lstm_pred'].dropna(), bottom_percentile)
            
            # Use calculated percentiles for thresholds
            print(f"Using percentile-based thresholds: Buy > {buy_threshold:.6f} (top {100-top_percentile:.1f}%), "
                  f"Sell < {sell_threshold:.6f} (bottom {bottom_percentile:.1f}%)")
            
            # Buy when predicted return is above top percentile
            result.loc[result['lstm_pred'] >= buy_threshold, 'signal'] = 1
            
            # Sell when predicted return is below bottom percentile
            result.loc[result['lstm_pred'] <= sell_threshold, 'signal'] = -1
        else:
            # Use dynamic threshold if predictions are very skewed
            pred_std = result['lstm_pred'].std()
            if pred_std < 0.005:  # Very small variation in predictions
                dynamic_threshold = max(0.001, pred_std * 2)
                print(f"Using dynamic threshold {dynamic_threshold:.6f} based on prediction std dev {pred_std:.6f}")
                signal_threshold = dynamic_threshold
                
            # Buy when predicted return is above threshold
            result.loc[result['lstm_pred'] >= signal_threshold, 'signal'] = 1
            
            # Sell when predicted return is below negative threshold
            result.loc[result['lstm_pred'] <= -signal_threshold, 'signal'] = -1
        
        # For compatibility with hybrid model, also keep lstm_signal column
        result['lstm_signal'] = result['signal']
        
        # Calculate signal statistics
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        neutral_signals = (result['signal'] == 0).sum()
        total_signals = len(result)
        trading_frequency = (buy_signals + sell_signals) / total_signals
        
        print(f"\nLSTM Signal Statistics:")
        print(f"Buy Signals: {buy_signals} ({buy_signals/total_signals:.2%} of data)")
        print(f"Sell Signals: {sell_signals} ({sell_signals/total_signals:.2%} of data)")
        print(f"Neutral Signals: {neutral_signals} ({neutral_signals/total_signals:.2%} of data)")
        print(f"Trading Frequency: {trading_frequency:.2%}")
        
        # Check if we meet minimum trading frequency
        if trading_frequency < min_trading_freq and min_trading_freq > 0:
            print(f"Warning: Trading frequency {trading_frequency:.2%} is below target {min_trading_freq:.2%}")
            if percentile_based:
                print(f"Already using percentile-based thresholds, can't auto-adjust further")
            else:
                # Adjust thresholds to increase trading frequency
                print(f"Switching to percentile-based thresholds to ensure minimum trading frequency")
                return self.generate_trading_signals(df, percentile_based=True, min_trading_freq=min_trading_freq)
        
        return result

    def predict(self, df, price_col='close'):
        """
        Generate predictions from the LSTM model.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            price_col (str): Column name for price
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure the model is built
        if self.model is None:
            print("Error: Model not trained yet")
            return result
        
        # Use the detected price column from training if available
        if hasattr(self, 'detected_price_col'):
            if price_col != self.detected_price_col:
                print(f"Note: Using price column '{self.detected_price_col}' from training instead of '{price_col}'")
                price_col = self.detected_price_col
        # Fallback logic if no detected price column
        elif price_col not in result.columns:
            # Try common price column names
            price_alternatives = ['close', 'price', 'adjclose', 'adjusted_close']
            for alt in price_alternatives:
                if alt in result.columns:
                    print(f"Price column '{price_col}' not found. Using '{alt}' instead.")
                    price_col = alt
                    break
        
        # Add the same enhanced features we used in training
        print("Adding enhanced features for LSTM prediction...")
        
        try:
            # Price momentum and trend indicators
            result['short_ma'] = result[price_col].rolling(window=5).mean()
            result['medium_ma'] = result[price_col].rolling(window=15).mean()
            result['long_ma'] = result[price_col].rolling(window=30).mean()
            
            # Trend directions (1 for uptrend, -1 for downtrend, 0 for sideways)
            result['short_trend'] = np.where(result['short_ma'] > result['short_ma'].shift(1), 1, 
                                        np.where(result['short_ma'] < result['short_ma'].shift(1), -1, 0))
            result['medium_trend'] = np.where(result['medium_ma'] > result['medium_ma'].shift(1), 1, 
                                          np.where(result['medium_ma'] < result['medium_ma'].shift(1), -1, 0))
            result['long_trend'] = np.where(result['long_ma'] > result['long_ma'].shift(1), 1, 
                                        np.where(result['long_ma'] < result['long_ma'].shift(1), -1, 0))
        except Exception as e:
            print(f"Warning: Error adding trend indicators: {str(e)}")
        
        # If we don't have feature columns defined yet, use all numeric columns
        if not hasattr(self, 'feature_cols') or not self.feature_cols:
            print("Warning: feature_cols not set. Using all numeric features including enhanced ones.")
            # Remove date and other non-numeric columns
            exclude_cols = [col for col in result.columns if any(
                date_str in col.lower() for date_str in ['date', 'time', 'timestamp'])]
            
            # Get all numeric columns
            self.feature_cols = [col for col in result.select_dtypes(include=['float64', 'float32', 'int']).columns 
                               if col not in exclude_cols]
        
        # Prepare data for prediction
        data = result[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        
        # Create sequences for prediction
        X = []
        valid_indices = []
        
        for i in range(len(data) - self.window_size + 1):
            window = data.iloc[i:i+self.window_size].values
            X.append(window)
            valid_indices.append(i + self.window_size - 1)
        
        if not X:
            print("Error: Not enough data for prediction")
            return result
        
        # Convert to numpy array
        X = np.array(X)
        
        print(f"Making PyTorch predictions with shape {X.shape}")
        
        # Get the number of features from X
        _, _, n_features_X = X.shape
        
        # If the feature scaler exists, get its expected number of features
        n_features_expected = self.feature_scaler.n_features_in_ if hasattr(self.feature_scaler, 'n_features_in_') else None
        
        # Handle feature mismatch
        if n_features_expected is not None and n_features_X != n_features_expected:
            print(f"Warning: Feature mismatch detected. Model expects {n_features_expected} features, but got {n_features_X}.")
            print("Attempting to pad or trim features to match...")
            
            # Create a new array with the expected number of features
            X_adjusted = np.zeros((X.shape[0], X.shape[1], n_features_expected))
            
            # Copy available features
            min_features = min(n_features_X, n_features_expected)
            X_adjusted[:, :, :min_features] = X[:, :, :min_features]
            
            if hasattr(self, 'feature_names') and len(self.feature_names) == n_features_expected:
                print(f"Missing features: {self.feature_names[min_features:]}")
            
            # Use the adjusted X for prediction
            X = X_adjusted
            print(f"Adjusted input shape to {X.shape}")
        
        # Scale features
        try:
            X_scaled, _ = self._scale_data(X, is_train=False)
            
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            
            # Add predictions to result DataFrame
            result['lstm_pred'] = np.nan
            
            # Use iloc instead of loc for integer-based indexing
            for i, pred_idx in enumerate(valid_indices):
                if pred_idx < len(result):
                    # Handle both 2D array predictions[i][0] and flattened predictions[i]
                    pred_value = predictions[i][0] if predictions.ndim > 1 else predictions[i]
                    result.iloc[pred_idx, result.columns.get_loc('lstm_pred')] = pred_value
        except Exception as e:
            print(f"Error during PyTorch prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Filling predictions with zeros")
            result['lstm_pred'] = 0
        
        # Generate trading signals using percentile-based thresholds for better performance
        result = self.generate_trading_signals(result, percentile_based=True, min_trading_freq=0.05)
        
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
        buy_signals = df[df['signal'] == 1]
        plt.scatter(buy_signals.index, buy_signals[price_col], 
                   color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = df[df['signal'] == -1]
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