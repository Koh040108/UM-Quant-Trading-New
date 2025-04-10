"""
Configuration parameters for the trading strategy.
"""

# Data parameters
DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"

# Trading parameters
TRADING_FEE = 0.0006  # 0.06%
MIN_TRADE_FREQUENCY = 0.03  # 3% of data rows

# Model parameters
HMM_STATES = 5  # Number of hidden states for HMM
LOOKBACK_WINDOW = 30  # Days of historical data to consider
TRAIN_TEST_SPLIT = 0.7  # 70% training, 30% testing
FORWARD_TEST_DAYS = 365  # 1 year of forward testing

# Success criteria
MIN_SHARPE_RATIO = 1.8
MAX_DRAWDOWN_LIMIT = -0.4  # -40%

# Cryptocurrencies to analyze
CRYPTOCURRENCIES = ["BTC", "ETH"]

# Data sources and their API configurations
API_CONFIG = {
    "cryptoquant": {
        "api_key": "",  # Add your API key here
        "base_url": "https://api.cryptoquant.com/v1/"
    },
    "glassnode": {
        "api_key": "",  # Add your API key here
        "base_url": "https://api.glassnode.com/v1/"
    },
    "coinglass": {
        "api_key": "",  # Add your API key here
        "base_url": "https://open-api.coinglass.com/api/v3/"
    }
}

# Data intervals (in minutes)
DATA_INTERVALS = {
    "1h": 60,
    "4h": 240,
    "1d": 1440
}

# Default interval to use
DEFAULT_INTERVAL = "4h" 