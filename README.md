# Crypto Trading Alpha Strategy using HMM/ML

## Overview
This project develops a Machine Learning model that analyzes on-chain crypto data to generate alpha trading strategies. It incorporates Hidden Markov Models (HMMs) to identify deterministic patterns in market movements and extract implicit indicators from noisy data.

## Project Structure
- `data/`: Raw and processed datasets
- `src/`: Source code for data processing, model development, and strategy implementation
- `models/`: Saved model files
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `results/`: Trading performance metrics and visualizations

## Requirements
- Data intervals ≤ 1 day
- Trade signal frequency ≥ 3% per data row
- Trading fees: 0.06%
- Success criteria:
  - Sharpe Ratio (SR) ≥ 1.8
  - Maximum Drawdown (MDD) ≥ -40%
  - Trade Frequency ≥ 3% per data row

## Installation
```bash
pip install -r requirements.txt
```

## Data Source
The project uses the cybotrade-datasource library which provides a unified interface to access multiple on-chain data providers including:
- CryptoQuant
- Glassnode
- Coinglass

To use the data source:

1. Obtain a Cybotrade API key
2. Set the API key as an environment variable:
   ```bash
   export CYBOTRADE_API_KEY="your_api_key_here"
   ```
   or pass it directly to the program with the `--cybotrade_api_key` flag

## Implementation Steps
1. Data collection and preprocessing
2. Feature engineering
3. HMM model development
4. Trading strategy implementation
5. Backtesting and optimization
6. Forward testing

## Usage
```bash
# Run with default settings
python src/main.py --crypto BTC --interval 4h

# Fetch fresh data
python src/main.py --crypto BTC --interval 1d --fetch_data

# Run with custom parameters
python src/main.py --crypto ETH --interval 1h --states 7 --threshold 0.001 --save_model

# Specify Cybotrade API key directly
python src/main.py --crypto BTC --interval 4h --cybotrade_api_key "your_api_key_here" --fetch_data
```

## Exploration Notebooks
- `notebooks/hmm_strategy_exploration.ipynb`: Explore the HMM-based trading strategy
- `notebooks/cybotrade_data_exploration.ipynb`: Explore on-chain data using the cybotrade-datasource API 