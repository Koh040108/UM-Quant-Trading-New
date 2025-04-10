# Crypto Trading Alpha Strategy using HMM/ML

## Overview
This project develops a Machine Learning model that analyzes on-chain crypto data to generate alpha trading strategies. It incorporates Hidden Markov Models (HMMs) to identify deterministic patterns in market movements and extract implicit indicators from noisy data.

## Features
- Automated data fetching from on-chain sources
- Robust feature engineering with technical indicators
- Hidden Markov Model for regime detection
- Advanced backtesting with fee modeling
- Interactive visualizations and HTML reports
- Outlier detection and handling
- Performance metrics against industry benchmarks

## Project Structure
- `data/`: Raw and processed datasets
- `src/`: Source code for data processing, model development, and strategy implementation
  - `cybotrade_fetcher.py`: Data fetching from on-chain sources
  - `feature_engineering.py`: Feature creation and preprocessing
  - `hmm_model.py`: HMM model implementation
  - `visualization.py`: Data visualization and dashboard creation
  - `main.py`: Main execution script
  - `config.py`: Configuration parameters
- `models/`: Saved model files
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `results/`: Trading performance metrics, HTML reports, and visualizations

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
   or create a `.env` file in the project root:
   ```
   CYBOTRADE_API_KEY=your_api_key_here
   ```

## Implementation Steps
1. Data collection and preprocessing
2. Feature engineering
3. HMM model development
4. Trading strategy implementation
5. Backtesting and optimization
6. Performance visualization and reporting

## Usage
```bash
# Run with default settings
python src/main.py --crypto BTC --interval 4h

# Refresh data from APIs
python src/main.py --crypto BTC --interval 1d --refresh_data

# Run with custom parameters
python src/main.py --crypto ETH --interval 1h --states 7 --threshold 0.001 --save_model

# Specify Cybotrade API key directly
python src/main.py --crypto BTC --interval 4h --cybotrade_api_key "your_api_key_here" --refresh_data

# Run without generating visualizations
python src/main.py --crypto BTC --interval 4h --skip_plots
```

## Running with Existing Data
If you don't have an API key or want to quickly test the model with existing data:

```bash
python run_model_with_data.py
```

This script uses the data already in the `data/` directory without attempting to fetch new data.

## Visualization
The project generates comprehensive visualization dashboards:

- **Price and HMM States**: Visualizes the detected market regimes
- **Trading Signals**: Shows buy/sell points on the price chart
- **Portfolio Growth**: Tracks strategy performance vs buy & hold
- **Performance Metrics**: Bar charts of key performance indicators

All visualizations are saved as both PNG files and combined into an HTML report for easy sharing.

## Exploration Notebooks
- `notebooks/hmm_strategy_exploration.ipynb`: Explore the HMM-based trading strategy
- `notebooks/cybotrade_data_exploration.ipynb`: Explore on-chain data using the cybotrade-datasource API
- `notebooks/visualization_examples.ipynb`: Examples of the visualization capabilities 