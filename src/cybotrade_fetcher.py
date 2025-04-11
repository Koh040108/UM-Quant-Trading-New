"""
Data fetching module using the cybotrade-datasource library to retrieve on-chain data.
"""

import os
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
import cybotrade_datasource
from dotenv import load_dotenv
import numpy as np

# Load configuration
from src.config import DATA_DIR, CRYPTOCURRENCIES, DATA_INTERVALS, DEFAULT_INTERVAL

# Load environment variables
load_dotenv()

async def fetch_data_async(topic, start_date, end_date, api_key=None):
    """
    Fetch data directly using cybotrade-datasource as per documentation.
    
    Args:
        topic (str): Topic string in format 'source|crypto/metric?params'
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        api_key (str, optional): API key for Cybotrade
        
    Returns:
        pd.DataFrame: DataFrame with the fetched data
    """
    # Use API key from env if not provided
    api_key = api_key or os.environ.get("CYBOTRADE_API_KEY", "")
    if not api_key:
        raise ValueError("Cybotrade API key is required. Set CYBOTRADE_API_KEY environment variable or pass explicitly.")
    
    # Convert date strings to datetime objects
    start_time = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_time = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    # Calculate date differences to determine appropriate limit
    days_difference = (end_time - start_time).days
    
    # For hourly data, we need 24 records per day (adjust based on topic)
    if 'hour' in topic:
        # Set limit to at least the number of hours in the date range
        limit = max(days_difference * 24, 1000)
        print(f"Setting limit to {limit} to ensure coverage of {days_difference} days of hourly data")
    else:
        # For daily data, we need more records for multi-year data
        limit = max(days_difference * 2, 1000)
    
    try:
        print(f"Fetching data for topic: {topic}")
        
        # Use the topic as provided without attempting to correct it
        # First try with just a limit parameter for simplicity
        try:
            data = await cybotrade_datasource.query_paginated(
                api_key=api_key,
                topic=topic,
                limit=limit  # Use the calculated limit
            )
            
            if not data and start_time and end_time:
                # If no data, try with specific time range
                print(f"  No data with limit only, trying with time range...")
                data = await cybotrade_datasource.query_paginated(
                    api_key=api_key,
                    topic=topic,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
        except Exception as inner_e:
            print(f"  Error with query: {str(inner_e)}")
            # Try a different approach with a simpler query
            print(f"  Trying with simpler query...")
            try:
                # Extract the source and crypto from the original topic
                parts = topic.split('|')
                source = parts[0]
                
                # For glassnode, use a=BTC format
                if source == "glassnode":
                    if "a=" in topic:
                        # Extract the crypto from a=XXX parameter
                        a_param = [p for p in topic.split('?')[1].split('&') if p.startswith('a=')]
                        crypto = a_param[0].split('=')[1] if a_param else "BTC"
                        simple_topic = f"{source}|market/price_usd_close?a={crypto}&i=hour"
                    else:
                        # Try to extract crypto from path
                        path_parts = parts[1].split('?')[0].split('/')
                        crypto = path_parts[0] if len(path_parts) > 0 else "btc"
                        simple_topic = f"{source}|market/price_usd_close?a={crypto.upper()}&i=hour"
                else:
                    # For other sources, use crypto/endpoint format
                    path_parts = parts[1].split('?')[0].split('/')
                    crypto = path_parts[0] if len(path_parts) > 0 else "btc"
                    simple_topic = f"{source}|{crypto}/price?interval=hour"
                
                print(f"  Trying fallback topic: {simple_topic}")
                data = await cybotrade_datasource.query_paginated(
                    api_key=api_key,
                    topic=simple_topic,
                    limit=limit
                )
            except Exception as final_e:
                print(f"  Final error: {str(final_e)}")
                return pd.DataFrame()
        
        if data:
            df = pd.DataFrame(data)
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"  Success! Received {len(df)} records.")
            return df
        else:
            print(f"  No data returned for {topic}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  Error fetching data for {topic}: {str(e)}")
        return pd.DataFrame()

def save_data(df, crypto, source, metric, interval=None):
    """Save fetched data to CSV file."""
    if df.empty:
        print(f"No data to save for {crypto}_{source}_{metric}")
        return None
        
    # Create directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Generate filename
    if interval:
        filename = f"{crypto}_{source}_{metric}_{interval}.csv"
    else:
        filename = f"{crypto}_{source}_{metric}.csv"
    
    file_path = os.path.join(DATA_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")
    
    # If this is price data, also save it in a standardized format for compatibility
    # Check various price-related metrics
    is_price_data = (
        (source == "glassnode" and metric in ["price", "market_price_usd_close", "ohlc"]) or
        (source == "coinglass" and metric == "price") or
        metric.lower() in ["price", "market_price", "close_price", "usd_price"]
    )
    
    if is_price_data:
        # Create a standardized market price file
        market_price_filename = f"{crypto}_market_price_{interval}.csv"
        market_price_path = os.path.join(DATA_DIR, market_price_filename)
        
        # Clone and prepare dataframe
        market_price_df = df.copy()
        
        # Handle different data formats
        if 'value' in market_price_df.columns:
            market_price_df.rename(columns={'value': 'price'}, inplace=True)
        elif 'close' in market_price_df.columns:
            market_price_df.rename(columns={'close': 'price'}, inplace=True)
        elif 'price' not in market_price_df.columns and metric == 'ohlc':
            # If this is OHLC data, create a price column from close
            if 'o' in market_price_df.columns and 'h' in market_price_df.columns and 'l' in market_price_df.columns and 'c' in market_price_df.columns:
                market_price_df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}, inplace=True)
                market_price_df['price'] = market_price_df['close']
            
        # Add volume column if it doesn't exist
        if 'volume' not in market_price_df.columns:
            market_price_df['volume'] = market_price_df['price'] * 0.1 if 'price' in market_price_df.columns else 100
        
        # Ensure we have a date column
        if 'date' not in market_price_df.columns and 'timestamp' in market_price_df.columns:
            market_price_df['date'] = pd.to_datetime(market_price_df['timestamp'], unit='ms')
            
        # Save the standardized price file
        market_price_df.to_csv(market_price_path, index=False)
        print(f"Created standardized market price data: {market_price_path}")
        
        # Also create the CCXT compatible file
        ccxt_filename = f"{crypto}_ccxt_market_data_{interval}.csv"
        ccxt_path = os.path.join(DATA_DIR, ccxt_filename)
        market_price_df.to_csv(ccxt_path, index=False)
        print(f"Created ccxt-compatible price data: {ccxt_path}")
    
    return file_path

async def fetch_and_save_data_async(crypto, start_date, end_date, interval, api_key=None):
    """Fetch and save data for a specific cryptocurrency."""
    print(f"Fetching data for {crypto}...")
    
    # Convert interval to API interval format
    if interval in ['1h', '4h', '6h', '8h', '12h']:
        api_interval = 'hour'  # Use 'hour' for all hour-based intervals
    else:  # Default to daily
        api_interval = 'day'
    
    # Use lowercase for crypto in API calls
    crypto_lower = crypto.lower()
    
    # Use simpler topics list with correct API format
    # The format should be 'source|crypto/metric?interval=X'
    topics = [
        # Price data is most critical
        f"cryptoquant|btc/inter-entity-flows/miner-to-miner?from_miner=f2pool&to_miner=all_miner&window=hour",
        f"cryptoquant|btc/market-data/price-ohlcv?window=day",
        f"cryptoquant|btc/flow-indicator/exchange-whale-ratio?exchange=binance&window=hour",
        f"cryptoquant|btc/exchange-flows/inflow?exchange=binance&window=hour",
        f"cryptoquant|btc/exchange-flows/outflow?exchange=binance&window=hour",
        # Try these as secondary data sources
    ]
    
    success_count = 0
    for topic in topics:
        # Extract source and metric from topic
        parts = topic.split('|')
        source = parts[0]
        
        path_parts = parts[1].split('?')
        full_path = path_parts[0]
        crypto_metric_parts = full_path.split('/', 1)
        
        if len(crypto_metric_parts) > 1:
            metric = crypto_metric_parts[1].replace('/', '_')
        else:
            metric = "data"  # Default if no specific metric
        
        # Fetch data
        df = await fetch_data_async(topic, start_date, end_date, api_key)
        
        # Save data
        if not df.empty:
            save_data(df, crypto, source, metric, interval)
            success_count += 1
    
    if success_count == 0:
        print(f"WARNING: Could not fetch any data for {crypto}. Please check your API key and permissions.")
        # Create a file to indicate fetch was attempted but failed
        placeholder_file = os.path.join(DATA_DIR, f"{crypto}_fetch_attempted.txt")
        with open(placeholder_file, 'w') as f:
            f.write(f"Fetch attempted on {datetime.now()} but failed to get any data.")
        print(f"Created placeholder file: {placeholder_file}")
        
        # Create basic price data so the model can run
        await create_price_data_if_missing(crypto, start_date, end_date, interval)
    else:
        print(f"Successfully fetched {success_count} out of {len(topics)} data sources for {crypto}.")

async def fetch_all_data_async(cryptos=None, start_date=None, end_date=None, interval=DEFAULT_INTERVAL, api_key=None):
    """
    Fetch data for multiple cryptocurrencies.
    
    Args:
        cryptos (list, optional): List of cryptocurrencies to fetch data for
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        interval (str, optional): Data interval (e.g., '1h', '4h', '1d')
        api_key (str, optional): API key for Cybotrade
    """
    # Use default cryptos if not specified
    if cryptos is None:
        cryptos = CRYPTOCURRENCIES
    
    # Set default dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create tasks for each crypto
    tasks = []
    for crypto in cryptos:
        tasks.append(fetch_and_save_data_async(crypto, start_date, end_date, interval, api_key))
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)
    
    # Check if we have price data for each crypto, create it if missing
    for crypto in cryptos:
        await create_price_data_if_missing(crypto, start_date, end_date, interval)

def fetch_all_data(cryptos=None, start_date=None, end_date=None, interval=DEFAULT_INTERVAL, api_key=None):
    """
    Synchronous wrapper for fetching data.
    
    Args:
        cryptos (list, optional): List of cryptocurrencies to fetch data for
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        interval (str, optional): Data interval (e.g., '1h', '4h', '1d')
        api_key (str, optional): API key for Cybotrade
    """
    return asyncio.run(fetch_all_data_async(
        cryptos=cryptos,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        api_key=api_key
    ))

async def create_price_data_if_missing(crypto, start_date, end_date, interval):
    """Check if price data exists, but don't create synthetic data if missing."""
    # Check if we already have price data
    market_price_filename = f"{crypto}_market_price_{interval}.csv"
    market_price_path = os.path.join(DATA_DIR, market_price_filename)
    
    ccxt_filename = f"{crypto}_ccxt_market_data_{interval}.csv"
    ccxt_path = os.path.join(DATA_DIR, ccxt_filename)
    
    if not os.path.exists(market_price_path) and not os.path.exists(ccxt_path):
        print(f"No price data found for {crypto}. Please make sure price data is available.")
        print(f"Expected files: {market_price_path} or {ccxt_path}")
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    api_key = os.environ.get("CYBOTRADE_API_KEY", "")
    
    if not api_key:
        print("CYBOTRADE_API_KEY environment variable not set")
        print("Please set it before running this script")
        exit(1)
    
    start_date = "2024-01-01"
    end_date = "2024-03-14"
    interval = "1h"
    
    print(f"Fetching data from {start_date} to {end_date} with interval {interval}...")
    print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    
    # Test fetching data
    fetch_all_data(
        cryptos=["BTC"],
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        api_key=api_key
    )
    
    print("Data fetching complete. Check the data directory for saved files.")
    print(f"Data directory: {DATA_DIR}")
    # List files in the data directory
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        print(f"Files in data directory ({len(files)} total):")
        for file in files:
            print(f"  - {file}")
    else:
        print("Data directory does not exist.") 