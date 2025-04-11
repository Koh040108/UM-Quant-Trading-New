"""
Script to fetch 3 years of data for the HMM model.
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append('.')

from src.cybotrade_fetcher import fetch_all_data
from src.config import CRYPTOCURRENCIES, DEFAULT_INTERVAL

def fetch_three_years_data():
    """Fetch 3 years of data for modeling."""
    # Load environment variables for API keys
    load_dotenv()
    
    # Calculate date ranges for 3 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    print(f"Attempting to fetch 3 years of data from {start_date} to {end_date}")
    
    # Check API key
    api_key = os.environ.get("CYBOTRADE_API_KEY", "")
    if not api_key:
        # Create a placeholder API key for testing
        # In production, this should be removed and a real API key should be provided
        print("WARNING: No CYBOTRADE_API_KEY environment variable set.")
        print("Setting a placeholder key for testing purposes only.")
        print("This will create synthetic data instead of fetching real data.")
        api_key = "placeholder_key_for_testing_only"
    else:
        print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    
    # Fetch data for each cryptocurrency
    print(f"Fetching data for cryptocurrencies: {', '.join(CRYPTOCURRENCIES)}")
    
    # Use 4-hour intervals for a good balance of detail and performance
    interval = DEFAULT_INTERVAL  # Using the default interval from config (usually '1h')
    
    fetch_all_data(
        cryptos=CRYPTOCURRENCIES,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        api_key=api_key
    )
    
    print("\nData fetching complete!")
    print(f"Data from {start_date} to {end_date} has been saved to the data directory.")
    print("You can now run the HMM model with 3 years of data for better results.")

if __name__ == "__main__":
    fetch_three_years_data() 