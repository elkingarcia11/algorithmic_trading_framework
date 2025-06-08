"""
Options Fetcher Module
Handles fetching options data from the API
"""

import os
from polygon_rate_limiter import PolygonRateLimiter
from datetime import datetime, timedelta
from polygon import RESTClient
from dotenv import load_dotenv
import pandas as pd

class PolygonOptionsFetcher:
    def __init__(self):
        """
        Initialize the OptionsTracker class
        """
        # Load environment variables
        load_dotenv()

        # Create data directory if it doesn't exist
        os.makedirs("data/options", exist_ok=True)

        # Initialize the clientget
        self.client = RESTClient(os.getenv("POLYGON_API_KEY"))

        # Initialize rate limiter
        self.rate_limiter = PolygonRateLimiter(max_requests=5, interval=60)
    
    def fetch_ohlcv(self, option_symbol, date):
        """
        Fetch 1 minute ohlcv for the option symbol for the date specified.
        """

        csv_filename = f"data/options/{option_symbol}_{date}.csv"
        
        # Always fetch fresh data for backtesting
        print(f"📥 Fetching OHLCV data for {option_symbol} for {date}")
        
        # Fetch data from API for the date range
        aggs = []
        try:
            for a in self.client.list_aggs(
                f"O:{option_symbol}",
                1,
                "minute",
                date,
                date,
                adjusted="true",
                sort="asc",
                limit=50000,
            ):
                aggs.append(a)
        except Exception as e:
            print(f"❌ Error fetching data for {option_symbol}: {e}")
            return
        
        if not aggs:
            print(f"⚠️  No data available for {option_symbol} for {date}")
            return
            
        # Save to csv (overwrite existing data)
        with open(csv_filename, 'w') as f:
            # Write header
            f.write("timestamp,open,high,low,close,volume\n")
            
            # Write all fetched data (no rounding)
            for agg in aggs:
                f.write(f"{agg.timestamp},{agg.open},{agg.high},{agg.low},{agg.close},{agg.volume}\n")
        
        print(f"✅ Saved {len(aggs)} OHLCV records for {option_symbol}")

    def get_option_price(self, option_symbol, timestamp):
        """
        Get the option price by fetching the day's options data from the csv file or from the API and searching for the timestamp within the days data.
        Use Case: For backtesting, we need to get the option price for a specific timestamp.
        """

        # Convert timestamp to date
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        # if file does not exist, fetch it
        if not os.path.exists(f"data/options/{option_symbol}_{date}.csv"):
            self.rate_limiter.wait_if_needed()
            self.fetch_ohlcv(option_symbol, date)
        # Read csv file
        data = pd.read_csv(f"data/options/{option_symbol}_{date}.csv")
        # Find row with timestamp
        row = data[data['timestamp'] == timestamp]
        # Return price
        return row['close'].values[0]

    def generate_option_symbol_for_date(self, symbol, date, strike_price, option_type):
        """
        Generate an option symbol based on the symbol, date, strike price, option type, and days until expiry
        """
        # Format date as YYMMDD
        date_str = date.strftime('%y%m%d')
        
        # Format strike price as 8-digit string with 3 decimal places
        strike_str = f"{int(strike_price * 1000):08d}"
        
        # Validate option type
        option_type = option_type.upper()
        if option_type not in ['C', 'P']:
            raise ValueError("option_type must be 'C' for call or 'P' for put")
        
        # Combine all parts
        option_symbol = f"{symbol.upper()}{date_str}{option_type}{strike_str}"
        
        return option_symbol
    
    def return_next_expiry_date(self, date, days_til_expiry):
        """
        Return the next expiry date based on the date and days until expiry
        Use Case: For generating option symbol based on a specific date
        """
        # Add days_til_expiry to base date
        expiry_date = date + timedelta(days=days_til_expiry)
        # If saturday or sunday, add 2 days
        if expiry_date.weekday() == 5:
            expiry_date += timedelta(days=2)
        elif expiry_date.weekday() == 6:
            expiry_date += timedelta(days=2)
        return expiry_date
        
    def fetch_option_data(self):
        """
        Fetch option data for a specific timestamp
        """
        # Ask user for input for symbol, date, strike price, option type, days til expiry
        symbol = input("Enter the option symbol: ")
        date = input("Enter the expiration date: (Format: 2025-01-01)")
        strike_price = input("Enter the strike price: ")
        option_type = input("Enter the option type: (C or P)")
        option_symbol = self.generate_option_symbol_for_date(symbol, date, strike_price, option_type)
        # Ask user for what date they want to fetch option symbol for:
        date = input("Enter the date to fetch option symbol for: (Format: 2025-01-01)")
        self.fetch_ohlcv(option_symbol, date)
        
        
        
if __name__ == "__main__":
    options_fetcher = PolygonOptionsFetcher()
    options_fetcher.fetch_option_data()