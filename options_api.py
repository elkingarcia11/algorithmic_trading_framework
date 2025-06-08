import os
from rate_limiter import RateLimiter
from datetime import datetime, timedelta
from polygon import RESTClient
from dotenv import load_dotenv
import pandas as pd

class OptionsAPI:
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
        self.rate_limiter = RateLimiter(max_requests=5, interval=60)
    
    def fetch_ohlcv(self, option_symbol, timestamp):
        """
        Fetch 1 minute ohlcv for the option symbol between start and end dates.
        """
        # Convert timestamp to date
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

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
        Get the option price for a specific timestamp
        """

        # Convert timestamp to date
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        # if file does not exist, fetch it
        if not os.path.exists(f"data/options/{option_symbol}_{date}.csv"):
            self.rate_limiter.wait_if_needed()
            self.fetch_ohlcv(option_symbol, timestamp)
        # Read csv file
        data = pd.read_csv(f"data/options/{option_symbol}_{date}.csv")
        # Find row with timestamp
        row = data[data['timestamp'] == timestamp]
        # Return price
        return row['close'].values[0]

    def generate_option_symbol_for_date(self, symbol, strike_price, option_type, timestamp, days_til_expiry):
        """
        Generate an option symbol for a specific base date (for backtesting)
        """
        # Convert timestamp to datetime object (keep as datetime for calculations)
        base_date = datetime.fromtimestamp(timestamp)
        
        # Add days_til_expiry to base date
        expiry_date = base_date + timedelta(days=days_til_expiry)

        # Check if expiry is a weekend and adjust
        if expiry_date.weekday() == 5:  # Saturday
            # Make it Monday (add 2 days)
            expiry_date += timedelta(days=2)
        elif expiry_date.weekday() == 6:  # Sunday
            # Make it Tuesday (add 2 days)
            expiry_date += timedelta(days=2)

        # Format date as YYMMDD
        date_str = expiry_date.strftime('%y%m%d')
        
        # Format strike price as 8-digit string with 3 decimal places
        strike_str = f"{int(strike_price * 1000):08d}"
        
        # Validate option type
        option_type = option_type.upper()
        if option_type not in ['C', 'P']:
            raise ValueError("option_type must be 'C' for call or 'P' for put")
        
        # Combine all parts
        option_symbol = f"{symbol.upper()}{date_str}{option_type}{strike_str}"
        
        return option_symbol