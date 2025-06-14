"""
Schwab Stock Data Fetcher Module
Handles fetching stock data from Schwab API
"""

import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List
from schwab_auth import SchwabAuth

class StockDataFetcher:
    def __init__(self, symbols_filepath: str, intervals_to_fetch: List[str] = ["5m","10m", "15m", "30m"], intervals_to_aggregate_to: List[str] = ["1h", "4h"], start_date: str = "2025-01-01", end_date: str = "2025-01-01"):
        """
        Initialize the StockDataFetcher class
        
        Args:
            symbols_filepath: Path to file containing symbols to fetch
            intervals_to_fetch: List of time intervals to fetch
            intervals_to_aggregate_to: List of time intervals to aggregate to
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.schwab_auth = SchwabAuth()
        self.symbols = self.get_symbols_from_file(symbols_filepath)
        self.intervals_to_fetch = intervals_to_fetch
        self.intervals_to_aggregate_to = intervals_to_aggregate_to
        self.start_date = start_date
        self.end_date = end_date
        self.data_directory = "data"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)

        for symbol in self.symbols:
            self.get_price_history_from_schwab(symbol)
            self.aggregate_data(symbol, start_interval=self.intervals_to_fetch[0])

        print("🎉 All data fetched and aggregated successfully")

    def get_symbols_from_file(self, symbols_filepath: str) -> List[str]:
        """Get symbols from file separated by commas"""
        with open(symbols_filepath, 'r') as file:
            symbols = file.read().split(',')
            return [symbol.strip() for symbol in symbols if symbol.strip()]
    
    def _extract_frequency_number(self, interval: str) -> int:
        """Extract numeric frequency from interval string (e.g., '5m' -> 5)"""
        try:
            return int(interval.replace('m', '').replace('h', '').replace('d', ''))
        except ValueError:
            print(f"⚠️  Invalid interval format: {interval}, defaulting to 1")
            return 1
    
    def get_price_history_from_schwab(self, symbol: str) -> bool:
        """
        Retrieve price history from Schwab API for each interval from start date to end date
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            
        Returns:
            bool: True if successful, False if failed
        """
        if not symbol:
            print("❌ Invalid symbol provided")
            return False
            
        symbol = symbol.upper()
        
        # Validate credentials first
        if not self.schwab_auth.validate_credentials():
            print("❌ Schwab credentials validation failed")
            return False
        
        # Check if we're authenticated
        if not self.schwab_auth.is_authenticated():
            print("❌ Not authenticated with Schwab API")
            return False
        
        headers = self.schwab_auth.get_auth_headers()
        if not headers:
            print("❌ No valid authentication headers available")
            return False
        
        # Display token info for debugging
        token_info = self.schwab_auth.get_token_info()
        if token_info['valid']:
            remaining_minutes = token_info['seconds_remaining'] / 60
            print(f"🔐 Authentication valid - token expires in {remaining_minutes:.1f} minutes")
        else:
            print("❌ Token validation failed")
            return False
        
        url = "https://api.schwabapi.com/marketdata/v1/pricehistory"

        try:
            # Convert string dates to datetime objects
            start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError as e:
            print(f"❌ Invalid date format: {e}")
            return False

        # For each interval, fetch the data from the Schwab API
        for interval in intervals_to_fetch:
            print(f"📊 Processing interval: {interval} for {symbol}")
            all_candles = []
            current_start_dt = start_date_dt
            
            # Extract numeric frequency for API
            frequency_num = self._extract_frequency_number(interval)
            
            while current_start_dt <= end_date_dt:
                current_end_dt = current_start_dt + timedelta(days=10)
                
                # Don't exceed the overall end date
                if current_end_dt > end_date_dt:
                    current_end_dt = end_date_dt
                
                # Convert start and end dates to UNIX epoch milliseconds
                start_time_ms = int(current_start_dt.timestamp() * 1000)
                end_time_ms = int(current_end_dt.timestamp() * 1000)
                
                params = {
                    'symbol': symbol,
                    'periodType': 'day',
                    'period': 10,
                    'frequencyType': 'minute',
                    'frequency': frequency_num,  # Use numeric frequency
                    'startDate': start_time_ms,
                    'endDate': end_time_ms,
                    'needExtendedHoursData': 'false',
                    'needPreviousClose': 'false'
                }
                
                print(f"📡 Fetching price history for {symbol} ({interval}) from {current_start_dt.date()} to {current_end_dt.date()}")
                
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    # Sleep for 1 second to avoid rate limiting
                    time.sleep(1)

                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'candles' in data and data['candles']:
                            candles = data['candles']
                            print(f"✅ Retrieved {len(candles)} candles from Schwab API")
                            all_candles.extend(candles)
                        else:
                            print("📊 No candle data found in API response")
                    else:
                        print(f"❌ API request failed: {response.status_code}")
                        if response.text:
                            print(f"Response: {response.text[:200]}...")
                        return False
                        
                except requests.exceptions.RequestException as e:
                    print(f"❌ Network error fetching price history: {e}")
                    return False
                except Exception as e:
                    print(f"❌ Unexpected error fetching price history: {e}")
                    return False
            
                # Move to next time window
                current_start_dt = current_end_dt + timedelta(days=1)  # Add 1 day to avoid overlap
    
            # Process and save the data if we have candles
            if all_candles:
                try:
                    # Convert candles to DataFrame with proper structure
                    df_data = []
                    for candle in all_candles:
                        df_data.append({
                            'timestamp': candle.get('datetime', 0),
                            'datetime': datetime.fromtimestamp(candle.get('datetime', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            'open': candle.get('open', 0),
                            'high': candle.get('high', 0),
                            'low': candle.get('low', 0),
                            'close': candle.get('close', 0),
                            'volume': candle.get('volume', 0)
                        })
                    
                    new_df = pd.DataFrame(df_data)
                    
                    # Sort by timestamp and remove duplicates
                    new_df = new_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                    
                    # Check if existing file exists
                    csv_filename = f"{self.data_directory}/{symbol}_{interval}.csv"
                    if os.path.exists(csv_filename):
                        print(f"📂 Found existing data file for {symbol}_{interval}")
                        # Read existing data
                        existing_df = pd.read_csv(csv_filename)
                        # Convert timestamp to numeric for proper comparison
                        existing_df['timestamp'] = pd.to_numeric(existing_df['timestamp'])
                        
                        # Combine existing and new data
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        # Sort by timestamp and remove duplicates
                        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                        
                        print(f"📊 Combined data: {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total records")
                        # Save combined data
                        combined_df.to_csv(csv_filename, index=False)
                        print(f"💾 Updated {csv_filename} with combined data")
                    else:
                        # Save new data
                        new_df.to_csv(csv_filename, index=False)
                        print(f"💾 Created new file {csv_filename} with {len(new_df)} records")
                    
                except Exception as e:
                    print(f"❌ Error processing data: {e}")
                    return False
            else:
                print(f"⚠️  No data retrieved for {symbol}_{interval}")
        
        return True

    def aggregate_data(self, symbol: str, start_interval: str):
        """
        take start interval and aggregate to all other intervals

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            start_interval: Start interval (e.g., '1m')

        """
        # Read in the data from the start interval
        df = pd.read_csv(f"{self.data_directory}/{symbol}_{start_interval}.csv")
        for interval in self.intervals_to_aggregate_to:
            start_interval_frequency = self._extract_frequency_number(start_interval)
            interval_frequency = self._extract_frequency_number(interval)
            # If start interval < interval and interval % start_interval == 0 and interval is not in the data directory, then aggregate
            if start_interval_frequency < interval_frequency and interval_frequency % start_interval_frequency == 0 and not os.path.exists(f"{self.data_directory}/{symbol}_{interval}.csv"):
                print(f"🔄 Aggregating {symbol}_{start_interval} to {symbol}_{interval}")
                
                # Aggregate the data to the interval to aggregate to
                df = df.resample(interval).agg({
                    'open': 'first',
                    'high': 'max',  
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })

                # Save the aggregated data  
                df.to_csv(f"{self.data_directory}/{symbol}_{interval}.csv", index=False)
                print(f"💾 Created new file {symbol}_{interval}.csv with {len(df)} records")

if __name__ == "__main__": 
    stock_data_fetcher = StockDataFetcher(symbols_filepath="symbols_to_fetch.txt", intervals_to_fetch=["5m","10m", "15m", "30m"], intervals_to_aggregate_to=["1h", "4h"], start_date="2025-01-01", end_date="2025-06-13")