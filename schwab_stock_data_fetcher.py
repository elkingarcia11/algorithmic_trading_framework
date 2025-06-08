import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List
from schwab_auth import SchwabAuth
import time

class StockDataFetcher:
    def __init__(self, data_directory: str = "data", symbols: List[str] = ["MSFT", "NVDA", "AAPL", "AMZN", "META", "TSLA", "QQQ", "SPY"], intervals: List[str] = ["1m", "5m", "10m", "15m", "30m"], start_date: str = "2025-01-01", end_date: str = "2025-06-06"):
        """
        Initialize the StockDataFetcher class
        
        Args:
            data_directory: Directory to save CSV files (default: "data")
            symbols: List of stock symbols to fetch (default: ["MSFT", "NVDA", "AAPL", "AMZN", "META", "TSLA", "QQQ", "SPY"])
            intervals: List of time intervals to fetch (default: ["1m", "5m", "10m", "15m", "30m"])
            start_date: Start date in YYYY-MM-DD format (default: "2025-01-01")
            end_date: End date in YYYY-MM-DD format (default: "2025-06-06")
        """
        self.data_directory = data_directory
        self.intervals = intervals
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.schwab_auth = SchwabAuth()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)

    def validate_date_format(self, date_str: str) -> bool:
        """Validate if the date string is in correct format (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def fetch_list_of_symbols(self, symbols: List[str]) -> dict:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            dict: Results for each symbol (True/False)
        """
        if not symbols or not isinstance(symbols, list):
            print("❌ Invalid symbols list")
            return {}
            
        results = {}
        
        print(f"🚀 Starting data fetch for {len(symbols)} symbols")
        
        for symbol in symbols:
            if not symbol or not symbol.strip():
                print(f"⚠️  Skipping empty symbol")
                continue
                
            symbol = symbol.strip().upper()
            success = self.get_price_history_from_schwab(symbol)
            results[symbol] = success
            
        # Print summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        print(f"📊 Summary: {successful}/{total} symbols processed successfully")
        
        return results

    def get_available_data_files(self) -> List[str]:
        """Get list of available data files in the data directory"""
        try:
            files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
            return sorted(files)
        except FileNotFoundError:
            print(f"❌ Data directory not found: {self.data_directory}")
            return []

    def get_symbol_info(self, symbol: str) -> dict:
        """Get information about available data for a symbol"""
        symbol = symbol.upper().strip()
        info = {
            'symbol': symbol,
            'available_timeframes': [],
            'files': []
        }
        
        available_files = self.get_available_data_files()
        symbol_files = [f for f in available_files if f.startswith(f"{symbol}_")]
        
        for file in symbol_files:
            # Extract timeframe from filename (e.g., "AAPL_1m.csv" -> "1m")
            parts = file.replace('.csv', '').split('_')
            if len(parts) >= 2:
                timeframe = parts[1]
                info['available_timeframes'].append(timeframe)
                info['files'].append(file)
        
        return info
    
    def check_authentication_status(self) -> bool:
        """Check and display authentication status"""
        print("🔐 Checking Schwab API authentication status...")
        
        # Validate credentials
        if not self.schwab_auth.validate_credentials():
            return False
        
        # Check authentication
        if not self.schwab_auth.is_authenticated():
            print("❌ Not authenticated - please ensure tokens are set up")
            return False
        
        # Get token info
        token_info = self.schwab_auth.get_token_info()
        if token_info['valid']:
            remaining_minutes = token_info['seconds_remaining'] / 60
            print(f"✅ Authentication successful")
            print(f"   Token expires in: {remaining_minutes:.1f} minutes")
            print(f"   Created at: {token_info['created_at']}")
            return True
        else:
            print(f"❌ Token validation failed: {token_info.get('error', 'Unknown error')}")
            return False

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
        if not symbol or not symbol.strip():
            print("❌ Invalid symbol provided")
            return False
            
        symbol = symbol.strip().upper()
        
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
        for interval in self.intervals:
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
                    
                    df = pd.DataFrame(df_data)
                    
                    # Sort by timestamp and remove duplicates
                    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                    
                    # Save to CSV
                    csv_filename = f"{self.data_directory}/{symbol}_{interval}.csv"
                    df.to_csv(csv_filename, index=False)
                    print(f"💾 Saved {len(df)} records to {csv_filename}")
                    
                except Exception as e:
                    print(f"❌ Error processing/saving data for {symbol}_{interval}: {e}")
                    return False
            else:
                print(f"⚠️  No data retrieved for {symbol}_{interval}")
        
        return True


def historical_data_fetcher():
    """Main function for testing"""
    # Ask user for symbol input otherwise default to these symbols to fetch
    symbols_input = input("Enter symbols to fetch (default: MSFT, NVDA, AAPL, AMZN, META, TSLA, QQQ, SPY): ").strip()
    if not symbols_input:
        symbols = ["MSFT", "NVDA", "AAPL", "AMZN", "META", "TSLA", "QQQ", "SPY"]
    else:
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    # Ask user for interval input otherwise default to these intervals to fetch
    intervals_input = input("Enter intervals to fetch (default: 1m, 5m, 10m, 15m, 30m): ").strip()
    if not intervals_input:
        intervals = ["1m", "5m", "10m", "15m", "30m"]
    else:
        intervals = [s.strip() for s in intervals_input.split(",") if s.strip()]
    
    # Ask for date range
    start_date = input("Enter start date (YYYY-MM-DD, default: 2025-01-01): ").strip()
    if not start_date:
        start_date = "2025-01-01"
    
    end_date = input("Enter end date (YYYY-MM-DD, default: 2025-06-06): ").strip()
    if not end_date:
        end_date = "2025-06-06"

    # Create fetcher instance
    fetcher = StockDataFetcher(
        symbols=symbols, 
        intervals=intervals, 
        start_date=start_date, 
        end_date=end_date
    )
    
    # Validate dates
    if not fetcher.validate_date_format(start_date):
        print(f"❌ Invalid start date format: {start_date}")
        return
    
    if not fetcher.validate_date_format(end_date):
        print(f"❌ Invalid end date format: {end_date}")
        return
    
    print(f"\n🎯 Configuration:")
    print(f"   Symbols: {symbols}")
    print(f"   Intervals: {intervals}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Data directory: {fetcher.data_directory}")
    
    # Check authentication status
    print(f"\n🔐 Authentication Check:")
    if not fetcher.check_authentication_status():
        print("\n❌ Authentication failed - cannot proceed with data fetching")
        print("Please ensure you have:")
        print("   1. schwab_credentials.env file with SCHWAB_APP_KEY and SCHWAB_APP_SECRET")
        print("   2. schwab_refresh_token.txt file with valid refresh token")
        print("   3. Valid Schwab API access")
        return
    
    # Fetch the data
    print(f"\n🚀 Starting data fetching process...")
    results = fetcher.fetch_list_of_symbols(symbols)
    
    print(f"\n🏁 Final Results:")
    for symbol, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {symbol}: {status}")

if __name__ == "__main__":
    historical_data_fetcher()