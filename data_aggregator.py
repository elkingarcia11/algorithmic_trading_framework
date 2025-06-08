import os
import pandas as pd
from datetime import datetime
from typing import List, Optional

class DataAggregator:
    """
    Class for aggregating stock data from one timeframe to another
    """
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize the DataAggregator
        
        Args:
            data_directory: Directory containing the stock data CSV files
        """
        self.data_directory = data_directory
        self.supported_source_timeframes = ["1m", "5m", "10m", "15m", "30m"]
        self.supported_target_timeframes = ["1h", "4h", "1d"]
        
        # Ensure data directory exists
        if not os.path.exists(self.data_directory):
            print(f"❌ Data directory '{self.data_directory}' not found!")
          
    def aggregate_custom(self, source_timeframe: str, target_timeframes: List[str]) -> bool:
        """
        Aggregate data from source timeframe to target timeframes
        
        Args:
            source_timeframe: Source timeframe (e.g., "1m", "5m")
            target_timeframes: List of target timeframes (e.g., ["1h", "4h"])
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🔄 Starting aggregation from {source_timeframe} to {target_timeframes}...")
        
        # Validate timeframes
        if source_timeframe not in self.supported_source_timeframes:
            print(f"❌ Unsupported source timeframe: {source_timeframe}")
            return False
        
        for tf in target_timeframes:
            if tf not in self.supported_target_timeframes:
                print(f"❌ Unsupported target timeframe: {tf}")
                return False
        
        # Get source files
        source_files = self._get_files_by_timeframe(source_timeframe)
        
        if not source_files:
            print(f"❌ No {source_timeframe} data files found!")
            return False
        
        print(f"📁 Found {len(source_files)} {source_timeframe} files to process")
        
        success_count = 0
        
        for file in source_files:
            try:
                print(f"\n📊 Processing {file}...")
                
                # Read and validate the data
                df, symbol = self._read_and_validate_file(file)
                if df is None:
                    continue
                
                print(f"   📈 Processing {len(df)} {source_timeframe} records")
                
                # Aggregate to each target timeframe
                for target_tf in target_timeframes:
                    aggregated_data = self._aggregate_to_timeframe(df, target_tf.upper())
                    self._save_aggregated_data(aggregated_data, symbol, target_tf)
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
                continue
        
        print(f"\n✅ Aggregation complete! Successfully processed {success_count}/{len(source_files)} files")
        return success_count > 0
    
    def _get_files_by_timeframe(self, timeframe: str) -> List[str]:
        """Get all files for a specific timeframe"""
        pattern = f"_{timeframe}.csv"
        return [f for f in os.listdir(self.data_directory) 
                if f.endswith(pattern) and os.path.isfile(os.path.join(self.data_directory, f))]
    
    def _read_and_validate_file(self, filename: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Read and validate a data file
        
        Returns:
            tuple: (DataFrame, symbol) or (None, None) if invalid
        """
        try:
            # Read the file
            file_path = os.path.join(self.data_directory, filename)
            df = pd.read_csv(file_path)
            
            # Extract symbol from filename (e.g., "MSFT_5m.csv" -> "MSFT")
            symbol = filename.split('_')[0]
            
            # Validate required columns
            required_columns = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                print(f"❌ Missing required columns in {filename}")
                print(f"   Required: {required_columns}")
                print(f"   Found: {list(df.columns)}")
                return None, None
            
            if len(df) == 0:
                print(f"⚠️  Empty file: {filename}")
                return None, None
            
            # Convert timestamp to datetime for proper grouping
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Remove any rows with invalid timestamps
            df = df.dropna(subset=['dt'])
            
            if len(df) == 0:
                print(f"⚠️  No valid timestamps in {filename}")
                return None, None
            
            # Sort by timestamp to ensure proper order
            df = df.sort_values('dt').reset_index(drop=True)
            
            return df, symbol
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            return None, None
    
    def _aggregate_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aggregate data to specified timeframe using proper OHLCV logic
        
        Args:
            df: DataFrame with source data
            timeframe: Target timeframe ('1H', '4H', '1D')
            
        Returns:
            DataFrame with aggregated data
        """
        # Set datetime as index for resampling
        df_indexed = df.set_index('dt')
        
        # Define aggregation rules for OHLCV data
        agg_rules = {
            'timestamp': 'first',    # Use timestamp of first bar in the period
            'open': 'first',         # First open price of the period
            'high': 'max',           # Highest high price of the period  
            'low': 'min',            # Lowest low price of the period
            'close': 'last',         # Last close price of the period
            'volume': 'sum'          # Total volume for the period
        }
        
        # Resample to target timeframe
        aggregated = df_indexed.resample(timeframe).agg(agg_rules)
        
        # Remove any periods with no data (NaN values)
        aggregated = aggregated.dropna()
        
        # Reset index to get datetime back as a column
        aggregated = aggregated.reset_index()
        
        # Recreate the datetime string column to match original format
        aggregated['datetime'] = aggregated['dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns to match original format
        final_columns = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        aggregated = aggregated[final_columns]
        
        # Convert numeric columns to appropriate types
        for col in ['open', 'high', 'low', 'close']:
            aggregated[col] = aggregated[col].round(2)
        
        aggregated['volume'] = aggregated['volume'].astype(int)
        aggregated['timestamp'] = aggregated['timestamp'].astype(int)
        
        return aggregated
    
    def _save_aggregated_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save aggregated data to CSV file"""
        if len(data) > 0:
            output_path = os.path.join(self.data_directory, f'{symbol}_{timeframe}.csv')
            data.to_csv(output_path, index=False)
            print(f"   💾 Saved {len(data)} {timeframe} records to {symbol}_{timeframe}.csv")
        else:
            print(f"   ⚠️  No {timeframe} data generated for {symbol}")
    
    def get_available_files(self) -> dict:
        """Get information about available data files"""
        files_info = {}
        
        for timeframe in self.supported_source_timeframes + self.supported_target_timeframes:
            files = self._get_files_by_timeframe(timeframe)
            symbols = [f.split('_')[0] for f in files]
            files_info[timeframe] = {
                'count': len(files),
                'files': files,
                'symbols': symbols
            }
        
        return files_info
    
    def print_status(self) -> None:
        """Print status of available data files"""
        print(f"📊 DataAggregator Status - Directory: {self.data_directory}")
        print("=" * 60)
        
        files_info = self.get_available_files()
        
        for timeframe in sorted(files_info.keys()):
            info = files_info[timeframe]
            print(f"{timeframe:>4}: {info['count']} files")
            if info['symbols']:
                symbols_str = ', '.join(info['symbols'][:5])
                if len(info['symbols']) > 5:
                    symbols_str += f" ... (+{len(info['symbols']) - 5} more)"
                print(f"      Symbols: {symbols_str}")
        
        print("=" * 60)


def test_aggregation():
    """Test function to run the aggregation"""
    # Create aggregator instance
    aggregator = DataAggregator("data")
    
    # Print current status
    aggregator.print_status()
    
    # Run 5m to 1h/4h aggregation
    print("\n" + "="*60)
    success = aggregator.aggregate_custom("5m", ["1h", "4h"])
    
    if success:
        print("\n🎉 Aggregation completed successfully!")
        print("\n📊 Updated status:")
        aggregator.print_status()
    else:
        print("\n❌ Aggregation failed!")


if __name__ == "__main__":
    test_aggregation() 