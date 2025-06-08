import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self):
        """Initialize the TechnicalIndicators class"""
        pass
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average (EMA) from data"""
        if isinstance(data, pd.DataFrame):
            # If DataFrame is passed, use the 'close' column
            series = data['close']
        else:
            # If Series is passed, use it directly
            series = data
            
        if period <= 0:
            raise ValueError("Period must be greater than 0")
            
        ema = series.ewm(span=period, adjust=False).mean()
        return ema

    def calculate_vwma(self, data, period):
        """Calculate Volume Weighted Moving Average (VWMA) from data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("VWMA requires DataFrame with 'close' and 'volume' columns")
            
        if 'close' not in data.columns or 'volume' not in data.columns:
            raise ValueError("Data must contain 'close' and 'volume' columns")
            
        if period <= 0:
            raise ValueError("Period must be greater than 0")
        
        # Calculate rolling sums
        price_volume = (data['close'] * data['volume']).rolling(window=period).sum()
        volume_sum = data['volume'].rolling(window=period).sum()
        
        # Handle division by zero
        vwma = np.where(volume_sum != 0, price_volume / volume_sum, np.nan)
        
        return pd.Series(vwma, index=data.index)

    def calculate_macd_line_and_signal(self, data):
        """Calculate MACD Line and Signal from data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("MACD requires DataFrame with 'close' column")
            
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        ema_12 = self.calculate_ema(data['close'], 12)
        ema_26 = self.calculate_ema(data['close'], 26)
        macd_line = ema_12 - ema_26
        macd_signal = self.calculate_ema(macd_line, 9)
        return macd_line, macd_signal

    def calculate_roc(self, data, period):
        """Calculate Rate of Change (ROC) from data - Fixed to calculate for entire series"""
        if isinstance(data, pd.DataFrame):
            # If DataFrame is passed, use the 'close' column
            series = data['close']
        else:
            # If Series is passed, use it directly
            series = data
            
        if period <= 0:
            raise ValueError("Period must be greater than 0")
        
        if len(series) < period:
            raise ValueError(f"Data length ({len(series)}) must be greater than period ({period})")
        
        # Calculate ROC for the entire series - FIXED from original bug
        roc = ((series - series.shift(period)) / series.shift(period)) * 100
        
        return roc

    def calculate_indicators(self, data_source, ema_period, vwma_period, roc_period):
        """Import data source from csv file, calculate indicators and save to another csv file"""
        try:
            data = pd.read_csv(data_source)
        except FileNotFoundError:
            print(f"❌ File not found: {data_source}")
            return False
        except pd.errors.EmptyDataError:
            print(f"❌ Empty data file: {data_source}")
            return False
        
        if data.empty:
            print(f"⚠️  No data in file: {data_source}")
            return False
        
        # Validate required columns
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        try:
            # Calculate all indicators
            data[f'ema_{ema_period}'] = self.calculate_ema(data, ema_period)
            data[f'vwma_{vwma_period}'] = self.calculate_vwma(data, vwma_period)    
            data['macd_line'], data['macd_signal_line'] = self.calculate_macd_line_and_signal(data)
            data[f'roc_{roc_period}'] = self.calculate_roc(data, roc_period)
            
            # Generate output filename
            output_filename = f"{data_source.split('.')[0]}/ema_{ema_period}_vwma_{vwma_period}_roc_{roc_period}.csv"
            
            # Save to csv file (no rounding - preserve full precision)
            data.to_csv(output_filename, index=False)
            
            print(f"✅ Indicators calculated and saved to: {output_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return False