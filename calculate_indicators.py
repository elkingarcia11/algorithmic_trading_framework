"""
Indicator Calculator Module
Handles calculating technical indicators from stock data
"""

import os
import pandas as pd
import numpy as np
from typing import Union, Tuple

class IndicatorCalculator:
    def __init__(self):
        """Initialize the TechnicalIndicators class"""
        pass
    
    def calculate_ema(self, data: Union[pd.DataFrame, pd.Series], period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA) from data
        
        Args:
            data: DataFrame with 'close' column or Series of prices
            period: EMA period (must be > 0)
            
        Returns:
            Series containing EMA values
            
        Raises:
            ValueError: If period <= 0
        """
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

    def calculate_vwma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA) from data
        
        Args:
            data: DataFrame with 'close' and 'volume' columns
            period: VWMA period (must be > 0)
            
        Returns:
            Series containing VWMA values
            
        Raises:
            ValueError: If period <= 0 or required columns missing
        """
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

    def calculate_macd_line_and_signal(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MACD Line and Signal from data
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Tuple of (MACD line, Signal line)
            
        Raises:
            ValueError: If required column missing
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("MACD requires DataFrame with 'close' column")
            
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        ema_12 = self.calculate_ema(data['close'], 12)
        ema_26 = self.calculate_ema(data['close'], 26)
        macd_line = ema_12 - ema_26
        macd_signal = self.calculate_ema(macd_line, 9)
        return macd_line, macd_signal

    def calculate_roc(self, data: Union[pd.DataFrame, pd.Series], period: int) -> pd.Series:
        """
        Calculate Rate of Change (ROC) from data
        
        Args:
            data: DataFrame with 'close' column or Series of prices
            period: ROC period (must be > 0)
            
        Returns:
            Series containing ROC values
            
        Raises:
            ValueError: If period <= 0 or data length < period
        """
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
        
        # Calculate ROC for the entire series
        roc = ((series - series.shift(period)) / series.shift(period)) * 100
        
        return roc

    def calculate_indicators(self, data_source: str, ema_period: int, vwma_period: int, roc_period: int) -> bool:
        """
        Import data source from csv file, calculate indicators and save to another csv file
        
        Args:
            data_source: Path to input CSV file
            ema_period: EMA period
            vwma_period: VWMA period
            roc_period: ROC period
            
        Returns:
            bool: True if successful, False otherwise
        """
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
            
            # Generate output filename and ensure directory exists
            output_dir = os.path.dirname(data_source)
            output_filename = f"{output_dir}/ema_{ema_period}_vwma_{vwma_period}_roc_{roc_period}.csv"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to csv file (no rounding - preserve full precision)
            data.to_csv(output_filename, index=False)
            
            print(f"✅ Indicators calculated and saved to: {output_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return False

    def run_calculation(self) -> None:
        """Run the technical indicators generator with user input"""
        print("\n🎯 Technical Indicators Generator")
        print("=" * 60)
        
        # Get data source
        data_source = input("Enter the data source file path: ").strip()
        if not data_source:
            print("❌ No data source provided")
            return
            
        # Get EMA period
        ema_input = input("Enter the EMA period: ").strip()
        if not ema_input:
            print("❌ No EMA period provided")
            return
        try:
            ema_period = int(ema_input)
            if ema_period <= 0:
                raise ValueError("Period must be greater than 0")
        except ValueError as e:
            print(f"❌ Invalid EMA period: {e}")
            return
            
        # Get VWMA period
        vwma_input = input("Enter the VWMA period: ").strip()
        if not vwma_input:
            print("❌ No VWMA period provided")
            return
        try:
            vwma_period = int(vwma_input)
            if vwma_period <= 0:
                raise ValueError("Period must be greater than 0")
        except ValueError as e:
            print(f"❌ Invalid VWMA period: {e}")
            return
            
        # Get ROC period
        roc_input = input("Enter the ROC period: ").strip()
        if not roc_input:
            print("❌ No ROC period provided")
            return
        try:
            roc_period = int(roc_input)
            if roc_period <= 0:
                raise ValueError("Period must be greater than 0")
        except ValueError as e:
            print(f"❌ Invalid ROC period: {e}")
            return
            
        # Confirm generation
        print("\n🔍 Generation Configuration:")
        print(f"   Data Source: {data_source}")
        print(f"   EMA Period: {ema_period}")
        print(f"   VWMA Period: {vwma_period}")
        print(f"   ROC Period: {roc_period}")
        
        confirm = input("\nProceed with indicator generation? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Generation cancelled")
            return
            
        # Run generation
        success = self.calculate_indicators(data_source, ema_period, vwma_period, roc_period)
        if success:
            print("\n✅ Generation completed successfully!")
        else:
            print("\n❌ Generation failed")

if __name__ == "__main__":
    calculator = IndicatorCalculator()
    calculator.run_calculation()