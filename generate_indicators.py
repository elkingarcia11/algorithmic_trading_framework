"""
Indicator Generator Module
Handles generating indicator files for all symbols and timeframes for different combinations of ema, vwma, roc periods
"""

import time
import os
from calculate_indicators import IndicatorCalculator

class IndicatorGenerator:
    def __init__(self, symbols_filepath: str = "symbols_to_fetch.txt", ema_periods: list[int] = None, vwma_periods: list[int] = None, roc_periods: list[int] = None, time_frames: list[str] = None) -> None:
        
        self.symbols = self.get_symbols_from_file(symbols_filepath)
        self.ema_periods = ema_periods or [3, 5, 6, 7, 9, 10, 14, 21, 28, 35, 42, 49]
        self.vwma_periods = vwma_periods or [5, 10, 12, 17, 21, 28, 35, 42, 49]
        self.roc_periods = roc_periods or [3, 5, 7, 8, 10, 14, 21, 28, 35, 42, 49]
        self.time_frames = time_frames or ["5m", "10m", "15m", "30m", "1h", "4h"]

        for symbol in self.symbols:
            for timeframe in self.time_frames:
                for ema_period in self.ema_periods:
                    self.calculate_ema(symbol, timeframe, ema_period)
                    for vwma_period in self.vwma_periods:
                        self.calculate_vwma(symbol, timeframe, vwma_period)
                        for roc_period in self.roc_periods:
                            self.calculate_roc(symbol, timeframe, roc_period)

    def calculate_ema(self, symbol: str, timeframe: str, ema_period: int) -> None:
        """Calculate EMA for a given symbol and timeframe"""
        df = pd.read_csv(f"data/{symbol}_{timeframe}.csv")
        df[f'ema_{ema_period}'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        df.to_csv(f"data/{symbol}_{timeframe}.csv", index=False)

    def calculate_vwma(self, symbol: str, timeframe: str, vwma_period: int) -> None:
        """Calculate VWMA for a given symbol and timeframe"""
        df = pd.read_csv(f"data/{symbol}_{timeframe}.csv")
        df[f'vwma_{vwma_period}'] = df['close'].ewm(span=vwma_period, adjust=False).mean()
        df.to_csv(f"data/{symbol}_{timeframe}.csv", index=False)

    def calculate_roc(self, symbol: str, timeframe: str, roc_period: int) -> None:
        """Calculate ROC for a given symbol and timeframe"""
        df = pd.read_csv(f"data/{symbol}_{timeframe}.csv")
        df[f'roc_{roc_period}'] = df['close'].pct_change(periods=roc_period)
        df.to_csv(f"data/{symbol}_{timeframe}.csv", index=False)

    def get_symbols_from_file(self, symbols_filepath: str) -> list[str]:
        """Get symbols from file separated by commas"""
        with open(symbols_filepath, 'r') as file:
            symbols = file.read().split(',')
            return [symbol.strip() for symbol in symbols if symbol.strip()]
        
        
if __name__ == "__main__":
    generator = IndicatorGenerator()
    generator.run_generation()