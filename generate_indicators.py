"""
Indicator Generator Module
Handles generating indicator files for all symbols and timeframes for different combinations of ema, vwma, roc periods
"""
import pandas as pd
import os
from itertools import product, repeat
from typing import Dict, List, Tuple

class IndicatorGenerator:
    def __init__(self, symbols_filepath: str = "symbols_to_fetch.txt", ema_periods: list[int] = None, signal_periods: list[int] = None, vwma_periods: list[int] = None, roc_periods: list[int] = None, time_frames: list[str] = None) -> None:
        
        self.symbols = self.get_symbols_from_file(symbols_filepath)
        self.ema_periods = ema_periods or list(range(3, 50))
        self.signal_periods = signal_periods or [5, 7, 9, 12]
        self.vwma_periods = vwma_periods or [5, 10, 12, 17, 21, 28, 35, 42, 49]     
        self.roc_periods = roc_periods or [3, 5, 7, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.time_frames = time_frames or ["5m", "10m", "15m", "30m", "1h", "4h"]
        self.macd_combos = self.generate_macd_combos()

    def process_symbol_timeframe(self, symbol: str, timeframe: str) -> None:
        """Process all indicators for a single symbol and timeframe"""
        try:
            # Load data once
            filepath = f"data/{symbol}_{timeframe}.csv"
            if not os.path.exists(filepath):
                print(f"⚠️ File not found: {filepath}")
                return
                
            df = pd.read_csv(filepath)
            
            # Calculate all EMAs
            for ema_period in self.ema_periods:
                df[f'ema_{ema_period}'] = df['close'].ewm(span=ema_period, adjust=False).mean()
            
            # Calculate all VWMAs
            for vwma_period in self.vwma_periods:
                df[f'vwma_{vwma_period}'] = df['close'].ewm(span=vwma_period, adjust=False).mean()
            
            # Calculate all ROCs
            for roc_period in self.roc_periods:
                df[f'roc_{roc_period}'] = df['close'].pct_change(periods=roc_period)
            
            # Calculate all MACDs
            for macd_combo in self.macd_combos:
                fast_ema = macd_combo['fast_ema']
                slow_ema = macd_combo['slow_ema']
                signal_ema = macd_combo['signal_ema']
                
                # Calculate MACD line
                df[f'macd_line_{fast_ema}_{slow_ema}'] = (
                    df['close'].ewm(span=fast_ema, adjust=False).mean() -
                    df['close'].ewm(span=slow_ema, adjust=False).mean()
                )
                
                # Calculate Signal line
                df[f'macd_signal_{fast_ema}_{slow_ema}_{signal_ema}'] = (
                    df[f'macd_line_{fast_ema}_{slow_ema}'].ewm(span=signal_ema, adjust=False).mean()
                )
            
            # Write all indicators at once
            df.to_csv(filepath, index=False)
            print(f"✅ Processed {symbol} {timeframe}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol} {timeframe}: {str(e)}")

    def generate_indicators(self) -> None:
        """Generate all indicators for all symbols and timeframes"""
        total_combinations = len(self.symbols) * len(self.time_frames)
        processed = 0
        
        for symbol in self.symbols:
            for timeframe in self.time_frames:
                self.process_symbol_timeframe(symbol, timeframe)
                processed += 1
                progress = (processed / total_combinations) * 100
                print(f"📊 Progress: {progress:.1f}%")

    def get_symbols_from_file(self, symbols_filepath: str) -> list[str]:
        """Get symbols from file separated by commas"""
        with open(symbols_filepath, 'r') as file:
            symbols = file.read().split(',')
            return [symbol.strip() for symbol in symbols if symbol.strip()]

    def generate_macd_combos(self) -> List[Dict]:
        """Generate MACD combinations"""
        combos = []
        for fast_ema in self.ema_periods:
            for slow_ema in self.ema_periods:
                for signal_ema in self.signal_periods:
                    if fast_ema < slow_ema:
                        combos.append({
                            'fast_ema': fast_ema,
                            'slow_ema': slow_ema,
                            'signal_ema': signal_ema
                        })
        return combos

if __name__ == "__main__":
    generator = IndicatorGenerator()
    generator.generate_indicators()