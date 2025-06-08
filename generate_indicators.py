"""
Indicator Generator Module
Handles generating indicator files for all symbols and timeframes for different combinations of ema, vwma, roc periods
"""

import time
import os
from calculate_indicators import IndicatorCalculator

class IndicatorGenerator:
    def __init__(self, symbols: list[str] = None, ema_periods: list[int] = None, vwma_periods: list[int] = None, roc_periods: list[int] = None, time_frames: list[str] = None) -> None:
        self.symbols = symbols or []
        self.ema_periods = ema_periods or [3, 5, 6, 7, 9, 10, 14, 21, 28, 35, 42, 49]
        self.vwma_periods = vwma_periods or [5, 10, 12, 17, 21, 28, 35, 42, 49]
        self.roc_periods = roc_periods or [3, 5, 7, 8, 10, 14, 21, 28, 35, 42, 49]
        self.time_frames = time_frames or ["1m", "5m", "10m", "15m", "30m", "1h", "4h"]
        self.indicator_calculator = IndicatorCalculator()
        
    def generate_all_indicators(self, symbols: list[str]):
        """Generate indicator files for all symbols and timeframes"""
        # Track total combinations
        total_combinations = len(self.ema_periods) * len(self.vwma_periods) * len(self.roc_periods)
        total_files = total_combinations * len(symbols) * len(self.time_frames)
        
        print(f"📊 Starting indicator generation for {len(symbols)} symbols")
        print(f"   Timeframes: {', '.join(self.time_frames)}")
        print(f"   Combinations per symbol: {total_combinations}")
        print(f"   Total files to generate: {total_files}")
        print("=" * 80)
        
        start_time = time.time()
        files_generated = 0
        
        # Process each symbol
        for symbol in symbols:
            print(f"\n🔄 Processing {symbol}...")
            symbol_start = time.time()
            
            # Generate indicators for this symbol
            self.generate_indicator_files(symbol)
            
            # Calculate progress
            symbol_time = time.time() - symbol_start
            files_generated += total_combinations * len(self.time_frames)
            progress = (files_generated / total_files) * 100
            
            print(f"✅ {symbol} complete in {symbol_time:.1f} seconds")
            print(f"📈 Overall progress: {progress:.1f}%")
            
        # Print final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("🎉 INDICATOR GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Files generated: {files_generated}")
        print(f"Average time per file: {total_time/files_generated:.2f} seconds")
        print(f"Average time per symbol: {total_time/len(symbols):.1f} seconds")
        print("=" * 80)

    def generate_indicator_files(self, symbol: str) -> bool:
        """Generate all indicator combinations for a symbol across all timeframes"""
        print(f"🔄 Generating indicator files for {symbol}...")
        
        # Create a list of all combinations of ema, vwma, roc periods
        combinations = []
        for ema_period in self.ema_periods:
            for vwma_period in self.vwma_periods:
                for roc_period in self.roc_periods:
                    combinations.append((ema_period, vwma_period, roc_period))

        total_combinations = len(combinations) * len(self.time_frames)
        current_combination = 0
        
        print(f"📊 Generating {len(combinations)} parameter combinations across {len(self.time_frames)} timeframes")
        print(f"   Total files to generate: {total_combinations}")
        
        # For each combination, calculate indicators for each timeframe
        for combination in combinations:
            ema_period, vwma_period, roc_period = combination
            
            # For each time frame, generate indicator file
            for time_frame in self.time_frames:
                current_combination += 1
                progress = (current_combination / total_combinations) * 100
                
                print(f"🔍 Progress: {progress:.1f}% - Generating {symbol} {time_frame} EMA:{ema_period} VWMA:{vwma_period} ROC:{roc_period}")
                
                data_source = f"data/{symbol}_{time_frame}.csv"
                
                # Check if data file exists
                if not os.path.exists(data_source):
                    print(f"⚠️  Skipping {data_source} - file not found")
                    continue
                
                # Calculate indicators and save to file
                success = self.indicator_calculator.calculate_indicators(data_source, ema_period, vwma_period, roc_period)
                if not success:
                    print(f"❌ Failed to generate indicators for {data_source}")
                    continue
                
        print(f"✅ Indicator file generation complete for {symbol}")
        return True

    def run_generation(self) -> None:
        """Run the indicator generator with user input"""
        print("\n🎯 Indicator Generator Tool")
        print("=" * 60)
        
        # Get symbols
        symbols_input = input("Enter the symbols (comma separated, e.g., AAPL,NVDA,MSFT,AMZN,META,TSLA,QQQ,SPY): ").strip()
        if not symbols_input:
            print("❌ No symbols provided")
            return
        symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
        
        # Get EMA periods
        ema_input = input("Enter the EMA periods (comma separated, e.g., 3,5,6,7,9,10,14,21,28,35,42,49): ").strip()
        if not ema_input:
            print("❌ No EMA periods provided")
            return
        try:
            ema_periods = [int(p.strip()) for p in ema_input.split(",") if p.strip()]
        except ValueError:
            print("❌ Invalid EMA periods - must be comma-separated integers")
            return
            
        # Get VWMA periods
        vwma_input = input("Enter the VWMA periods (comma separated, e.g., 5,10,12,17,21,28,35,42,49): ").strip()
        if not vwma_input:
            print("❌ No VWMA periods provided")
            return
        try:
            vwma_periods = [int(p.strip()) for p in vwma_input.split(",") if p.strip()]
        except ValueError:
            print("❌ Invalid VWMA periods - must be comma-separated integers")
            return
            
        # Get ROC periods
        roc_input = input("Enter the ROC periods (comma separated, e.g., 3,5,7,8,10,14,21,28,35,42,49): ").strip()
        if not roc_input:
            print("❌ No ROC periods provided")
            return
        try:
            roc_periods = [int(p.strip()) for p in roc_input.split(",") if p.strip()]
        except ValueError:
            print("❌ Invalid ROC periods - must be comma-separated integers")
            return
            
        # Get timeframes
        tf_input = input("Enter the timeframes (comma separated, e.g., 1m,5m,10m,15m,30m,1h,4h): ").strip()
        if not tf_input:
            print("❌ No timeframes provided")
            return
        time_frames = [tf.strip() for tf in tf_input.split(",") if tf.strip()]
        
        # Validate timeframes
        valid_timeframes = ["1m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]
        invalid_tfs = [tf for tf in time_frames if tf not in valid_timeframes]
        if invalid_tfs:
            print(f"❌ Invalid timeframes: {', '.join(invalid_tfs)}")
            print(f"   Valid timeframes: {', '.join(valid_timeframes)}")
            return
            
        # Confirm generation
        print("\n🔍 Generation Configuration:")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   EMA Periods: {', '.join(map(str, ema_periods))}")
        print(f"   VWMA Periods: {', '.join(map(str, vwma_periods))}")
        print(f"   ROC Periods: {', '.join(map(str, roc_periods))}")
        print(f"   Timeframes: {', '.join(time_frames)}")
        
        confirm = input("\nProceed with indicator generation? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Generation cancelled")
            return
            
        # Create generator and run
        generator = IndicatorGenerator(symbols, ema_periods, vwma_periods, roc_periods, time_frames)
        generator.generate_all_indicators(symbols)

if __name__ == "__main__":
    generator = IndicatorGenerator()
    generator.run_generation()