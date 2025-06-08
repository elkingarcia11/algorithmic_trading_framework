from strategy_optimizer import StrategyOptimizer
import time
from datetime import datetime

def generate_all_indicators(symbols: list[str]):
    """Generate indicator files for all symbols and timeframes"""
    
    # Initialize optimizer
    optimizer = StrategyOptimizer()
    
    # Track total combinations
    total_combinations = len(optimizer.ema_periods) * len(optimizer.vwma_periods) * len(optimizer.roc_periods)
    total_files = total_combinations * len(symbols) * len(optimizer.time_frames)
    
    print(f"📊 Starting indicator generation for {len(symbols)} symbols")
    print(f"   Timeframes: {', '.join(optimizer.time_frames)}")
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
        optimizer.generate_indicator_files(symbol)
        
        # Calculate progress
        symbol_time = time.time() - symbol_start
        files_generated += total_combinations * len(optimizer.time_frames)
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

if __name__ == "__main__":
    generate_all_indicators(["AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA", "QQQ", "SPY"]) 