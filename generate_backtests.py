"""
Backtest Strategy Module
Handles backtesting strategies
"""
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time
from itertools import product
import multiprocessing
import numpy as np

def get_optimal_workers():
    """
    Calculate optimal number of worker processes
    Returns number of CPU cores - 1 to leave one core free for system
    """
    return max(1, multiprocessing.cpu_count() - 1)

class BacktestStrategy:
    def __init__(self, ema_periods: list[int] = None, vwma_periods: list[int] = None, roc_periods: list[int] = None, fast_emas: list[int] = None, slow_emas: list[int] = None, signal_emas: list[int] = None):
        # Use optimized parameter ranges if none provided
        self.ema_periods = ema_periods or [3, 5, 8, 10, 12, 14, 16, 18, 20]
        self.vwma_periods = vwma_periods or [3, 5, 8, 10, 12, 14, 16, 18, 20]
        self.roc_periods = roc_periods or [8, 10, 12]
        self.fast_emas = fast_emas or [12, 14, 16, 18, 20]
        self.slow_emas = slow_emas or [26, 28, 30, 32, 34]
        self.signal_emas = signal_emas or [9, 10, 11, 12, 13]

    def process_combination(self, args):
        """
        Process a single combination
        """
        df, symbol, timeframe, ema_period, vwma_period, roc_period, fast_ema, slow_ema, signal_ema = args
        try:
            # Pre-calculate indicator names
            ema_col = f'ema_{ema_period}'
            vwma_col = f'vwma_{vwma_period}'
            roc_col = f'roc_{roc_period}'
            macd_line_col = f'macd_line_{fast_ema}_{slow_ema}'
            macd_signal_col = f'macd_signal_{fast_ema}_{slow_ema}_{signal_ema}'
            
            # Validate required columns
            required_columns = ['close', 'open', 'high', 'low', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing_columns = [col for col in required_columns if col not in df.columns]
                print(f"❌ Missing required columns in {symbol}_{timeframe}: {missing_columns}")
                return None
            
            # Validate indicator columns
            required_indicators = [ema_col, vwma_col, roc_col, macd_line_col, macd_signal_col]
            if not all(ind in df.columns for ind in required_indicators):
                missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
                print(f"❌ Missing indicators in {symbol}_{timeframe}: {missing_indicators}")
                return None
            
            # Run backtest
            max_open_candles = 0
            max_loss_percentage = 0
            max_win_percentage = 0
            total_open_candles = 0
            total_profit_percentage = 0
            total_trade_count = 0
            total_win_count = 0
            total_loss_count = 0

            open_position = False
            open_candles = 0
            entry_price = 0
            
            # Pre-fetch columns for faster access
            close_prices = df['close'].values
            ema_values = df[ema_col].values
            vwma_values = df[vwma_col].values
            roc_values = df[roc_col].values
            macd_values = df[macd_line_col].values
            macd_signal_values = df[macd_signal_col].values
            
            for i in range(len(df)):
                if not open_position and self.buy_signal(
                    ema_values[i], vwma_values[i], roc_values[i], 
                    macd_values[i], macd_signal_values[i]
                ):
                    open_position = True
                    open_candles = 0
                    entry_price = close_prices[i]
                elif open_position and self.sell_signal(
                    ema_values[i], vwma_values[i], roc_values[i], 
                    macd_values[i], macd_signal_values[i]
                ):
                    open_position = False
                    max_open_candles = max(max_open_candles, open_candles)
                    total_open_candles += open_candles
                    open_candles = 0
                    exit_price = close_prices[i]
                    profit_percentage = (exit_price - entry_price) / entry_price * 100
                    total_trade_count += 1
                    if profit_percentage > 0:
                        total_win_count += 1
                        max_win_percentage = max(max_win_percentage, profit_percentage)
                    else:
                        max_loss_percentage = min(max_loss_percentage, profit_percentage)
                        total_loss_count += 1
                    total_profit_percentage += profit_percentage
                elif open_position:
                    open_candles += 1

            if total_trade_count == 0:
                print(f"ℹ️ No trades executed for {symbol}_{timeframe} EMA:{ema_period} VWMA:{vwma_period} ROC:{roc_period} MACD:{fast_ema}_{slow_ema}_{signal_ema}")
                return None

            average_open_candles = total_open_candles / total_trade_count
            average_profit_percentage = total_profit_percentage / total_trade_count
            win_rate = total_win_count / total_trade_count * 100
            loss_rate = total_loss_count / total_trade_count * 100

            result = {
                'max_open_candles': max_open_candles,
                'average_open_candles': average_open_candles,
                'average_profit_percentage': average_profit_percentage,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'total_trade_count': total_trade_count,
                'max_win_percentage': max_win_percentage,
                'max_loss_percentage': max_loss_percentage
            }

            # Format the result as a CSV row (without datetime)
            csv_row = f"{symbol}_{timeframe},{ema_period},{vwma_period},{roc_period},{fast_ema},{slow_ema},{signal_ema},{result['max_open_candles']},{result['average_open_candles']},{result['average_profit_percentage']},{result['win_rate']},{result['loss_rate']},{result['total_trade_count']},{result['max_win_percentage']},{result['max_loss_percentage']}\n"
            
            # Write result immediately to file
            results_file = os.path.abspath('backtest_results.csv')
            try:
                with open(results_file, 'a') as f:
                    f.write(csv_row)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"✅ Written result for {symbol}_{timeframe} EMA:{ema_period} VWMA:{vwma_period} ROC:{roc_period} MACD:{fast_ema}_{slow_ema}_{signal_ema}")
            except Exception as e:
                print(f"❌ Error writing result to file: {e}")
            
            return csv_row
        except Exception as e:
            print(f"❌ Error processing {symbol}_{timeframe} EMA:{ema_period} VWMA:{vwma_period} ROC:{roc_period} MACD:{fast_ema}_{slow_ema}_{signal_ema}: {str(e)}")
            return None

    def backtest(self, symbol, timeframe) -> bool:
        """
        Backtest the strategy in parallel for multiple data files
        """
        results_file = os.path.abspath(f'data/backtest_results_{symbol}_{timeframe}.csv')
        print(f"\nResults will be written to: {results_file}")
        
        # Write header only if file does not exist
        if not os.path.exists(results_file):
            print(f"Creating new results file: {results_file}")
            with open(results_file, 'w') as f:
                f.write("symbol,ema_period,vwma_period,roc_period,fast_ema,slow_ema,signal_ema,max_open_candles,average_open_candles,average_profit_percentage,win_rate,loss_rate,total_trade_count,max_win_percentage,max_loss_percentage\n")
                f.flush()
                os.fsync(f.fileno())
        else:
            print(f"Using existing results file: {results_file}")
        
        # Get data file
        df = pd.read_csv(f"data/{symbol}_{timeframe}.csv")
        if df is None or df.empty:
            print(f"❌ Empty dataframe for {f'data/{symbol}_{timeframe}.csv'}")
            return False
            
        # Calculate total combinations
        total_strategy_combos = len(self.ema_periods) * len(self.vwma_periods) * len(self.roc_periods) * len(self.fast_emas) * len(self.slow_emas) * len(self.signal_emas)
        total_combinations = total_strategy_combos
        
        print(f"\nBacktest Configuration:")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Total strategy combinations: {total_strategy_combos:,}")
        print(f"Total combinations to process: {total_combinations:,}")
        
        start_time = time.time()
        processed = 0
        successful = 0
        failed = 0
        errors = []
        
        print("\nStarting parallel processing...")
        
        # Create all combinations of files and strategy parameters
        all_args = []
        for ema_period, vwma_period, roc_period, fast_ema, slow_ema, signal_ema in product(
            self.ema_periods,
            self.vwma_periods,
            self.roc_periods,
            self.fast_emas,
            self.slow_emas,
            self.signal_emas
        ):
            all_args.extend([(df, symbol, timeframe, ema_period, vwma_period, roc_period, fast_ema, slow_ema, signal_ema)])
        
        # Process all combinations in parallel with batch writing
        batch_size = 1000  # Write results in batches of 1000
        results_batch = []
        last_write_time = time.time()
        write_interval = 60  # Write every 60 seconds if batch is not full
        
        # Calculate optimal number of workers
        num_workers = get_optimal_workers()
        print(f"\nUsing {num_workers} worker processes")
        
        # Split work into chunks for better parallelization
        chunk_size = max(1, len(all_args) // (num_workers * 4))  # Divide work into 4x number of workers
        print(f"Processing in chunks of {chunk_size} combinations")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process chunks in parallel
            for i in range(0, len(all_args), chunk_size):
                chunk = all_args[i:i + chunk_size]
                chunk_results = list(executor.map(self.process_combination, chunk))
                
                # Process results from this chunk
                for result in chunk_results:
                    if result:
                        successful += 1
                        results_batch.append(result)
                    else:
                        failed += 1
                    processed += 1
                    
                    current_time = time.time()
                    # Write batch if it's full or enough time has passed
                    if len(results_batch) >= batch_size or (current_time - last_write_time) >= write_interval:
                        if results_batch:  # Only write if we have results
                            with open(results_file, 'a') as f:
                                f.writelines(results_batch)
                                f.flush()
                                os.fsync(f.fileno())
                            results_batch = []
                            last_write_time = current_time
                    
                    # Print progress
                    if processed % 100 == 0:
                        elapsed_time = time.time() - start_time
                        combinations_per_second = processed / elapsed_time
                        remaining_combinations = total_combinations - processed
                        estimated_time_remaining = remaining_combinations / combinations_per_second if combinations_per_second > 0 else 0
                        
                        print(f"Progress: {processed:,}/{total_combinations:,} "
                            f"({(processed/total_combinations)*100:.2f}%) "
                            f"ETA: {estimated_time_remaining/60:.1f} minutes")
        
        # Write any remaining results
        if results_batch:
            with open(results_file, 'a') as f:
                f.writelines(results_batch)
                f.flush()
                os.fsync(f.fileno())
        
        total_time = time.time() - start_time
        print(f"\n✅ Backtest completed in {total_time/60:.1f} minutes")
        print(f"Total combinations processed: {processed:,}")
        print(f"Successful: {successful:,}")
        print(f"Failed: {failed:,}")
        
        if errors:
            print("\nErrors encountered:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"- {error}")
            if len(errors) > 10:
                print(f"... and {len(errors) - 10} more errors")
        
        return True

    def buy_signal(self, ema: float, vwma: float, roc: float, macd: float, macd_signal: float) -> bool:
        """
        Check if we should buy
        """
        # Buy conditions
        return ema > vwma and roc > 0 and macd > macd_signal

    def sell_signal(self, ema: float, vwma: float, roc: float, macd: float, macd_signal: float) -> bool:
        """
        Check if we should sell
        """
        # Sell conditions
        return sum([ema < vwma, roc < 0, macd < macd_signal]) >= 2
        

if __name__ == "__main__":
    print("Starting backtest script...")

    backtest_strategy = BacktestStrategy(
        ema_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
        vwma_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
        roc_periods=[8, 10, 12],
        fast_emas=[12, 14, 16, 18, 20],
        slow_emas=[26, 28, 30, 32, 34],
        signal_emas=[9, 10, 11, 12, 13]
    )
    print("Backtest strategy initialized")
    print("Starting backtest...")
    # Run intraday backtests for SPY
    backtest_strategy.backtest("SPY", "5m")