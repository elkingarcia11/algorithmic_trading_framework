import logging
import os
from typing import List, Optional
from market_data_fetcher import MarketDataFetcher
from generate_indicators import IndicatorGenerator
from generate_backtests import BacktestStrategy
from analyze_results import BacktestAnalyzer
import csv
import pandas as pd
from datetime import datetime, timedelta
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_timeframe(timeframe: str) -> bool:
    """Validate if the timeframe is in the correct format."""
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    return timeframe in valid_timeframes

def validate_symbol(symbol: str) -> bool:
    """Validate if the symbol is in the correct format."""
    return bool(symbol and symbol.isalpha() and 1 <= len(symbol) <= 5)

def workflow(
    symbols_filepath:str= "symbols_to_fetch.txt",
    ema_periods: Optional[List[int]] = None,
    vwma_periods: Optional[List[int]] = None,
    roc_periods: Optional[List[int]] = None,
    fast_emas: Optional[List[int]] = None,
    slow_emas: Optional[List[int]] = None,
    signal_emas: Optional[List[int]] = None,
    time_frames: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    start_date: str = "2025-06-09",
    end_date: str = "2025-06-09"
) -> None:
    """
    Execute the complete trading strategy workflow.
    
    Args:
        symbols_filepath: Path to file containing symbols to analyze
        ema_periods: List of EMA periods to test
        vwma_periods: List of VWMA periods to test
        roc_periods: List of ROC periods to test
        fast_emas: List of fast EMA periods to test
        slow_emas: List of slow EMA periods to test
        signal_emas: List of signal EMA periods to test
        time_frames: List of timeframes to analyze
        symbols: List of symbols to analyze (if None, reads from symbols_filepath)
        start_date: Start date for backtesting
        end_date: End date for backtesting
    """
    try:
        # Get symbols from file if not provided
        if symbols is None:
            if not os.path.exists(symbols_filepath):
                raise FileNotFoundError(f"Symbols file not found: {symbols_filepath}")
            with open(symbols_filepath, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
        
        if not symbols:
            raise ValueError("No symbols provided or found in file")

        # Validate symbols
        for symbol in symbols:
            if not validate_symbol(symbol):
                raise ValueError(f"Invalid symbol format: {symbol}")

        market_data_fetcher = MarketDataFetcher()

        # Optimized parameter ranges
        ema_periods = ema_periods or [5,6,7]
        vwma_periods = vwma_periods or [6,7,8,15,16,17]
        roc_periods = roc_periods or [5,6,7,8,15,16,17]
        fast_emas = fast_emas or [14,15,16,25,26,27]
        slow_emas = slow_emas or [38,39,40]
        signal_emas = signal_emas or [10,11,12,15,16,17]
        time_frames = time_frames or ["1m", "5m"]

        # Validate timeframes
        for timeframe in time_frames:
            if not validate_timeframe(timeframe):
                raise ValueError(f"Invalid timeframe: {timeframe}")

        indicator_generator = IndicatorGenerator(
            ema_periods, vwma_periods, roc_periods,
            fast_emas, slow_emas, signal_emas
        )
        backtest_strategy = BacktestStrategy(
            ema_periods, vwma_periods, roc_periods,
            fast_emas, slow_emas, signal_emas
        )

        total_operations = len(symbols) * len(time_frames)
        completed_operations = 0

        for symbol in symbols:
            for timeframe in time_frames:
                try:
                    logger.info(f"Processing {symbol} on {timeframe} timeframe")
                    
                    # Fetch market data
                    logger.info(f"Fetching market data for {symbol}")
                    market_data_fetcher.get_price_history_from_schwab(
                        symbol, start_date, end_date, timeframe
                    )
                    
                    # Generate indicators
                    logger.info(f"Generating indicators for {symbol}")
                    indicator_generator.generate_indicators(symbol, timeframe)
                    
                    # Run backtest
                    logger.info(f"Running backtest for {symbol}")
                    backtest_strategy.backtest(symbol, timeframe)
                    
                    # Analyze results
                    logger.info(f"Analyzing results for {symbol}")
                    BacktestAnalyzer(
                        symbol, 
                        timeframe,
                        ema_periods=ema_periods,
                        vwma_periods=vwma_periods,
                        roc_periods=roc_periods,
                        fast_emas=fast_emas,
                        slow_emas=slow_emas,
                        signal_emas=signal_emas
                    ).analyze_backtest_results()
                    
                    completed_operations += 1
                    logger.info(f"Progress: {completed_operations}/{total_operations} operations completed")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} on {timeframe}: {str(e)}")
                    continue

        logger.info("Workflow completed successfully")

    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

def main():
    """Main entry point for the workflow."""
    try:
        workflow(time_frames=["1m"], symbols=["SPY"], start_date="2025-05-01", end_date="2025-06-17")
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

def run_daily_workflow():
    """
    Run the workflow from 2025-05-01 to 2025-06-17, one day at a time,
    and save the top 3 total_profit_percentage entries from each day to final_backtest_results.csv
    """
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 6, 17)
    current_date = start_date
    
    final_results_file = "final_backtest_results.csv"
    if not os.path.exists(final_results_file):
        with open(final_results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'symbol', 'ema_period', 'vwma_period', 'roc_period', 'fast_ema', 'slow_ema', 'signal_ema',
                'max_open_candles', 'average_open_candles', 'average_profit_percentage', 'total_profit_percentage',
                'win_rate', 'loss_rate', 'total_trade_count', 'max_win_percentage', 'max_loss_percentage'
            ])
    
    while current_date <= end_date:
        # Ensure current_date doesn't exceed end_date
        if current_date > end_date:
            current_date = end_date
            
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n📅 Processing date: {date_str}")
        
        # Delete and recreate the data directory for a clean run
        if os.path.exists('data'):
            shutil.rmtree('data')
        os.makedirs('data', exist_ok=True)
        
        try:
            # Run the workflow for this date
            workflow(time_frames=["1m"], symbols=["SPY"], start_date=date_str, end_date=date_str)
            
            # Process results for each symbol and timeframe
            for symbol in ['SPY']:
                for timeframe in ['1m']:
                    results_file = f"data/backtest_results_{symbol}_{timeframe}.csv"
                    if os.path.exists(results_file):
                        df = pd.read_csv(results_file)
                        if not df.empty:
                            top_3 = df.nlargest(3, 'total_profit_percentage')
                            top_3.to_csv(final_results_file, mode='a', header=False, index=False)
                            print(f"✅ Saved top 3 results for {symbol}_{timeframe} on {date_str}")
                            print(f"\nTop 3 results for {symbol}_{timeframe} on {date_str}:")
                            for _, row in top_3.iterrows():
                                print(f"Total Profit: {row['total_profit_percentage']:.2%} | "
                                      f"EMA: {row['ema_period']} | "
                                      f"VWMA: {row['vwma_period']} | "
                                      f"ROC: {row['roc_period']} | "
                                      f"MACD: {row['fast_ema']}_{row['slow_ema']}_{row['signal_ema']}")
        except Exception as e:
            print(f"❌ Error processing date {date_str}: {str(e)}")
        current_date += timedelta(days=1)
    print("\n✅ Daily workflow completed. Results saved to final_backtest_results.csv")

if __name__ == "__main__":
    # Run the daily workflow
    #run_daily_workflow()
    workflow(time_frames=["1m"], symbols=["SPY"], start_date="2025-05-01", end_date="2025-06-17")