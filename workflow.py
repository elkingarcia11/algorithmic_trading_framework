import logging
import os
from typing import List, Optional
from market_data_fetcher import MarketDataFetcher
from generate_indicators import IndicatorGenerator
from generate_backtests import BacktestStrategy
from analyze_results import BacktestAnalyzer

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
    symbols_filepath: str = "symbols_to_fetch.txt",
    ema_periods: Optional[List[int]] = None,
    vwma_periods: Optional[List[int]] = None,
    roc_periods: Optional[List[int]] = None,
    fast_emas: Optional[List[int]] = None,
    slow_emas: Optional[List[int]] = None,
    signal_emas: Optional[List[int]] = None,
    time_frames: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    start_date: str = "2025-06-16",
    end_date: str = "2025-06-16"
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
        ema_periods = ema_periods or [3, 5, 8, 10, 12, 14, 16, 18, 20]
        vwma_periods = vwma_periods or [16, 17, 18]
        roc_periods = roc_periods or [3, 5, 8, 10, 12, 14, 16, 18, 20]
        fast_emas = fast_emas or [12, 14, 16, 18, 20]
        slow_emas = slow_emas or [26, 28, 30, 32, 34]
        signal_emas = signal_emas or [9, 10, 11, 12, 13]
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
                    BacktestAnalyzer(symbol, timeframe).analyze_backtest_results()
                    
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
        workflow()
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()