from technical_indicators import TechnicalIndicators
from options_api import OptionsAPI
import pandas as pd
import os
from datetime import datetime

class StrategyOptimizer:
    def __init__(self, output_directory: str = "optimization_results"):
        """Initialize the StrategyOptimizer class"""
        self.options_api = OptionsAPI()
        self.technical_indicators = TechnicalIndicators()
        self.time_frames = ["1m", "5m", "10m", "15m", "30m", "1h", "4h"]
        self.ema_periods = [3, 5, 6, 7, 9, 10, 14, 21, 28, 35, 42, 49]
        self.vwma_periods = [5, 10, 12, 17, 21, 28, 35, 42, 49]
        self.roc_periods = [3, 5, 7, 8, 10, 14, 21, 28, 35, 42, 49]
        
        # Create output directory for results
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Initialize results storage
        self.optimization_results = []
        
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
                success = self.technical_indicators.calculate_indicators(data_source, ema_period, vwma_period, roc_period)
                if not success:
                    print(f"❌ Failed to generate indicators for {data_source}")
                    continue
                
        print(f"✅ Indicator file generation complete for {symbol}")
        return True
        
    def backtest_combinations(self, symbol: str) -> str:
        """Backtest all indicator combinations for a symbol"""
        print(f"🔄 Starting backtesting for {symbol}...")
        
        # Clear previous results
        self.optimization_results = []
        
        # Generate output filename
        output_path = os.path.join(self.output_directory, f"{symbol}_optimization.txt")
        
        # Create a list of all combinations of ema, vwma, roc periods
        combinations = []
        for ema_period in self.ema_periods:
            for vwma_period in self.vwma_periods:
                for roc_period in self.roc_periods:
                    combinations.append((ema_period, vwma_period, roc_period))

        total_combinations = len(combinations) * len(self.time_frames)
        current_combination = 0
        
        print(f"📊 Testing {len(combinations)} parameter combinations across {len(self.time_frames)} timeframes")
        print(f"   Total tests: {total_combinations}")
        
        # Start writing results to file
        with open(output_path, 'w') as f:
            self._write_header(f, symbol, len(combinations))
            
            # For each combination, backtest each timeframe
            for combination in combinations:
                ema_period, vwma_period, roc_period = combination
                
                # For each time frame, run backtest
                for time_frame in self.time_frames:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    print(f"🔍 Progress: {progress:.1f}% - Testing {symbol} {time_frame} EMA:{ema_period} VWMA:{vwma_period} ROC:{roc_period}")
                    
                    # Path to the indicator file
                    indicator_file = f"data/{symbol}_{time_frame}/ema_{ema_period}_vwma_{vwma_period}_roc_{roc_period}.csv"
                    
                    # Check if indicator file exists
                    if not os.path.exists(indicator_file):
                        print(f"⚠️  Skipping {indicator_file} - file not found")
                        continue
                    
                    # Run backtest
                    result = self.backtest_strategy(indicator_file, symbol, ema_period, vwma_period, roc_period, time_frame)
                    
                    if result:
                        self.optimization_results.append(result)
                        self._write_result_to_file(f, result)
            
            # Write summary statistics
            self._write_summary_to_file(f, symbol)
        
        print(f"✅ Backtesting complete! Results saved to: {output_path}")
        print(f"📈 Tested {len(self.optimization_results)} valid combinations")
        
        return output_path
        
    def optimize_indicators(self, symbol: str) -> str:
        """Run complete optimization process: generate indicators then backtest"""
        print(f"🔄 Starting complete optimization for {symbol}...")
        
        # Step 1: Generate all indicator files
        if not self.generate_indicator_files(symbol):
            print("❌ Failed to generate indicator files")
            return None
            
        # Step 2: Run backtesting on all combinations
        results_file = self.backtest_combinations(symbol)
        
        return results_file

    def backtest_strategy(self, file: str, symbol: str, ema_period: int, vwma_period: int, 
                         roc_period: int, time_frame: str) -> dict:
        """Backtest strategy and return results"""
        try:
            data = pd.read_csv(file)
        except FileNotFoundError:
            print(f"❌ File not found: {file}")
            return None
        
        if data.empty:
            print(f"⚠️  No data in file: {file}")
            return None
            
        position_open = False
        entries = []
        entry = {}

        for index, row in data.iterrows():
            if position_open:
                # Get option symbol in entry
                option_symbol = entry['option_symbol']
                
                try:
                    # Get current option price
                    current_price = self.options_api.get_option_price(option_symbol, row['timestamp'])
                    # Calculate unrealized profit/loss percentage
                    entry['profit'] = (current_price - entry['entry_price']) / entry['entry_price'] * 100
                except Exception as e:
                    # Close position due to data error
                    entries.append(entry)
                    position_open = False
                    entry = {}
                    continue
                
                # Count how many sell conditions are true
                sell_conditions = [
                    row[f'ema_{ema_period}'] < row[f'vwma_{vwma_period}'],
                    row['macd_line'] < row['macd_signal_line'],
                    row[f'roc_{roc_period}'] < 0
                ]
                
                # Close position if any 2 of 3 sell conditions are true OR unrealized loss >= 5%
                if sum(sell_conditions) >= 2 or entry['profit'] <= -5:
                    entries.append(entry)
                    position_open = False
                    entry = {}
            else:
                # Count how many buy conditions are true
                buy_conditions = [
                    row[f'ema_{ema_period}'] > row[f'vwma_{vwma_period}'],
                    row['macd_line'] > row['macd_signal_line'],
                    row[f'roc_{roc_period}'] > 0
                ]
                if sum(buy_conditions) >= 3:
                    position_open = True
                    option_symbol = self.options_api.generate_option_symbol_for_date(symbol, row['close'], 'C', row['timestamp'], 2)
                    
                    try:
                        # Get price for option symbol on this datetime
                        entry['entry_price'] = self.options_api.get_option_price(option_symbol, row['timestamp'])
                        entry['option_symbol'] = option_symbol
                    except Exception as e:
                        position_open = False
                        continue
                    
        # Calculate summary statistics
        result = self._calculate_summary_stats(entries, symbol, ema_period, vwma_period, roc_period, time_frame)
        return result

    def _calculate_summary_stats(self, entries: list, symbol: str, ema_period: int, 
                                vwma_period: int, roc_period: int, time_frame: str) -> dict:
        """Calculate summary statistics for the backtest"""
        if len(entries) == 0:
            return {
                'symbol': symbol,
                'timeframe': time_frame,
                'ema_period': ema_period,
                'vwma_period': vwma_period,
                'roc_period': roc_period,
                'total_trades': 0,
                'total_profit': 0,
                'win_rate': 0,
                'average_profit': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0
            }
            
        profits = [entry['profit'] for entry in entries]
        total_profit = sum(profits)
        winning_trades = sum(1 for profit in profits if profit > 0)
        losing_trades = sum(1 for profit in profits if profit <= 0)
        win_rate = (winning_trades / len(entries)) * 100
        average_profit = total_profit / len(entries)
        
        # Additional metrics
        max_profit = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        # Calculate profit factor (gross profit / gross loss)
        gross_profit = sum(profit for profit in profits if profit > 0)
        gross_loss = abs(sum(profit for profit in profits if profit < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Simple Sharpe ratio approximation
        profit_std = pd.Series(profits).std() if len(profits) > 1 else 0
        sharpe_ratio = average_profit / profit_std if profit_std > 0 else 0
        
        return {
            'symbol': symbol,
            'timeframe': time_frame,
            'ema_period': ema_period,
            'vwma_period': vwma_period,
            'roc_period': roc_period,
            'total_trades': len(entries),
            'total_profit': round(total_profit, 2),
            'win_rate': round(win_rate, 2),
            'average_profit': round(average_profit, 2),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }

    def _write_header(self, file, symbol: str, total_combinations: int):
        """Write header information to the results file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write("=" * 80 + "\n")
        file.write(f"STRATEGY OPTIMIZATION RESULTS\n")
        file.write(f"Symbol: {symbol}\n")
        file.write(f"Generated: {timestamp}\n")
        file.write(f"Total Parameter Combinations: {total_combinations}\n")
        file.write(f"Timeframes: {', '.join(self.time_frames)}\n")
        file.write("=" * 80 + "\n\n")

    def _write_result_to_file(self, file, result: dict):
        """Write individual result to file"""
        file.write(f"Timeframe: {result['timeframe']} | ")
        file.write(f"EMA: {result['ema_period']} | ")
        file.write(f"VWMA: {result['vwma_period']} | ")
        file.write(f"ROC: {result['roc_period']}\n")
        file.write(f"  Trades: {result['total_trades']} | ")
        file.write(f"Profit: {result['total_profit']}% | ")
        file.write(f"Win Rate: {result['win_rate']}% | ")
        file.write(f"Avg: {result['average_profit']}% | ")
        file.write(f"PF: {result['profit_factor']} | ")
        file.write(f"Sharpe: {result['sharpe_ratio']}\n")
        file.write("-" * 80 + "\n")

    def _write_summary_to_file(self, file, symbol: str):
        """Write summary statistics to file"""
        if not self.optimization_results:
            file.write("\nNo valid results to summarize.\n")
            return
        
        file.write("\n" + "=" * 80 + "\n")
        file.write("OPTIMIZATION SUMMARY\n")
        file.write("=" * 80 + "\n")
        
        # Sort by total profit descending
        sorted_results = sorted(self.optimization_results, key=lambda x: x['total_profit'], reverse=True)
        
        file.write(f"\nTOP 10 BEST PERFORMING COMBINATIONS (by Total Profit):\n")
        file.write("-" * 80 + "\n")
        for i, result in enumerate(sorted_results[:10], 1):
            file.write(f"{i:2d}. {result['timeframe']:>4} | ")
            file.write(f"EMA:{result['ema_period']:2d} VWMA:{result['vwma_period']:2d} ROC:{result['roc_period']:2d} | ")
            file.write(f"Profit: {result['total_profit']:7.2f}% | ")
            file.write(f"Trades: {result['total_trades']:3d} | ")
            file.write(f"Win: {result['win_rate']:5.1f}%\n")
        
        # Sort by win rate
        sorted_by_winrate = sorted(self.optimization_results, key=lambda x: x['win_rate'], reverse=True)
        file.write(f"\nTOP 10 HIGHEST WIN RATES:\n")
        file.write("-" * 80 + "\n")
        for i, result in enumerate(sorted_by_winrate[:10], 1):
            file.write(f"{i:2d}. {result['timeframe']:>4} | ")
            file.write(f"EMA:{result['ema_period']:2d} VWMA:{result['vwma_period']:2d} ROC:{result['roc_period']:2d} | ")
            file.write(f"Win Rate: {result['win_rate']:5.1f}% | ")
            file.write(f"Trades: {result['total_trades']:3d} | ")
            file.write(f"Profit: {result['total_profit']:7.2f}%\n")
        
        # Overall statistics
        total_trades = sum(r['total_trades'] for r in self.optimization_results)
        avg_profit = sum(r['total_profit'] for r in self.optimization_results) / len(self.optimization_results)
        profitable_strategies = sum(1 for r in self.optimization_results if r['total_profit'] > 0)
        
        file.write(f"\nOVERALL STATISTICS:\n")
        file.write("-" * 80 + "\n")
        file.write(f"Total Strategies Tested: {len(self.optimization_results)}\n")
        file.write(f"Profitable Strategies: {profitable_strategies} ({profitable_strategies/len(self.optimization_results)*100:.1f}%)\n")
        file.write(f"Total Trades Across All Tests: {total_trades}\n")
        file.write(f"Average Profit Across All Strategies: {avg_profit:.2f}%\n")
        file.write(f"Best Single Strategy Profit: {sorted_results[0]['total_profit']:.2f}%\n")
        file.write(f"Worst Single Strategy Profit: {sorted_results[-1]['total_profit']:.2f}%\n")

    def print_summary(self, entries, symbol, ema_period, vwma_period, roc_period):
        """Print summary of entries (legacy method - kept for compatibility)"""
        print(f"Symbol: {symbol}")
        print(f"EMA Period: {ema_period}")
        print(f"VWMA Period: {vwma_period}")
        print(f"ROC Period: {roc_period}")
        print(f"Total entries: {len(entries)}")
        
        # Avoid division by zero
        if len(entries) == 0:
            print("No trades executed")
            return
            
        total_profit = sum(entry['profit'] for entry in entries)
        winning_trades = sum(1 for entry in entries if entry['profit'] > 0)
        losing_trades = sum(1 for entry in entries if entry['profit'] <= 0)
        win_rate = (winning_trades / len(entries)) * 100
        average_profit = total_profit / len(entries)
        
        print(f"Total profit: {total_profit}%")
        print(f"Win rate: {win_rate}%")
        print(f"Average profit: {average_profit}%")
        print(f"Total losing trades: {losing_trades}")
        print(f"Total winning trades: {winning_trades}")
        print("-" * 50)


# Test function
def test_optimization():
    """Test the strategy optimization with file output"""
    optimizer = StrategyOptimizer()
    
    # Test with a single symbol
    symbol = "AAPL"
    print(f"🧪 Testing optimization for {symbol}...")
    
    # Test generating indicators only
    print("\n=== Testing Indicator Generation ===")
    optimizer.generate_indicator_files(symbol)
    
    # Test backtesting only
    print("\n=== Testing Backtesting ===")
    results_file = optimizer.backtest_combinations(symbol)
    print(f"📄 Results saved to: {results_file}")
    
    # Test complete optimization
    print("\n=== Testing Complete Optimization ===")
    results_file = optimizer.optimize_indicators(symbol)
    print(f"📄 Results saved to: {results_file}")


if __name__ == "__main__":
    test_optimization()