"""
Backtest Strategy Module
Handles backtesting strategies
"""

import os
import pandas as pd
import numpy as np
import time
from typing import Dict, List

class BacktestStrategy:
    def __init__(self):
        self.initial_capital = 100000  # $100k initial capital
        self.position_size = 0.1  # 10% of capital per trade
        self.commission = 0.001  # 0.1% commission per trade
        
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'stop_loss_count': 0,
                'stop_loss_pct': 0,
                'avg_drawdown_from_max': 0
            }
            
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        # Count stop loss triggers
        stop_loss_trades = [t for t in trades if t.get('exit_reason') == "Stop Loss"]
        stop_loss_count = len(stop_loss_trades)
        stop_loss_pct = (stop_loss_count / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate average drawdown from max price
        avg_drawdown_from_max = sum(t['drawdown_from_max'] for t in trades) / total_trades
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate P&L metrics
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        # Calculate drawdown
        cumulative_returns = np.cumsum([t['pnl'] for t in trades])
        max_drawdown = 0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        # Calculate Sharpe Ratio (assuming daily returns)
        returns = pd.Series([t['pnl'] for t in trades])
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 else 0
        
        # Calculate total return
        total_return = (cumulative_returns[-1] / self.initial_capital) * 100 if len(cumulative_returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'stop_loss_count': stop_loss_count,
            'stop_loss_pct': stop_loss_pct,
            'avg_drawdown_from_max': avg_drawdown_from_max
        }
        
    def backtest(self, data: pd.DataFrame, ema_period: int, vwma_period: int, roc_period: int) -> Dict:
        """Run backtest on the data"""
        trades = []
        position = None
        entry_price = 0
        entry_time = None
        stop_loss_pct = 0.05  # 5% stop loss
        
        # Track capital over time
        capital = self.initial_capital
        capital_history = [{'timestamp': data.index[0], 'capital': capital}]
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]
            
            # Get indicator values
            ema = data[f'ema_{ema_period}'].iloc[i]
            vwma = data[f'vwma_{vwma_period}'].iloc[i]
            roc = data[f'roc_{roc_period}'].iloc[i]
            macd_line = data['macd_line'].iloc[i]
            macd_signal_line = data['macd_signal_line'].iloc[i]
            
            # Trading logic
            if position is None:  # No position
                if roc > 0 and ema > vwma and macd_line > macd_signal_line:  # Buy signal
                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    trade_size = capital * self.position_size
                    commission = trade_size * self.commission
                    capital -= commission  # Deduct commission from capital
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'position': position,
                        'size': trade_size,
                        'commission': commission,
                        'max_price': current_price,  # Highest price seen during trade
                        'max_price_time': current_time,  # When max price occurred
                        'drawdown_from_max': 0  # Will be calculated at exit
                    })
                    
            elif position == 'long':  # In long position
                # Update max price if current price is higher
                if current_price > trades[-1]['max_price']:
                    trades[-1]['max_price'] = current_price
                    trades[-1]['max_price_time'] = current_time
                
                # Check stop loss
                price_change = (current_price - entry_price) / entry_price
                stop_loss_triggered = price_change <= -stop_loss_pct
                
                # Check indicator signals
                ema_sell = ema < vwma
                roc_sell = roc < 0
                macd_sell = macd_line < macd_signal_line
                
                # Count how many sell signals we have
                sell_signals = sum([ema_sell, roc_sell, macd_sell])
                
                # Exit if stop loss hit or 2+ sell signals
                if stop_loss_triggered or sell_signals >= 2:
                    exit_price = current_price
                    trade_size = trades[-1]['size']
                    pnl = (exit_price - entry_price) / entry_price * trade_size
                    commission = trade_size * self.commission
                    pnl -= commission  # Deduct commission from P&L
                    capital += pnl  # Add P&L to capital
                    
                    # Calculate drawdown from max price
                    max_price = trades[-1]['max_price']
                    drawdown_from_max = (max_price - exit_price) / max_price
                    
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': "Stop Loss" if stop_loss_triggered else f"{sell_signals} Sell Signals",
                        'drawdown_from_max': drawdown_from_max * 100  # Convert to percentage
                    })
                    position = None
                    
            # Record capital at each step
            capital_history.append({
                'timestamp': current_time,
                'capital': capital
            })
                    
        # Close any open position at the end
        if position is not None:
            exit_price = data['close'].iloc[-1]
            trade_size = trades[-1]['size']
            pnl = (exit_price - entry_price) / entry_price * trade_size
            commission = trade_size * self.commission
            pnl -= commission
            capital += pnl
            
            # Calculate final drawdown from max price
            max_price = trades[-1]['max_price']
            drawdown_from_max = (max_price - exit_price) / max_price
            
            trades[-1].update({
                'exit_time': data.index[-1],
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': "End of Data",
                'drawdown_from_max': drawdown_from_max * 100  # Convert to percentage
            })
            
        # Calculate metrics including capital
        metrics = self.calculate_metrics(trades)
        metrics['final_capital'] = capital
        metrics['total_return_pct'] = ((capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown from capital history
        capital_values = [h['capital'] for h in capital_history]
        peak = capital_values[0]
        max_drawdown = 0
        
        for value in capital_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        metrics['max_drawdown_pct'] = max_drawdown * 100
        
        return metrics

    def run_backtests(symbols: List[str]):
        """Run backtests for all indicator files"""
        backtester = BacktestStrategy()
        results = []
        
        # Load existing results if any
        if os.path.exists("results.csv"):
            existing_results = pd.read_csv("results.csv")
            print(f"📊 Loaded {len(existing_results)} existing results")
        else:
            existing_results = pd.DataFrame()
        
        # Track progress
        total_files = 0
        processed_files = 0
        skipped_files = 0
        
        # Count total files first
        for symbol in symbols:
            for timeframe in ["1m", "5m", "10m", "15m", "30m", "1h", "4h"]:
                indicator_dir = f"data/{symbol}_{timeframe}"
                if os.path.exists(indicator_dir):
                    total_files += len([f for f in os.listdir(indicator_dir) if f.endswith('.csv')])
        
        print(f"📊 Starting backtests for {len(symbols)} symbols")
        print(f"   Total indicator files to process: {total_files}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Process each symbol
        for symbol in symbols:
            print(f"\n🔄 Processing {symbol}...")
            symbol_start = time.time()
            
            # Process each timeframe
            for timeframe in ["1m", "5m", "10m", "15m", "30m", "1h", "4h"]:
                indicator_dir = f"data/{symbol}_{timeframe}"
                if not os.path.exists(indicator_dir):
                    continue
                    
                # Process each indicator file
                for filename in os.listdir(indicator_dir):
                    if not filename.endswith('.csv'):
                        continue
                        
                    # Parse parameters from filename
                    params = filename.replace('.csv', '').split('_')
                    ema_period = int(params[1])
                    vwma_period = int(params[3])
                    roc_period = int(params[5])
                    
                    # Check if this test has already been run
                    test_exists = False
                    if not existing_results.empty:
                        mask = (
                            (existing_results['symbol'] == symbol) &
                            (existing_results['timeframe'] == timeframe) &
                            (existing_results['ema_period'] == ema_period) &
                            (existing_results['vwma_period'] == vwma_period) &
                            (existing_results['roc_period'] == roc_period)
                        )
                        if mask.any():
                            test_exists = True
                            skipped_files += 1
                            processed_files += 1
                            progress = (processed_files / total_files) * 100
                            print(f"⏭️  Skipping {symbol} {timeframe} EMA:{ema_period} VWma:{vwma_period} ROC:{roc_period} (already exists)")
                            continue
                    
                    # Load and process data
                    filepath = os.path.join(indicator_dir, filename)
                    data = pd.read_csv(filepath, index_col=0)
                    data.index = pd.to_datetime(data.index.astype(int), unit='ms')
                    
                    # Run backtest
                    metrics = backtester.backtest(data, ema_period, vwma_period, roc_period)
                    
                    # Create result row
                    result = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'ema_period': ema_period,
                        'vwma_period': vwma_period,
                        'roc_period': roc_period,
                        **metrics
                    }
                    
                    # Append to results list
                    results.append(result)
                    
                    # Save results after each test
                    if not existing_results.empty:
                        # Append new result to existing results
                        existing_results = pd.concat([existing_results, pd.DataFrame([result])], ignore_index=True)
                        # Sort and save
                        existing_results = existing_results.sort_values('total_return', ascending=False)
                        existing_results.to_csv("results.csv", index=False)
                    else:
                        # Create new results file
                        pd.DataFrame([result]).to_csv("results.csv", index=False)
                        existing_results = pd.read_csv("results.csv")
                    
                    # Update progress
                    processed_files += 1
                    progress = (processed_files / total_files) * 100
                    print(f"🔍 Progress: {progress:.1f}% - {symbol} {timeframe} EMA:{ema_period} VWma:{vwma_period} ROC:{roc_period}")
                    
            # Print symbol summary
            symbol_time = time.time() - symbol_start
            print(f"✅ {symbol} complete in {symbol_time:.1f} seconds")
        
        # Print final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("🎉 BACKTESTING COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Files processed: {processed_files}")
        print(f"Files skipped: {skipped_files}")
        print(f"Average time per file: {total_time/(processed_files-skipped_files):.2f} seconds")
        print(f"Results saved to: results.csv")
        print("=" * 80)
        
        # Print top 5 strategies
        print("\n🏆 TOP 5 STRATEGIES:")
        print("=" * 80)
        top_5 = pd.read_csv("results.csv").head()
        for _, row in top_5.iterrows():
            print(f"Symbol: {row['symbol']} {row['timeframe']}")
            print(f"Parameters: EMA:{row['ema_period']} VWMA:{row['vwma_period']} ROC:{row['roc_period']}")
            print(f"Return: {row['total_return']:.2f}% | Win Rate: {row['win_rate']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"Stop Loss: {row['stop_loss_count']} trades ({row['stop_loss_pct']:.1f}%)")
            print(f"Avg Drawdown from Max: {row['avg_drawdown_from_max']:.1f}%")
            print("-" * 40)

    def test_single_backtest():
        """Test backtest on a single indicator file"""
        backtester = BacktestStrategy()
        
        # Test parameters
        symbol = "AAPL"
        timeframe = "1h"
        ema_period = 3
        vwma_period = 5
        roc_period = 3
        
        # Load data
        filepath = f"data/{symbol}_{timeframe}/ema_{ema_period}_vwma_{vwma_period}_roc_{roc_period}.csv"
        print(f"📊 Testing {filepath}")
        print("=" * 80)
        
        data = pd.read_csv(filepath, index_col=0)
        data.index = pd.to_datetime(data.index.astype(int), unit='ms')
        
        # Run backtest
        metrics = backtester.backtest(data, ema_period, vwma_period, roc_period)
        
        # Create results DataFrame
        results_df = pd.DataFrame([{
            'symbol': symbol,
            'timeframe': timeframe,
            'ema_period': ema_period,
            'vwma_period': vwma_period,
            'roc_period': roc_period,
            **metrics
        }])
        
        # Save to results.csv
        results_df.to_csv("results.csv", index=False)
        
        # Print detailed results
        print(f"\n🔍 Strategy Details:")
        print(f"Symbol: {symbol} {timeframe}")
        print(f"Parameters: EMA:{ema_period} VWMA:{vwma_period} ROC:{roc_period}")
        print("\n📈 Performance Metrics:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"Stop Loss Triggers: {metrics['stop_loss_count']} ({metrics['stop_loss_pct']:.1f}%)")
        print(f"Avg Drawdown from Max: {metrics['avg_drawdown_from_max']:.1f}%")
        print("\n💰 Trade Statistics:")
        print(f"Average Win: ${metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"\n✅ Results saved to results.csv")

if __name__ == "__main__":
    backtest_strategy = BacktestStrategy()
    backtest_strategy.run_backtests(["AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA", "QQQ", "SPY"])  # Full backtest 