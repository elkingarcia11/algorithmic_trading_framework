import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import sys


class BacktestAnalyzer:
    def __init__(self, symbol, timeframe, ema_periods=None, vwma_periods=None, roc_periods=None, 
                 fast_emas=None, slow_emas=None, signal_emas=None):
        """
        Initialize the BacktestAnalyzer with symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'SPY')
            timeframe (str): Timeframe for analysis (e.g., '5m', '15m')
            ema_periods (list): List of EMA periods used in backtest
            vwma_periods (list): List of VWMA periods used in backtest
            roc_periods (list): List of ROC periods used in backtest
            fast_emas (list): List of fast EMA periods used in backtest
            slow_emas (list): List of slow EMA periods used in backtest
            signal_emas (list): List of signal EMA periods used in backtest
        """
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.analysis_dir = f"data/backtest_results_{self.symbol}_{self.timeframe}"
        self.input_file = f"data/backtest_results_{self.symbol}_{self.timeframe}.csv"
        
        # Store parameter ranges
        self.ema_periods = ema_periods or [3, 5, 8, 10, 12, 14, 16, 18, 20]
        self.vwma_periods = vwma_periods or [16, 17, 18]
        self.roc_periods = roc_periods or [3, 5, 8, 10, 12, 14, 16, 18, 20]
        self.fast_emas = fast_emas or [12, 14, 16, 18, 20]
        self.slow_emas = slow_emas or [26, 28, 30, 32, 34]
        self.signal_emas = signal_emas or [9, 10, 11, 12, 13]
        
        # Create analysis directory
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Validate input file exists
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Backtest results file not found: {self.input_file}")

    def create_ema_vwma_heatmap(self, df):
        """Create heatmaps comparing EMA and VWMA periods."""
        output_file_profit = os.path.join(self.analysis_dir, 'ema_vwma_profit_heatmap.png')
        output_file_total_profit = os.path.join(self.analysis_dir, 'ema_vwma_total_profit_heatmap.png')
        output_file_winrate = os.path.join(self.analysis_dir, 'ema_vwma_winrate_heatmap.png')
        
        # Filter data to only include the periods we want to analyze
        filtered_df = df[df['ema_period'].isin(self.ema_periods) & 
                        df['vwma_period'].isin(self.vwma_periods)]
        
        if filtered_df.empty:
            print("No data available for EMA vs VWMA heatmap")
            return
        
        # Create profit percentage heatmap
        profit_pivot = filtered_df.pivot_table(
            values='average_profit_percentage',
            index='ema_period',
            columns='vwma_period',
            aggfunc='mean'
        )
        
        # Create total profit percentage heatmap
        total_profit_pivot = filtered_df.pivot_table(
            values='total_profit_percentage',
            index='ema_period',
            columns='vwma_period',
            aggfunc='mean'
        )
        
        # Create win rate heatmap
        winrate_pivot = filtered_df.pivot_table(
            values='win_rate',
            index='ema_period',
            columns='vwma_period',
            aggfunc='mean'
        )
        
        # Ensure all periods are in the pivot tables
        profit_pivot = profit_pivot.reindex(index=self.ema_periods, columns=self.vwma_periods)
        total_profit_pivot = total_profit_pivot.reindex(index=self.ema_periods, columns=self.vwma_periods)
        winrate_pivot = winrate_pivot.reindex(index=self.ema_periods, columns=self.vwma_periods)
        
        # Skip if all values are NaN
        if profit_pivot.isna().all().all() or total_profit_pivot.isna().all().all() or winrate_pivot.isna().all().all():
            print("All values are NaN for EMA vs VWMA heatmap")
            return
        
        # Plot average profit percentage heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(profit_pivot,
                    annot=True,
                    fmt='.2%',
                    cmap='RdYlGn',
                    center=profit_pivot.mean().mean(),
                    cbar_kws={'label': 'Average Profit Percentage'})
        plt.title('EMA vs VWMA Period - Average Profit Percentage')
        plt.xlabel('VWMA Period')
        plt.ylabel('EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_profit, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot total profit percentage heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(total_profit_pivot,
                    annot=True,
                    fmt='.2%',
                    cmap='RdYlGn',
                    center=total_profit_pivot.mean().mean(),
                    cbar_kws={'label': 'Total Profit Percentage'})
        plt.title('EMA vs VWMA Period - Total Profit Percentage')
        plt.xlabel('VWMA Period')
        plt.ylabel('EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_total_profit, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot win rate heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(winrate_pivot,
                    annot=True,
                    fmt='.1f',
                    cmap='RdYlGn',
                    center=winrate_pivot.mean().mean(),
                    cbar_kws={'label': 'Win Rate (%)'})
        plt.title('EMA vs VWMA Period - Win Rate')
        plt.xlabel('VWMA Period')
        plt.ylabel('EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_winrate, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"EMA vs VWMA average profit heatmap saved to: {output_file_profit}")
        print(f"EMA vs VWMA total profit heatmap saved to: {output_file_total_profit}")
        print(f"EMA vs VWMA win rate heatmap saved to: {output_file_winrate}")

    def create_roc_lineplots(self, df):
        """Create line plots for ROC period analysis."""
        output_file_winrate = os.path.join(self.analysis_dir, 'roc_winrate_lineplot.png')
        output_file_profit = os.path.join(self.analysis_dir, 'roc_profit_lineplot.png')
        
        # Filter data to only include the periods we want to analyze
        filtered_df = df[df['roc_period'].isin(self.roc_periods)]
        
        if filtered_df.empty:
            print("No data available for ROC period analysis")
            return
        
        plt.figure(figsize=(10, 6))
        win_rate_data = filtered_df.groupby('roc_period')['win_rate'].mean()
        plt.plot(win_rate_data.index, win_rate_data.values, marker='o')
        plt.title('ROC Period vs Win Rate')
        plt.xlabel('ROC Period')
        plt.ylabel('Win Rate (%)')
        plt.grid(True)
        plt.savefig(output_file_winrate, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        profit_data = filtered_df.groupby('roc_period')['average_profit_percentage'].mean()
        plt.plot(profit_data.index, profit_data.values, marker='o')
        plt.title('ROC Period vs Average Profit Percentage')
        plt.xlabel('ROC Period')
        plt.ylabel('Average Profit Percentage')
        plt.grid(True)
        plt.savefig(output_file_profit, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC period vs win rate line plot saved to: {output_file_winrate}")
        print(f"ROC period vs average profit percentage line plot saved to: {output_file_profit}")

    def create_fast_slow_ema_heatmaps(self, df, signal_ema_value=10):
        """Create heatmaps for fast and slow EMA analysis."""
        output_file_profit = os.path.join(self.analysis_dir, f'fast_slow_ema_profit_heatmap_signal{signal_ema_value}.png')
        output_file_winrate = os.path.join(self.analysis_dir, f'fast_slow_ema_winrate_heatmap_signal{signal_ema_value}.png')
        
        # Filter data to only include the periods we want to analyze
        filtered_df = df[(df['signal_ema'] == signal_ema_value) & 
                        df['fast_ema'].isin(self.fast_emas) & 
                        df['slow_ema'].isin(self.slow_emas)]
        
        if filtered_df.empty:
            print(f"No data available for Fast vs Slow EMA heatmap (Signal EMA={signal_ema_value})")
            return
        
        profit_pivot = filtered_df.pivot_table(
            values='average_profit_percentage',
            index='fast_ema',
            columns='slow_ema',
            aggfunc='mean'
        )
        winrate_pivot = filtered_df.pivot_table(
            values='win_rate',
            index='fast_ema',
            columns='slow_ema',
            aggfunc='mean'
        )
        
        # Ensure all periods are in the pivot tables
        profit_pivot = profit_pivot.reindex(index=self.fast_emas, columns=self.slow_emas)
        winrate_pivot = winrate_pivot.reindex(index=self.fast_emas, columns=self.slow_emas)
        
        # Skip if all values are NaN
        if profit_pivot.isna().all().all() or winrate_pivot.isna().all().all():
            print(f"All values are NaN for Fast vs Slow EMA heatmap (Signal EMA={signal_ema_value})")
            return
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(profit_pivot,
                    annot=True,
                    fmt='.2%',
                    cmap='RdYlGn',
                    center=profit_pivot.mean().mean(),
                    cbar_kws={'label': 'Average Profit Percentage'})
        plt.title(f'Fast EMA vs Slow EMA - Average Profit Percentage (Signal EMA={signal_ema_value})')
        plt.xlabel('Slow EMA Period')
        plt.ylabel('Fast EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_profit, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(winrate_pivot,
                    annot=True,
                    fmt='.1f',
                    cmap='RdYlGn',
                    center=winrate_pivot.mean().mean(),
                    cbar_kws={'label': 'Win Rate (%)'})
        plt.title(f'Fast EMA vs Slow EMA - Win Rate (Signal EMA={signal_ema_value})')
        plt.xlabel('Slow EMA Period')
        plt.ylabel('Fast EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_winrate, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Fast EMA vs Slow EMA profit heatmap (Signal EMA={signal_ema_value}) saved to: {output_file_profit}")
        print(f"Fast EMA vs Slow EMA win rate heatmap (Signal EMA={signal_ema_value}) saved to: {output_file_winrate}")

    def create_signal_ema_heatmaps(self, df):
        """Create heatmaps for signal EMA analysis."""
        output_file_profit = os.path.join(self.analysis_dir, 'signal_ema_profit_heatmap.png')
        output_file_winrate = os.path.join(self.analysis_dir, 'signal_ema_winrate_heatmap.png')
        
        # Filter data to only include the periods we want to analyze
        filtered_df = df[df['signal_ema'].isin(self.signal_emas) & 
                        df['vwma_period'].isin(self.vwma_periods)]
        
        if filtered_df.empty:
            print("No data available for Signal EMA heatmap")
            return
        
        profit_pivot = filtered_df.pivot_table(
            values='average_profit_percentage',
            index='signal_ema',
            columns='vwma_period',
            aggfunc='mean'
        )
        winrate_pivot = filtered_df.pivot_table(
            values='win_rate',
            index='signal_ema',
            columns='vwma_period',
            aggfunc='mean'
        )
        
        # Ensure all periods are in the pivot tables
        profit_pivot = profit_pivot.reindex(index=self.signal_emas, columns=self.vwma_periods)
        winrate_pivot = winrate_pivot.reindex(index=self.signal_emas, columns=self.vwma_periods)
        
        # Skip if all values are NaN
        if profit_pivot.isna().all().all() or winrate_pivot.isna().all().all():
            print("All values are NaN for Signal EMA heatmap")
            return
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(profit_pivot,
                    annot=True,
                    fmt='.2%',
                    cmap='RdYlGn',
                    center=profit_pivot.mean().mean(),
                    cbar_kws={'label': 'Average Profit Percentage'})
        plt.title('Signal EMA vs VWMA Period - Average Profit Percentage')
        plt.xlabel('VWMA Period')
        plt.ylabel('Signal EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_profit, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(winrate_pivot,
                    annot=True,
                    fmt='.1f',
                    cmap='RdYlGn',
                    center=winrate_pivot.mean().mean(),
                    cbar_kws={'label': 'Win Rate (%)'})
        plt.title('Signal EMA vs VWMA Period - Win Rate')
        plt.xlabel('VWMA Period')
        plt.ylabel('Signal EMA Period')
        plt.tight_layout()
        plt.savefig(output_file_winrate, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Signal EMA vs VWMA profit heatmap saved to: {output_file_profit}")
        print(f"Signal EMA vs VWMA win rate heatmap saved to: {output_file_winrate}")

    def create_macd_heatmaps(self, df):
        """Create heatmaps for MACD parameters (fast EMA, slow EMA, signal EMA)."""
        # Create separate heatmaps for each signal EMA value
        for signal_ema in self.signal_emas:
            output_file_profit = os.path.join(self.analysis_dir, f'macd_profit_heatmap_signal{signal_ema}.png')
            output_file_total_profit = os.path.join(self.analysis_dir, f'macd_total_profit_heatmap_signal{signal_ema}.png')
            output_file_winrate = os.path.join(self.analysis_dir, f'macd_winrate_heatmap_signal{signal_ema}.png')
            
            # Filter data for this signal EMA value
            filtered_df = df[(df['signal_ema'] == signal_ema) & 
                           df['fast_ema'].isin(self.fast_emas) & 
                           df['slow_ema'].isin(self.slow_emas)]
            
            if filtered_df.empty:
                print(f"No data available for MACD heatmap (Signal EMA={signal_ema})")
                continue
            
            # Create profit percentage heatmap
            profit_pivot = filtered_df.pivot_table(
                values='average_profit_percentage',
                index='fast_ema',
                columns='slow_ema',
                aggfunc='mean'
            )
            
            # Create total profit percentage heatmap
            total_profit_pivot = filtered_df.pivot_table(
                values='total_profit_percentage',
                index='fast_ema',
                columns='slow_ema',
                aggfunc='mean'
            )
            
            # Create win rate heatmap
            winrate_pivot = filtered_df.pivot_table(
                values='win_rate',
                index='fast_ema',
                columns='slow_ema',
                aggfunc='mean'
            )
            
            # Ensure all periods are in the pivot tables
            profit_pivot = profit_pivot.reindex(index=self.fast_emas, columns=self.slow_emas)
            total_profit_pivot = total_profit_pivot.reindex(index=self.fast_emas, columns=self.slow_emas)
            winrate_pivot = winrate_pivot.reindex(index=self.fast_emas, columns=self.slow_emas)
            
            # Skip if all values are NaN
            if profit_pivot.isna().all().all() or total_profit_pivot.isna().all().all() or winrate_pivot.isna().all().all():
                print(f"All values are NaN for MACD heatmap (Signal EMA={signal_ema})")
                continue
            
            # Plot average profit percentage heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(profit_pivot,
                        annot=True,
                        fmt='.2%',
                        cmap='RdYlGn',
                        center=profit_pivot.mean().mean(),
                        cbar_kws={'label': 'Average Profit Percentage'})
            plt.title(f'MACD Fast vs Slow EMA - Average Profit Percentage (Signal EMA={signal_ema})')
            plt.xlabel('Slow EMA Period')
            plt.ylabel('Fast EMA Period')
            plt.tight_layout()
            plt.savefig(output_file_profit, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot total profit percentage heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(total_profit_pivot,
                        annot=True,
                        fmt='.2%',
                        cmap='RdYlGn',
                        center=total_profit_pivot.mean().mean(),
                        cbar_kws={'label': 'Total Profit Percentage'})
            plt.title(f'MACD Fast vs Slow EMA - Total Profit Percentage (Signal EMA={signal_ema})')
            plt.xlabel('Slow EMA Period')
            plt.ylabel('Fast EMA Period')
            plt.tight_layout()
            plt.savefig(output_file_total_profit, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot win rate heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(winrate_pivot,
                        annot=True,
                        fmt='.1f',
                        cmap='RdYlGn',
                        center=winrate_pivot.mean().mean(),
                        cbar_kws={'label': 'Win Rate (%)'})
            plt.title(f'MACD Fast vs Slow EMA - Win Rate (Signal EMA={signal_ema})')
            plt.xlabel('Slow EMA Period')
            plt.ylabel('Fast EMA Period')
            plt.tight_layout()
            plt.savefig(output_file_winrate, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"MACD average profit heatmap (Signal EMA={signal_ema}) saved to: {output_file_profit}")
            print(f"MACD total profit heatmap (Signal EMA={signal_ema}) saved to: {output_file_total_profit}")
            print(f"MACD win rate heatmap (Signal EMA={signal_ema}) saved to: {output_file_winrate}")

    def analyze_parameter_frequencies(self, df):
        """Analyze the frequency and performance of different parameter values."""
        parameters = ['vwma_period', 'roc_period', 'fast_ema', 'slow_ema', 'signal_ema']

        print("\nParameter Frequency Analysis:")
        print("=" * 50)

        for param in parameters:
            print(f"\n{param.upper()} Analysis:")
            print("-" * 30)

            param_stats = df.groupby(param).agg({
                'win_rate': 'mean',
                'average_profit_percentage': 'mean',
                'total_profit_percentage': 'mean'
            }).round(4)

            best_profit = param_stats.loc[param_stats['average_profit_percentage'].idxmax()]
            print(f"Best Average Profit Period: {param_stats['average_profit_percentage'].idxmax()}")
            print(f"  - Average Profit: {best_profit['average_profit_percentage']:.4%}")
            print(f"  - Total Profit: {best_profit['total_profit_percentage']:.4%}")
            print(f"  - Win Rate: {best_profit['win_rate']:.4%}")

            best_total_profit = param_stats.loc[param_stats['total_profit_percentage'].idxmax()]
            print(f"\nBest Total Profit Period: {param_stats['total_profit_percentage'].idxmax()}")
            print(f"  - Total Profit: {best_total_profit['total_profit_percentage']:.4%}")
            print(f"  - Average Profit: {best_total_profit['average_profit_percentage']:.4%}")
            print(f"  - Win Rate: {best_total_profit['win_rate']:.4%}")

            best_winrate = param_stats.loc[param_stats['win_rate'].idxmax()]
            print(f"\nBest Win Rate Period: {param_stats['win_rate'].idxmax()}")
            print(f"  - Win Rate: {best_winrate['win_rate']:.4%}")
            print(f"  - Average Profit: {best_winrate['average_profit_percentage']:.4%}")
            print(f"  - Total Profit: {best_winrate['total_profit_percentage']:.4%}")

            print("\nAll periods sorted by total profit:")
            print(param_stats.sort_values('total_profit_percentage', ascending=False))

        # Create all visualizations
        self.create_ema_vwma_heatmap(df)
        self.create_roc_lineplots(df)
        self.create_fast_slow_ema_heatmaps(df, signal_ema_value=10)
        self.create_signal_ema_heatmaps(df)
        self.create_macd_heatmaps(df)

    def analyze_backtest_results(self):
        """Main method to analyze backtest results."""
        # Read the original CSV file directly
        df = pd.read_csv(self.input_file)
        
        # Analyze parameters
        self.analyze_parameter_frequencies(df)


def main():
    symbol = "SPY"
    timeframe = "1m"
    ema_periods = [3, 5, 8, 10, 12, 14, 16, 18, 20]
    vwma_periods = [16, 17, 18]
    roc_periods = [3, 5, 8, 10, 12, 14, 16, 18, 20]
    fast_emas = [12, 14, 16, 18, 20]
    slow_emas = [26, 28, 30, 32, 34]
    signal_emas = [9, 10, 11, 12, 13]
    try:
        analyzer = BacktestAnalyzer(symbol, timeframe, ema_periods, vwma_periods, roc_periods, fast_emas, slow_emas, signal_emas)
        analyzer.analyze_backtest_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()