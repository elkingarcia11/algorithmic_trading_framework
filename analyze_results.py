import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import sys


class BacktestAnalyzer:
    def __init__(self, symbol, timeframe):
        """
        Initialize the BacktestAnalyzer with symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'SPY')
            timeframe (str): Timeframe for analysis (e.g., '5m', '15m')
        """
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.analysis_dir = f"data/backtest_results_{self.symbol}_{self.timeframe}"
        self.input_file = f"data/backtest_results_{self.symbol}_{self.timeframe}.csv"
        
        # Create analysis directory
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Validate input file exists
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Backtest results file not found: {self.input_file}")

    def create_ema_vwma_heatmap(self, df):
        """Create heatmap comparing EMA and VWMA periods."""
        output_file = os.path.join(self.analysis_dir, 'ema_vwma_heatmap.png')
        pivot_table = df.pivot_table(
            values='average_profit_percentage',
            index='ema_period',
            columns='vwma_period',
            aggfunc='mean'
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table,
                    annot=True,
                    fmt='.2%',
                    cmap='RdYlGn',
                    center=pivot_table.mean().mean(),
                    cbar_kws={'label': 'Average Profit Percentage'})
        plt.title('EMA vs VWMA Period - Average Profit Percentage')
        plt.xlabel('VWMA Period')
        plt.ylabel('EMA Period')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to: {output_file}")

    def create_roc_lineplots(self, df):
        """Create line plots for ROC period analysis."""
        output_file_winrate = os.path.join(self.analysis_dir, 'roc_winrate_lineplot.png')
        output_file_profit = os.path.join(self.analysis_dir, 'roc_profit_lineplot.png')
        
        plt.figure(figsize=(10, 6))
        win_rate_data = df.groupby('roc_period')['win_rate'].mean()
        plt.plot(win_rate_data.index, win_rate_data.values, marker='o')
        plt.title('ROC Period vs Win Rate')
        plt.xlabel('ROC Period')
        plt.ylabel('Win Rate (%)')
        plt.grid(True)
        plt.savefig(output_file_winrate, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        profit_data = df.groupby('roc_period')['average_profit_percentage'].mean()
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
        
        filtered_df = df[df['signal_ema'] == signal_ema_value]
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
        
        profit_pivot = df.pivot_table(
            values='average_profit_percentage',
            index='signal_ema',
            columns='vwma_period',
            aggfunc='mean'
        )
        winrate_pivot = df.pivot_table(
            values='win_rate',
            index='signal_ema',
            columns='vwma_period',
            aggfunc='mean'
        )
        
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
                'average_profit_percentage': 'mean'
            }).round(4)

            best_profit = param_stats.loc[param_stats['average_profit_percentage'].idxmax()]
            print(f"Best Profit Period: {param_stats['average_profit_percentage'].idxmax()}")
            print(f"  - Average Profit: {best_profit['average_profit_percentage']:.4%}")
            print(f"  - Win Rate: {best_profit['win_rate']:.4%}")

            best_winrate = param_stats.loc[param_stats['win_rate'].idxmax()]
            print(f"\nBest Win Rate Period: {param_stats['win_rate'].idxmax()}")
            print(f"  - Win Rate: {best_winrate['win_rate']:.4%}")
            print(f"  - Average Profit: {best_winrate['average_profit_percentage']:.4%}")

            print("\nAll periods sorted by average profit:")
            print(param_stats.sort_values('average_profit_percentage', ascending=False))

        # Create all visualizations
        self.create_ema_vwma_heatmap(df)
        self.create_roc_lineplots(df)
        self.create_fast_slow_ema_heatmaps(df, signal_ema_value=10)
        self.create_signal_ema_heatmaps(df)

    def clean_backtest_results(self):
        """Clean and process backtest results."""
        output_file = os.path.join(self.analysis_dir, 'cleaned_backtest_results.csv')

        # Read the CSV file
        df = pd.read_csv(self.input_file)

        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)

        # Save cleaned results
        df.to_csv(output_file, index=False)

        print("Cleaning complete:")
        print(f"Initial rows: {initial_rows}")
        print(f"Final rows: {final_rows}")
        print(f"Removed rows: {initial_rows - final_rows}")
        print(f"Results saved to: {output_file}")

        return df

    def analyze_backtest_results(self):
        """Main method to analyze backtest results."""
        # Clean the data
        df = self.clean_backtest_results()
        
        # Analyze parameters
        self.analyze_parameter_frequencies(df)


def main():
    symbol = "SPY"
    timeframe = "5m"
    try:
        analyzer = BacktestAnalyzer(symbol, timeframe)
        analyzer.analyze_backtest_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()