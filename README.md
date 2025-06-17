# Algorithmic Trading Framework

A comprehensive framework for backtesting trading strategies using technical indicators, featuring data fetching, indicator generation, backtesting, and results analysis.

## Features

- **Data Management**:

  - Organized data storage by symbol and timeframe
  - Automatic directory creation and management
  - Data validation and cleaning

- **Technical Indicators**:

  - EMA (Exponential Moving Average)
  - VWMA (Volume Weighted Moving Average)
  - ROC (Rate of Change)
  - Fast/Slow EMA combinations
  - Signal EMA for trade confirmation

- **Backtesting**:

  - Multiple timeframe support
  - Configurable parameter ranges
  - Comprehensive performance metrics
  - Parallel processing support

- **Analysis**:
  - Parameter optimization analysis
  - Performance visualization
  - Heatmap generation
  - Win rate analysis
  - Profit analysis

## Project Structure

```
algorithmic_trading_framework/
├── data/                      # Directory for storing price data and results
│   ├── backtest_results_*     # Backtest results by symbol and timeframe
│   └── indicators_*          # Generated indicators
├── market_data_fetcher.py    # Data fetching functionality
├── generate_indicators.py    # Technical indicator generation
├── generate_backtests.py     # Backtesting implementation
├── analyze_results.py        # Results analysis and visualization
├── workflow.py              # Main workflow orchestration
└── requirements.txt         # Project dependencies
```

## Prerequisites

- Python 3.8+
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/algorithmic_trading_framework.git
cd algorithmic_trading_framework
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Running the Complete Workflow

```python
from workflow import workflow

# Run with default settings
workflow()

# Or customize parameters
workflow(
    symbols=["SPY", "QQQ"],
    time_frames=["5m", "15m"],
    ema_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
    vwma_periods=[16, 17, 18],
    roc_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
    fast_emas=[12, 14, 16, 18, 20],
    slow_emas=[26, 28, 30, 32, 34],
    signal_emas=[9, 10, 11, 12, 13]
)
```

### 2. Individual Components

#### Generate Indicators

```python
from generate_indicators import IndicatorGenerator

generator = IndicatorGenerator(
    ema_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
    vwma_periods=[16, 17, 18],
    roc_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
    fast_emas=[12, 14, 16, 18, 20],
    slow_emas=[26, 28, 30, 32, 34],
    signal_emas=[9, 10, 11, 12, 13]
)
generator.generate_indicators("SPY", "5m")
```

#### Run Backtests

```python
from generate_backtests import BacktestStrategy

backtester = BacktestStrategy(
    ema_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
    vwma_periods=[16, 17, 18],
    roc_periods=[3, 5, 8, 10, 12, 14, 16, 18, 20],
    fast_emas=[12, 14, 16, 18, 20],
    slow_emas=[26, 28, 30, 32, 34],
    signal_emas=[9, 10, 11, 12, 13]
)
backtester.backtest("SPY", "5m")
```

#### Analyze Results

```python
from analyze_results import BacktestAnalyzer

analyzer = BacktestAnalyzer("SPY", "5m")
analyzer.analyze_backtest_results()
```

## Configuration

### Supported Timeframes

- 1 minute (1m)
- 5 minutes (5m)
- 15 minutes (15m)
- 30 minutes (30m)
- 1 hour (1h)
- 4 hours (4h)
- 1 day (1d)

### Default Parameters

#### Indicator Periods

- EMA: [3, 5, 8, 10, 12, 14, 16, 18, 20]
- VWMA: [16, 17, 18]
- ROC: [3, 5, 8, 10, 12, 14, 16, 18, 20]
- Fast EMA: [12, 14, 16, 18, 20]
- Slow EMA: [26, 28, 30, 32, 34]
- Signal EMA: [9, 10, 11, 12, 13]

## Analysis Output

The framework generates various analysis outputs:

1. **Heatmaps**:

   - EMA vs VWMA period analysis
   - Fast vs Slow EMA analysis
   - Signal EMA analysis

2. **Line Plots**:

   - ROC period vs Win Rate
   - ROC period vs Average Profit

3. **Parameter Analysis**:

   - Best performing parameter combinations
   - Win rate analysis
   - Profit analysis

4. **Performance Metrics**:
   - Win rate
   - Average profit percentage
   - Parameter frequency analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Schwab for providing the data API
- Inspired by various algorithmic trading frameworks and strategies
