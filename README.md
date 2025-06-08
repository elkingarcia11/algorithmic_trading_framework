# Comprehensive Backtester

A comprehensive backtesting framework for algorithmic trading strategies, featuring data fetching from Schwab and Polygon.io, technical indicator generation, and strategy backtesting capabilities.

## Features

- **Data Fetching**:

  - Schwab API integration
    - Supports multiple timeframes (1m to 1d)
    - Automatic data deduplication
    - Configurable date ranges
    - Default symbols: MSFT, NVDA, AAPL, AMZN, META, TSLA, QQQ, SPY
  - Polygon.io integration (coming soon)
    - Real-time and historical data
    - Rate-limited API access
    - Multiple data types (stocks, options, forex)
    - Extended historical data coverage

- **Data Aggregation**: Aggregate data across different timeframes

  - OHLCV (Open, High, Low, Close, Volume) aggregation
  - Proper timestamp handling
  - Data validation and error handling

- **Technical Indicators**: Calculate various technical indicators:

  - EMA (Exponential Moving Average)
  - VWMA (Volume Weighted Moving Average)
  - MACD (Moving Average Convergence Divergence)
  - ROC (Rate of Change)

- **Backtesting**: Test trading strategies with customizable parameters

  - Multiple timeframe support
  - Configurable position sizing
  - Commission handling
  - Stop loss implementation
  - Support for both stock and options trading (coming soon)

- **Performance Metrics**: Comprehensive performance analysis including:
  - Win rate
  - Profit factor
  - Maximum drawdown
  - Sharpe ratio
  - Stop loss statistics
  - Average drawdown from max
  - Total return

## Project Structure

```
comprehensive_backtester/
├── data/                      # Directory for storing price data
├── calculate_indicators.py    # Technical indicator calculations
├── data_aggregator.py        # Data aggregation across timeframes
├── generate_indicators.py    # Generate indicator files
├── generate_backtests.py     # Run backtests on strategies
├── schwab_stock_data_fetcher.py  # Fetch data from Schwab
├── schwab_auth.py           # Schwab API authentication
├── polygon_options_fetcher.py  # Fetch options data from Polygon.io
├── rate_limiter.py         # API rate limiting implementation
└── requirements.txt         # Project dependencies
```

## Prerequisites

- Python 3.8+
- API Credentials:
  - Schwab API:
    - `schwab_credentials.env` file with:
      ```
      SCHWAB_APP_KEY=your_app_key_here
      SCHWAB_APP_SECRET=your_app_secret_here
      ```
    - `schwab_refresh_token.txt` file with:
      ```
      your_refresh_token_here
      ```
  - Polygon.io API (coming soon):
    - `polygon_credentials.env` file with:
      ```
      POLYGON_API_KEY=your_api_key_here
      ```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/comprehensive_backtester.git
cd comprehensive_backtester
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Set up API credentials:
   - Schwab API:
     - Create `schwab_credentials.env` in the project root:
       ```
       SCHWAB_APP_KEY=your_app_key_here
       SCHWAB_APP_SECRET=your_app_secret_here
       ```
     - Create `schwab_refresh_token.txt` in the project root:
       ```
       your_refresh_token_here
       ```
   - Polygon.io API (coming soon):
     - Create `polygon_credentials.env` in the project root:
       ```
       POLYGON_API_KEY=your_api_key_here
       ```

## Usage

### 1. Fetch Historical Data

#### Schwab Data

```python
from schwab_stock_data_fetcher import StockDataFetcher

# Initialize with default settings
fetcher = StockDataFetcher()

# Or customize settings
fetcher = StockDataFetcher(
    data_directory="data",
    symbols=["AAPL", "NVDA", "MSFT"],
    intervals=["1m", "5m", "15m"],
    start_date="2024-01-01",
    end_date="2024-03-01"
)

# Fetch data
fetcher.run_fetch()
```

#### Polygon.io Data (coming soon)

```python
from polygon_options_fetcher import PolygonOptionsFetcher

# Initialize with default settings
fetcher = PolygonOptionsFetcher()

# Or customize settings
fetcher = PolygonOptionsFetcher(
    data_directory="data",
    symbols=["AAPL", "NVDA"],
    start_date="2024-01-01",
    end_date="2024-03-01"
)

# Fetch data
fetcher.run_fetch()
```

### 2. Generate Technical Indicators

```python
from generate_indicators import IndicatorGenerator

# Initialize with default settings
generator = IndicatorGenerator()

# Or customize settings
generator = IndicatorGenerator(
    symbols=["AAPL", "NVDA"],
    ema_periods=[3, 5, 10, 20],
    vwma_periods=[5, 10, 20],
    roc_periods=[3, 5, 10],
    time_frames=["1m", "5m", "15m"]
)

# Generate indicators
generator.run_generation()
```

### 3. Run Backtests

```python
from generate_backtests import BacktestStrategy

# Initialize backtester
backtester = BacktestStrategy()

# Run backtests for specific symbols
backtester.run_backtests(["AAPL", "NVDA"])
```

## Configuration

### Supported Timeframes

- 1 minute (1m)
- 5 minutes (5m)
- 10 minutes (10m)
- 15 minutes (15m)
- 30 minutes (30m)
- 1 hour (1h)
- 2 hours (2h)
- 4 hours (4h)
- 1 day (1d)

### Default Parameters

- Initial capital: $100,000
- Position size: 10% of capital per trade
- Commission: 0.1% per trade
- Stop loss: 5% per trade

### Default Indicator Periods

- EMA: [3, 5, 6, 7, 9, 10, 14, 21, 28, 35, 42, 49]
- VWMA: [5, 10, 12, 17, 21, 28, 35, 42, 49]
- ROC: [3, 5, 7, 8, 10, 14, 21, 28, 35, 42, 49]

## Performance Metrics

The backtester calculates the following metrics:

- Total trades
- Win rate
- Profit factor
- Average win/loss
- Maximum drawdown
- Sharpe ratio
- Total return
- Stop loss statistics
- Average drawdown from max price

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
- Thanks to Polygon.io for providing the options data API
- Inspired by various algorithmic trading frameworks and strategies
