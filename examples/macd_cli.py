from src.strategies import run_macd_analysis
from src.utils import download_stock_data

# Download data
df = download_stock_data("AAPL", "2023-01-01")

# Run MACD analysis
results = run_macd_analysis(
    df,
    fast_period=12,
    slow_period=26,
    signal_period=9,
    symbol="AAPL"
)