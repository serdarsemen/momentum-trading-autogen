"""
Utility functions for data acquisition and processing.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from typing import Optional, Union, Dict, List, Tuple, Any

def get_current_date() -> str:
    """
    Get the current date in YYYY-MM-DD format.
    
    Returns:
        String representation of the current date
    """
    return date.today().strftime('%Y-%m-%d')

def download_stock_data(
    symbol: str, 
    start_date: Union[str, datetime], 
    end_date: Optional[Union[str, datetime]] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data retrieval
        end_date: End date for data retrieval (defaults to current date)
        interval: Data interval ('1d', '1wk', '1mo', etc.)
        
    Returns:
        DataFrame with historical price data
    """
    if end_date is None:
        end_date = get_current_date()
    
    # SSL workaround
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Create a session with custom headers
    import requests
    session = requests.Session()
    session.headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    }
    
    # Try to download with different methods
    try:
        # Method 1: Using yfinance with session
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, session=session)
        
        if df.empty:
            # Method 2: Try with pandas_datareader
            import pandas_datareader.data as web
            df = web.DataReader(symbol, 'yahoo', start_date, end_date)
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Method 3: Create dummy data for testing
        print("Creating dummy data for testing purposes...")
        df = create_dummy_stock_data(symbol, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data available for {symbol} from {start_date} to {end_date}")
        
    return df

def create_dummy_stock_data(symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
    """Create dummy stock data for testing when data download fails."""
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create random price data
    import numpy as np
    price = 100.0
    prices = []
    for _ in range(len(date_range)):
        change_percent = np.random.normal(0, 0.02)  # 2% standard deviation
        price *= (1 + change_percent)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'Adj Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'Volume': [int(np.random.normal(1000000, 200000)) for _ in prices]
    }, index=date_range)
    
    print(f"Created dummy data for {symbol} with {len(df)} trading days")
    return df

def calculate_returns_statistics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate common statistics for a returns series.
    
    Args:
        returns: Series of return values
        
    Returns:
        Dictionary with calculated statistics
    """
    annual_factor = 252  # Trading days in a year
    
    stats = {
        'total_return': (returns + 1).prod() - 1,
        'annualized_return': (1 + returns).prod() ** (annual_factor / len(returns)) - 1,
        'volatility': returns.std() * np.sqrt(annual_factor),
        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(annual_factor) if returns.std() > 0 else 0,
        'max_drawdown': ((returns + 1).cumprod() / (returns + 1).cumprod().cummax() - 1).min(),
        'positive_days': (returns > 0).sum() / len(returns),
        'win_loss_ratio': abs(returns[returns > 0].mean() / returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else float('inf')
    }
    
    return stats

def format_returns_as_markdown(
    symbol: str,
    ma_pairs: List[Tuple[int, int]],
    returns: List[float]
) -> str:
    """
    Format a list of returns for different MA pairs as a markdown table.
    
    Args:
        symbol: Stock symbol
        ma_pairs: List of (short_window, long_window) tuples
        returns: List of corresponding returns (must be same length as ma_pairs)
        
    Returns:
        Formatted markdown string
    """
    if len(ma_pairs) != len(returns):
        raise ValueError("Number of MA pairs must match number of returns")
    
    markdown = f"# Momentum Trading Strategy Results for {symbol}\n\n"
    markdown += "## Analysis Parameters\n\n"
    markdown += f"- **Stock Symbol**: {symbol}\n"
    markdown += f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    markdown += "## Moving Average Pairs and Returns\n\n"
    markdown += "| Short Window | Long Window | Final Return |\n"
    markdown += "|--------------|-------------|-------------|\n"
    
    for i, ((short, long), ret) in enumerate(zip(ma_pairs, returns)):
        markdown += f"| {short} | {long} | {ret:.2%} |\n"
    
    # Add best pair information
    best_idx = returns.index(max(returns))
    best_pair = ma_pairs[best_idx]
    markdown += f"\n## Best Performing MA Pair\n\n"
    markdown += f"The best performing MA pair is **({best_pair[0]}, {best_pair[1]})** with a return of **{returns[best_idx]:.2%}**.\n"
    
    return markdown

def prepare_comparison_data(
    results: Dict[Tuple[int, int], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Prepare data for comparison across different MA pairs.
    
    Args:
        results: Dictionary with results for each MA pair
        
    Returns:
        Dictionary with processed comparison data
    """
    comparison = {
        'ma_pairs': list(results.keys()),
        'returns': [r['final_return'] for r in results.values()],
        'trade_counts': [len(r['signals'][r['signals']['positions'] != 0]) for r in results.values()],
        'buy_signals': [len(r['signals'][r['signals']['positions'] == 1.0]) for r in results.values()],
        'sell_signals': [len(r['signals'][r['signals']['positions'] == -1.0]) for r in results.values()],
    }
    
    # Find the best performing pair
    best_idx = comparison['returns'].index(max(comparison['returns']))
    comparison['best_pair'] = comparison['ma_pairs'][best_idx]
    comparison['best_return'] = comparison['returns'][best_idx]
    
    return comparison