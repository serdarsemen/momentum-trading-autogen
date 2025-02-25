# examples/multi_stock_analysis.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import run_strategy_analysis

def analyze_stock(symbol, start_date, end_date, ma_pair, output_dir):
    """Analyze a single stock with the given MA pair."""
    try:
        print(f"Analyzing {symbol} with MA pair {ma_pair}...")
        
        # Create stock-specific directory
        stock_dir = os.path.join(output_dir, symbol)
        os.makedirs(stock_dir, exist_ok=True)
        
        # Download data
        df = download_stock_data(symbol, start_date, end_date)
        print(f"Downloaded {len(df)} days of data for {symbol}")
        
        # Run strategy
        result = run_strategy_analysis(
            df,
            ma_pair[0],
            ma_pair[1],
            symbol=symbol,
            save_results=True,
            output_dir=stock_dir
        )
        
        return symbol, result
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return symbol, None

def run_multi_stock_analysis():
    """
    Run momentum analysis for multiple stocks.
    Uses environment variables to get parameters.
    """
    # Get parameters from environment variables or use defaults
    start_date = os.environ.get('ANALYSIS_START_DATE', '2024-01-01')
    end_date = os.environ.get('ANALYSIS_END_DATE', get_current_date())
    short_window = int(os.environ.get('ANALYSIS_SHORT_WINDOW', '10'))
    long_window = int(os.environ.get('ANALYSIS_LONG_WINDOW', '50'))
    output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', 'output')
    
    ma_pair = (short_window, long_window)
    
    # Default stock symbols
    symbols = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # If a specific symbol was set, add it to the front
    specific_symbol = os.environ.get('ANALYSIS_SYMBOL')
    if specific_symbol and specific_symbol not in symbols:
        symbols.insert(0, specific_symbol)
    
    print(f"Running momentum analysis for multiple stocks with MA pair {ma_pair}")
    
    # Configure SSL
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each stock (parallel for efficiency)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_stock = {
            executor.submit(analyze_stock, symbol, start_date, end_date, ma_pair, output_dir): symbol
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_stock):
            symbol, result = future.result()
            if result:
                results[symbol] = result
    
    # Create summary table
    summary = []
    for symbol, result in results.items():
        if result:
            summary.append({
                'Symbol': symbol,
                'Return': result['final_return'],
                'Buy Signals': len(result['signals'][result['signals']['positions'] == 1.0]),
                'Sell Signals': len(result['signals'][result['signals']['positions'] == -1.0])
            })
    
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.sort_values('Return', ascending=False, inplace=True)
        
        # Save summary
        summary_path = os.path.join(output_dir, "multi_stock_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print("\nResults Summary:")
        print(summary_df)
        print(f"Summary saved to {summary_path}")
    else:
        print("No valid results generated for any stocks.")
    
    print("Analysis complete!")
    return results

if __name__ == "__main__":
    run_multi_stock_analysis()