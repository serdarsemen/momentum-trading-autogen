"""
Example of running a MACD trading strategy analysis.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data
from src.strategies import run_macd_analysis

def main():
    # Example parameters
    symbol = "NVDA"
    start_date = "2023-01-01"
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    try:
        # Download data
        df = download_stock_data(symbol, start_date)
        
        # Run MACD analysis
        results = run_macd_analysis(
            df,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            symbol=symbol,
            save_results=True,
            output_dir="output"
        )
        
        # Print results
        print(f"\nResults for {symbol} MACD Strategy:")
        print(f"Final Return: {results['final_return']:.2%}")
        print(f"Strategy plots saved to: {results['strategy_plot_path']}")
        print(f"Data saved to: {results['csv_path']}")
        
    except Exception as e:
        print(f"Error running MACD analysis: {e}")

if __name__ == "__main__":
    main()