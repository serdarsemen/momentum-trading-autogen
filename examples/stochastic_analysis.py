"""
Example of running a Stochastic Oscillator trading strategy analysis.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data
from src.strategies import run_stochastic_analysis

def main():
    # Example parameters
    symbol = "NVDA"
    start_date = "2023-01-01"
    k_period = 14
    d_period = 3
    smooth_k = 3
    
    try:
        # Download data
        df = download_stock_data(symbol, start_date)
        
        # Run Stochastic analysis
        results = run_stochastic_analysis(
            df,
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
            symbol=symbol,
            save_results=True,
            output_dir="output"
        )
        
        # Print results
        print(f"\nResults for {symbol} Stochastic Strategy:")
        print(f"Final Return: {results['final_return']:.2%}")
        print(f"Strategy plots saved to: {results['strategy_plot_path']}")
        print(f"Data saved to: {results['csv_path']}")
        
    except Exception as e:
        print(f"Error running Stochastic analysis: {e}")

if __name__ == "__main__":
    main()