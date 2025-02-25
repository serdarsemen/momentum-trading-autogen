# examples/manual_momentum_analysis.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import run_strategy_analysis

def run_manual_analysis():
    """
    Run a manual analysis without using AutoGen agents, useful for testing.
    Uses environment variables to get parameters.
    """
    # Get parameters from environment variables or use defaults
    symbol = os.environ.get('ANALYSIS_SYMBOL', 'NVDA')
    start_date = os.environ.get('ANALYSIS_START_DATE', '2024-01-01')
    end_date = os.environ.get('ANALYSIS_END_DATE', get_current_date())
    short_window = int(os.environ.get('ANALYSIS_SHORT_WINDOW', '5'))
    long_window = int(os.environ.get('ANALYSIS_LONG_WINDOW', '20'))
    output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', 'output')
    
    print(f"Running manual momentum analysis for {symbol} ({start_date} to {end_date})")
    print(f"Parameters: short_window={short_window}, long_window={long_window}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Configure SSL for Yahoo Finance
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
            
        # Now download data
        df = download_stock_data(symbol, start_date, end_date)
        print(f"Downloaded {len(df)} days of data")
        
        # Run strategy
        results = run_strategy_analysis(
            df,
            short_window,
            long_window,
            symbol=symbol,
            save_results=True,
            output_dir=output_dir
        )
        
        # Print results
        print(f"Analysis complete!")
        print(f"Final return: {results['final_return']:.2%}")
        print(f"Results saved to {output_dir}")
        
        # Show plot
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Error in manual analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_manual_analysis()