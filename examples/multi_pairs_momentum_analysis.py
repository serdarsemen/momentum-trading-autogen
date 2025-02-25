# examples/multi_pairs_momentum_analysis.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date, format_returns_as_markdown
from src.strategies import run_strategy_analysis

def run_multi_pair_analysis():
    """
    Run momentum analysis with multiple MA pairs.
    Uses environment variables to get parameters.
    """
    # Get parameters from environment variables or use defaults
    symbol = os.environ.get('ANALYSIS_SYMBOL', 'NVDA')
    start_date = os.environ.get('ANALYSIS_START_DATE', '2024-01-01')
    end_date = os.environ.get('ANALYSIS_END_DATE', get_current_date())
    output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', 'output')
    
    # Default MA pairs
    ma_pairs = [(5, 20), (10, 50), (20, 100), (50, 200)]
    
    print(f"Running momentum analysis for {symbol} with multiple MA pairs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure SSL
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download data once
    try:
        df = download_stock_data(symbol, start_date, end_date)
        print(f"Downloaded {len(df)} days of data")
        
        # Results container
        results = {}
        returns = []
        
        # Analyze each MA pair
        for short_window, long_window in ma_pairs:
            print(f"Analyzing MA pair ({short_window}, {long_window})...")
            result = run_strategy_analysis(
                df.copy(),  # Use a copy to avoid modifying original
                short_window,
                long_window,
                symbol=symbol,
                save_results=True,
                output_dir=output_dir
            )
            
            results[(short_window, long_window)] = result
            returns.append(result['final_return'])
            
            print(f"Final return: {result['final_return']:.2%}")
        
        # Create markdown report
        report = format_returns_as_markdown(symbol, ma_pairs, returns)
        
        # Save report
        report_path = os.path.join(output_dir, "momentum_analysis_report.md")
        with open(report_path, "w") as f:
            f.write(report)
            
        print(f"Analysis complete! Report saved to {report_path}")
        
        # Find best pair
        best_idx = returns.index(max(returns))
        best_pair = ma_pairs[best_idx]
        print(f"Best performing pair: {best_pair} with return {returns[best_idx]:.2%}")
        
        return results, report
        
    except Exception as e:
        print(f"Error in multi-pair analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    run_multi_pair_analysis()