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

def run_manual_analysis(args, agents):
    """
    Run a manual analysis with AutoGen agents.

    Args:
        args: Command line arguments
        agents: Dictionary of AutoGen agents
    """
    # Get parameters from args or use defaults
    symbol = args.symbol
    start_date = args.start
    end_date = args.end or get_current_date()
    short_window = args.short
    long_window = args.long
    output_dir = args.output

    print(f"Running manual momentum analysis for {symbol} ({start_date} to {end_date})")
    print(f"Parameters: short_window={short_window}, long_window={long_window}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download data
    print("Downloading data...")
    df = download_stock_data(symbol, start_date, end_date)
    print(f"Downloaded {len(df)} days of data")

    # Run strategy
    print("Running strategy...")
    results = run_strategy_analysis(
        df,
        short_window,
        long_window,
        symbol=symbol,
        save_results=True,
        output_dir=output_dir
    )

    return results

if __name__ == "__main__":
    run_manual_analysis()
