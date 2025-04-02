"""
Example of running a momentum trading strategy analysis with multiple MA pairs.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import create_agents, run_momentum_analysis
from src.utils import (
    get_current_date, 
    download_stock_data, 
    format_returns_as_markdown,
    prepare_comparison_data
)
from src.strategies import run_strategy_analysis

def run_manual_multiple_pairs_analysis():
    """
    Run a manual analysis with multiple MA pairs without using AutoGen agents.
    Useful for testing or demonstration.
    """
    # Set parameters
    symbol = "NVDA"
    start_date = "2024-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    ma_pairs = [(5, 20), (10, 50), (20, 100), (50, 200)]
    
    print(f"Running manual momentum analysis for {symbol} ({start_date} to {end_date})")
    print(f"Parameters: {len(ma_pairs)} MA pairs: {ma_pairs}")
    
    # Download data
    print("Downloading data...")
    df = download_stock_data(symbol, start_date, end_date)
    print(f"Downloaded {len(df)} days of data")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Run strategy for each pair
    results = {}
    for short_window, long_window in ma_pairs:
        print(f"Running strategy for MA pair ({short_window}, {long_window})...")
        result = run_strategy_analysis(
            df, 
            short_window, 
            long_window, 
            symbol=symbol,
            save_results=True,
            output_dir="output"
        )
        results[(short_window, long_window)] = result
    
    # Create comparison data
    comparison = prepare_comparison_data(results)
    
    # Generate markdown report
    returns_list = [results[pair]['final_return'] for pair in ma_pairs]
    markdown_report = format_returns_as_markdown(symbol, ma_pairs, returns_list)
    
    # Save markdown report
    with open("output/momentum_analysis_report.md", "w") as f:
        f.write(markdown_report)
    
    # Print results
    print(f"Analysis complete!")
    print(f"Best performing pair: {comparison['best_pair']} with return: {comparison['best_return']:.2%}")
    print(f"Results saved to output directory")
    print(f"Report saved to output/momentum_analysis_report.md")
    
    # Create summary dataframe for display
    summary_df = pd.DataFrame({
        'Short Window': [p[0] for p in ma_pairs],
        'Long Window': [p[1] for p in ma_pairs],
        'Return': [f"{r:.2%}" for r in returns_list],
        'Buy Signals': comparison['buy_signals'],
        'Sell Signals': comparison['sell_signals'],
        'Total Trades': comparison['trade_counts']
    })
    
    print("\nSummary:")
    print(summary_df)

def run_agent_multiple_pairs_analysis():
    """
    Run a momentum trading strategy analysis with multiple MA pairs using AutoGen agents.
    """
    # Check for API key
    # api_key = os.environ.get("OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # print("Error: OPENAI_API_KEY environment variable not set")
        # print("Please set your OpenAI API key as an environment variable:")
        # print("export OPENAI_API_KEY='your-api-key'")
        print("API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Set parameters
    symbol = "NVDA"
    start_date = "2024-01-01"
    end_date = get_current_date()
    ma_pairs = [(5, 20), (10, 50), (20, 100), (50, 200)]
    
    print(f"Running agent-based momentum analysis for {symbol}")
    print(f"Parameters: {len(ma_pairs)} MA pairs: {ma_pairs}")
    
    # Create agents
    agents = create_agents(api_key=api_key, work_dir="output")
    
    # Run analysis
    print("Starting agent conversation...")
    result = run_momentum_analysis(
        agents,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        ma_pairs=ma_pairs
    )
    
    print("Analysis complete!")
    print("Check the output directory for results")
    
    return result

if __name__ == "__main__":
    # Choose which analysis to run
    run_agent_multiple_pairs_analysis()
    # Uncomment to run manual analysis instead
    # run_manual_multiple_pairs_analysis()