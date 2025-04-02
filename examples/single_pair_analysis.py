"""
Example of running a momentum trading strategy analysis with a single MA pair.
"""

import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import create_agents, run_momentum_analysis
from src.utils import get_current_date, download_stock_data
from src.strategies import (
    momentum_trading_strategy, 
    compute_returns,
    plot_trading_strategy,
    plot_buy_sell_signals,
    run_strategy_analysis
)

def run_manual_analysis():
    """
    Run a manual analysis without using AutoGen agents, useful for testing.
    """
    # Set parameters
    symbol = "NVDA"
    start_date = "2024-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    short_window = 5
    long_window = 20
    
    print(f"Running manual momentum analysis for {symbol} ({start_date} to {end_date})")
    print(f"Parameters: short_window={short_window}, long_window={long_window}")
    
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
        output_dir="output"
    )
    
    # Print results
    print("Analysis complete!")
    print("Final return: {results['final_return']:.2%}")
    print("Results saved to output directory")
    
    # Show plots
    plt.show()

def run_agent_analysis():
    """
    Run a momentum trading strategy analysis using AutoGen agents.
    """
    # Check for API key
    # api_key = os.environ.get("OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # print("Error: OPENAI_API_KEY environment variable not set")
        # print("Please set your OpenAI API key as an environment variable:")
        # print("export OPENAI_API_KEY='your-api-key'")
        print("Error: AZURE_OPENAI_API_KEY environment variable not set")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Set parameters
    symbol = "NVDA"
    start_date = "2024-01-01"
    end_date = get_current_date()
    ma_pair = (5, 20)  # (short_window, long_window)
    
    print(f"Running agent-based momentum analysis for {symbol}")
    print(f"Parameters: MA pair={ma_pair}")
    
    # Create agents
    agents = create_agents(api_key=api_key, work_dir="output")
    
    # Run analysis
    print("Starting agent conversation...")
    result = run_momentum_analysis(
        agents,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        ma_pairs=[ma_pair]
    )
    
    print("Analysis complete!")
    print("Check the output directory for results")
    
    return result

if __name__ == "__main__":
    # Choose which analysis to run
    run_agent_analysis()
    # Uncomment to run manual analysis instead
    # run_manual_analysis()