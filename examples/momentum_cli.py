# examples/momentum_cli.py
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.manual_momentum_analysis import run_manual_analysis
from examples.multi_pairs_momentum_analysis import run_multi_pair_analysis 
from examples.multi_stock_analysis import run_multi_stock_analysis

def main():
    parser = argparse.ArgumentParser(description='Momentum Trading Strategy Analysis')
    
    # Main command argument
    parser.add_argument('command', choices=['single', 'pairs', 'stocks'], 
                        help='Analysis type: single (one pair for one stock), pairs (multiple MA pairs for one stock), stocks (one pair for multiple stocks)')
    
    # Optional arguments
    parser.add_argument('--symbol', type=str, default='NVDA',
                        help='Stock symbol to analyze (default: NVDA)')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='Start date in YYYY-MM-DD format (default: 2024-01-01)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date in YYYY-MM-DD format (default: current date)')
    parser.add_argument('--short', type=int, default=5,
                        help='Short window size for MA (default: 5)')
    parser.add_argument('--long', type=int, default=20,
                        help='Long window size for MA (default: 20)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Set output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set environment variables for functions to use if needed
    os.environ['ANALYSIS_SYMBOL'] = args.symbol
    os.environ['ANALYSIS_START_DATE'] = args.start
    if args.end:
        os.environ['ANALYSIS_END_DATE'] = args.end
    os.environ['ANALYSIS_SHORT_WINDOW'] = str(args.short)
    os.environ['ANALYSIS_LONG_WINDOW'] = str(args.long)
    os.environ['ANALYSIS_OUTPUT_DIR'] = args.output
    
    # Run requested analysis
    if args.command == 'single':
        print(f"Running single pair analysis for {args.symbol} with MA ({args.short}, {args.long})")
        # Since original functions take no arguments, pass through environment variables
        run_manual_analysis()
    
    elif args.command == 'pairs':
        print(f"Running multi-pair analysis for {args.symbol}")
        run_multi_pair_analysis()
    
    elif args.command == 'stocks':
        print(f"Running multi-stock analysis with MA ({args.short}, {args.long})")
        run_multi_stock_analysis()

if __name__ == "__main__":
    main()