"""
Command-line interface for Fair Value Gap (FVG) trading strategy analysis.
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import run_fvg_analysis

def validate_date(date_str):
    """Validate and parse date string."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError as e:
        print(f"Invalid date format: {date_str}. Please use YYYY-MM-DD format.")
        sys.exit(1)

def validate_timeframe(timeframe):
    """Validate the timeframe parameter."""
    valid_timeframes = ['15m', '30m', '1h', '4h', '1d']
    if timeframe not in valid_timeframes:
        print(f"Invalid timeframe: {timeframe}. Must be one of: {', '.join(valid_timeframes)}")
        sys.exit(1)
    return timeframe

def prepare_stock_data(symbol, start_date, end_date, timeframe='4h'):
    """Download and prepare stock data."""
    try:
        return download_stock_data(symbol, start_date, end_date, interval=timeframe)
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Fair Value Gap (FVG) Trading Strategy Analysis'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol to analyze (default: BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='4h',
        help='Timeframe for analysis (15m, 30m, 1h, 4h, 1d) (default: 4h)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2024-01-01',
        help='Start date in YYYY-MM-DD format (default: 2024-01-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (default: current date)'
    )
    parser.add_argument(
        '--min-gap',
        type=float,
        default=0.001,
        help='Minimum gap size as percentage (default: 0.001)'
    )
    parser.add_argument(
        '--max-fvgs',
        type=int,
        default=3,
        help='Maximum number of active FVGs to track (default: 3)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.02,
        help='Stop loss percentage (default: 0.02)'
    )
    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.03,
        help='Take profit percentage (default: 0.03)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    args = parser.parse_args()
    
    # Validate timeframe
    timeframe = validate_timeframe(args.timeframe)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Validate dates
    start_date = validate_date(args.start)
    end_date = validate_date(args.end) if args.end else get_current_date()
    
    print(f"\nAnalyzing {args.symbol} with FVG parameters:")
    print(f"Timeframe: {timeframe}")
    print(f"Minimum gap size: {args.min_gap}")
    print(f"Maximum active FVGs: {args.max_fvgs}")
    print(f"Stop loss: {args.stop_loss}")
    print(f"Take profit: {args.take_profit}\n")
    
    # Prepare data
    df = prepare_stock_data(args.symbol, start_date, end_date, timeframe)
    if df is None:
        return
    
    # Run analysis
    results = run_fvg_analysis(
        df,
        min_gap_size=args.min_gap,
        max_active_fvgs=args.max_fvgs,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        symbol=f"{args.symbol}_{timeframe}",  # Include timeframe in output files
        save_results=True,
        output_dir=args.output
    )
    
    # Print results
    print(f"\nResults for {args.symbol} FVG Strategy ({timeframe}):")
    print(f"Final Return: {results['final_return']:.2%}")
    print(f"Total Bullish FVGs: {len(results['bullish_fvgs'])}")
    print(f"Total Bearish FVGs: {len(results['bearish_fvgs'])}")
    print(f"Strategy plots saved to: {results['strategy_plot_path']}")
    print(f"Data saved to: {results['csv_path']}")

if __name__ == "__main__":
    main()
