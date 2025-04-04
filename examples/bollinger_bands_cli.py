"""
Command-line interface for Bollinger Bands trading strategy analysis.
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import run_bollinger_bands_analysis

def validate_date(date_str):
    """Validate and parse date string."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError as e:
        print(f"Invalid date format: {date_str}. Please use YYYY-MM-DD format.")
        sys.exit(1)

def prepare_stock_data(symbol, start_date, end_date):
    """Download and prepare stock data."""
    try:
        return download_stock_data(symbol, start_date, end_date)
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def run_single_analysis(args):
    """Run analysis for a single stock with specified parameters."""
    try:
        # Validate dates
        start_date = validate_date(args.start)
        end_date = validate_date(args.end) if args.end else get_current_date()
        
        # Prepare data
        df = prepare_stock_data(args.symbol, start_date, end_date)
        if df is None:
            return
            
        print(f"\nAnalyzing {args.symbol} with Bollinger Bands parameters:")
        print(f"Window: {args.window}")
        print(f"Standard Deviations: {args.num_std}")
        print(f"Use ATR: {args.use_atr}")
        if args.use_atr:
            print(f"ATR Period: {args.atr_period}\n")
        
        # Run analysis
        results = run_bollinger_bands_analysis(
            df,
            window=args.window,
            num_std=args.num_std,
            use_atr=args.use_atr,
            atr_period=args.atr_period,
            symbol=args.symbol,
            save_results=True,
            output_dir=args.output
        )
        
        # Print results
        print(f"Analysis Results for {args.symbol}:")
        print(f"Final Return: {results['final_return']:.2%}")
        
    except Exception as e:
        print(f"Error running analysis: {e}")

def run_multi_parameter_analysis(args):
    """Run analysis with multiple parameter combinations."""
    try:
        # Validate dates
        start_date = validate_date(args.start)
        end_date = validate_date(args.end) if args.end else get_current_date()
        
        # Prepare data
        df = prepare_stock_data(args.symbol, start_date, end_date)
        if df is None:
            return
            
        print(f"\nRunning multi-parameter analysis for {args.symbol}")
        
        # Define parameter combinations to test
        windows = [10, 20, 50]
        std_devs = [1.5, 2.0, 2.5]
        
        results_summary = []
        
        # Test different parameter combinations
        for window in windows:
            for std in std_devs:
                print(f"\nTesting parameters: Window={window}, StdDev={std}")
                
                results = run_bollinger_bands_analysis(
                    df.copy(),
                    window=window,
                    num_std=std,
                    use_atr=args.use_atr,
                    atr_period=args.atr_period,
                    symbol=args.symbol,
                    save_results=True,
                    output_dir=args.output
                )
                
                results_summary.append({
                    'window': window,
                    'std_dev': std,
                    'final_return': results['final_return']
                })
        
        # Print summary
        print("\nParameter Optimization Results:")
        for result in sorted(results_summary, key=lambda x: x['final_return'], reverse=True):
            print(f"Window: {result['window']}, StdDev: {result['std_dev']:.1f}, "
                  f"Return: {result['final_return']:.2%}")
            
    except Exception as e:
        print(f"Error running analysis: {e}")

def run_multi_stock_analysis(args):
    """Run analysis on multiple stocks with same parameters."""
    try:
        # Validate dates
        start_date = validate_date(args.start)
        end_date = validate_date(args.end) if args.end else get_current_date()
        
        stocks = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"\nAnalyzing multiple stocks with parameters:")
        print(f"Window: {args.window}")
        print(f"Standard Deviations: {args.num_std}\n")
        
        results_summary = []
        
        for symbol in stocks:
            print(f"\nAnalyzing {symbol}...")
            
            # Prepare data
            df = prepare_stock_data(symbol, start_date, end_date)
            if df is None:
                continue
            
            # Run analysis
            results = run_bollinger_bands_analysis(
                df,
                window=args.window,
                num_std=args.num_std,
                use_atr=args.use_atr,
                atr_period=args.atr_period,
                symbol=symbol,
                save_results=True,
                output_dir=args.output
            )
            
            results_summary.append({
                'symbol': symbol,
                'final_return': results['final_return']
            })
        
        # Print summary
        print("\nMulti-Stock Analysis Results:")
        for result in sorted(results_summary, key=lambda x: x['final_return'], reverse=True):
            print(f"Symbol: {result['symbol']}, Return: {result['final_return']:.2%}")
            
    except Exception as e:
        print(f"Error running analysis: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Bollinger Bands Trading Strategy Analysis'
    )
    
    # Main command argument
    parser.add_argument(
        'command',
        choices=['single', 'optimize', 'stocks'],
        help='Analysis type: single (one stock), optimize (parameter optimization), '
             'stocks (multiple stocks)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--symbol',
        type=str,
        default='NVDA',
        help='Stock symbol to analyze (default: NVDA)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='NVDA,AAPL,MSFT',
        help='Comma-separated list of stock symbols (default: NVDA,AAPL,MSFT)'
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
        '--window',
        type=int,
        default=20,
        help='Window size for moving average (default: 20)'
    )
    parser.add_argument(
        '--num-std',
        type=float,
        default=2.0,
        help='Number of standard deviations for bands (default: 2.0)'
    )
    parser.add_argument(
        '--use-atr',
        action='store_true',
        help='Use ATR for position sizing'
    )
    parser.add_argument(
        '--atr-period',
        type=int,
        default=14,
        help='Period for ATR calculation if used (default: 14)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Run appropriate analysis based on command
    if args.command == 'single':
        print(f"Running single stock analysis for {args.symbol}")
        run_single_analysis(args)
    
    elif args.command == 'optimize':
        print(f"Running parameter optimization for {args.symbol}")
        run_multi_parameter_analysis(args)
    
    elif args.command == 'stocks':
        print(f"Running multi-stock analysis")
        run_multi_stock_analysis(args)

if __name__ == "__main__":
    main()