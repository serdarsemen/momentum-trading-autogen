"""
Command-line interface for Stochastic Oscillator trading strategy analysis.
"""

import sys
import os
import argparse
from datetime import datetime, date
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import run_stochastic_analysis

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
        # Convert dates to string format if they're date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')
            
        df = download_stock_data(symbol, start_date, end_date)
        
        if df.empty:
            print(f"No data available for {symbol}")
            return None
            
        # Ensure the index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Ensure we have the required price column
        if 'Close' in df.columns:
            df['price'] = df['Close']
        elif 'Adj Close' in df.columns:
            df['price'] = df['Adj Close']
        else:
            print(f"No price data found for {symbol}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error preparing data for {symbol}: {e}")
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
            
        print(f"\nAnalyzing {args.symbol} with Stochastic parameters:")
        print(f"K Period: {args.k_period}")
        print(f"D Period: {args.d_period}")
        print(f"Smooth K: {args.smooth_k}")
        print(f"Oversold: {args.oversold}")
        print(f"Overbought: {args.overbought}\n")
        
        # Run analysis
        results = run_stochastic_analysis(
            df,
            k_period=args.k_period,
            d_period=args.d_period,
            smooth_k=args.smooth_k,
            oversold=args.oversold,
            overbought=args.overbought,
            symbol=args.symbol,
            save_results=True,
            output_dir=args.output
        )
        
        # Print results
        print(f"Analysis Results for {args.symbol}:")
        print(f"Final Return: {results['final_return']:.2%}")
        print(f"Strategy plots saved to: {results['strategy_plot_path']}")
        print(f"Data saved to: {results['csv_path']}")
        
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
        k_periods = [5, 9, 14, 21]
        d_periods = [3, 5, 9]
        smooth_ks = [1, 2, 3]
        
        results_summary = []
        
        # Test different parameter combinations
        for k in k_periods:
            for d in d_periods:
                for s in smooth_ks:
                    print(f"\nTesting parameters: K={k}, D={d}, Smooth={s}")
                    
                    results = run_stochastic_analysis(
                        df.copy(),  # Use copy to prevent modifications to original data
                        k_period=k,
                        d_period=d,
                        smooth_k=s,
                        oversold=args.oversold,
                        overbought=args.overbought,
                        symbol=args.symbol,
                        save_results=True,
                        output_dir=args.output
                    )
                    
                    results_summary.append({
                        'k_period': k,
                        'd_period': d,
                        'smooth_k': s,
                        'final_return': results['final_return']
                    })
        
        # Print summary of results
        print("\nParameter Optimization Results:")
        print("K Period | D Period | Smooth K | Return")
        print("-" * 45)
        
        # Sort by return
        results_summary.sort(key=lambda x: x['final_return'], reverse=True)
        
        for result in results_summary:
            print(f"{result['k_period']:8d} | {result['d_period']:8d} | "
                  f"{result['smooth_k']:8d} | {result['final_return']:7.2%}")
        
    except Exception as e:
        print(f"Error running multi-parameter analysis: {e}")

def run_multi_stock_analysis(args):
    """Run analysis on multiple stocks with same parameters."""
    try:
        # Validate dates
        start_date = validate_date(args.start)
        end_date = validate_date(args.end) if args.end else get_current_date()
        
        stocks = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"\nAnalyzing multiple stocks with parameters:")
        print(f"K Period: {args.k_period}")
        print(f"D Period: {args.d_period}")
        print(f"Smooth K: {args.smooth_k}\n")
        
        results_summary = []
        
        for symbol in stocks:
            print(f"\nAnalyzing {symbol}...")
            
            # Prepare data
            df = prepare_stock_data(symbol, start_date, end_date)
            if df is None:
                continue
            
            # Run analysis
            results = run_stochastic_analysis(
                df,
                k_period=args.k_period,
                d_period=args.d_period,
                smooth_k=args.smooth_k,
                oversold=args.oversold,
                overbought=args.overbought,
                symbol=symbol,
                save_results=True,
                output_dir=args.output
            )
            
            results_summary.append({
                'symbol': symbol,
                'final_return': results['final_return']
            })
        
        if results_summary:
            # Print summary
            print("\nResults Summary:")
            print("Symbol | Return")
            print("-" * 20)
            
            # Sort by return
            results_summary.sort(key=lambda x: x['final_return'], reverse=True)
            
            for result in results_summary:
                print(f"{result['symbol']:6s} | {result['final_return']:7.2%}")
        else:
            print("No valid results generated for any stocks.")
        
    except Exception as e:
        print(f"Error running multi-stock analysis: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Stochastic Oscillator Trading Strategy Analysis'
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
        '--k-period',
        type=int,
        default=14,
        help='Look-back period for %K (default: 14)'
    )
    parser.add_argument(
        '--d-period',
        type=int,
        default=3,
        help='Period for %D moving average (default: 3)'
    )
    parser.add_argument(
        '--smooth-k',
        type=int,
        default=3,
        help='Smoothing period for %K (default: 3)'
    )
    parser.add_argument(
        '--oversold',
        type=int,
        default=20,
        help='Oversold threshold (default: 20)'
    )
    parser.add_argument(
        '--overbought',
        type=int,
        default=80,
        help='Overbought threshold (default: 80)'
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

