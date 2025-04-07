"""
Command-line interface for RSI (Relative Strength Index) trading strategy analysis.
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import rsi_trading_strategy, compute_returns
from src.agents.setup_agents import create_agents

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

def run_single_analysis(args, agents):
    """Run analysis for a single stock with specified parameters."""
    try:
        # Validate dates
        start_date = validate_date(args.start)
        end_date = validate_date(args.end) if args.end else get_current_date()

        # Prepare data
        df = prepare_stock_data(args.symbol, start_date, end_date)
        if df is None:
            return

        print(f"\nAnalyzing {args.symbol} with RSI parameters:")
        print(f"Period: {args.period}")
        print(f"Oversold: {args.oversold}")
        print(f"Overbought: {args.overbought}\n")

        # Run analysis
        signals = rsi_trading_strategy(
            df,
            period=args.period,
            oversold=args.oversold,
            overbought=args.overbought
        )
        final_return, cum_returns = compute_returns(signals)

        # Print results
        print(f"Analysis Results for {args.symbol}:")
        print(f"Final Return: {final_return:.2%}")

        # Save results if output directory is specified
        if args.output:
            output_file = os.path.join(args.output, f"rsi_results_{args.symbol}.csv")
            signals.to_csv(output_file)
            print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error running analysis: {e}")

def run_multi_parameter_analysis(args, agents):
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
        periods = [9, 14, 21, 25]
        oversold_levels = [20, 25, 30, 35]
        overbought_levels = [65, 70, 75, 80]

        results_summary = []

        # Test different parameter combinations
        for period in periods:
            for oversold in oversold_levels:
                for overbought in overbought_levels:
                    if oversold >= overbought:
                        continue

                    print(f"\nTesting parameters: Period={period}, "
                          f"Oversold={oversold}, Overbought={overbought}")

                    signals = rsi_trading_strategy(
                        df.copy(),
                        period=period,
                        oversold=oversold,
                        overbought=overbought
                    )
                    final_return, _ = compute_returns(signals)

                    results_summary.append({
                        'period': period,
                        'oversold': oversold,
                        'overbought': overbought,
                        'final_return': final_return
                    })

        # Print summary
        print("\nParameter Optimization Results:")
        for result in sorted(results_summary, key=lambda x: x['final_return'], reverse=True):
            print(f"Period: {result['period']}, Oversold: {result['oversold']}, "
                  f"Overbought: {result['overbought']}, Return: {result['final_return']:.2%}")

    except Exception as e:
        print(f"Error running analysis: {e}")

def run_multi_stock_analysis(args, agents):
    """Run analysis on multiple stocks with same parameters."""
    try:
        # Validate dates
        start_date = validate_date(args.start)
        end_date = validate_date(args.end) if args.end else get_current_date()

        stocks = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"\nAnalyzing multiple stocks with parameters:")
        print(f"Period: {args.period}")
        print(f"Oversold: {args.oversold}")
        print(f"Overbought: {args.overbought}\n")

        results_summary = []

        for symbol in stocks:
            print(f"\nAnalyzing {symbol}...")

            # Prepare data
            df = prepare_stock_data(symbol, start_date, end_date)
            if df is None:
                continue

            # Run analysis
            signals = rsi_trading_strategy(
                df,
                period=args.period,
                oversold=args.oversold,
                overbought=args.overbought
            )
            final_return, _ = compute_returns(signals)

            results_summary.append({
                'symbol': symbol,
                'final_return': final_return
            })

        # Print summary
        print("\nMulti-Stock Analysis Results:")
        for result in sorted(results_summary, key=lambda x: x['final_return'], reverse=True):
            print(f"Symbol: {result['symbol']}, Return: {result['final_return']:.2%}")

    except Exception as e:
        print(f"Error running analysis: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='RSI (Relative Strength Index) Trading Strategy Analysis'
    )

    # Main command argument
    parser.add_argument(
        'command',
        choices=['single', 'optimize', 'stocks'],
        help='Analysis type: single (one stock), optimize (parameter optimization), '
             'stocks (multiple stocks)'
    )

    # Add provider argument with azure as default
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'azure', 'gemini', 'groq'],
        default='azure',
        help='LLM provider to use (default: azure)'
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
        '--period',
        type=int,
        default=14,
        help='RSI period (default: 14)'
    )
    parser.add_argument(
        '--oversold',
        type=int,
        default=30,
        help='Oversold threshold (default: 30)'
    )
    parser.add_argument(
        '--overbought',
        type=int,
        default=70,
        help='Overbought threshold (default: 70)'
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

    # Create agents with specified provider
    agents = create_agents(provider=args.provider)

    # Run appropriate analysis based on command
    if args.command == 'single':
        print(f"Running single stock analysis for {args.symbol} using {args.provider}")
        run_single_analysis(args, agents)

    elif args.command == 'optimize':
        print(f"Running parameter optimization for {args.symbol} using {args.provider}")
        run_multi_parameter_analysis(args, agents)

    elif args.command == 'stocks':
        print(f"Running multi-stock analysis using {args.provider}")
        run_multi_stock_analysis(args, agents)

if __name__ == "__main__":
    main()


