"""
Command-line interface for MACD (Moving Average Convergence Divergence) trading strategy analysis.
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data, get_current_date
from src.strategies import macd_trading_strategy, compute_returns
from src.agents.setup_agents import create_agents, get_default_model, AVAILABLE_MODELS

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

        print(f"\nAnalyzing {args.symbol} with MACD parameters:")
        print(f"Fast Period: {args.fast_period}")
        print(f"Slow Period: {args.slow_period}")
        print(f"Signal Period: {args.signal_period}")
        print(f"Histogram Threshold: {args.threshold}\n")

        # Run analysis
        results = run_macd_analysis(
            df,
            fast_period=args.fast_period,
            slow_period=args.slow_period,
            signal_period=args.signal_period,
            histogram_threshold=args.threshold,
            symbol=args.symbol,
            save_results=True,
            output_dir=args.output
        )

        # Print results
        print(f"\nResults for {args.symbol} MACD Strategy:")
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
        fast_periods = [8, 12, 15]
        slow_periods = [21, 26, 30]
        signal_periods = [7, 9, 11]
        thresholds = [0.0, 0.1, 0.2]

        results_summary = []

        # Test different parameter combinations
        for fast in fast_periods:
            for slow in slow_periods:
                if fast >= slow:  # Skip invalid combinations
                    continue
                for signal in signal_periods:
                    for threshold in thresholds:
                        print(f"\nTesting parameters: Fast={fast}, Slow={slow}, "
                              f"Signal={signal}, Threshold={threshold}")

                        results = run_macd_analysis(
                            df.copy(),
                            fast_period=fast,
                            slow_period=slow,
                            signal_period=signal,
                            histogram_threshold=threshold,
                            symbol=args.symbol,
                            save_results=True,
                            output_dir=args.output
                        )

                        results_summary.append({
                            'fast_period': fast,
                            'slow_period': slow,
                            'signal_period': signal,
                            'threshold': threshold,
                            'final_return': results['final_return']
                        })

        # Print summary
        print("\nParameter Optimization Results:")
        for result in sorted(results_summary, key=lambda x: x['final_return'], reverse=True):
            print(f"Fast: {result['fast_period']}, Slow: {result['slow_period']}, "
                  f"Signal: {result['signal_period']}, Threshold: {result['threshold']}, "
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
        print(f"Fast Period: {args.fast_period}")
        print(f"Slow Period: {args.slow_period}")
        print(f"Signal Period: {args.signal_period}")
        print(f"Histogram Threshold: {args.threshold}\n")

        results_summary = []

        for symbol in stocks:
            print(f"\nAnalyzing {symbol}...")

            # Prepare data
            df = prepare_stock_data(symbol, start_date, end_date)
            if df is None:
                continue

            # Run analysis
            results = run_macd_analysis(
                df,
                fast_period=args.fast_period,
                slow_period=args.slow_period,
                signal_period=args.signal_period,
                histogram_threshold=args.threshold,
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

def select_provider():
    """Display provider selection menu and return chosen provider."""
    print("\nSelect LLM Provider:")
    print("1. Azure OpenAI")
    print("2. OpenAI")
    print("3. Google Gemini")
    print("4. Groq")

    while True:
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if 1 <= choice <= 4:
                provider_map = {
                    1: 'azure',
                    2: 'openai',
                    3: 'gemini',
                    4: 'groq'
                }
                selected_provider = provider_map[choice]
                default_model = get_default_model(selected_provider)
                print(f"\nSelected Provider: {selected_provider.upper()}")
                print(f"Default Model: {default_model}")
                print(f"Available Models: {', '.join(AVAILABLE_MODELS[selected_provider])}")
                return selected_provider
            else:
                print("Invalid choice. Please enter a number between 1-4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    parser = argparse.ArgumentParser(
        description='MACD (Moving Average Convergence Divergence) Trading Strategy Analysis'
    )

    # Main command argument
    parser.add_argument(
        'command',
        choices=['single', 'optimize', 'stocks'],
        help='Analysis type: single (one stock), optimize (parameter optimization), '
             'stocks (multiple stocks)'
    )

    # Remove the provider argument since we're using interactive menu

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
        '--fast-period',
        type=int,
        default=12,
        help='Fast EMA period (default: 12)'
    )
    parser.add_argument(
        '--slow-period',
        type=int,
        default=26,
        help='Slow EMA period (default: 26)'
    )
    parser.add_argument(
        '--signal-period',
        type=int,
        default=9,
        help='Signal line period (default: 9)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Histogram threshold for signals (default: 0.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )

    args = parser.parse_args()

    # Get provider through interactive menu
    provider = select_provider()

    # Create agents with specified provider
    agents = create_agents(provider=provider)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Run appropriate analysis based on command
    if args.command == 'single':
        print(f"\nRunning single stock analysis for {args.symbol}")
        print(f"Using Provider: {provider.upper()}")
        print(f"Using Model: {get_default_model(provider)}\n")
        run_single_analysis(args, agents)

    elif args.command == 'optimize':
        print(f"\nRunning parameter optimization for {args.symbol}")
        print(f"Using Provider: {provider.upper()}")
        print(f"Using Model: {get_default_model(provider)}\n")
        run_multi_parameter_analysis(args, agents)

    elif args.command == 'stocks':
        print(f"\nRunning multi-stock analysis")
        print(f"Using Provider: {provider.upper()}")
        print(f"Using Model: {get_default_model(provider)}\n")
        run_multi_stock_analysis(args, agents)

if __name__ == "__main__":
    main()
