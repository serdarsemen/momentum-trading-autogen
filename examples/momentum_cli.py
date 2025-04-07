# examples/momentum_cli.py
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.manual_momentum_analysis import run_manual_analysis
from examples.multi_pairs_momentum_analysis import run_multi_pair_analysis
from examples.multi_stock_analysis import run_multi_stock_analysis
from src.agents.setup_agents import create_agents, AVAILABLE_MODELS, get_default_model

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
                print("Invalid choice. Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    parser = argparse.ArgumentParser(description='Momentum Trading Strategy Analysis')

    # Main command argument
    parser.add_argument(
        'command',
        choices=['single', 'pairs', 'stocks'],
        help='Analysis type: single (one pair for one stock), pairs (multiple MA pairs for one stock), stocks (one pair for multiple stocks)'
    )

    # Remove the provider argument since we're using interactive menu

    # Add model argument
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to use (if not specified, will use provider default)'
    )

    # Add Docker usage argument
    parser.add_argument(
        '--use-docker',
        action='store_true',
        help='Use Docker for code execution (default: False)'
    )

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

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Get provider through interactive menu
    provider = select_provider()

    # Create agents with Docker configuration
    agents = create_agents(
        provider=provider,
        model=args.model,
        use_docker=args.use_docker,
        work_dir=args.output
    )

    # Run requested analysis
    if args.command == 'single':
        print(f"\nRunning single pair analysis for {args.symbol}")
        print(f"Using Provider: {provider.upper()}")
        print(f"Using Model: {get_default_model(provider)}\n")
        run_manual_analysis(args, agents)

    elif args.command == 'pairs':
        print(f"\nRunning multi-pair analysis for {args.symbol}")
        print(f"Using Provider: {provider.upper()}")
        print(f"Using Model: {get_default_model(provider)}\n")
        run_multi_pair_analysis(args, agents)

    elif args.command == 'stocks':
        print(f"\nRunning multi-stock analysis")
        print(f"Using Provider: {provider.upper()}")
        print(f"Using Model: {get_default_model(provider)}\n")
        run_multi_stock_analysis(args, agents)

if __name__ == "__main__":
    main()
