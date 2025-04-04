"""
Trading strategy implementations.
"""

from .momentum_trading_strategy import (
    momentum_trading_strategy,
    compute_returns,
    plot_trading_strategy,
    plot_buy_sell_signals,
    plot_cumulative_returns,
    run_strategy_analysis
)
from .rsi_strategy import (
    rsi_trading_strategy,
    calculate_rsi
)
from .macd_strategy import (
    macd_trading_strategy,
    calculate_macd,
    plot_macd_strategy,
    run_macd_analysis
)
from .stochastic_strategy import (
    stochastic_trading_strategy,
    calculate_stochastic,
    plot_stochastic_strategy,
    run_stochastic_analysis
)

__all__ = [
    'momentum_trading_strategy',
    'compute_returns',
    'plot_trading_strategy',
    'plot_buy_sell_signals',
    'plot_cumulative_returns',
    'run_strategy_analysis',
    'rsi_trading_strategy',
    'calculate_rsi',
    'macd_trading_strategy',
    'calculate_macd',
    'plot_macd_strategy',
    'run_macd_analysis',
    'stochastic_trading_strategy',
    'calculate_stochastic',
    'plot_stochastic_strategy',
    'run_stochastic_analysis'
]
