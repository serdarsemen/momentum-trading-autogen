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
from .bollinger_bands_strategy import (
    bollinger_bands_strategy,
    calculate_bollinger_bands,
    plot_bollinger_bands_strategy,
    run_bollinger_bands_analysis
)
from .fvg_strategy import (
    fvg_trading_strategy,
    identify_fvg,
    plot_fvg_strategy,
    run_fvg_analysis
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
    'run_stochastic_analysis',
    'bollinger_bands_strategy',
    'calculate_bollinger_bands',
    'plot_bollinger_bands_strategy',
    'run_bollinger_bands_analysis',
    'fvg_trading_strategy',
    'identify_fvg',
    'plot_fvg_strategy',
    'run_fvg_analysis'
]
