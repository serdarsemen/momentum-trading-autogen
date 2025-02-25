"""
Utility functions and helpers.
"""

from .data_utils import (
    get_current_date,
    download_stock_data,
    calculate_returns_statistics,
    format_returns_as_markdown,
    prepare_comparison_data
)

__all__ = [
    'get_current_date',
    'download_stock_data',
    'calculate_returns_statistics',
    'format_returns_as_markdown',
    'prepare_comparison_data'
]