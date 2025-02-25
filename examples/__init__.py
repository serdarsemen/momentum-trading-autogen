"""
Example scripts demonstrating the momentum trading strategy analysis.
"""

# Import examples for easy reference
from .single_pair_analysis import run_manual_analysis, run_agent_analysis
from .multiple_pairs_analysis import run_manual_multiple_pairs_analysis, run_agent_multiple_pairs_analysis

__all__ = [
    'run_manual_analysis',
    'run_agent_analysis',
    'run_manual_multiple_pairs_analysis',
    'run_agent_multiple_pairs_analysis'
]