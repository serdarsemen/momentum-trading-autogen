"""
AutoGen agents module.
"""

from .setup_agents import create_agents, create_group_chat, run_momentum_analysis
from .prompt_templates import (
    CODE_GENERATOR_PROMPT,
    CRITIC_AGENT_PROMPT,
    COMPARER_AGENT_PROMPT
)

__all__ = [
    'create_agents',
    'create_group_chat',
    'run_momentum_analysis',
    'CODE_GENERATOR_PROMPT',
    'CRITIC_AGENT_PROMPT',
    'COMPARER_AGENT_PROMPT'
]