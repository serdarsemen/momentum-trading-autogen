"""
Agent setup and initialization for the AutoGen framework.
"""

import os
from typing import List, Tuple, Dict, Any, Optional

from autogen import (
    AssistantAgent, 
    UserProxyAgent, 
    GroupChat, 
    GroupChatManager,
    ConversableAgent
)
from autogen.coding import LocalCommandLineCodeExecutor

from .prompt_templates import (
    CODE_GENERATOR_PROMPT, 
    CRITIC_AGENT_PROMPT, 
    COMPARER_AGENT_PROMPT
)

def create_agents(
    api_key: str, 
    model: str = "gpt-4o",
    work_dir: str = "output"
) -> Dict[str, ConversableAgent]:
    """
    Create and initialize the AutoGen agents for the momentum trading analysis.
    
    Args:
        api_key: OpenAI API key
        model: Model to use for the assistants
        work_dir: Directory for code execution
        
    Returns:
        Dictionary containing all the initialized agents
    """
    # Set up config list for the LLM
    config_list = [{"model": model, "api_key": api_key}]
    
    # Initialize the code executor
    code_executor = LocalCommandLineCodeExecutor(
        timeout=60,
        work_dir=work_dir,
    )
    
    # Create the Code Generator agent
    code_generator = AssistantAgent(
        name="Code_generator",
        system_message=CODE_GENERATOR_PROMPT,
        llm_config={
            "config_list": config_list
        },
        human_input_mode="NEVER"
    )
    
    # Create the Code Executor agent
    code_executor_agent = UserProxyAgent(
        name="Code_executor",
        code_execution_config={
            "executor": code_executor
        },
        llm_config=False,
        human_input_mode="NEVER"
    )
    
    # Create the Critic agent
    critic = AssistantAgent(
        name="Critic_agent",
        system_message=CRITIC_AGENT_PROMPT,
        llm_config={
            "config_list": config_list
        },
        human_input_mode="NEVER"
    )
    
    # Create the Comparer agent
    comparer = AssistantAgent(
        name="Comparer",
        system_message=COMPARER_AGENT_PROMPT,
        llm_config={
            "config_list": config_list
        },
        human_input_mode="NEVER"
    )
    
    # Group the agents together
    agents = {
        "code_generator": code_generator,
        "code_executor": code_executor_agent,
        "critic": critic,
        "comparer": comparer
    }
    
    return agents

def create_group_chat(agents: Dict[str, ConversableAgent], config_list: List[Dict[str, str]]) -> GroupChatManager:
    """
    Create a GroupChat and Manager from the given agents.
    
    Args:
        agents: Dictionary of agents to include in the group chat
        config_list: LLM config list
        
    Returns:
        GroupChatManager instance
    """
    # Create a list of agents for the group chat
    agent_list = list(agents.values())
    
    # Create the group chat
    groupchat = GroupChat(
        agents=agent_list,
        messages=[],
        max_round=20
    )
    
    # Create the group chat manager
    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config={"config_list": config_list}
    )
    
    return manager

def run_momentum_analysis(
    agents: Dict[str, ConversableAgent],
    symbol: str = "NVDA",
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    ma_pairs: List[Tuple[int, int]] = [(5, 20), (10, 50), (20, 100), (50, 200)],
) -> Dict[str, Any]:
    """
    Run the momentum trading strategy analysis with the given parameters.
    
    Args:
        agents: Dictionary of agents
        symbol: Stock symbol to analyze
        start_date: Start date for historical data
        end_date: End date for historical data (defaults to current date)
        ma_pairs: List of (short window, long window) pairs to analyze
        
    Returns:
        Dictionary with results from the analysis
    """
    # Create the GroupChatManager
    api_key = agents["code_generator"].llm_config["config_list"][0]["api_key"]
    config_list = [{"model": "gpt-4o", "api_key": api_key}]
    manager = create_group_chat(agents, config_list)
    
    # Format the MA pairs for the prompt
    ma_pairs_str = ", ".join([f"({short}, {long})" for short, long in ma_pairs])
    
    # Create the analysis message with enhanced instructions
    if end_date:
        date_range = f"from {start_date} to {end_date}"
    else:
        date_range = f"since {start_date}"
        
    message = f"""Let's proceed step by step:
1- Get the current date.
2- Propose a Python code implementation of a momentum trading strategy with 2 moving averages: short and long.
3- Create the Python file with this structure:
   - Use 'import pandas as pd, numpy as np, matplotlib.pyplot as plt, yfinance as yf'
   - Import ssl and add 'try: _create_unverified_https_context = ssl._create_unverified_context; except AttributeError: pass; else: ssl._create_default_https_context = _create_unverified_https_context' to handle SSL issues
   - For Yahoo Finance data, use proper error handling and consider using a try-except block
   - Save in a file called 'momentum_trading_strategy.py'
4- Apply this code to {symbol} historical price {date_range}, with the following pairs of moving averages: {ma_pairs_str}.
5- For each pair of moving averages, save the results in a csv file called '{symbol.lower()}_trading_strategy_{{pair_of_moving_average}}.csv'
6- For each pair of moving averages, plot this trading strategy and save it in a file called '{symbol.lower()}_trading_strategy_{{pair_of_moving_average}}.png'
7- For each pair of moving averages, calculate buy and sell signals. Plot them and save it in a file called 'buy_sell_signals_{{pair_of_moving_average}}.png'
8- For each pair of moving averages, compute the final return of the strategy, and provide these results in a markdown format.

Make sure to save the strategy implementation file properly before executing it.
"""
    
    # Initiate the chat with the message
    result = agents["code_executor"].initiate_chat(manager, message=message)
    
    # Return the result
    return result