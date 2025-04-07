"""
Agent setup and initialization for the AutoGen framework with multiple LLM providers.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
from openai import AzureOpenAI, OpenAI
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
    COMPARER_AGENT_PROMPT,
    FORECASTING_AGENT_PROMPT
)

# Available models for each provider
AVAILABLE_MODELS = {
    'azure': [
        'gpt-4o',
        'gpt-4-turbo',
        'gpt-35-turbo'
    ],
    'openai': [
        'gpt-4-0125-preview',
        'gpt-4-turbo-preview',
        'gpt-4',
        'gpt-3.5-turbo'
    ],
    'gemini': [
        'gemini-2.5-pro-preview-03-25',
        'gemini-1.5-pro',
        'gemini-pro'
    ],
    'groq': [
        'llama-3.3-70b-versatile',
        'gemma-7b-it'
    ]
}

def get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    defaults = {
        'azure': 'gpt-4',
        'openai': 'gpt-4-0125-preview',
        'gemini': 'gemini-2.5-pro-preview-03-25',
        'groq': 'llama-3.3-70b-versatile'
    }
    return defaults.get(provider.lower())

# Load environment variables from .env file
load_dotenv()

def get_api_key(provider: str) -> str:
    """
    Get API key for the specified provider from environment variables.

    Args:
        provider: LLM provider ('openai', 'azure', 'gemini', or 'groq')

    Returns:
        API key for the specified provider
    """
    env_var_map = {
        'openai': 'OPENAI_API_KEY',
        'azure': 'AZURE_OPENAI_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'groq': 'GROQ_API_KEY'
    }

    env_var = env_var_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"API key not found in .env file for provider: {provider}. "
                        f"Please set {env_var} in your .env file.")

    return api_key

def setup_llm_config(
    provider: str,
    model: str = None,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview"
) -> List[Dict[str, Any]]:
    """
    Set up LLM configuration based on the provider.
    """
    api_key = get_api_key(provider)

    if provider.lower() == 'azure':
        if not azure_endpoint:
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_deployment:
            azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

        if not azure_endpoint or not azure_deployment:
            raise ValueError("Azure OpenAI requires endpoint URL and deployment name. "
                           "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT in .env file.")

        return [{
            "model": azure_deployment,
            "api_key": api_key,
            "api_type": "azure",
            "api_version": azure_api_version,
            "base_url": azure_endpoint
        }]

    elif provider.lower() == 'openai':
        return [{
            "model": model or "gpt-4",
            "api_key": api_key,
            "api_type": "openai"
        }]

    elif provider.lower() == 'gemini':
        genai.configure(api_key=api_key)
        return [{
            "model": model or "gemini-2.5-pro-preview-03-25",
            "api_key": api_key,
            "api_type": "google"
        }]

    elif provider.lower() == 'groq':
        return [{
            "model": model or "llama-3.3-70b-versatile",
            "api_key": api_key,
            "api_type": "groq"
        }]

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_agents(
    provider: str = "openai",
    model: str = None,
    work_dir: str = "output",
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview",
    use_docker: bool = False  # Added parameter to control Docker usage
) -> Dict[str, ConversableAgent]:
    """
    Create and initialize the AutoGen agents with support for multiple LLM providers.

    Args:
        provider: LLM provider name
        model: Model name to use
        work_dir: Working directory for code execution
        azure_endpoint: Azure OpenAI endpoint URL
        azure_deployment: Azure OpenAI deployment name
        azure_api_version: Azure OpenAI API version
        use_docker: Whether to use Docker for code execution (default: False)
    """
    # Set up LLM configuration
    config_list = setup_llm_config(
        provider,
        model,
        azure_endpoint,
        azure_deployment,
        azure_api_version
    )

    # Create base LLM configuration
    llm_config = {
        "seed": 42,
        "config_list": config_list,
        "temperature": 0.7
    }

    # Create agents with the configured LLM
    agents = {}
    agent_configs = [
        ("code_generator", CODE_GENERATOR_PROMPT),
        ("critic", CRITIC_AGENT_PROMPT),
        ("comparer", COMPARER_AGENT_PROMPT),
        ("forecasting_agent", FORECASTING_AGENT_PROMPT)
    ]

    for name, prompt in agent_configs:
        agent_config = llm_config.copy()
        agents[name] = AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=agent_config
        )

    # Add the code executor agent with Docker configuration
    code_execution_config = {
        "work_dir": work_dir,
        "timeout": 60,
        "last_n_messages": 3,
        "use_docker": use_docker  # Control Docker usage
    }

    agents["code_executor"] = UserProxyAgent(
        name="code_executor",
        human_input_mode="NEVER",
        code_execution_config=code_execution_config
    )

    return agents

def create_group_chat(
    agents: Dict[str, ConversableAgent],
    provider: str = "openai",
    config_list: Optional[List[Dict[str, Any]]] = None
) -> GroupChatManager:
    """
    Create a GroupChat and Manager from the given agents.

    Args:
        agents: Dictionary of agents to include in the group chat
        provider: LLM provider ('openai' or 'gemini')
        config_list: Optional LLM config list

    Returns:
        GroupChatManager instance
    """
    # Create a list of agents for the group chat
    agent_list = list(agents.values())

    # If no config_list provided, extract from code_generator agent
    if config_list is None:
        config_list = agents["code_generator"].llm_config["config_list"]

    # Create the group chat
    groupchat = GroupChat(
        agents=agent_list,
        messages=[],
        max_round=20
    )

    # Create the group chat manager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={
            "config_list": config_list,
            "provider": provider
        }
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
