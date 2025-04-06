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

    Args:
        provider: LLM provider ('openai', 'azure', 'gemini', or 'groq')
        model: Model name (optional, provider-specific default will be used if not specified)
        azure_endpoint: Azure OpenAI endpoint URL (optional, required for Azure)
        azure_deployment: Azure deployment name (optional, required for Azure)
        azure_api_version: Azure API version (optional)

    Returns:
        Configuration list for the LLM
    """
    api_key = get_api_key(provider)

    if provider.lower() == 'openai':
        default_model = "gpt-4" if not model else model
        return [{
            "model": default_model,
            "api_key": api_key,
            "client": OpenAI(api_key=api_key)
        }]

    elif provider.lower() == 'azure':
        if not azure_endpoint:
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_deployment:
            azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

        if not azure_endpoint or not azure_deployment:
            raise ValueError("Azure OpenAI requires endpoint URL and deployment name. "
                           "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT in .env file.")

        client = AzureOpenAI(
            api_key=api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint
        )

        return [{
            "model": azure_deployment,
            "api_key": api_key,
            "api_version": azure_api_version,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "client": client,
            "provider": "azure"
        }]

    elif provider.lower() == 'gemini':
        default_model = "gemini-2.5-pro-preview-03-25" if not model else model   #"gemini-pro"
        genai.configure(api_key=api_key)
        return [{
            "model": default_model,
            "api_key": api_key,
            "config_list": genai.get_default_generation_config(),
            "provider": "gemini"
        }]

    elif provider.lower() == 'groq':
        default_model = "llama-3.3-70b-versatile" if not model else model   # "mixtral-8x7b-32768"
        client = Groq(api_key=api_key)
        return [{
            "model": default_model,
            "api_key": api_key,
            "client": client,
            "provider": "groq"
        }]

    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'azure', 'gemini', or 'groq'")

def create_agents(
    provider: str = "openai",
    model: str = None,
    work_dir: str = "output",
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview"
) -> Dict[str, ConversableAgent]:
    """
    Create and initialize the AutoGen agents with support for multiple LLM providers.

    Args:
        provider: LLM provider ('openai', 'azure', 'gemini', or 'groq')
        model: Model name (optional)
        work_dir: Directory for code execution
        azure_endpoint: Azure OpenAI endpoint URL
        azure_deployment: Azure deployment name
        azure_api_version: Azure API version

    Returns:
        Dictionary containing all the initialized agents
    """
    # Set up LLM configuration
    config_list = setup_llm_config(
        provider,
        model,
        azure_endpoint,
        azure_deployment,
        azure_api_version
    )

    # Initialize the code executor
    code_executor = LocalCommandLineCodeExecutor(
        timeout=60,
        work_dir=work_dir,
    )

    # Create agents with the configured LLM
    agents = {}
    agent_configs = [
        ("code_generator", CODE_GENERATOR_PROMPT),
        ("critic", CRITIC_AGENT_PROMPT),
        ("comparer", COMPARER_AGENT_PROMPT),
        ("forecasting_agent", FORECASTING_AGENT_PROMPT)
    ]

    for name, prompt in agent_configs:
        agents[name] = AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config={
                "config_list": config_list,
                "provider": provider
            }
        )

    # Add the code executor agent
    agents["code_executor"] = UserProxyAgent(
        name="Code_executor",
        code_execution_config={
            "executor": code_executor
        },
        llm_config=False,
        human_input_mode="NEVER"
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
