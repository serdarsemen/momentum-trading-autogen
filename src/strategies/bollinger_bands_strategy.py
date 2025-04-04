"""
Bollinger Bands trading strategy implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

def calculate_bollinger_bands(
    data: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands components.
    
    Args:
        data: Price series
        window: Period for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    # Calculate middle band (simple moving average)
    middle_band = data.rolling(window=window).mean()
    
    # Calculate standard deviation
    rolling_std = data.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return middle_band, upper_band, lower_band

def bollinger_bands_strategy(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    use_atr: bool = False,
    atr_period: int = 14
) -> pd.DataFrame:
    """
    Implement Bollinger Bands trading strategy.
    
    Args:
        df: DataFrame with price data
        window: Period for moving average
        num_std: Number of standard deviations for bands
        use_atr: Whether to use ATR for position sizing
        atr_period: Period for ATR calculation if used
        
    Returns:
        DataFrame with signals and indicators
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # Calculate Bollinger Bands
    middle_band, upper_band, lower_band = calculate_bollinger_bands(
        signals['price'],
        window,
        num_std
    )
    
    # Store Bollinger Bands components
    signals['middle_band'] = middle_band
    signals['upper_band'] = upper_band
    signals['lower_band'] = lower_band
    
    # Calculate bandwidth
    signals['bandwidth'] = (upper_band - lower_band) / middle_band
    
    # Calculate %B indicator
    signals['percent_b'] = (signals['price'] - lower_band) / (upper_band - lower_band)
    
    # Generate signals
    signals['signal'] = 0.0
    
    # Buy signal when price crosses below lower band
    signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1.0
    
    # Sell signal when price crosses above upper band
    signals.loc[signals['price'] > signals['upper_band'], 'signal'] = 0.0
    
    # Calculate positions
    signals['positions'] = signals['signal'].diff()
    
    # Add ATR-based position sizing if requested
    if use_atr:
        signals['atr'] = calculate_atr(df, atr_period)
        signals['position_size'] = 1.0 / signals['atr']
        signals['position_size'] = signals['position_size'] / signals['position_size'].mean()
        signals['positions'] = signals['positions'] * signals['position_size']
    
    return signals

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with High, Low, Close prices
        period: Period for ATR calculation
        
    Returns:
        Series containing ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def plot_bollinger_bands_strategy(
    signals: pd.DataFrame,
    title: str = 'Bollinger Bands Strategy',
    filename: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Bollinger Bands strategy components and signals.
    
    Args:
        signals: DataFrame with signals from bollinger_bands_strategy()
        title: Plot title
        filename: If provided, save plot to this file
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    fig.suptitle(title)
    
    # Plot price and bands
    ax1.plot(signals['price'], label='Price', color='black', alpha=0.7)
    ax1.plot(signals['middle_band'], label='Middle Band', color='blue', alpha=0.6)
    ax1.plot(signals['upper_band'], label='Upper Band', color='gray', linestyle='--')
    ax1.plot(signals['lower_band'], label='Lower Band', color='gray', linestyle='--')
    
    # Plot buy/sell signals
    buy_signals = signals[signals['positions'] == 1.0]
    sell_signals = signals[signals['positions'] == -1.0]
    
    ax1.scatter(buy_signals.index, buy_signals['price'],
                marker='^', color='g', s=100, label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['price'],
                marker='v', color='r', s=100, label='Sell Signal')
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot %B indicator
    ax2.plot(signals['percent_b'], label='%B', color='blue')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3)
    
    ax2.set_ylabel('%B')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    
    return fig, (ax1, ax2)

def run_bollinger_bands_analysis(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    use_atr: bool = False,
    atr_period: int = 14,
    symbol: str = 'STOCK',
    save_results: bool = True,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Run a complete analysis of the Bollinger Bands trading strategy.
    
    Args:
        df: DataFrame with price data
        window: Period for moving average
        num_std: Number of standard deviations for bands
        use_atr: Whether to use ATR for position sizing
        atr_period: Period for ATR calculation if used
        symbol: Stock symbol for naming output files
        save_results: Whether to save results to files
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    # Generate signals
    signals = bollinger_bands_strategy(
        df,
        window,
        num_std,
        use_atr,
        atr_period
    )
    
    # Compute returns
    from .momentum_trading_strategy import compute_returns
    final_return, cumulative_returns = compute_returns(signals)
    
    # Base filename for outputs
    base_filename = f"{symbol.lower()}_bollinger_{window}_{num_std}"
    
    results = {
        'signals': signals,
        'final_return': final_return,
        'cumulative_returns': cumulative_returns,
        'parameters': {
            'window': window,
            'num_std': num_std,
            'use_atr': use_atr,
            'atr_period': atr_period,
            'symbol': symbol
        }
    }
    
    if save_results:
        # Save signals to CSV
        signals.to_csv(f"{output_dir}/{base_filename}_signals.csv")
        
        # Create and save strategy plot
        fig, _ = plot_bollinger_bands_strategy(
            signals,
            title=f"Bollinger Bands Strategy: {symbol}",
            filename=f"{output_dir}/{base_filename}_plot.png"
        )
        plt.close(fig)
    
    return results