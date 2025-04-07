import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta
import ssl

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import necessary modules from your project
from src.utils import download_stock_data, get_current_date, prepare_comparison_data
from src.strategies import momentum_trading_strategy, compute_returns, run_strategy_analysis
from src.agents.setup_agents import AVAILABLE_MODELS, get_default_model

# Configure SSL for Yahoo Finance
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set page config with minimal UI
st.set_page_config(
    page_title="Momentum Trading Strategy Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 4px;
    }
    .stDateInput > div > div > input {
        border-radius: 4px;
    }
    .stSlider > div {
        padding-left: 0;
        padding-right: 0;
    }
    .stButton > button {
        border-radius: 4px;
        background-color: #ff4b4b;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] > div {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stMetric"] label {
        font-size: 1rem;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        flex: 1;
        margin: 0 0.5rem;
        text-align: center;
    }
    .metric-card:first-child {
        margin-left: 0;
    }
    .metric-card:last-child {
        margin-right: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #333;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .dataframe {
        width: 100%;
    }
    .alert {
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        color: #721c24;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .info-alert {
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        color: #0c5460;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Sidebar
with st.sidebar:
    st.markdown("## Strategy Parameters")

    # LLM Provider selection
    st.markdown("### LLM Provider")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=['azure', 'openai', 'gemini', 'groq'],
        index=0,  # Default to azure
        help="Choose the LLM provider for analysis",
        label_visibility="collapsed"
    )

    # Model selection based on provider
    st.markdown("### LLM Model")
    available_models = AVAILABLE_MODELS[llm_provider]
    default_model = get_default_model(llm_provider)
    default_index = available_models.index(default_model) if default_model in available_models else 0

    llm_model = st.selectbox(
        "Select Model",
        options=available_models,
        index=default_index,
        help=f"Choose the specific model for {llm_provider}",
        label_visibility="collapsed"
    )

    # Analysis type selection
    st.markdown("### Analysis Type")
    analysis_type = st.radio(
        "Analysis Type",
        ["Single Pair", "Multiple Pairs", "Multiple Stocks"],
        label_visibility="collapsed"
    )

    # Stock symbol
    st.markdown("### Stock Symbol")
    symbol = st.text_input("Stock Symbol", value="NVDA", label_visibility="collapsed")

    # Date range
    st.markdown("### Start Date")
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365),
        label_visibility="collapsed"
    ).strftime('%Y-%m-%d')

    st.markdown("### End Date")
    end_date = st.date_input(
        "End Date",
        datetime.now(),
        label_visibility="collapsed"
    ).strftime('%Y-%m-%d')

    # MA parameters
    if analysis_type == "Single Pair":
        st.markdown("### Short Window")
        short_window = st.slider("Short Window", 5, 50, 5, 5, label_visibility="collapsed")

        st.markdown("### Long Window")
        long_window = st.slider("Long Window", 20, 200, 20, 5, label_visibility="collapsed")

        ma_pairs = [(short_window, long_window)]

    elif analysis_type == "Multiple Pairs":
        st.markdown("### MA Pairs")

        ma_pairs = []
        if st.checkbox("MA Pair (5, 20)", value=True):
            ma_pairs.append((5, 20))
        if st.checkbox("MA Pair (10, 50)", value=True):
            ma_pairs.append((10, 50))
        if st.checkbox("MA Pair (20, 100)", value=True):
            ma_pairs.append((20, 100))
        if st.checkbox("MA Pair (50, 200)", value=True):
            ma_pairs.append((50, 200))

        # Custom MA pair
        if st.checkbox("Add Custom MA Pair"):
            st.markdown("#### Custom MA Pair")
            custom_short = st.number_input("Short", 5, 100, 15, 5)
            custom_long = st.number_input("Long", 20, 300, 60, 5)
            ma_pairs.append((custom_short, custom_long))

        if not ma_pairs:
            ma_pairs = [(5, 20)]  # Default if nothing selected

    else:  # Multiple Stocks
        st.markdown("### Stocks")

        stocks = []
        if st.checkbox("Include NVDA", value=True):
            stocks.append("NVDA")
        if st.checkbox("Include AAPL"):
            stocks.append("AAPL")
        if st.checkbox("Include MSFT"):
            stocks.append("MSFT")
        if st.checkbox("Include GOOGL"):
            stocks.append("GOOGL")
        if st.checkbox("Include AMZN"):
            stocks.append("AMZN")

        # Custom stock
        if st.checkbox("Add Custom Stock"):
            custom_stock = st.text_input("Custom Stock Symbol")
            if custom_stock and custom_stock not in stocks:
                stocks.append(custom_stock)

        if not stocks:
            stocks = ["NVDA"]  # Default if nothing selected

        st.markdown("### Short Window")
        short_window = st.slider("", 5, 50, 5, 5, label_visibility="collapsed")

        st.markdown("### Long Window")
        long_window = st.slider("", 20, 200, 20, 5, label_visibility="collapsed")

        ma_pairs = [(short_window, long_window)]

    # Run button
    run_analysis = st.button("Run Analysis")

# Main area
st.title("Momentum Trading Strategy Analysis")

# Function to run single pair analysis
def run_single_pair_analysis(symbol, start_date, end_date, short_window, long_window):
    try:
        df = download_stock_data(symbol, start_date, end_date)

        if df.empty:
            st.error(f"No data available for {symbol} in the selected date range.")
            return None

        # Run strategy
        result = run_strategy_analysis(
            df,
            short_window,
            long_window,
            symbol=symbol,
            save_results=True,
            output_dir="output"
        )

        return result
    except Exception as e:
        st.error(f"Error analyzing {symbol} with MA pair ({short_window}, {long_window}): {str(e)}")
        return None

# Function to run multiple pair analysis
def run_multiple_pairs_analysis(symbol, start_date, end_date, ma_pairs):
    try:
        df = download_stock_data(symbol, start_date, end_date)

        if df.empty:
            st.error(f"No data available for {symbol} in the selected date range.")
            return None, None

        # Run strategy for each pair
        results = {}
        for short_window, long_window in ma_pairs:
            result = run_strategy_analysis(
                df.copy(),
                short_window,
                long_window,
                symbol=symbol,
                save_results=True,
                output_dir="output"
            )
            results[(short_window, long_window)] = result

        # Create comparison data
        comparison = prepare_comparison_data(results)

        return results, comparison
    except Exception as e:
        st.error(f"Error analyzing {symbol} with multiple MA pairs: {str(e)}")
        return None, None

# Function to run multiple stocks analysis
def run_multiple_stocks_analysis(stocks, start_date, end_date, ma_pair):
    results = {}
    summary = []

    for stock in stocks:
        try:
            stock_dir = os.path.join("output", stock)
            os.makedirs(stock_dir, exist_ok=True)

            df = download_stock_data(stock, start_date, end_date)

            if df.empty:
                st.warning(f"No data available for {stock} in the selected date range.")
                continue

            # Run strategy
            result = run_strategy_analysis(
                df,
                ma_pair[0],
                ma_pair[1],
                symbol=stock,
                save_results=True,
                output_dir=stock_dir
            )

            results[stock] = result

            # Add to summary
            summary.append({
                'Symbol': stock,
                'Return': result['final_return'],
                'Buy Signals': len(result['signals'][result['signals']['positions'] == 1.0]),
                'Sell Signals': len(result['signals'][result['signals']['positions'] == -1.0]),
                'Total Trades': len(result['signals'][result['signals']['positions'] != 0])
            })
        except Exception as e:
            st.error(f"Error analyzing {stock}: Cannot save file to a non-existent directory: 'output/{stock}'")

    return results, pd.DataFrame(summary) if summary else None

# Run analysis when button is clicked
if run_analysis:
    if analysis_type == "Single Pair":
        st.subheader(f"Analysis for {symbol} with MA Pair ({ma_pairs[0][0]}, {ma_pairs[0][1]})")
        result = run_single_pair_analysis(symbol, start_date, end_date, ma_pairs[0][0], ma_pairs[0][1])

        if result:
            # Calculate metrics
            final_return = result['final_return']
            daily_returns = result['signals']['strategy_returns'].dropna()
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
            max_drawdown = ((result['signals']['cumulative_strategy_return'].cummax() - result['signals']['cumulative_strategy_return']) / result['signals']['cumulative_strategy_return'].cummax()).max() if 'cumulative_strategy_return' in result['signals'].columns else 0
            win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns[daily_returns != 0]) * 100 if len(daily_returns[daily_returns != 0]) > 0 else 0

            # Display metrics in a row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("##### Total Return")
                st.markdown(f"### {final_return:.2%}")

            with col2:
                st.markdown("##### Sharpe Ratio")
                st.markdown(f"### {sharpe:.2f}")

            with col3:
                st.markdown("##### Max Drawdown")
                st.markdown(f"### {max_drawdown:.2%}")

            with col4:
                st.markdown("##### Win Rate")
                st.markdown(f"### {win_rate:.2f}%")

            # Display price and trading signals chart
            st.subheader(f"{symbol} Price and Trading Signals")

            fig = go.Figure()

            # Add price line
            fig.add_trace(go.Scatter(
                x=result['signals'].index,
                y=result['signals']['price'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=1.5)
            ))

            # Add MA lines
            fig.add_trace(go.Scatter(
                x=result['signals'].index,
                y=result['signals']['short_mavg'],
                mode='lines',
                name=f'SMA {ma_pairs[0][0]}',
                line=dict(color='blue', width=1.5)
            ))

            fig.add_trace(go.Scatter(
                x=result['signals'].index,
                y=result['signals']['long_mavg'],
                mode='lines',
                name=f'SMA {ma_pairs[0][1]}',
                line=dict(color='red', width=1.5)
            ))

            # Add buy signals
            buy_signals = result['signals'][result['signals']['positions'] == 1.0]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ))

            # Add sell signals
            sell_signals = result['signals'][result['signals']['positions'] == -1.0]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ))

            fig.update_layout(
                height=450,
                xaxis_title='Date',
                yaxis_title='Price',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display portfolio performance chart
            st.subheader("Portfolio Performance vs Buy & Hold")

            # Calculate buy & hold performance
            initial_investment = 100000
            buy_hold_df = pd.DataFrame(index=result['signals'].index)
            buy_hold_df['price'] = result['signals']['price']
            buy_hold_df['buy_hold_return'] = buy_hold_df['price'] / buy_hold_df['price'].iloc[0] - 1
            buy_hold_df['buy_hold_value'] = (buy_hold_df['buy_hold_return'] + 1) * initial_investment

            # Strategy performance
            strategy_value = result['signals']['cumulative_strategy_return'] * initial_investment if 'cumulative_strategy_return' in result['signals'].columns else None

            if strategy_value is not None:
                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(
                    x=result['signals'].index,
                    y=strategy_value,
                    mode='lines',
                    name='Strategy Performance',
                    line=dict(color='green', width=1.5)
                ))

                fig2.add_trace(go.Scatter(
                    x=buy_hold_df.index,
                    y=buy_hold_df['buy_hold_value'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=1.5)
                ))

                fig2.update_layout(
                    height=350,
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='white',
                    hovermode='x unified'
                )

                st.plotly_chart(fig2, use_container_width=True)

    elif analysis_type == "Multiple Pairs":
        st.subheader(f"Analysis for {symbol} with Multiple MA Pairs")
        results, comparison = run_multiple_pairs_analysis(symbol, start_date, end_date, ma_pairs)

        if results and comparison:
            # Display performance summary table
            st.subheader("Performance Summary")

            # Prepare the data
            returns_list = [results[pair]['final_return'] for pair in ma_pairs]

            # Create the table
            table_data = {
                'Short Window': [p[0] for p in ma_pairs],
                'Long Window': [p[1] for p in ma_pairs],
                'Return': [f"{r:.2%}" for r in returns_list],
                'Buy Signals': comparison['buy_signals'],
                'Sell Signals': comparison['sell_signals'],
                'Total Trades': comparison['trade_counts']
            }

            # Display as a stylized table
            summary_df = pd.DataFrame(table_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Display best pair
            st.markdown(
                f"""<div class="info-alert">Best performing pair: <b>({comparison['best_pair'][0]}, {comparison['best_pair'][1]})</b> with return: <b>{comparison['best_return']:.2%}</b></div>""",
                unsafe_allow_html=True
            )

            # Returns comparison bar chart
            st.subheader("Returns Comparison")

            fig3 = go.Figure()

            # Color palette for the bars
            colors = ['#1f77b4', '#64b5f6', '#e74c3c', '#ffb74d']

            for i, pair in enumerate(ma_pairs):
                fig3.add_trace(go.Bar(
                    x=[f"({pair[0]}, {pair[1]})"],
                    y=[results[pair]['final_return'] * 100],
                    name=f"MA Pair ({pair[0]}, {pair[1]})",
                    text=[f"{results[pair]['final_return']:.2%}"],
                    textposition='auto',
                    marker_color=colors[i % len(colors)]
                ))

            fig3.update_layout(
                height=400,
                xaxis_title='MA Pairs',
                yaxis_title='Return (%)',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig3, use_container_width=True)

            # Cumulative returns comparison
            st.subheader("Cumulative Returns Comparison")

            fig4 = go.Figure()

            for i, pair in enumerate(ma_pairs):
                if 'cumulative_strategy_return' in results[pair]['signals'].columns:
                    fig4.add_trace(go.Scatter(
                        x=results[pair]['signals'].index,
                        y=results[pair]['signals']['cumulative_strategy_return'],
                        mode='lines',
                        name=f"MA Pair ({pair[0]}, {pair[1]})",
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ))

            fig4.update_layout(
                height=400,
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='white',
                hovermode='x unified'
            )

            st.plotly_chart(fig4, use_container_width=True)

    else:  # Multiple Stocks
        st.subheader(f"Analysis for Multiple Stocks with MA Pair ({ma_pairs[0][0]}, {ma_pairs[0][1]})")

        # Run the analysis
        try:
            results, summary_df = run_multiple_stocks_analysis(stocks, start_date, end_date, ma_pairs[0])

            if results and summary_df is not None:
                # Sort by return
                summary_df = summary_df.sort_values('Return', ascending=False)

                # Display summary table
                st.subheader("Performance Summary")

                # Format the dataframe for display
                display_df = summary_df.copy()
                display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:.2%}")

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Display best stock
                best_stock = summary_df.iloc[0]['Symbol']
                best_return = summary_df.iloc[0]['Return']
                st.markdown(
                    f"""<div class="info-alert">Best performing stock: <b>{best_stock}</b> with return: <b>{best_return:.2%}</b></div>""",
                    unsafe_allow_html=True
                )

                # Plot returns comparison
                st.subheader("Returns Comparison")

                fig5 = go.Figure()

                colors = ['#1f77b4', '#64b5f6', '#e74c3c', '#ffb74d', '#9c27b0']

                for i, (idx, row) in enumerate(summary_df.iterrows()):
                    fig5.add_trace(go.Bar(
                        x=[row['Symbol']],
                        y=[row['Return'] * 100],
                        name=row['Symbol'],
                        text=[f"{row['Return']:.2%}"],
                        textposition='auto',
                        marker_color=colors[i % len(colors)]
                    ))

                fig5.update_layout(
                    height=400,
                    xaxis_title='Stocks',
                    yaxis_title='Return (%)',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='white'
                )

                st.plotly_chart(fig5, use_container_width=True)

                # Plot cumulative returns for all stocks
                st.subheader("Cumulative Returns Comparison")

                fig6 = go.Figure()

                for i, stock in enumerate(summary_df['Symbol']):
                    if 'cumulative_strategy_return' in results[stock]['signals'].columns:
                        fig6.add_trace(go.Scatter(
                            x=results[stock]['signals'].index,
                            y=results[stock]['signals']['cumulative_strategy_return'],
                            mode='lines',
                            name=stock,
                            line=dict(color=colors[i % len(colors)], width=1.5)
                        ))

                fig6.update_layout(
                    height=400,
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='white',
                    hovermode='x unified'
                )

                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.error("No valid results were generated for any stocks.")
        except Exception as e:
            st.error(f"Error analyzing stocks: {str(e)}")
else:
    # Display information about the strategy
    st.markdown("### About Momentum Trading Strategy")

    st.markdown("""
    This dashboard implements a momentum trading strategy based on Moving Average (MA) crossovers:

    * When the short-term MA crosses above the long-term MA, a buy signal is generated
    * When it crosses below, a sell signal is triggered

    Performance metrics:

    * **Total Return**: Overall strategy return
    * **Sharpe Ratio**: Risk-adjusted return (higher is better)
    * **Max Drawdown**: Largest percentage drop from peak
    * **Win Rate**: Percentage of profitable trades
    """)
