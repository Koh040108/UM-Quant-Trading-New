"""
Visualization module for displaying trading strategy results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from src.config import RESULTS_DIR

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")


def money_formatter(x, pos):
    """Format numbers as currency."""
    if abs(x) >= 1e6:
        return f'${x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'${x/1e3:.1f}K'
    else:
        return f'${x:.2f}'


def percentage_formatter(x, pos):
    """Format numbers as percentages."""
    return f'{x*100:.1f}%'


def plot_price_and_states(df, title='BTC Price and HMM States', save_path=None):
    """
    Plot price chart with HMM states overlay.
    
    Args:
        df (pd.DataFrame): DataFrame with price and state data
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    if 'state' not in df.columns or 'price' not in df.columns:
        print("Error: DataFrame must contain 'state' and 'price' columns")
        return
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    
    # Plot price
    ax1.plot(df['date'], df['price'], 'steelblue', linewidth=1.5, label='Price')
    
    # Format date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Format y-axis as money
    ax1.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Create a twin axis for the state
    ax2 = ax1.twinx()
    ax2.set_ylabel('HMM State', fontsize=12)
    
    # Plot states as a step function
    ax2.step(df['date'], df['state'], 'r-', linewidth=1.0, alpha=0.7, label='State')
    
    # Add a scatter plot with colors for different states
    unique_states = df['state'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
    
    for i, state in enumerate(unique_states):
        mask = df['state'] == state
        ax2.scatter(df.loc[mask, 'date'], df.loc[mask, 'state'], 
                   color=colors[i], label=f'State {state}', s=30, zorder=5)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_trading_signals(df, title='Trading Signals', save_path=None):
    """
    Plot price chart with buy/sell signals.
    
    Args:
        df (pd.DataFrame): DataFrame with price and signal data
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    if 'price' not in df.columns or 'signal' not in df.columns:
        print("Error: DataFrame must contain 'price' and 'signal' columns")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    
    # Plot price
    ax.plot(df['date'], df['price'], 'steelblue', linewidth=1.5, label='Price')
    
    # Plot buy signals
    buy_signals = df[df['signal'] == 1]
    ax.scatter(buy_signals['date'], buy_signals['price'], marker='^', color='green', s=100, label='Buy', zorder=5)
    
    # Plot sell signals
    sell_signals = df[df['signal'] == -1]
    ax.scatter(sell_signals['date'], sell_signals['price'], marker='v', color='red', s=100, label='Sell', zorder=5)
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Format y-axis as money
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_performance_metrics(performance, title='Performance Metrics', save_path=None):
    """
    Plot performance metrics.
    
    Args:
        performance (dict): Dictionary with performance metrics
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    # Extract metrics
    metrics = {k: v for k, v in performance.items() if isinstance(v, (int, float))}
    
    # Sort metrics by value
    sorted_metrics = sorted(metrics.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set colors based on values
    colors = ['green' if v > 0 else 'red' for _, v in sorted_metrics]
    
    # Create bar plot
    bars = ax.barh([m[0] for m in sorted_metrics], [m[1] for m in sorted_metrics], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01 if width > 0 else width * 0.99
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    ax.set_xlabel('Value', fontsize=12)
    
    # Format x-axis for percentages
    if any('Return' in m or 'Sharpe' in m or 'Drawdown' in m for m in metrics.keys()):
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_portfolio_growth(results, title='Portfolio Growth', save_path=None):
    """
    Plot portfolio value over time.
    
    Args:
        results (pd.DataFrame): DataFrame with portfolio results
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    if 'portfolio_value' not in results.columns:
        print("Error: DataFrame must contain 'portfolio_value' column")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    
    # Plot portfolio value
    ax.plot(results['date'], results['portfolio_value'], 'forestgreen', linewidth=2, label='Portfolio Value')
    
    # Plot buy-and-hold benchmark if available
    if 'buy_hold_value' in results.columns:
        ax.plot(results['date'], results['buy_hold_value'], 'gray', linewidth=1.5, linestyle='--', label='Buy & Hold')
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Format y-axis as money
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def create_performance_dashboard(results, performance, crypto='BTC'):
    """
    Create a comprehensive performance dashboard with both Matplotlib and Plotly visualizations.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        performance (dict): Dictionary with performance metrics
        crypto (str): Cryptocurrency symbol
        
    Returns:
        str: Path to the dashboard directory
    """
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(RESULTS_DIR, f"{crypto}_results_{timestamp}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create and save static plots with Matplotlib
    # Plot price and states
    if 'hmm_state' in results.columns:
        state_column = 'hmm_state'
        # Backward compatibility with 'state' column
        if 'state' not in results.columns:
            results['state'] = results[state_column]
            
        price_states_path = os.path.join(save_dir, f"{crypto}_price_states.png")
        plot_price_and_states(results, title=f'{crypto} Price and HMM States', save_path=price_states_path)
    
    # Plot trading signals
    if 'signal' in results.columns:
        signals_path = os.path.join(save_dir, f"{crypto}_signals.png")
        plot_trading_signals(results, title=f'{crypto} Trading Signals', save_path=signals_path)
    
    # Plot portfolio growth
    if 'portfolio_value' in results.columns:
        portfolio_path = os.path.join(save_dir, f"{crypto}_portfolio.png")
        plot_portfolio_growth(results, title=f'{crypto} Portfolio Growth', save_path=portfolio_path)
    
    # Plot performance metrics
    metrics_path = os.path.join(save_dir, f"{crypto}_metrics.png")
    plot_performance_metrics(performance, title=f'{crypto} Performance Metrics', save_path=metrics_path)
    
    # Create an HTML report
    html_path = os.path.join(save_dir, f"{crypto}_report.html")
    create_html_report(results, performance, crypto, save_dir, html_path)
    
    print(f"Matplotlib dashboard saved to {save_dir}")
    
    # Create and save interactive Plotly visualizations
    create_plotly_trading_dashboard(results, performance, crypto, save_dir)
    
    return save_dir


def create_html_report(results, performance, crypto, img_dir, output_path):
    """
    Create a simple HTML report with the dashboard visualizations.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        performance (dict): Dictionary with performance metrics
        crypto (str): Cryptocurrency symbol
        img_dir (str): Directory with saved images
        output_path (str): Path to save the HTML report
    """
    # Get relative paths to images
    rel_dir = os.path.basename(img_dir)
    img_files = {
        'price_states': f"{crypto}_price_states.png",
        'signals': f"{crypto}_signals.png",
        'portfolio': f"{crypto}_portfolio.png",
        'metrics': f"{crypto}_metrics.png"
    }
    
    # Format performance metrics table
    metrics_html = "<table class='metrics-table'>"
    metrics_html += "<tr><th>Metric</th><th>Value</th></tr>"
    
    for metric, value in performance.items():
        if isinstance(value, (int, float)):
            # Format percentage values
            if 'return' in metric.lower() or 'drawdown' in metric.lower():
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.4f}"
            
            metrics_html += f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>"
    
    metrics_html += "</table>"
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{crypto} Trading Strategy Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .dashboard-img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border: 1px solid #ddd;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .metrics-table th {{
                background-color: #f2f2f2;
            }}
            .metrics-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .section {{
                margin: 40px 0;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 0.8em;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{crypto} Trading Strategy Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {metrics_html}
            </div>
            
            <div class="section">
                <h2>Portfolio Growth</h2>
                <img src="{img_files['portfolio']}" class="dashboard-img" alt="Portfolio Growth">
            </div>
            
            <div class="section">
                <h2>Price and HMM States</h2>
                <img src="{img_files['price_states']}" class="dashboard-img" alt="Price and HMM States">
            </div>
            
            <div class="section">
                <h2>Trading Signals</h2>
                <img src="{img_files['signals']}" class="dashboard-img" alt="Trading Signals">
            </div>
            
            <div class="section">
                <h2>Performance Metrics Visualization</h2>
                <img src="{img_files['metrics']}" class="dashboard-img" alt="Performance Metrics">
            </div>
            
            <div class="footer">
                <p>This report was automatically generated by the HMM Crypto Trading Strategy.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report created at {output_path}")
    return output_path


def create_plotly_trading_dashboard(results, performance, crypto='BTC', save_dir=None):
    """
    Create an interactive Plotly dashboard showing trading signals and performance.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        performance (dict): Dictionary with performance metrics
        crypto (str): Cryptocurrency symbol
        save_dir (str): Directory to save the HTML file
        
    Returns:
        str: Path to the saved HTML file
    """
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(RESULTS_DIR, f"{crypto}_dashboard_{timestamp}")
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create main price and signals chart
    price_signals_fig = create_plotly_price_signals(results, crypto)
    
    # Create portfolio performance chart
    portfolio_fig = create_plotly_portfolio_growth(results, crypto)
    
    # Create performance metrics chart
    metrics_fig = create_plotly_performance_metrics(performance)
    
    # Save the figures as HTML
    price_signals_path = os.path.join(save_dir, f"{crypto}_price_signals.html")
    portfolio_path = os.path.join(save_dir, f"{crypto}_portfolio.html")
    metrics_path = os.path.join(save_dir, f"{crypto}_metrics.html")
    
    price_signals_fig.write_html(price_signals_path)
    portfolio_fig.write_html(portfolio_path)
    metrics_fig.write_html(metrics_path)
    
    print(f"Plotly dashboard created in {save_dir}")
    
    return save_dir


def create_plotly_price_signals(results, crypto='BTC'):
    """
    Create an interactive Plotly chart showing price movement and trading signals.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        crypto (str): Cryptocurrency symbol
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Check if required columns exist
    required_cols = ['date', 'signal', 'hmm_state']
    price_col = None
    
    if 'price' in results.columns:
        price_col = 'price'
    elif 'close' in results.columns:
        price_col = 'close'
    
    if price_col is None or not all(col in results.columns for col in required_cols):
        print("Warning: Required columns missing for plotting")
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results[price_col],
            mode='lines',
            name=f'{crypto} Price',
            line=dict(color='royalblue', width=1.5)
        )
    )
    
    # Add buy signals
    buy_signals = results[results['signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['date'],
                y=buy_signals[price_col],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='triangle-up',
                    line=dict(color='darkgreen', width=1)
                )
            )
        )
    
    # Add sell signals
    sell_signals = results[results['signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['date'],
                y=sell_signals[price_col],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='triangle-down',
                    line=dict(color='darkred', width=1)
                )
            )
        )
    
    # Add HMM states on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results['hmm_state'],
            mode='lines',
            name='HMM State',
            line=dict(color='purple', width=1, dash='dot'),
            opacity=0.7
        ),
        secondary_y=True
    )
    
    # Set titles and labels
    fig.update_layout(
        title=f'{crypto} Price with HMM Trading Signals',
        xaxis_title='Date',
        yaxis_title=f'{crypto} Price (USD)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text=f"{crypto} Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="HMM State", secondary_y=True)
    
    return fig


def create_plotly_portfolio_growth(results, crypto='BTC'):
    """
    Create an interactive Plotly chart showing portfolio growth.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        crypto (str): Cryptocurrency symbol
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Check if we need to calculate portfolio value
    if 'portfolio_value' not in results.columns and 'strategy_value' in results.columns:
        # Use strategy_value column instead
        results = results.copy()
        results['portfolio_value'] = results['strategy_value']
    elif 'portfolio_value' not in results.columns and 'cumulative_returns' in results.columns:
        # Calculate portfolio value from cumulative returns
        results = results.copy()
        initial_value = 10000  # Assume $10,000 starting capital
        results['portfolio_value'] = initial_value * (1 + results['cumulative_returns'])
    
    if 'portfolio_value' not in results.columns:
        print("Warning: portfolio_value column missing and cannot be calculated")
        # Create a simple figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Portfolio value data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Portfolio Growth - Data Missing")
        return fig
    
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results['portfolio_value'],
            mode='lines',
            name='Strategy',
            line=dict(color='green', width=2)
        )
    )
    
    # Add buy-and-hold line if available
    if 'buy_hold_value' in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=results['buy_hold_value'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=1.5, dash='dash')
            )
        )
    
    # Add position information if available
    if 'position' in results.columns:
        # Get position change points
        position_changes = results[results['position'].diff() != 0]
        
        # Add position annotations
        for idx, row in position_changes.iterrows():
            position_text = "LONG" if row['position'] > 0 else "SHORT" if row['position'] < 0 else "CASH"
            position_color = "green" if row['position'] > 0 else "red" if row['position'] < 0 else "yellow"
            
            fig.add_annotation(
                x=row['date'],
                y=row['portfolio_value'],
                text=position_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor=position_color,
                arrowsize=1,
                arrowwidth=1,
                ax=0,
                ay=-40
            )
    
    # Set titles and layout
    fig.update_layout(
        title=f'{crypto} Trading Strategy Performance',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (USD)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Format y-axis as currency
    fig.update_yaxes(
        tickprefix='$',
        tickformat=',.2f'
    )
    
    return fig


def create_plotly_performance_metrics(performance):
    """
    Create an interactive Plotly chart showing performance metrics.
    
    Args:
        performance (dict): Dictionary with performance metrics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Convert performance dict to DataFrame for plotting
    metrics = {k: v for k, v in performance.items() if isinstance(v, (int, float))}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    
    # Sort by absolute value (using pandas sort_values instead of argsort)
    metrics_df = metrics_df.sort_values('Value', key=abs, ascending=False)
    
    # Create color map (green for positive, red for negative)
    colors = ['green' if v > 0 else 'red' for v in metrics_df['Value']]
    
    # Create bar chart
    fig = px.bar(
        metrics_df,
        y='Metric',
        x='Value',
        orientation='h',
        color_discrete_sequence=colors,
        labels={'Value': 'Value', 'Metric': ''},
        title='Strategy Performance Metrics'
    )
    
    # Add value labels
    fig.update_traces(
        texttemplate='%{x:.4f}',
        textposition='outside'
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Value',
        yaxis=dict(
            categoryorder='total ascending'
        ),
        height=600,
        margin=dict(l=200)
    )
    
    return fig 