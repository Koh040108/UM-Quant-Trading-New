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

from config import RESULTS_DIR

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


def create_performance_dashboard(results, performance, crypto='BTC', save_dir=None):
    """
    Create a complete performance dashboard with multiple plots.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        performance (dict): Dictionary with performance metrics
        crypto (str): Cryptocurrency symbol
        save_dir (str): Directory to save the figures
    """
    # Create save directory if needed
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(RESULTS_DIR, f"{crypto}_dashboard_{timestamp}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot price and states
    if 'state' in results.columns:
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
    
    print(f"Dashboard saved to {save_dir}")
    
    # Create an HTML report
    html_path = os.path.join(save_dir, f"{crypto}_report.html")
    create_html_report(results, performance, crypto, save_dir, html_path)
    
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