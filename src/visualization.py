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
    
    plt.show()  # Display the plot
    
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
    
    plt.show()  # Display the plot
    
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
    
    plt.show()  # Display the plot
    
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
    Create a complete performance dashboard with multiple visualizations.
    
    Args:
        results (pd.DataFrame): DataFrame with trading results
        performance (dict): Dictionary with performance metrics
        crypto (str): Cryptocurrency symbol
        
    Returns:
        str: Path to the dashboard directory
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dashboard_dir = os.path.join(RESULTS_DIR, f"{crypto}_dashboard_{timestamp}")
    
    if not os.path.exists(dashboard_dir):
        os.makedirs(dashboard_dir)
    
    # Check if we have the required data
    if results is None or performance is None:
        print("Warning: Missing data for dashboard creation")
        # Create a simple HTML file with the warning
        with open(os.path.join(dashboard_dir, "dashboard_error.html"), "w") as f:
            f.write("<html><body><h1>Dashboard Error</h1><p>Missing required data for dashboard creation.</p></body></html>")
        return dashboard_dir
    
    try:
        # Traditional matplotlib visualizations
        print("Creating traditional matplotlib visualizations...")
        
        # Create price and signal chart
        if 'signal' in results.columns:
            fig = plot_trading_signals(results, title=f'{crypto} Trading Signals')
            if fig:
                plt.figure(fig.number)
                signals_path = os.path.join(dashboard_dir, f"{crypto}_signals.png")
                plt.savefig(signals_path, dpi=300, bbox_inches='tight')
                print(f"  Saved trading signals chart to {signals_path}")
        
        # Create portfolio value chart
        if 'portfolio_value' in results.columns:
            plt.figure(figsize=(14, 8))
            plt.plot(results['date'], results['portfolio_value'], 'g-', label='Portfolio Value')
            
            if 'buy_hold_value' in results.columns:
                plt.plot(results['date'], results['buy_hold_value'], 'b--', label='Buy & Hold')
            
            plt.title(f'{crypto} Portfolio Growth', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value ($)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            portfolio_path = os.path.join(dashboard_dir, f"{crypto}_portfolio.png")
            plt.savefig(portfolio_path, dpi=300, bbox_inches='tight')
            print(f"  Saved portfolio chart to {portfolio_path}")
        
        # Create HMM state visualization if available
        if 'hmm_state' in results.columns and 'price' in results.columns:
            fig = plot_price_and_states(results, title=f'{crypto} Price and HMM States')
            if fig:
                plt.figure(fig.number)
                states_path = os.path.join(dashboard_dir, f"{crypto}_states.png")
                plt.savefig(states_path, dpi=300, bbox_inches='tight')
                print(f"  Saved HMM states chart to {states_path}")
        
        # Create performance metrics chart
        if performance:
            fig = plot_performance_metrics(performance, title=f'{crypto} Performance Metrics')
            if fig:
                plt.figure(fig.number)
                metrics_path = os.path.join(dashboard_dir, f"{crypto}_metrics.png")
                plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
                print(f"  Saved performance metrics chart to {metrics_path}")
    except Exception as e:
        print(f"Error creating matplotlib visualizations: {str(e)}")
    
    try:
        # Create Plotly interactive visualizations
        print("Creating interactive Plotly visualizations...")
        plotly_dir = os.path.join(dashboard_dir, "plotly")
        
        if not os.path.exists(plotly_dir):
            os.makedirs(plotly_dir)
            
        # Create price and signals chart
        try:
            price_signals_fig = create_plotly_price_signals(results, crypto)
            if price_signals_fig:
                price_signals_path = os.path.join(plotly_dir, f"{crypto}_price_signals.html")
                price_signals_fig.write_html(price_signals_path)
                print(f"  Saved interactive price signals chart to {price_signals_path}")
        except Exception as e:
            print(f"  Error creating price signals chart: {str(e)}")
        
        # Create portfolio growth chart
        try:
            portfolio_fig = create_plotly_portfolio_growth(results, crypto)
            if portfolio_fig:
                portfolio_path = os.path.join(plotly_dir, f"{crypto}_portfolio.html")
                portfolio_fig.write_html(portfolio_path)
                print(f"  Saved interactive portfolio chart to {portfolio_path}")
        except Exception as e:
            print(f"  Error creating portfolio chart: {str(e)}")
        
        # Create performance metrics chart
        try:
            metrics_fig = create_plotly_performance_metrics(performance)
            if metrics_fig:
                metrics_path = os.path.join(plotly_dir, f"{crypto}_metrics.html")
                metrics_fig.write_html(metrics_path)
                print(f"  Saved interactive metrics chart to {metrics_path}")
        except Exception as e:
            print(f"  Error creating metrics chart: {str(e)}")
            
        # Create HTML report
        try:
            report_path = create_html_report(results, performance, crypto, plotly_dir, 
                                           os.path.join(dashboard_dir, f"{crypto}_report.html"))
            print(f"  Saved HTML report to {report_path}")
        except Exception as e:
            print(f"  Error creating HTML report: {str(e)}")
    except Exception as e:
        print(f"Error creating Plotly visualizations: {str(e)}")
    
    print(f"Dashboard created in {dashboard_dir}")
    return dashboard_dir


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
    required_cols = ['date', 'signal']
    optional_cols = ['hmm_state', 'lstm_signal', 'prediction', 'probability']
    price_col = None
    
    # Find price column
    for col in ['price', 'close', 'price_usd_close', 'value']:
        if col in results.columns:
            price_col = col
            break
    
    if price_col is None or 'date' not in results.columns or 'signal' not in results.columns:
        print(f"Warning: Required columns missing for plotting. Found columns: {results.columns.tolist()}")
        # Create a simple figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Cannot create chart: Missing required columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Price and Signals - Data Missing")
        return fig
    
    # Create figure with secondary y-axis (only if we have state data)
    has_state = 'hmm_state' in results.columns
    fig = make_subplots(specs=[[{"secondary_y": has_state}]])
    
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
    
    # Add HMM states on secondary y-axis (if available)
    if 'hmm_state' in results.columns:
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
        # Set secondary y-axis title
        fig.update_yaxes(title_text="HMM State", secondary_y=True)
    
    # Add LSTM predictions if available
    if 'lstm_prediction' in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=results['lstm_prediction'],
                mode='lines',
                name='LSTM Prediction',
                line=dict(color='orange', width=1, dash='dot'),
                opacity=0.7
            ),
            secondary_y=True
        )
    
    # Set titles and labels
    model_type = "HMM" if "hmm_state" in results.columns else "ML"
    if "lstm_prediction" in results.columns or "lstm_signal" in results.columns:
        model_type = "LSTM"
    
    fig.update_layout(
        title=f'{crypto} Price with {model_type} Trading Signals',
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
    
    # Set y-axes title (always set the primary y-axis)
    fig.update_yaxes(title_text=f"{crypto} Price (USD)", secondary_y=False)
    
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
    # Make a copy to avoid modifying the original
    results = results.copy()
    
    # Check if we need to calculate portfolio value
    if 'portfolio_value' not in results.columns:
        price_col = None
        for col in ['price', 'close', 'price_usd_close', 'value']:
            if col in results.columns:
                price_col = col
                break
                
        if price_col is None:
            print("Warning: No price column found for portfolio calculation")
            # Create a simple figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="Portfolio value data not available - No price column",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Portfolio Growth - Data Missing")
            return fig
            
        # Try to calculate portfolio value using different available columns
        if 'strategy_value' in results.columns:
            # Use strategy_value column
            results['portfolio_value'] = results['strategy_value']
            print("Using 'strategy_value' for portfolio calculations")
        elif 'cumulative_returns' in results.columns:
            # Calculate portfolio value from cumulative returns
            initial_value = 10000  # Assume $10,000 starting capital
            results['portfolio_value'] = initial_value * (1 + results['cumulative_returns'])
            print("Calculated portfolio value from 'cumulative_returns'")
        elif 'signal' in results.columns:
            # Calculate portfolio value from signals and price
            print("Calculating portfolio value from signals and price...")
            initial_value = 10000  # Assume $10,000 starting capital
            
            # Initialize columns for position and portfolio value
            results['position'] = 0
            results['portfolio_value'] = initial_value
            
            # Fill in positions - 1 for long, -1 for short, 0 for cash
            for i in range(1, len(results)):
                if results.iloc[i-1]['signal'] == 1:  # Buy signal
                    results.loc[results.index[i], 'position'] = 1
                elif results.iloc[i-1]['signal'] == -1:  # Sell signal
                    results.loc[results.index[i], 'position'] = -1 if 'no_shorts' not in results.columns else 0
                else:
                    results.loc[results.index[i], 'position'] = results.iloc[i-1]['position']
            
            # Calculate daily returns based on position
            results['daily_return'] = results[price_col].pct_change() * results['position']
            
            # Calculate cumulative returns and portfolio value
            results['cumulative_returns'] = (1 + results['daily_return'].fillna(0)).cumprod() - 1
            results['portfolio_value'] = initial_value * (1 + results['cumulative_returns'])
        else:
            print("Warning: Cannot calculate portfolio value - missing required columns")
            # Create a simple figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="Portfolio value data not available - Missing required columns",
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
            name='Portfolio Value',
            line=dict(color='green', width=2)
        )
    )
    
    # Add a horizontal line for the initial value
    initial_value = results['portfolio_value'].iloc[0]
    fig.add_shape(
        type="line",
        x0=results['date'].iloc[0],
        y0=initial_value,
        x1=results['date'].iloc[-1],
        y1=initial_value,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    # Get the buy & hold ending value
    if 'buy_hold_value' in results.columns:
        buy_hold_value = results['buy_hold_value']
        
        # Add buy & hold line
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=buy_hold_value,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=1.5, dash='dot')
            )
        )
    
    # Set titles and labels
    fig.update_layout(
        title=f'{crypto} Portfolio Growth',
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
    fig.update_yaxes(tickprefix='$')
    
    return fig


def create_plotly_performance_metrics(performance):
    """
    Create an interactive Plotly chart showing performance metrics.
    
    Args:
        performance (dict): Dictionary with performance metrics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if not performance or not isinstance(performance, dict):
        print("Warning: No performance metrics provided")
        # Create a simple figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Performance metrics not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Performance Metrics - Data Missing")
        return fig
        
    # Extract numeric metrics
    metrics = {k: v for k, v in performance.items() if isinstance(v, (int, float)) and not pd.isna(v)}
    
    if not metrics:
        print("Warning: No numeric performance metrics found")
        # Create a simple figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric performance metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Performance Metrics - No Data")
        return fig
    
    # Sort metrics by absolute value
    sorted_metrics = sorted(metrics.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create figure for bar chart
    fig = go.Figure()
    
    # Create separate traces for positive and negative metrics
    positive_metrics = [(k, v) for k, v in sorted_metrics if v >= 0]
    negative_metrics = [(k, v) for k, v in sorted_metrics if v < 0]
    
    if positive_metrics:
        fig.add_trace(
            go.Bar(
                x=[m[0] for m in positive_metrics],
                y=[m[1] for m in positive_metrics],
                name='Positive Metrics',
                marker_color='green'
            )
        )
    
    if negative_metrics:
        fig.add_trace(
            go.Bar(
                x=[m[0] for m in negative_metrics],
                y=[m[1] for m in negative_metrics],
                name='Negative Metrics',
                marker_color='red'
            )
        )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(metrics) - 0.5,
        y1=0,
        line=dict(color="white", width=1)
    )
    
    # Set titles and labels
    fig.update_layout(
        title='Trading Strategy Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add data labels
    for i, (metric, value) in enumerate(sorted_metrics):
        text_position = 'outside' if value >= 0 else 'outside'
        fig.add_annotation(
            x=metric,
            y=value,
            text=f"{value:.4f}",
            showarrow=False,
            yshift=10 if value >= 0 else -10,
            font=dict(color='white')
        )
    
    return fig 