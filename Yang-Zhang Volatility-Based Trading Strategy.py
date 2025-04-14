import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from IPython.display import display

# FMP API Key
API_KEY = "YOU_FMP_API_KEY_HERE"
BASE_URL = "https://financialmodelingprep.com/api/v3"

def fetch_data(symbol, start_date, end_date=None, include_vix=True):
    """Fetch historical price data using FMP API with optional VIX data"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    # Fetch primary symbol data
    url = f"{BASE_URL}/historical-price-full/{symbol}"
    params = {
        'from': start_date,
        'to': end_date,
        'apikey': API_KEY
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    
    data = response.json()
    if 'historical' not in data or len(data['historical']) == 0:
        raise Exception(f"No historical data found for {symbol}")
    
    # Convert to DataFrame and format
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    # Rename columns to match standard OHLC format
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Fetch VIX data if requested
    if include_vix:
        try:
            print("Downloading VIX data...")
            vix_url = f"{BASE_URL}/historical-price-full/index/%5EVIX"
            vix_response = requests.get(vix_url, params=params)
            
            if vix_response.status_code == 200:
                vix_data = vix_response.json()
                if 'historical' in vix_data and len(vix_data['historical']) > 0:
                    vix_df = pd.DataFrame(vix_data['historical'])
                    vix_df['date'] = pd.to_datetime(vix_df['date'])
                    vix_df = vix_df.sort_values('date')
                    vix_df.set_index('date', inplace=True)
                    
                    # Add VIX close to main dataframe
                    df['VIX'] = vix_df['close']
                    
                    # Create synthetic VIX3M (not available directly in FMP)
                    df['VIX3M'] = df['VIX'].rolling(window=63).mean()  # ~3 months
                    
                    # Calculate VIX ratio (VIX/VIX3M)
                    df['VIX_Ratio'] = df['VIX'] / df['VIX3M']
                    
                    print("VIX data downloaded and processed")
                else:
                    print("No VIX data found")
            else:
                print(f"Error fetching VIX data: {vix_response.status_code}")
        except Exception as e:
            print(f"Error processing VIX data: {str(e)}")
    
    print(f"Downloaded {len(df)} days of data")
    return df

def calculate_yang_zhang_volatility(data, window=10):
    """
    Calculate Yang-Zhang volatility, which combines overnight and intraday volatility
    
    Parameters:
    - data: DataFrame with OHLC prices
    - window: Lookback period for volatility calculation
    
    Returns:
    - Series of annualized Yang-Zhang volatility
    """
    # Ensure window is at least 2
    window = max(2, window)
    
    # Extract price data
    open_price = data['Open']
    high_price = data['High']
    low_price = data['Low']
    close_price = data['Close']
    
    # Calculate logarithmic returns for different components
    log_ho = np.log(high_price / open_price)
    log_lo = np.log(low_price / open_price)
    log_co = np.log(close_price / open_price)
    log_oc = np.log(open_price / close_price.shift(1))
    
    # Calculate components for Yang-Zhang volatility
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)  # Rogers-Satchell volatility
    close_vol = log_co**2  # Close-to-open (overnight) volatility
    open_vol = log_oc**2   # Open-to-close (intraday) volatility
    
    # Calculate rolling values
    rs_roll = rs.rolling(window=window).mean()
    close_roll = close_vol.rolling(window=window).mean()
    open_roll = open_vol.rolling(window=window).mean()
    
    # Calculate Yang-Zhang volatility with optimal k parameter
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_vol = np.sqrt(open_roll + k * close_roll + (1 - k) * rs_roll)
    
    # Annualize volatility (multiply by sqrt(252))
    yz_vol_annualized = yz_vol * np.sqrt(252)
    
    return yz_vol_annualized

def generate_signals(data, window_short=8, window_long=21, vol_window=10, vol_lookback=252):
    """
    Generate trading signals based on Yang-Zhang volatility and market regime
    
    Parameters:
    - data: DataFrame with OHLC prices and VIX data
    - window_short: Short-term momentum window
    - window_long: Long-term momentum window
    - vol_window: Window for Yang-Zhang volatility calculation
    - vol_lookback: Lookback period for volatility regime determination
    
    Returns:
    - DataFrame with signals and indicators
    """
    df = data.copy()
    
    # Calculate returns
    df['Return'] = df['Close'].pct_change()
    
    # Calculate Yang-Zhang volatility
    df['YZ_Vol'] = calculate_yang_zhang_volatility(df, window=vol_window)
    
    # Calculate moving averages
    df['EMA_Short'] = df['Close'].ewm(span=window_short, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=window_long, adjust=False).mean()
    
    # Calculate oscillator based on EMAs
    df['Oscillator'] = ((df['EMA_Short'] - df['EMA_Long']) / df['EMA_Long']) * 100
    
    # Calculate volatility regime
    df['Vol_Percentile'] = df['YZ_Vol'].rolling(window=vol_lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], 
        raw=False
    )
    
    # Determine market conditions using VIX ratio if available
    if 'VIX_Ratio' in df.columns:
        # VIX_Ratio > 1 indicates backwardation (high fear)
        # VIX_Ratio < 1 indicates contango (low fear)
        df['Market_Fear'] = df['VIX_Ratio'].rolling(window=21).mean()
    else:
        # Fallback to using just volatility percentile
        df['Market_Fear'] = df['Vol_Percentile']
    
    # Initialize signal and position columns
    df['Signal'] = 0
    df['Position'] = 0
    
    # Generate signals based on oscillator and volatility conditions
    # Buy when:
    # 1. Oscillator crosses above zero
    # 2. Volatility is not extremely high
    
    # Sell when:
    # 1. Oscillator crosses below zero
    # 2. Volatility spikes to extreme levels
    
    for i in range(max(window_long, vol_lookback) + 1, len(df)):
        # Current and previous values
        curr_osc = df['Oscillator'].iloc[i]
        prev_osc = df['Oscillator'].iloc[i-1]
        
        vol_pct = df['Vol_Percentile'].iloc[i]
        market_fear = df['Market_Fear'].iloc[i]
        
        # Entry conditions
        buy_signal = (prev_osc < 0 and curr_osc >= 0) and vol_pct < 0.8
        
        # Add more conservative entry in high volatility regimes
        if 'VIX_Ratio' in df.columns:
            # In times of market stress (backwardation), be more cautious
            if market_fear > 1.1:  # Strong backwardation
                buy_signal = buy_signal and curr_osc > 2  # Stronger momentum required
        
        # Exit conditions - if either of these is true, exit
        sell_signal = (prev_osc > 0 and curr_osc <= 0)  # Oscillator turns negative
        vol_exit = vol_pct > 0.9  # Volatility spike exit
        
        # Set signals
        if buy_signal:
            df.iloc[i, df.columns.get_loc('Signal')] = 1
        elif sell_signal or vol_exit:
            df.iloc[i, df.columns.get_loc('Signal')] = -1
    
    # Calculate positions (1 = long, 0 = flat)
    position = 0
    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        
        if signal == 1:  # Buy signal
            position = 1
        elif signal == -1:  # Sell signal
            position = 0
            
        df.iloc[i, df.columns.get_loc('Position')] = position
    
    return df

def backtest_strategy(data):
    """
    Backtest the strategy with execution at next day's open
    
    Parameters:
    - data: DataFrame with signals
    
    Returns:
    - DataFrame with backtest results
    """
    df = data.copy()
    
    # Calculate next day open for execution
    df['Next_Open'] = df['Open'].shift(-1)
    
    # Initialize tracking columns
    df['Trade_Entry'] = np.nan
    df['Trade_Exit'] = np.nan
    df['Trade_Return'] = np.nan
    
    # Track trades
    in_position = False
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)-1):  # Skip first day and last day
        current_date = df.index[i]
        next_date = df.index[i+1]
        
        # Buy signal at today's close = entry at tomorrow's open
        if df['Signal'].iloc[i] == 1 and not in_position:
            entry_price = df['Next_Open'].iloc[i]
            entry_date = next_date
            in_position = True
            df.loc[next_date, 'Trade_Entry'] = entry_price
            
        # Sell signal at today's close = exit at tomorrow's open
        elif df['Signal'].iloc[i] == -1 and in_position:
            exit_price = df['Next_Open'].iloc[i]
            
            # Calculate trade return
            trade_return = (exit_price / entry_price) - 1
            
            # Record exit information
            df.loc[next_date, 'Trade_Exit'] = exit_price
            df.loc[next_date, 'Trade_Return'] = trade_return
            
            # Reset trade tracking
            in_position = False
            entry_price = 0
            entry_date = None
    
    # Calculate strategy returns
    df['Strategy_Return'] = np.nan
    
    # For days with a position, calculate returns from open to next day's open
    for i in range(1, len(df)-1):
        if df['Position'].iloc[i] == 1:
            # Return from today's open to tomorrow's open
            day_return = df['Next_Open'].iloc[i] / df['Open'].iloc[i] - 1
            df.loc[df.index[i+1], 'Strategy_Return'] = day_return
        else:
            df.loc[df.index[i+1], 'Strategy_Return'] = 0
    
    # Calculate cumulative returns
    df['Strategy_Equity'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    
    # Calculate buy and hold returns (open-to-open)
    df['BuyHold_Return'] = df['Open'].pct_change()
    df['BuyHold_Equity'] = (1 + df['BuyHold_Return'].fillna(0)).cumprod()
    
    # Calculate drawdowns
    df['Strategy_Peak'] = df['Strategy_Equity'].expanding().max()
    df['Strategy_Drawdown'] = df['Strategy_Equity'] / df['Strategy_Peak'] - 1
    
    df['BuyHold_Peak'] = df['BuyHold_Equity'].expanding().max()
    df['BuyHold_Drawdown'] = df['BuyHold_Equity'] / df['BuyHold_Peak'] - 1
    
    return df

def calculate_metrics(results):
    """
    Calculate performance metrics for both strategy and buy-hold
    
    Parameters:
    - results: DataFrame with backtest results
    
    Returns:
    - Dictionary with performance metrics
    """
    # Get trade returns
    trade_returns = results['Trade_Return'].dropna()
    strategy_returns = results['Strategy_Return'].dropna()
    buyhold_returns = results['BuyHold_Return'].dropna()
    
    # Default metrics
    if len(strategy_returns) == 0:
        return {
            'strategy': {
                'Total Return': 0,
                'Annual Return': 0,
                'Volatility': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0,
                'Number of Trades': 0,
                'Win Rate': 0,
                'Average Win': 0,
                'Average Loss': 0,
                'Profit Factor': 0
            },
            'buy_hold': {
                'Total Return': 0,
                'Annual Return': 0,
                'Volatility': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0
            }
        }
    
    # Calculate metrics for strategy
    strategy_metrics = {}
    
    # Total return
    strategy_metrics['Total Return'] = results['Strategy_Equity'].iloc[-1] - 1 if not results['Strategy_Equity'].empty else 0
    
    # Annualized return
    days = len(results)
    years = days / 252
    strategy_metrics['Annual Return'] = (1 + strategy_metrics['Total Return']) ** (1 / years) - 1
    
    # Volatility
    strategy_metrics['Volatility'] = strategy_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    strategy_metrics['Sharpe Ratio'] = strategy_metrics['Annual Return'] / strategy_metrics['Volatility'] \
                                     if strategy_metrics['Volatility'] > 0 else 0
    
    # Maximum drawdown
    strategy_metrics['Max Drawdown'] = results['Strategy_Drawdown'].min()
    
    # Trade statistics
    strategy_metrics['Number of Trades'] = len(trade_returns)
    strategy_metrics['Win Rate'] = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
    strategy_metrics['Average Win'] = trade_returns[trade_returns > 0].mean() \
                                    if len(trade_returns[trade_returns > 0]) > 0 else 0
    strategy_metrics['Average Loss'] = trade_returns[trade_returns < 0].mean() \
                                     if len(trade_returns[trade_returns < 0]) > 0 else 0
    
    # Profit factor
    total_gains = trade_returns[trade_returns > 0].sum() if len(trade_returns[trade_returns > 0]) > 0 else 0
    total_losses = abs(trade_returns[trade_returns < 0].sum()) if len(trade_returns[trade_returns < 0]) > 0 else 0
    strategy_metrics['Profit Factor'] = total_gains / total_losses if total_losses > 0 else float('inf')
    
    # Time in market
    strategy_metrics['Time in Market'] = (results['Position'] == 1).mean()
    
    # Calculate metrics for buy and hold
    buy_hold_metrics = {}
    
    # Total return
    buy_hold_metrics['Total Return'] = results['BuyHold_Equity'].iloc[-1] - 1 if not results['BuyHold_Equity'].empty else 0
    
    # Annualized return
    buy_hold_metrics['Annual Return'] = (1 + buy_hold_metrics['Total Return']) ** (1 / years) - 1
    
    # Volatility
    buy_hold_metrics['Volatility'] = buyhold_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    buy_hold_metrics['Sharpe Ratio'] = buy_hold_metrics['Annual Return'] / buy_hold_metrics['Volatility'] \
                                     if buy_hold_metrics['Volatility'] > 0 else 0
    
    # Maximum drawdown
    buy_hold_metrics['Max Drawdown'] = results['BuyHold_Drawdown'].min()
    
    # Convert values to percentages for display
    for key in ['Total Return', 'Annual Return', 'Volatility', 'Max Drawdown', 'Win Rate', 
               'Average Win', 'Average Loss', 'Time in Market']:
        if key in strategy_metrics:
            strategy_metrics[key] *= 100
    
    for key in ['Total Return', 'Annual Return', 'Volatility', 'Max Drawdown']:
        buy_hold_metrics[key] *= 100
    
    return {
        'strategy': strategy_metrics,
        'buy_hold': buy_hold_metrics
    }

def plot_results(data, metrics):
    """
    Plot backtest results showing strategy performance
    
    Parameters:
    - data: DataFrame with backtest results
    - metrics: Dictionary with performance metrics
    
    Returns:
    - Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1.5], hspace=0.3)
    
    # 1. Price chart with signals and YZ volatility
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['Close'], color='blue', alpha=0.7, label='Asset Price')
    
    # Right y-axis for volatility
    ax1_vol = ax1.twinx()
    ax1_vol.plot(data.index, data['YZ_Vol'], color='purple', alpha=0.5, label='YZ Volatility')
    ax1_vol.set_ylabel('Annualized Volatility', color='purple')
    
    # Add volatility percentile thresholds
    vol_high = data[data['Vol_Percentile'] >= 0.8]
    if not vol_high.empty:
        ax1_vol.scatter(vol_high.index, vol_high['YZ_Vol'], color='red', marker='_', alpha=0.5, label='High Vol (80%+)')
    
    # Mark entries and exits
    entries = data.dropna(subset=['Trade_Entry'])
    exits = data.dropna(subset=['Trade_Exit'])
    
    if not entries.empty:
        ax1.scatter(entries.index, entries['Trade_Entry'], color='green', marker='^', s=100, label='Entry')
    
    if not exits.empty:
        ax1.scatter(exits.index, exits['Trade_Exit'], color='red', marker='v', s=100, label='Exit')
    
    # Combine legends from both y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_vol.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title('Price Chart with Signals and Yang-Zhang Volatility')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # 2. Oscillator with signals
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(data.index, data['Oscillator'], color='blue', label='Oscillator')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Mark oscillator crossovers
    for i in range(1, len(data)):
        if data['Oscillator'].iloc[i-1] < 0 and data['Oscillator'].iloc[i] >= 0:
            ax2.scatter(data.index[i], data['Oscillator'].iloc[i], color='green', marker='o', s=50)
        elif data['Oscillator'].iloc[i-1] > 0 and data['Oscillator'].iloc[i] <= 0:
            ax2.scatter(data.index[i], data['Oscillator'].iloc[i], color='red', marker='o', s=50)
    
    ax2.set_title('Oscillator (EMA Differential)')
    ax2.set_ylabel('Oscillator Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Equity curves
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(data.index, data['Strategy_Equity'], color='blue', label='Strategy')
    ax3.plot(data.index, data['BuyHold_Equity'], color='green', alpha=0.7, label='Buy & Hold')
    
    # Highlight when strategy is in cash
    for i in range(len(data)-1):
        if data['Position'].iloc[i] == 0:
            ax3.axvspan(data.index[i], data.index[i+1], color='lightgray', alpha=0.3)
    
    ax3.set_title('Strategy vs Buy & Hold Performance')
    ax3.set_ylabel('Equity (Normalized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown comparison
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(data.index, data['Strategy_Drawdown'] * 100, color='blue', label='Strategy Drawdown')
    ax4.plot(data.index, data['BuyHold_Drawdown'] * 100, color='green', alpha=0.7, label='Buy & Hold Drawdown')
    ax4.set_title('Drawdown Comparison')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add comparative metrics as text
    metrics_text = (
        f"STRATEGY vs BUY & HOLD\n"
        f"{'='*30}\n"
        f"Total Return: {metrics['strategy']['Total Return']:.2f}% vs {metrics['buy_hold']['Total Return']:.2f}%\n"
        f"Annual Return: {metrics['strategy']['Annual Return']:.2f}% vs {metrics['buy_hold']['Annual Return']:.2f}%\n"
        f"Volatility: {metrics['strategy']['Volatility']:.2f}% vs {metrics['buy_hold']['Volatility']:.2f}%\n"
        f"Sharpe Ratio: {metrics['strategy']['Sharpe Ratio']:.2f} vs {metrics['buy_hold']['Sharpe Ratio']:.2f}\n"
        f"Max Drawdown: {metrics['strategy']['Max Drawdown']:.2f}% vs {metrics['buy_hold']['Max Drawdown']:.2f}%\n\n"
        f"STRATEGY DETAILS\n"
        f"{'='*20}\n"
        f"Trades: {int(metrics['strategy']['Number of Trades'])}\n"
        f"Win Rate: {metrics['strategy']['Win Rate']:.2f}%\n"
        f"Profit Factor: {metrics['strategy']['Profit Factor']:.2f}\n"
        f"Time in Market: {metrics['strategy']['Time in Market']:.2f}%"
    )
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.02, 0.02, metrics_text, transform=ax3.transAxes, fontsize=10,
               verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig

def run_strategy(symbol='BTCUSD', start_date='2018-01-01', end_date=None,
                window_short=8, window_long=21, vol_window=10, vol_lookback=252):
    """
    Run the complete Yang-Zhang volatility-based strategy
    
    Parameters:
    - symbol: Stock symbol (default: BTCUSD)
    - start_date: Start date in 'YYYY-MM-DD' format
    - end_date: End date in 'YYYY-MM-DD' format (default: today)
    - window_short: Short-term EMA window (default: 8)
    - window_long: Long-term EMA window (default: 21)
    - vol_window: Window for Yang-Zhang volatility calculation (default: 10)
    - vol_lookback: Lookback period for volatility regime determination (default: 252)
    
    Returns:
    - Tuple of (backtest_results, metrics)
    """
    # Set end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Running Yang-Zhang Volatility Strategy for {symbol}")
    print(f"Period: {start_date} to {end_date}")
    
    # Fetch historical data
    print("Fetching historical data...")
    data = fetch_data(symbol, start_date, end_date)
    
    # Generate signals
    print("Generating trading signals...")
    signals = generate_signals(
        data, 
        window_short=window_short,
        window_long=window_long,
        vol_window=vol_window,
        vol_lookback=vol_lookback
    )
    
    # Run backtest
    print("Running backtest with execution at next day's open...")
    backtest = backtest_strategy(signals)
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(backtest)
    
    # Print comparison results
    print("\nStrategy vs Buy & Hold Performance:")
    print("=" * 60)
    
    # Format metrics for display
    comparison_table = pd.DataFrame({
        'Strategy': [
            f"{metrics['strategy']['Total Return']:.2f}%",
            f"{metrics['strategy']['Annual Return']:.2f}%",
            f"{metrics['strategy']['Volatility']:.2f}%",
            f"{metrics['strategy']['Sharpe Ratio']:.2f}",
            f"{metrics['strategy']['Max Drawdown']:.2f}%"
        ],
        'Buy & Hold': [
            f"{metrics['buy_hold']['Total Return']:.2f}%",
            f"{metrics['buy_hold']['Annual Return']:.2f}%",
            f"{metrics['buy_hold']['Volatility']:.2f}%",
            f"{metrics['buy_hold']['Sharpe Ratio']:.2f}",
            f"{metrics['buy_hold']['Max Drawdown']:.2f}%"
        ]
    }, index=['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'])
    
    display(comparison_table)
    
    print("\nStrategy Details:")
    print("-" * 40)
    print(f"Number of Trades: {int(metrics['strategy']['Number of Trades'])}")
    print(f"Win Rate: {metrics['strategy']['Win Rate']:.2f}%")
    print(f"Avg Winning Trade: {metrics['strategy']['Average Win']:.2f}%")
    print(f"Avg Losing Trade: {metrics['strategy']['Average Loss']:.2f}%")
    print(f"Profit Factor: {metrics['strategy']['Profit Factor']:.2f}")
    print(f"Time in Market: {metrics['strategy']['Time in Market']:.2f}%")
    
    # Plot results
    print("\nGenerating charts...")
    fig = plot_results(backtest, metrics)
    plt.show()
    
    return backtest, metrics

# Run the strategy for Bitcoin USD
backtest_results, metrics = run_strategy(
    symbol='BTCUSD',  # FMP API uses BTCUSD without the hyphen
    start_date='2018-01-01',
    window_short=8,
    window_long=21,
    vol_window=10,
    vol_lookback=252
)
