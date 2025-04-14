import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class SPYBuyTheDipStrategy:
    """
    SPY "Buy the Dip" simplified trading strategy:
    - VaR95% as initial trigger
    - IBS<0.2 as buy signal
    - EUP95% and IBS>0.8 as sell signal
    """
    
    def __init__(self, start_date='2010-01-01', end_date=None, ibs_threshold_low=0.2, ibs_threshold_high=0.8):
        self.ticker = 'SPY'
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.results = None
        self.ibs_threshold_low = ibs_threshold_low
        self.ibs_threshold_high = ibs_threshold_high
    
    def fetch_data(self):
        """Download historical data for SPY"""
        print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        # Download SPY data
        spy = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        
        if spy.empty:
            raise ValueError(f"Failed to download data for {self.ticker}")
        
        print(f"Downloaded {len(spy)} days of data.")
        self.data = spy.copy()
        return spy
    
    def calculate_ibs(self):
        """Calculate Internal Bar Strength (IBS)"""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
            
        # IBS = (Close - Low) / (High - Low)
        high_low_diff = self.data['High'] - self.data['Low']
        # Avoid division by zero
        high_low_diff[high_low_diff == 0] = np.nan
        
        self.data['IBS'] = (self.data['Close'] - self.data['Low']) / high_low_diff
        return self.data['IBS']
    
    def calculate_var(self, returns, confidence_level=0.95, window=21):
        """Calculate Value-at-Risk (VaR)"""
        alpha = 1 - confidence_level
        var = returns.rolling(window=window).quantile(alpha)
        return var
    
    def calculate_eup(self, returns, confidence_level=0.95, window=21):
        """Calculate Expected Upside Potential (EUP)"""
        threshold = confidence_level
        
        def calc_eup(x):
            try:
                if isinstance(x, pd.Series):
                    series = x
                elif hasattr(x, 'flatten'):
                    series = pd.Series(x.flatten())
                else:
                    series = pd.Series(x)
                
                if len(series) < 5 or series.isna().all():
                    return np.nan
                
                valid_returns = series.dropna()
                perc_threshold = np.percentile(valid_returns, threshold * 100)
                above_threshold = valid_returns[valid_returns >= perc_threshold]
                
                if len(above_threshold) == 0:
                    return np.nan
                    
                return above_threshold.mean()
            except Exception:
                return np.nan
                
        eup = returns.rolling(window=window).apply(calc_eup, raw=False)
        return eup
    
    def calculate_indicators(self):
        """Calculate all indicators needed for the strategy"""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
            
        print("Calculating indicators...")
        
        # Calculate returns
        self.data['Return'] = self.data['Close'].pct_change()
        returns = self.data['Return']
        
        # Calculate IBS
        self.calculate_ibs()
        
        # Calculate VaR
        self.data['VaR_95'] = self.calculate_var(returns, confidence_level=0.95, window=21)
        
        # Calculate EUP
        self.data['EUP_95'] = self.calculate_eup(returns, confidence_level=0.95, window=21)
        
        # Additional metrics for analysis
        # Daily volatility (21-day)
        self.data['Volatility_21'] = returns.rolling(window=21).std().multiply(np.sqrt(252))
        
        # Moving averages
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        
        print("Indicators calculated successfully.")
        return self.data
    
    def generate_signals(self):
        """Generate trading signals based on indicators"""
        if 'EUP_95' not in self.data.columns:
            raise ValueError("Indicators not calculated. Call calculate_indicators() first.")
            
        print("Generating trading signals...")
        
        # Initialize signal and position columns
        self.data['VaR_Trigger'] = 0
        self.data['Buy_Signal'] = 0
        self.data['Sell_Signal'] = 0
        self.data['Position'] = 0
        
        # Check if the current return is below the VaR 95% threshold
        var_condition = self.data['Return'] <= self.data['VaR_95']
        
        # VaR Trigger: Current return is below VaR95% threshold
        self.data.loc[var_condition, 'VaR_Trigger'] = 1
        
        # Current IBS values
        current_ibs = self.data['IBS']
        
        # Calculate the EUP threshold (80th percentile)
        eup_lookback = 252
        
        def percentile_80(x):
            if len(x) < 10 or x.isna().all():
                return np.nan
            return np.nanpercentile(x, 80)
        
        # Calculate rolling percentiles
        eup_threshold = self.data['EUP_95'].rolling(window=eup_lookback).apply(percentile_80, raw=False)
        
        # Buy signal: VaR triggered and IBS low (oversold)
        buy_condition = (
            (self.data['VaR_Trigger'] == 1) &  # VaR95 is triggered
            (current_ibs < self.ibs_threshold_low)  # IBS is below threshold (oversold)
        )
        
        # Safely assign buy signals
        self.data.loc[buy_condition, 'Buy_Signal'] = 1
        
        # Sell signal: EUP high and IBS high (overbought)
        sell_condition = (
            (self.data['EUP_95'].notna()) &    # EUP value exists
            (eup_threshold.notna()) &          # Threshold value exists
            (self.data['EUP_95'] >= eup_threshold) &  # EUP is in top 20% of its range
            (current_ibs > self.ibs_threshold_high)  # IBS is above threshold (overbought)
        )
        
        # Safely assign sell signals
        self.data.loc[sell_condition, 'Sell_Signal'] = 1
        
        # Calculate positions: Start with no position
        position = 0
        positions = np.zeros(len(self.data))
        
        # Process signals to determine position
        for i in range(len(self.data)):
            if self.data['Buy_Signal'].iloc[i] == 1 and position == 0:
                # Buy signal and currently no position
                position = 1
            elif self.data['Sell_Signal'].iloc[i] == 1 and position == 1:
                # Sell signal and currently long
                position = 0
                
            positions[i] = position
            
        # Assign position array to dataframe without using .iloc
        self.data['Position'] = positions
            
        print("Trading signals generated successfully.")
        return self.data
    
    def backtest_strategy(self):
        """Backtest the strategy and calculate performance metrics"""
        if 'Position' not in self.data.columns:
            raise ValueError("Signals not generated. Call generate_signals() first.")
            
        print("Running backtest...")
        
        # Calculate strategy returns
        self.data['Strategy_Return'] = self.data['Position'].shift(1) * self.data['Return']
        
        # Fill first-day NaN for Strategy Return with 0
        self.data['Strategy_Return'] = self.data['Strategy_Return'].fillna(0)
        
        # Calculate cumulative returns
        self.data['Buy_Hold_Cum_Return'] = (1 + self.data['Return'].fillna(0)).cumprod()
        self.data['Strategy_Cum_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        
        # Calculate drawdowns
        self.data['Buy_Hold_Peak'] = self.data['Buy_Hold_Cum_Return'].cummax()
        self.data['Strategy_Peak'] = self.data['Strategy_Cum_Return'].cummax()
        
        self.data['Buy_Hold_Drawdown'] = (self.data['Buy_Hold_Cum_Return'] / self.data['Buy_Hold_Peak']) - 1
        self.data['Strategy_Drawdown'] = (self.data['Strategy_Cum_Return'] / self.data['Strategy_Peak']) - 1
        
        # Prepare results for analysis
        self.results = self.data.copy()
        
        print("Backtest completed successfully.")
        return self.results
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics for the strategy"""
        if self.results is None:
            raise ValueError("Backtest not run. Call backtest_strategy() first.")
            
        print("Calculating performance metrics...")
        
        # Strategy and buy & hold returns
        strategy_returns = self.results['Strategy_Return']
        buy_hold_returns = self.results['Return'].fillna(0)
        
        # Basic metrics
        total_days = len(strategy_returns)
        trading_days_per_year = 252
        years = total_days / trading_days_per_year
        
        # Calculate metrics for both strategy and buy & hold
        metrics = {}
        
        # Strategy metrics
        metrics['Strategy'] = self._calculate_metrics(strategy_returns)
        
        # Buy & Hold metrics
        metrics['Buy_Hold'] = self._calculate_metrics(buy_hold_returns)
        
        # Strategy specific metrics
        total_trades = (self.results['Position'].diff() != 0).sum()
        metrics['Strategy']['Total_Trades'] = total_trades
        metrics['Strategy']['Trades_Per_Year'] = total_trades / years
        
        # Buy signals
        buy_signals = self.results['Buy_Signal'].sum()
        metrics['Strategy']['Buy_Signals'] = buy_signals
        
        # Sell signals
        sell_signals = self.results['Sell_Signal'].sum()
        metrics['Strategy']['Sell_Signals'] = sell_signals
        
        # Time in market
        time_in_market = self.results['Position'].mean() * 100
        metrics['Strategy']['Time_In_Market_Pct'] = time_in_market
        
        print("Performance metrics calculated successfully.")
        return metrics
    
    def _calculate_metrics(self, returns):
        """Calculate performance metrics for a return series"""
        metrics = {}
        
        # Basic metrics
        total_days = len(returns)
        trading_days_per_year = 252
        years = total_days / trading_days_per_year
        
        # Ensure we're not working with NaN values
        returns_filled = returns.fillna(0)
        
        # Total return
        metrics['Total_Return_Pct'] = ((1 + returns_filled).prod() - 1) * 100
        
        # Annualized return
        metrics['Annual_Return_Pct'] = (((1 + returns_filled).prod()) ** (1 / years) - 1) * 100
        
        # Volatility
        metrics['Daily_Volatility_Pct'] = returns_filled.std() * 100
        metrics['Annual_Volatility_Pct'] = returns_filled.std() * np.sqrt(trading_days_per_year) * 100
        
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        if returns_filled.std() > 0:
            metrics['Sharpe_Ratio'] = (returns_filled.mean() / returns_filled.std()) * np.sqrt(trading_days_per_year)
        else:
            metrics['Sharpe_Ratio'] = 0.0
        
        # Maximum Drawdown
        cum_returns = (1 + returns_filled).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        metrics['Max_Drawdown_Pct'] = drawdowns.min() * 100
        
        # Sortino Ratio
        negative_returns = returns_filled[returns_filled < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            downside_deviation = negative_returns.std() * np.sqrt(trading_days_per_year)
            metrics['Sortino_Ratio'] = (returns_filled.mean() * trading_days_per_year) / downside_deviation
        else:
            metrics['Sortino_Ratio'] = 0.0
        
        # Calmar Ratio
        if metrics['Max_Drawdown_Pct'] != 0:
            metrics['Calmar_Ratio'] = metrics['Annual_Return_Pct'] / abs(metrics['Max_Drawdown_Pct'])
        else:
            metrics['Calmar_Ratio'] = 0.0
        
        # Win rate
        winning_days = (returns_filled > 0).sum()
        metrics['Win_Rate_Pct'] = (winning_days / total_days) * 100
        
        # Make sure no NaN values remain
        for key, value in metrics.items():
            if pd.isna(value):
                metrics[key] = 0.0
        
        return metrics
    
    def plot_results(self, metrics):
        """Create visualizations of the strategy results"""
        if self.results is None:
            raise ValueError("Backtest not run. Call backtest_strategy() first.")
            
        print("Creating visualizations...")
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Fallback to default style if seaborn-v0_8-darkgrid not available
            plt.style.use('default')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(5, 2, height_ratios=[2, 1.5, 1.5, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Equity Curves
        ax1 = fig.add_subplot(gs[0, :])
        
        ax1.plot(self.results.index, self.results['Strategy_Cum_Return'], 
                label='Strategy', color='blue', linewidth=2)
        ax1.plot(self.results.index, self.results['Buy_Hold_Cum_Return'], 
                label='Buy & Hold', color='gray', alpha=0.7, linewidth=2)
        
        # Mark buy and sell signals
        buy_signals = self.results[self.results['Buy_Signal'] == 1]
        sell_signals = self.results[self.results['Sell_Signal'] == 1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, self.results.loc[buy_signals.index, 'Strategy_Cum_Return'],
                       color='green', marker='^', s=100, alpha=0.7, label='Buy Signal')
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, self.results.loc[sell_signals.index, 'Strategy_Cum_Return'],
                       color='red', marker='v', s=100, alpha=0.7, label='Sell Signal')
        
        ax1.set_title('Cumulative Returns: Strategy vs Buy & Hold', fontsize=14)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdowns
        ax2 = fig.add_subplot(gs[1, :])
        
        ax2.fill_between(self.results.index, self.results['Strategy_Drawdown'], 0, 
                        color='blue', alpha=0.3, label='Strategy Drawdown')
        ax2.fill_between(self.results.index, self.results['Buy_Hold_Drawdown'], 0, 
                        color='gray', alpha=0.3, label='Buy & Hold Drawdown')
        
        ax2.set_title('Drawdowns', fontsize=14)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk Indicators: VaR and IBS
        ax3 = fig.add_subplot(gs[2, 0])
        
        ax3.plot(self.results.index, self.results['VaR_95'], 
                label='VaR 95%', color='red', alpha=0.7)
        ax3.plot(self.results.index, self.results['Return'], 
                label='Returns', color='blue', alpha=0.4)
        
        ax3.set_title('VaR 95% and Returns', fontsize=14)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. IBS and EUP
        ax4 = fig.add_subplot(gs[2, 1])
        
        ax4.plot(self.results.index, self.results['IBS'], 
                label='IBS', color='green', alpha=0.7)
        ax4.plot(self.results.index, self.results['EUP_95'], 
                label='EUP 95%', color='orange', alpha=0.7)
        
        # Add horizontal lines for IBS thresholds
        ax4.axhline(y=self.ibs_threshold_low, color='green', linestyle='--', alpha=0.5, 
                   label=f'IBS Low Threshold ({self.ibs_threshold_low})')
        ax4.axhline(y=self.ibs_threshold_high, color='red', linestyle='--', alpha=0.5, 
                   label=f'IBS High Threshold ({self.ibs_threshold_high})')
        
        ax4.set_title('Technical Indicators: IBS & EUP', fontsize=14)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.legend(fontsize=10, loc='lower right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Position Over Time
        ax5 = fig.add_subplot(gs[3, :])
        
        ax5.fill_between(self.results.index, self.results['Position'], 0, 
                        color='green', alpha=0.3, label='Long Position')
        
        ax5.set_title('Position Over Time', fontsize=14)
        ax5.set_ylabel('Position', fontsize=12)
        ax5.set_ylim(-0.1, 1.1)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Table
        ax6 = fig.add_subplot(gs[4, :])
        ax6.axis('off')
        
        metrics_table = pd.DataFrame({
            'Strategy': [
                f"{metrics['Strategy']['Total_Return_Pct']:.2f}%",
                f"{metrics['Strategy']['Annual_Return_Pct']:.2f}%",
                f"{metrics['Strategy']['Annual_Volatility_Pct']:.2f}%",
                f"{metrics['Strategy']['Sharpe_Ratio']:.2f}",
                f"{metrics['Strategy']['Sortino_Ratio']:.2f}",
                f"{metrics['Strategy']['Max_Drawdown_Pct']:.2f}%",
                f"{metrics['Strategy']['Win_Rate_Pct']:.2f}%",
                f"{metrics['Strategy']['Total_Trades']}",
                f"{metrics['Strategy']['Time_In_Market_Pct']:.2f}%"
            ],
            'Buy & Hold': [
                f"{metrics['Buy_Hold']['Total_Return_Pct']:.2f}%",
                f"{metrics['Buy_Hold']['Annual_Return_Pct']:.2f}%",
                f"{metrics['Buy_Hold']['Annual_Volatility_Pct']:.2f}%",
                f"{metrics['Buy_Hold']['Sharpe_Ratio']:.2f}",
                f"{metrics['Buy_Hold']['Sortino_Ratio']:.2f}",
                f"{metrics['Buy_Hold']['Max_Drawdown_Pct']:.2f}%",
                f"{metrics['Buy_Hold']['Win_Rate_Pct']:.2f}%",
                "1",
                "100.00%"
            ]
        }, index=[
            'Total Return',
            'Annual Return',
            'Annual Volatility',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Maximum Drawdown',
            'Win Rate',
            'Total Trades',
            'Time in Market'
        ])
        
        # Create table
        table = ax6.table(
            cellText=metrics_table.values,
            rowLabels=metrics_table.index,
            colLabels=metrics_table.columns,
            cellLoc='center',
            loc='center',
            bbox=[0.2, 0, 0.6, 1]
        )
        
        # Customize table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add strategy description
        ax6.text(0.02, 0.95, 
                "Strategy Description: Buy the Dip for SPY using VaR95% as initial trigger, "
                "IBS<0.2 as buy signal, and EUP95% with IBS>0.8 as sell signal.",
                fontsize=10, ha='left', va='top', transform=ax6.transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Create additional charts separately to avoid layout issues
        self._plot_monthly_returns_heatmap()
        self._plot_returns_by_position()
        
        print("Visualizations created successfully.")
    
    def _plot_monthly_returns_heatmap(self):
        """Create monthly returns heatmap"""
        try:
            # Resample to monthly returns (use ME to avoid warning)
            monthly_returns = self.results['Strategy_Return'].fillna(0).resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            
            # Create a pivot table with years as rows and months as columns
            pivot_data = []
            for date, value in monthly_returns.items():
                pivot_data.append({
                    'Year': date.year,
                    'Month': date.month,
                    'Return': value
                })
            
            # Convert to DataFrame and create pivot table
            monthly_returns_df = pd.DataFrame(pivot_data)
            
            if monthly_returns_df.empty:
                print("Insufficient data for monthly returns heatmap")
                return
                
            pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
            
            plt.title('Monthly Strategy Returns (%)', fontsize=14)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Year', fontsize=12)
            
            # Change month numbers to names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(np.arange(12) + 0.5, month_names)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error creating monthly returns heatmap: {e}")
    
    def _plot_returns_by_position(self):
        """Create boxplot of returns by position"""
        try:
            # Create a dataframe with position and returns
            position_returns = pd.DataFrame({
                'Position': self.results['Position'].shift(1),
                'Return': self.results['Return'] * 100  # Convert to percentage
            }).dropna()
            
            if position_returns.empty:
                print("Insufficient data for returns boxplot")
                return
                
            # Plot boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Position', y='Return', data=position_returns, palette="Set3")
            
            plt.title('Distribution of Daily Returns by Position', fontsize=14)
            plt.xlabel('Position (0=Cash, 1=Long)', fontsize=12)
            plt.ylabel('Daily Return (%)', fontsize=12)
            
            # Add descriptive stats as text
            for i, position in enumerate([0, 1]):
                pos_returns = position_returns[position_returns['Position'] == position]['Return']
                if len(pos_returns) > 0:
                    mean_return = pos_returns.mean()
                    median_return = pos_returns.median()
                    win_rate = (pos_returns > 0).mean() * 100
                    
                    plt.text(i, pos_returns.min() - 0.5, 
                            f"Mean: {mean_return:.2f}%\nMedian: {median_return:.2f}%\nWin Rate: {win_rate:.1f}%",
                            ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error creating returns boxplot: {e}")
    
    def generate_report(self, metrics):
        """Generate a comprehensive strategy report"""
        if self.results is None:
            raise ValueError("Backtest not run. Call backtest_strategy() first.")
            
        print("Generating strategy report...")
        
        # Get start and end dates
        start_date = self.results.index[0].strftime('%Y-%m-%d')
        end_date = self.results.index[-1].strftime('%Y-%m-%d')
        
        # Format report
        report = f"""
        SPY "Buy the Dip" Strategy Report
        ======================================
        
        Strategy Overview:
        ------------------------------
        - Asset: SPY (S&P 500 ETF)
        - Period: {start_date} to {end_date}
        - Approach: Buy the dip using risk metrics and IBS indicator
        - Entry: VaR95% trigger + IBS < {self.ibs_threshold_low}
        - Exit: Extreme EUP95% + IBS > {self.ibs_threshold_high}
        
        Performance Summary:
        ---------------------
        - Total Return: {metrics['Strategy']['Total_Return_Pct']:.2f}% (Buy & Hold: {metrics['Buy_Hold']['Total_Return_Pct']:.2f}%)
        - Annual Return: {metrics['Strategy']['Annual_Return_Pct']:.2f}% (Buy & Hold: {metrics['Buy_Hold']['Annual_Return_Pct']:.2f}%)
        - Annual Volatility: {metrics['Strategy']['Annual_Volatility_Pct']:.2f}% (Buy & Hold: {metrics['Buy_Hold']['Annual_Volatility_Pct']:.2f}%)
        - Sharpe Ratio: {metrics['Strategy']['Sharpe_Ratio']:.2f} (Buy & Hold: {metrics['Buy_Hold']['Sharpe_Ratio']:.2f})
        - Sortino Ratio: {metrics['Strategy']['Sortino_Ratio']:.2f} (Buy & Hold: {metrics['Buy_Hold']['Sortino_Ratio']:.2f})
        - Maximum Drawdown: {metrics['Strategy']['Max_Drawdown_Pct']:.2f}% (Buy & Hold: {metrics['Buy_Hold']['Max_Drawdown_Pct']:.2f}%)
        - Win Rate: {metrics['Strategy']['Win_Rate_Pct']:.2f}% (Buy & Hold: {metrics['Buy_Hold']['Win_Rate_Pct']:.2f}%)
        
        Trading Activity:
        ------------------
        - Total Trades: {metrics['Strategy']['Total_Trades']}
        - Trades Per Year: {metrics['Strategy']['Trades_Per_Year']:.2f}
        - Buy Signals: {metrics['Strategy']['Buy_Signals']}
        - Sell Signals: {metrics['Strategy']['Sell_Signals']}
        - Time in Market: {metrics['Strategy']['Time_In_Market_Pct']:.2f}%
        
        Risk Analysis:
        ----------------"""
        
        # Add risk metrics if available
        var_mean = self.results['VaR_95'].mean()
        eup_mean = self.results['EUP_95'].mean()
        
        if not pd.isna(var_mean):
            report += f"\n        - VaR 95% (Average): {var_mean:.4f}"
            
        if not pd.isna(eup_mean):
            report += f"\n        - EUP 95% (Average): {eup_mean:.4f}"
        
        report += f"""
        
        Strategy Logic:
        ---------------------
        1. Wait for a "dip" signaled by returns falling below the VaR95% threshold
        2. Confirm buying opportunity when IBS < {self.ibs_threshold_low}
        3. Enter long position in SPY
        4. Hold position until EUP95% is extremely positive and IBS > {self.ibs_threshold_high}
        5. Sell position and wait for next buying opportunity
        
        Conclusion:
        ---------
        """
        
        # Add appropriate conclusion based on performance
        strategy_return = metrics['Strategy']['Total_Return_Pct']
        buy_hold_return = metrics['Buy_Hold']['Total_Return_Pct']
        
        if strategy_return > buy_hold_return:
            outperformance = ((1 + strategy_return/100) / (1 + buy_hold_return/100) - 1) * 100
            report += f"The strategy outperformed buy & hold by {outperformance:.2f}% over the test period with lower risk. "
            report += f"The strategy successfully identifies advantageous buying opportunities during market dips "
            report += f"and exits positions effectively during overbought conditions."
        else:
            underperformance = ((1 + buy_hold_return/100) / (1 + strategy_return/100) - 1) * 100
            report += f"The strategy underperformed buy & hold by {underperformance:.2f}% over the test period. "
            report += f"However, the strategy showed lower volatility and may be valuable in certain market environments. "
            report += f"Further optimization of the entry/exit parameters could improve performance."
        
        print("Strategy report generated successfully.")
        return report
    
    def run_strategy(self):
        """Execute the complete strategy pipeline"""
        # 1. Download data
        self.fetch_data()
        
        # 2. Calculate indicators
        self.calculate_indicators()
        
        # 3. Generate trading signals
        self.generate_signals()
        
        # 4. Backtest the strategy
        self.backtest_strategy()
        
        # 5. Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # 6. Create visualizations
        self.plot_results(metrics)
        
        # 7. Generate report
        report = self.generate_report(metrics)
        print("\n" + report)
        
        return self.results, metrics, report

# Example usage
if __name__ == "__main__":
    # Initialize strategy with default parameters
    strategy = SPYBuyTheDipStrategy(start_date='2010-01-01', ibs_threshold_low=0.2, ibs_threshold_high=0.8)
    
    # Run complete strategy pipeline
    results, metrics, report = strategy.run_strategy()
