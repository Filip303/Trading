import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from datetime import datetime, timedelta

class NVDAMomentumVolatilityQualityStrategy:
    """
    NVIDIA-specific trading strategy combining:
    1. Momentum (price trend, relative strength)
    2. Volatility management (Yang-Zhang, VaR/CVaR)
    3. Quality metrics (growth, profitability)
    """
    
    def __init__(self, ticker="NVDA", start_date='2018-01-01'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.sector_data = None
        self.results = None
    
    def fetch_data(self):
        """Fetch historical price data for NVDA and benchmark indices"""
        print(f"Downloading data for {self.ticker}...")
        
        # Download NVDA data
        nvda = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        data = pd.DataFrame()
        
        # Save OHLC data
        data['Open'] = nvda['Open']
        data['High'] = nvda['High']
        data['Low'] = nvda['Low']
        data['Close'] = nvda['Close']
        data['Volume'] = nvda['Volume']
        
        # Download benchmark data (SPY, QQQ, SMH)
        print("Downloading benchmark data...")
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)['Close']
        qqq = yf.download('QQQ', start=self.start_date, end=self.end_date, progress=False)['Close']
        smh = yf.download('SMH', start=self.start_date, end=self.end_date, progress=False)['Close']
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)['Close']
        
        # Add to main dataframe
        data['SPY'] = spy
        data['QQQ'] = qqq
        data['SMH'] = smh
        data['VIX'] = vix
        
        # Drop rows with missing values
        data = data.dropna()
        print(f"Data downloaded. Shape: {data.shape}")
        
        self.data = data
        return data
    
    def get_sector_data(self):
        """Fetch data for semiconductor sector peers for relative strength"""
        tickers = ['AMD', 'INTC', 'TSM', 'MU', 'AVGO', 'QCOM', 'MRVL', 'AMAT', 'ASML']
        sector_data = pd.DataFrame()
        
        print("Downloading semiconductor sector data...")
        for ticker in tickers:
            try:
                price = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)['Close']
                sector_data[ticker] = price
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        # Add NVDA to the sector data
        sector_data[self.ticker] = self.data['Close']
        
        # Drop rows with missing values
        sector_data = sector_data.dropna()
        print(f"Sector data downloaded. Shape: {sector_data.shape}")
        
        self.sector_data = sector_data
        return sector_data
    
    def get_financial_metrics(self):
        """Fetch financial metrics using yfinance"""
        print("Fetching financial metrics...")
        
        try:
            # Get NVDA info
            nvda_info = yf.Ticker(self.ticker)
            
            # Get quarterly financials
            financials = nvda_info.quarterly_financials
            balance_sheet = nvda_info.quarterly_balance_sheet
            income_stmt = nvda_info.quarterly_income_stmt
            cashflow = nvda_info.quarterly_cashflow
            
            # Extract relevant metrics
            metrics = {}
            
            # Basic company info
            info = nvda_info.info
            metrics['gross_margin'] = info.get('grossMargins', 0) * 100
            metrics['operating_margin'] = info.get('operatingMargins', 0) * 100
            metrics['profit_margin'] = info.get('profitMargins', 0) * 100
            metrics['roe'] = info.get('returnOnEquity', 0) * 100
            
            # Calculate trailing 12 months values
            if len(income_stmt.columns) >= 4:
                # Revenue and net income
                latest_revenue = sum(income_stmt.loc['Total Revenue', income_stmt.columns[:4]])
                prev_year_revenue = sum(income_stmt.loc['Total Revenue', income_stmt.columns[4:8]]) if len(income_stmt.columns) >= 8 else None
                
                if prev_year_revenue and prev_year_revenue > 0:
                    metrics['revenue_growth'] = (latest_revenue / prev_year_revenue - 1) * 100
                else:
                    metrics['revenue_growth'] = None
            
            print("Financial metrics fetched successfully.")
            return metrics
            
        except Exception as e:
            print(f"Error fetching financial metrics: {e}")
            return {}
    
    def calculate_yang_zhang_volatility(self, window=21):
        """Calculate Yang-Zhang volatility"""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        open_price = self.data['Open']
        high_price = self.data['High']
        low_price = self.data['Low']
        close_price = self.data['Close']
        
        # Calculate logarithmic returns
        close_to_close = np.log(close_price / close_price.shift(1))
        overnight_jump = np.log(open_price / close_price.shift(1))
        intraday_return = np.log(close_price / open_price)
        
        # Calculate Rogers-Satchell volatility
        rogers_satchell = (np.log(high_price / close_price) * np.log(high_price / open_price) +
                          np.log(low_price / close_price) * np.log(low_price / open_price))
        
        # Calculate the three components of Yang-Zhang volatility
        overnight_vol = overnight_jump.rolling(window=window).var()
        open_close_vol = intraday_return.rolling(window=window).var()
        rs_vol = rogers_satchell.rolling(window=window).mean()
        
        # Combine the components with suggested weights
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yang_zhang_vol = np.sqrt(overnight_vol + k * open_close_vol + (1 - k) * rs_vol)
        
        return yang_zhang_vol * np.sqrt(252)  # Annualized
    
    def calculate_var_cvar(self, returns, confidence_level=0.95, window=21):
        """Calculate Value-at-Risk and Conditional Value-at-Risk"""
        alpha = 1 - confidence_level
        
        var = returns.rolling(window=window).quantile(alpha)
        
        def calc_cvar(x):
            try:
                x = pd.Series(x.flatten()) if hasattr(x, 'flatten') else pd.Series(x)
                return x[x <= np.percentile(x, alpha * 100)].mean()
            except Exception:
                return np.nan
        
        cvar = returns.rolling(window=window).apply(calc_cvar)
        
        return var, cvar
    
    def calculate_relative_strength(self):
        """Calculate relative strength of NVDA vs peers and indices"""
        if self.data is None or self.sector_data is None:
            raise ValueError("Data not loaded. Call fetch_data() and get_sector_data() first.")
        
        # Calculate returns
        nvda_returns = self.data['Close'].pct_change(21)  # 1-month return
        spy_returns = self.data['SPY'].pct_change(21)
        qqq_returns = self.data['QQQ'].pct_change(21)
        smh_returns = self.data['SMH'].pct_change(21)
        
        # Calculate relative strength
        rs_vs_spy = nvda_returns - spy_returns
        rs_vs_qqq = nvda_returns - qqq_returns
        rs_vs_smh = nvda_returns - smh_returns
        
        # Calculate sector relative strength
        sector_returns = self.sector_data.pct_change(21)
        nvda_sector_rank = sector_returns.rank(axis=1, ascending=False)[self.ticker]
        
        # Normalize rank to 0-1 scale (1 is best)
        normalized_rank = 1 - (nvda_sector_rank - 1) / (len(self.sector_data.columns) - 1)
        
        return {
            'rs_vs_spy': rs_vs_spy,
            'rs_vs_qqq': rs_vs_qqq, 
            'rs_vs_smh': rs_vs_smh,
            'sector_rank': nvda_sector_rank,
            'normalized_rank': normalized_rank
        }
    
    def generate_alpha_score(self, rs_data):
        """Generate a consolidated alpha score from different data sources"""
        # Get price momentum scores
        close_prices = self.data['Close']
        
        # Price momentum (1, 3, 6 month)
        momentum_1m = close_prices.pct_change(21)
        momentum_3m = close_prices.pct_change(63)
        momentum_6m = close_prices.pct_change(126)
        
        # Check if moving averages are in uptrend
        ma50 = close_prices.rolling(50).mean()
        ma200 = close_prices.rolling(200).mean()
        price_above_ma50 = (close_prices > ma50).astype(int)
        price_above_ma200 = (close_prices > ma200).astype(int)
        ma50_above_ma200 = (ma50 > ma200).astype(int)
        
        # Combine momentum signals (with more weight to recent periods)
        momentum_score = (
            0.5 * momentum_1m.rolling(window=5).mean() +
            0.3 * momentum_3m.rolling(window=5).mean() +
            0.2 * momentum_6m.rolling(window=5).mean()
        )
        
        # Handle all-NaN slices by replacing with 0
        momentum_rolling_min = momentum_score.rolling(252).min()
        momentum_rolling_max = momentum_score.rolling(252).max()
        
        # Avoid division by zero
        momentum_range = momentum_rolling_max - momentum_rolling_min
        
        # Normalize momentum score (handle cases where range is very small)
        normalized_momentum = pd.Series(index=momentum_score.index, dtype=float)
        for i in range(len(momentum_score)):
            if i < 252:  # Not enough data for rolling window
                normalized_momentum.iloc[i] = 0.5  # Neutral value
            else:
                score = momentum_score.iloc[i]
                min_val = momentum_rolling_min.iloc[i]
                max_val = momentum_rolling_max.iloc[i]
                
                if pd.isna(score) or pd.isna(min_val) or pd.isna(max_val):
                    normalized_momentum.iloc[i] = 0.5  # Neutral value
                elif abs(max_val - min_val) < 1e-6:  # Very small range
                    normalized_momentum.iloc[i] = 0.5  # Neutral value
                else:
                    normalized_momentum.iloc[i] = (score - min_val) / (max_val - min_val)
        
        # Relative strength score (weighted average)
        rs_score = (
            0.4 * rs_data['rs_vs_smh'] +
            0.3 * rs_data['rs_vs_qqq'] +
            0.2 * rs_data['rs_vs_spy'] +
            0.1 * rs_data['normalized_rank']
        )
        
        # Normalize relative strength (using the same approach)
        rs_rolling_min = rs_score.rolling(252).min()
        rs_rolling_max = rs_score.rolling(252).max()
        
        normalized_rs = pd.Series(index=rs_score.index, dtype=float)
        for i in range(len(rs_score)):
            if i < 252:  # Not enough data for rolling window
                normalized_rs.iloc[i] = 0.5  # Neutral value
            else:
                score = rs_score.iloc[i]
                min_val = rs_rolling_min.iloc[i]
                max_val = rs_rolling_max.iloc[i]
                
                if pd.isna(score) or pd.isna(min_val) or pd.isna(max_val):
                    normalized_rs.iloc[i] = 0.5  # Neutral value
                elif abs(max_val - min_val) < 1e-6:  # Very small range
                    normalized_rs.iloc[i] = 0.5  # Neutral value
                else:
                    normalized_rs.iloc[i] = (score - min_val) / (max_val - min_val)
        
        # Moving average trend score
        trend_score = (
            0.4 * price_above_ma50 +
            0.3 * price_above_ma200 +
            0.3 * ma50_above_ma200
        )
        
        # Combine scores with weights
        alpha_score = (
            0.4 * normalized_momentum +
            0.4 * normalized_rs +
            0.2 * trend_score
        )
        
        # Scale to 0-100 for easier interpretation
        alpha_score_scaled = alpha_score * 100
        
        return {
            'momentum_score': normalized_momentum,
            'rs_score': normalized_rs,
            'trend_score': trend_score,
            'alpha_score': alpha_score_scaled,
            'ma50': ma50,
            'ma200': ma200
        }
    
    def calculate_risk_score(self):
        """Calculate risk score using volatility metrics"""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        # Calculate returns
        returns = self.data['Close'].pct_change()
        
        # Calculate volatility measures
        daily_vol = returns.rolling(window=21).std() * np.sqrt(252)
        yang_zhang_vol = self.calculate_yang_zhang_volatility(window=21)
        var_95, cvar_95 = self.calculate_var_cvar(returns, confidence_level=0.95, window=21)
        
        # VIX ratio (current VIX / 1-month average)
        vix_ratio = self.data['VIX'] / self.data['VIX'].rolling(window=21).mean()
        
        # Replace NaN values with neutral values
        daily_vol = daily_vol.fillna(daily_vol.mean())
        yang_zhang_vol = yang_zhang_vol.fillna(yang_zhang_vol.mean())
        cvar_95 = cvar_95.fillna(cvar_95.mean())
        vix_ratio = vix_ratio.fillna(1.0)  # Neutral value for VIX ratio
        
        # Calculate z-scores with error handling
        def safe_z_score(series, window=252):
            result = pd.Series(index=series.index, dtype=float)
            for i in range(len(series)):
                if i < window:
                    result.iloc[i] = 0  # Default to neutral
                else:
                    window_data = series.iloc[i-window:i]
                    mean = window_data.mean()
                    std = window_data.std()
                    
                    if std <= 1e-6:  # Very small standard deviation
                        result.iloc[i] = 0
                    else:
                        result.iloc[i] = (series.iloc[i] - mean) / std
            return result
        
        vol_z_score = safe_z_score(daily_vol)
        yz_z_score = safe_z_score(yang_zhang_vol)
        cvar_z_score = safe_z_score(cvar_95)
        vix_z_score = safe_z_score(vix_ratio)
        
        # Higher risk score means higher risk
        risk_score = (
            0.30 * vol_z_score +
            0.30 * yz_z_score +
            0.25 * cvar_z_score +
            0.15 * vix_z_score
        )
        
        # Scale risk score to 0-100
        risk_min = risk_score.rolling(252).min().fillna(risk_score.min())
        risk_max = risk_score.rolling(252).max().fillna(risk_score.max())
        risk_range = risk_max - risk_min
        
        normalized_risk = pd.Series(index=risk_score.index, dtype=float)
        for i in range(len(risk_score)):
            min_val = risk_min.iloc[i]
            max_val = risk_max.iloc[i]
            score = risk_score.iloc[i]
            
            if pd.isna(score) or pd.isna(min_val) or pd.isna(max_val) or abs(max_val - min_val) < 1e-6:
                normalized_risk.iloc[i] = 50  # Default to moderate risk
            else:
                normalized_risk.iloc[i] = ((score - min_val) / (max_val - min_val)) * 100
        
        return {
            'daily_vol': daily_vol,
            'yang_zhang_vol': yang_zhang_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'vix_ratio': vix_ratio,
            'risk_score': normalized_risk
        }
    
    def calculate_position_size(self, alpha_score, risk_score):
        """Calculate optimal position size based on alpha and risk scores"""
        # Base position size based on alpha score
        base_position = alpha_score / 100
        
        # Adjust position based on risk
        risk_adjustment = 1 - (risk_score / 100) * 0.5  # Max 50% reduction due to risk
        
        # Calculate final position size (0 to 1)
        position_size = base_position * risk_adjustment
        
        # Ensure position_size is between 0 and 1
        position_size = position_size.clip(0, 1)
        
        # Define position bands manually instead of using pd.cut to avoid NaN issues
        position_bands = pd.Series(index=position_size.index, dtype=float)
        
        for i in range(len(position_size)):
            pos = position_size.iloc[i]
            
            if pd.isna(pos):
                position_bands.iloc[i] = 0  # Default to no position for NaN
            elif pos < 0.2:
                position_bands.iloc[i] = 0
            elif pos < 0.4:
                position_bands.iloc[i] = 0.25
            elif pos < 0.6:
                position_bands.iloc[i] = 0.5
            elif pos < 0.8:
                position_bands.iloc[i] = 0.75
            else:
                position_bands.iloc[i] = 1.0
        
        return position_size, position_bands
    
    def generate_signals(self, alpha_score, risk_score, position_size):
        """Generate trading signals based on alpha score, risk score, and position sizing"""
        # Ensure we have valid inputs
        alpha_score = alpha_score.fillna(50)  # Neutral value
        risk_score = risk_score.fillna(50)  # Neutral value
        
        # Calculate exponential moving averages of alpha score
        ema_fast = alpha_score.ewm(span=5, adjust=False).mean()
        ema_slow = alpha_score.ewm(span=21, adjust=False).mean()
        
        # Calculate volatility bands for the slow EMA
        volatility = alpha_score.rolling(window=21).std().fillna(5)  # Default volatility
        upper_band = ema_slow + 1.5 * volatility  # Less restrictive
        lower_band = ema_slow - 1.5 * volatility  # Less restrictive
        
        # Initial signal conditions - LESS RESTRICTIVE
        buy_signal = ((ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))) | \
                     ((ema_fast > lower_band) & (ema_fast.shift(1) <= lower_band)) | \
                     ((alpha_score > 60) & (alpha_score.shift(1) <= 60))  # New condition: Alpha score crosses above 60
        
        sell_signal = ((ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))) | \
                      ((ema_fast < upper_band) & (ema_fast.shift(1) >= upper_band)) | \
                      ((alpha_score < 40) & (alpha_score.shift(1) >= 40))  # New condition: Alpha score crosses below 40
        
        # Risk-based exit signal (exit when risk is extremely high)
        risk_exit = risk_score > 75  # Less restrictive
        
        # Combine signals
        final_buy = buy_signal & ~risk_exit
        final_sell = sell_signal | risk_exit
        
        # Avoid signals too close together (min 5 days apart - LESS RESTRICTIVE)
        min_days = 5
        buy_signals = final_buy.astype(int).copy()
        sell_signals = final_sell.astype(int).copy()
        
        for i in range(min_days, len(buy_signals)):
            if buy_signals.iloc[i] == 1:
                if buy_signals.iloc[i-min_days:i].sum() > 0:
                    buy_signals.iloc[i] = 0
            
            if sell_signals.iloc[i] == 1:
                if sell_signals.iloc[i-min_days:i].sum() > 0:
                    sell_signals.iloc[i] = 0
        
        return {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'buy_signal': buy_signals,
            'sell_signal': sell_signals,
            'risk_exit': risk_exit.astype(int)
        }
    
    def run_strategy(self):
        """Execute the full strategy pipeline"""
        # Fetch data
        self.fetch_data()
        self.get_sector_data()
        financial_metrics = self.get_financial_metrics()
        
        # Calculate relative strength
        rs_data = self.calculate_relative_strength()
        
        # Calculate alpha score
        alpha_data = self.generate_alpha_score(rs_data)
        
        # Calculate risk score
        risk_data = self.calculate_risk_score()
        
        # Calculate position size
        position_size, position_bands = self.calculate_position_size(
            alpha_data['alpha_score'], risk_data['risk_score']
        )
        
        # Generate signals
        signals = self.generate_signals(
            alpha_data['alpha_score'], risk_data['risk_score'], position_size
        )
        
        # Combine all results into a single DataFrame
        results = pd.DataFrame({
            'price': self.data['Close'],
            'volume': self.data['Volume'],
            'alpha_score': alpha_data['alpha_score'],
            'risk_score': risk_data['risk_score'],
            'position_size': position_size,
            'position_bands': position_bands,
            'ema_fast': signals['ema_fast'],
            'ema_slow': signals['ema_slow'],
            'upper_band': signals['upper_band'],
            'lower_band': signals['lower_band'],
            'buy_signal': signals['buy_signal'],
            'sell_signal': signals['sell_signal'],
            'ma50': alpha_data['ma50'],
            'ma200': alpha_data['ma200']
        })
        
        # Fill NaN values
        results = results.fillna(method='ffill')
        
        self.results = results
        return results, financial_metrics
    
    def plot_strategy_dashboard(self, financial_metrics=None):
        """Create comprehensive visualization of the strategy"""
        if self.results is None:
            raise ValueError("Strategy not run. Call run_strategy() first.")
        
        results = self.results
        
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Chart 1: Price with Signals and Position Size
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(results.index, results['price'], label='NVDA Price', color='blue')
        ax1.plot(results.index, results['ma50'], label='50-day MA', color='orange', alpha=0.7)
        ax1.plot(results.index, results['ma200'], label='200-day MA', color='red', alpha=0.7)
        
        # Mark buy and sell signals
        buy_signals = results[results['buy_signal'] == 1]
        sell_signals = results[results['sell_signal'] == 1]
        
        ax1.scatter(buy_signals.index, buy_signals['price'], 
                   marker='^', color='green', s=100, label='Buy Signal')
        ax1.scatter(sell_signals.index, sell_signals['price'], 
                   marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot position bands (colored background)
        date_range = results.index
        for i in range(1, len(date_range)):
            pos = results['position_bands'].iloc[i]
            color_intensity = pos  # 0 to 1
            if not pd.isna(color_intensity):
                ax1.axvspan(date_range[i-1], date_range[i], 
                           alpha=0.1*color_intensity, color='green', lw=0)
        
        ax1.set_title('NVDA Price with Trading Signals and Position Size', fontsize=14)
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Alpha Score with Signals
        ax2 = fig.add_subplot(gs[1, 0:2])
        ax2.plot(results.index, results['alpha_score'], label='Alpha Score', color='blue', alpha=0.7)
        ax2.plot(results.index, results['ema_fast'], label='Fast EMA', color='green', linewidth=1.5)
        ax2.plot(results.index, results['ema_slow'], label='Slow EMA', color='red', linewidth=1.5)
        ax2.plot(results.index, results['upper_band'], label='Upper Band', color='gray', linestyle='--', alpha=0.7)
        ax2.plot(results.index, results['lower_band'], label='Lower Band', color='gray', linestyle='--', alpha=0.7)
        
        # Add zone coloring
        ax2.axhspan(0, 20, color='red', alpha=0.1, label='Very Weak')
        ax2.axhspan(20, 40, color='orange', alpha=0.1, label='Weak')
        ax2.axhspan(40, 60, color='yellow', alpha=0.1, label='Neutral')
        ax2.axhspan(60, 80, color='lightgreen', alpha=0.1, label='Strong')
        ax2.axhspan(80, 100, color='green', alpha=0.1, label='Very Strong')
        
        ax2.set_title('Alpha Score with Signal Indicators', fontsize=14)
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Risk Score
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(results.index, results['risk_score'], label='Risk Score', color='red')
        
        # Add zone coloring
        ax3.axhspan(0, 20, color='green', alpha=0.1, label='Very Low Risk')
        ax3.axhspan(20, 40, color='lightgreen', alpha=0.1, label='Low Risk')
        ax3.axhspan(40, 60, color='yellow', alpha=0.1, label='Moderate Risk')
        ax3.axhspan(60, 80, color='orange', alpha=0.1, label='High Risk')
        ax3.axhspan(80, 100, color='red', alpha=0.1, label='Extreme Risk')
        
        ax3.set_title('Risk Score', fontsize=14)
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        # Chart 4: Position Size
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(results.index, results['position_size'], label='Position Size', color='green')
        ax4.set_title('Position Size (% of Capital)', fontsize=14)
        ax4.set_ylabel('Position Size')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        
        # Chart 5: Financial Metrics (if available)
        ax5 = fig.add_subplot(gs[3, 0:2])
        ax5.axis('off')
        
        if financial_metrics:
            metrics_text = "NVDA Financial Metrics (Latest Quarter):\n"
            metrics_text += f"Revenue Growth (YoY): {financial_metrics.get('revenue_growth', 'N/A')}%\n"
            metrics_text += f"Return on Equity: {financial_metrics.get('roe', 'N/A'):.1f}%\n"
            metrics_text += f"Gross Margin: {financial_metrics.get('gross_margin', 'N/A'):.1f}%\n"
            metrics_text += f"Operating Margin: {financial_metrics.get('operating_margin', 'N/A'):.1f}%\n"
            metrics_text += f"Profit Margin: {financial_metrics.get('profit_margin', 'N/A'):.1f}%"
        else:
            metrics_text = "Financial metrics not available."
        
        # Strategy status
        latest_date = results.index[-1].strftime('%Y-%m-%d')
        latest_price = results['price'].iloc[-1]
        latest_alpha = results['alpha_score'].iloc[-1]
        latest_risk = results['risk_score'].iloc[-1]
        latest_position = results['position_bands'].iloc[-1]
        
        # Determine current status
        if latest_position > 0:
            status = f"BULLISH - {int(latest_position * 100)}% Position"
        else:
            status = "NEUTRAL/BEARISH - 0% Position"
        
        status_text = f"NVDA Strategy Status (as of {latest_date}):\n"
        status_text += f"Current Price: ${latest_price:.2f}\n"
        status_text += f"Alpha Score: {latest_alpha:.1f}/100\n"
        status_text += f"Risk Score: {latest_risk:.1f}/100\n"
        status_text += f"Signal: {status}"
        
        # Add metrics and status to plot
        ax5.text(0.01, 0.99, metrics_text, va='top', ha='left', fontsize=12, 
                transform=ax5.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        ax5.text(0.5, 0.99, status_text, va='top', ha='left', fontsize=12, 
                transform=ax5.transAxes, bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        return fig
    
    def calculate_performance_metrics(self, returns, risk_free_rate=0.02):
        """Calculate performance metrics for a return series"""
        # Ensure we have enough data
        if len(returns) < 2 or returns.isna().all():
            return {
                'Total Return': 0.0,
                'Annual Return': 0.0,
                'Annual Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Calmar Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0
            }
            
        # Fill NaN values with 0 (no position)
        returns = returns.fillna(0)
            
        # Daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        
        # Total and annualized return
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Risk metrics
        excess_returns = returns - rf_daily
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() > 0 else 1e-6
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_days = sum(returns > 0)
        active_days = sum(returns != 0)
        win_rate = win_days / active_days if active_days > 0 else 0
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate
        }

    def backtest_strategy(self, initial_capital=100000):
        """Backtest the strategy with historical data"""
        if self.results is None:
            raise ValueError("Strategy not run. Call run_strategy() first.")
        
        # Initialize backtest variables
        results = self.results.copy()
        daily_returns = results['price'].pct_change().fillna(0)
        strategy_returns = []
        equity = initial_capital
        equity_curve = [initial_capital]
        positions = []
        trades = []
        
        position = 0
        entry_price = 0
        
        # Run backtest
        for i in range(1, len(results)):
            current_date = results.index[i]
            current_price = results['price'].iloc[i]
            current_position_size = results['position_bands'].iloc[i]
            
            # Process buy signals
            if position == 0 and results['buy_signal'].iloc[i] == 1:
                position = current_position_size
                entry_price = current_price
                trades.append({
                    'type': 'buy',
                    'date': current_date,
                    'price': current_price,
                    'size': position
                })
            
            # Process sell signals
            elif position > 0 and results['sell_signal'].iloc[i] == 1:
                # Calculate trade return
                trade_return = (current_price / entry_price - 1) * position
                trades.append({
                    'type': 'sell',
                    'date': current_date,
                    'price': current_price,
                    'size': position,
                    'return': trade_return
                })
                position = 0
                entry_price = 0
            
            # Position adjustment (if we're in a position and size changes significantly)
            elif position > 0 and abs(current_position_size - position) > 0.1:
                old_position = position
                position = current_position_size
                trades.append({
                    'type': 'adjust',
                    'date': current_date,
                    'price': current_price,
                    'old_size': old_position,
                    'new_size': position
                })
            
            # Calculate daily return for strategy
            if position > 0:
                daily_return = daily_returns.iloc[i] * position
            else:
                daily_return = 0
            
            strategy_returns.append(daily_return)
            positions.append(position)
            
            # Update equity
            equity *= (1 + daily_return)
            equity_curve.append(equity)
        
        # Convert to Series
        strategy_returns = pd.Series(strategy_returns, index=results.index[1:])
        positions = pd.Series(positions, index=results.index[1:])
        equity_curve = pd.Series(equity_curve, index=results.index)
        
        # Calculate buy & hold returns for comparison
        buy_hold_equity = initial_capital * (1 + daily_returns.fillna(0)).cumprod()
        
        # Calculate performance metrics
        perf_metrics = self.calculate_performance_metrics(strategy_returns)
        bh_metrics = self.calculate_performance_metrics(daily_returns.iloc[1:])
        
        # Calculate win rate and other trade statistics
        closed_trades = [t for t in trades if t['type'] == 'sell']
        winning_trades = [t for t in closed_trades if t.get('return', 0) > 0]
        
        total_trades = len(closed_trades)
        winning_trades_count = len(winning_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        # Calculate average trade metrics
        if total_trades > 0:
            avg_trade_return = sum(t.get('return', 0) for t in closed_trades) / total_trades
            avg_winning_trade = sum(t.get('return', 0) for t in winning_trades) / winning_trades_count if winning_trades_count > 0 else 0
            avg_losing_trade = sum(t.get('return', 0) for t in closed_trades if t.get('return', 0) <= 0) / (total_trades - winning_trades_count) if total_trades > winning_trades_count else 0
        else:
            avg_trade_return = 0
            avg_winning_trade = 0
            avg_losing_trade = 0
        
        # Add trade statistics to metrics
        perf_metrics['Total Trades'] = total_trades
        perf_metrics['Winning Trades'] = winning_trades_count
        perf_metrics['Win Rate (Trades)'] = win_rate
        perf_metrics['Average Trade Return'] = avg_trade_return
        perf_metrics['Average Winning Trade'] = avg_winning_trade
        perf_metrics['Average Losing Trade'] = avg_losing_trade
        
        backtest_results = {
            'strategy_returns': strategy_returns,
            'daily_returns': daily_returns[1:],
            'strategy_equity': equity_curve,
            'buy_hold_equity': buy_hold_equity,
            'positions': positions,
            'trades': trades,
            'metrics': perf_metrics,
            'bh_metrics': bh_metrics
        }
        
        return backtest_results

    def plot_backtest_results(self, backtest_results):
        """Visualize backtest results"""
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Subplot 1: Equity Curves (Log scale)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(backtest_results['strategy_equity'], 
                 label='Strategy', color='blue', linewidth=1.5)
        ax1.semilogy(backtest_results['buy_hold_equity'], 
                 label='Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
        ax1.set_title('Equity Curves (Log Scale)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        strategy_dd = (backtest_results['strategy_equity'] / 
                      backtest_results['strategy_equity'].expanding().max() - 1)
        bh_dd = (backtest_results['buy_hold_equity'] / 
                 backtest_results['buy_hold_equity'].expanding().max() - 1)
        
        ax2.fill_between(strategy_dd.index, strategy_dd, 0, 
                        color='blue', alpha=0.3, label='Strategy DD')
        ax2.fill_between(bh_dd.index, bh_dd, 0, 
                        color='gray', alpha=0.3, label='Buy & Hold DD')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Subplot 3: Position over time
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.fill_between(backtest_results['positions'].index, 
                        backtest_results['positions'], 
                        color='lightblue', alpha=0.5)
        ax3.set_title('Position Size Over Time')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Metrics Comparison
        ax4 = fig.add_subplot(gs[:, 1])
        metrics_comparison = pd.DataFrame({
            'Strategy': [
                f"{backtest_results['metrics']['Total Return']:.2%}",
                f"{backtest_results['metrics']['Annual Return']:.2%}",
                f"{backtest_results['metrics']['Annual Volatility']:.2%}",
                f"{backtest_results['metrics']['Total Trades']}",
                f"{backtest_results['metrics']['Win Rate (Trades)']:.2%}",
                f"{backtest_results['metrics']['Sharpe Ratio']:.2f}",
                f"{backtest_results['metrics']['Sortino Ratio']:.2f}",
                f"{backtest_results['metrics']['Calmar Ratio']:.2f}",
                f"{backtest_results['metrics']['Max Drawdown']:.2%}",
                f"{backtest_results['metrics']['Average Trade Return']:.2%}"
            ],
            'Buy & Hold': [
                f"{backtest_results['bh_metrics']['Total Return']:.2%}",
                f"{backtest_results['bh_metrics']['Annual Return']:.2%}",
                f"{backtest_results['bh_metrics']['Annual Volatility']:.2%}",
                "1",  # Buy & hold is just one trade
                "100.00%" if backtest_results['bh_metrics']['Total Return'] > 0 else "0.00%",  # Win rate depends on return
                f"{backtest_results['bh_metrics']['Sharpe Ratio']:.2f}",
                f"{backtest_results['bh_metrics']['Sortino Ratio']:.2f}",
                f"{backtest_results['bh_metrics']['Calmar Ratio']:.2f}",
                f"{backtest_results['bh_metrics']['Max Drawdown']:.2%}",
                f"{backtest_results['bh_metrics']['Total Return']:.2%}"  # Average trade is the total return
            ]
        }, index=[
            'Total Return',
            'Annual Return',
            'Annual Volatility',
            'Total Trades',
            'Win Rate',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Max Drawdown',
            'Avg Trade Return'
        ])
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=metrics_comparison.values,
                         rowLabels=metrics_comparison.index,
                         colLabels=metrics_comparison.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.2, 0, 0.8, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        return fig

    def generate_performance_report(self, backtest_results=None):
        """Generate a comprehensive performance report"""
        if backtest_results is None:
            if self.results is None:
                raise ValueError("Strategy not run. Call run_strategy() first.")
            backtest_results = self.backtest_strategy()
        
        # Get current status
        results = self.results
        latest_date = results.index[-1].strftime('%Y-%m-%d')
        latest_price = results['price'].iloc[-1]
        latest_alpha = results['alpha_score'].iloc[-1]
        latest_risk = results['risk_score'].iloc[-1]
        latest_position = results['position_bands'].iloc[-1]
        
        # Determine current status
        if latest_position > 0:
            status = f"BULLISH - {int(latest_position * 100)}% Position"
        else:
            status = "NEUTRAL/BEARISH - 0% Position"
        
        metrics = backtest_results['metrics']
        bh_metrics = backtest_results['bh_metrics']
        
        # Trade analysis
        trades = backtest_results['trades']
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        # Generate report
        report = f"""
NVDA Momentum-Volatility-Quality Strategy Performance Report
===========================================================
Date: {latest_date}

Current Status:
--------------
Current Price: ${latest_price:.2f}
Alpha Score: {latest_alpha:.1f}/100
Risk Score: {latest_risk:.1f}/100
Position: {status}

Performance Metrics:
-------------------
Total Return: {metrics['Total Return']:.2%} (Buy & Hold: {bh_metrics['Total Return']:.2%})
Annual Return: {metrics['Annual Return']:.2%} (Buy & Hold: {bh_metrics['Annual Return']:.2%})
Annual Volatility: {metrics['Annual Volatility']:.2%} (Buy & Hold: {bh_metrics['Annual Volatility']:.2%})
Sharpe Ratio: {metrics['Sharpe Ratio']:.2f} (Buy & Hold: {bh_metrics['Sharpe Ratio']:.2f})
Sortino Ratio: {metrics['Sortino Ratio']:.2f} (Buy & Hold: {bh_metrics['Sortino Ratio']:.2f})
Maximum Drawdown: {metrics['Max Drawdown']:.2%} (Buy & Hold: {bh_metrics['Max Drawdown']:.2%})
Win Rate: {metrics['Win Rate (Trades)']:.2%}

Trading Activity:
---------------
Total Buy Signals: {len(buy_trades)}
Total Sell Signals: {len(sell_trades)}
Total Trades: {metrics.get('Total Trades', 0)}
Winning Trades: {metrics.get('Winning Trades', 0)}
Average Trade Return: {metrics.get('Average Trade Return', 0):.2%}
Average Winning Trade: {metrics.get('Average Winning Trade', 0):.2%}
Average Losing Trade: {metrics.get('Average Losing Trade', 0):.2%}
"""

        # Show recent trades
        if len(trades) > 0:
            report += "\nRecent Trades (Last 5):\n"
            report += "---------------------\n"
            
            for trade in trades[-5:]:
                if trade['type'] == 'buy':
                    report += f"BUY: {trade['date'].strftime('%Y-%m-%d')} @ ${trade['price']:.2f}, Size: {trade['size']:.2f}\n"
                elif trade['type'] == 'sell':
                    report += f"SELL: {trade['date'].strftime('%Y-%m-%d')} @ ${trade['price']:.2f}, Size: {trade['size']:.2f}, Return: {trade['return']:.2%}\n"
                elif trade['type'] == 'adjust':
                    report += f"ADJUST: {trade['date'].strftime('%Y-%m-%d')} @ ${trade['price']:.2f}, Size: {trade['old_size']:.2f} -> {trade['new_size']:.2f}\n"
        
        return report

    def main(self):
        """Main execution method"""
        # Run strategy
        print("Running NVDA Momentum-Volatility-Quality Strategy...")
        results, financial_metrics = self.run_strategy()
        
        # Generate strategy dashboard
        print("\nGenerating strategy dashboard...")
        dashboard = self.plot_strategy_dashboard(financial_metrics)
        plt.show()
        
        # Run backtest
        print("\nRunning strategy backtest...")
        backtest_results = self.backtest_strategy()
        
        # Plot backtest results
        print("\nGenerating backtest results visualization...")
        backtest_plot = self.plot_backtest_results(backtest_results)
        plt.show()
        
        # Generate performance report
        print("\nGenerating performance report...")
        report = self.generate_performance_report(backtest_results)
        print(report)
        
        return results, backtest_results

if __name__ == "__main__":
    # Initialize and run the strategy
    nvda_strategy = NVDAMomentumVolatilityQualityStrategy()
    results, backtest_results = nvda_strategy.main()
