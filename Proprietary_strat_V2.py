import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

def fetch_data(start_date='2017-01-01', end_date='2025-04-12'):
    """Descarga datos incluyendo OHLC para calcular Yang-Zhang"""
    print("Descargando datos...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    data = pd.DataFrame(index=spy.index)
    data['SPY_Open'] = spy['Open']
    data['SPY_High'] = spy['High']
    data['SPY_Low'] = spy['Low']
    data['SPY_Close'] = spy['Close']

    other_tickers = ['XLK', 'XLY', 'XLI', 'XLP', 'XLU', 'XLV', '^VIX', '^VIX3M']

    for ticker in other_tickers:
        print(f"Descargando {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data[ticker] = df['Close']

    data = data.dropna()
    return data

def calculate_yang_zhang_volatility(data, window=10):
    """
    Calcula la volatilidad de Yang-Zhang que combina:
    - Volatilidad overnight (close-to-open)
    - Volatilidad intraday (open-to-close)
    - Volatilidad Rogers-Satchell
    """
    # Asegurar que window es al menos 2 para evitar división por cero
    window = max(2, window)
    
    # Logaritmos para cálculos diarios
    log_ho = np.log(data['SPY_High'] / data['SPY_Open'])
    log_lo = np.log(data['SPY_Low'] / data['SPY_Open'])
    log_oc = np.log(data['SPY_Close'] / data['SPY_Open'])
    log_co = np.log(data['SPY_Open'] / data['SPY_Close'].shift(1))
    
    # Volatilidad del rango
    rs = log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)
    open_vol = (log_co**2).rolling(window=window).mean()
    close_vol = (log_oc**2).rolling(window=window).mean()
    window_rs = rs.rolling(window=window).mean()
    
    # Este es el factor k de la fórmula de Yang-Zhang
    k = 0.34 / (1.34 + (window + 1)/(window - 1))
    yz_vol = np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs)
    
    # Anualizar la volatilidad (multiplicar por sqrt(252))
    yz_vol_annualized = yz_vol * np.sqrt(252)
    
    return yz_vol_annualized

def calculate_performance_metrics(returns, risk_free_rate=0.02):
    """Calcula métricas de rendimiento para una serie de retornos"""
    rf_daily = (1 + risk_free_rate) ** (1/252) - 1
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1/years) - 1
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    excess_returns = returns - rf_daily
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def calculate_oscillator(df):
    try:
        # Calcular volatilidad Yang-Zhang
        yz_vol = calculate_yang_zhang_volatility(df)
        
        # Normalizar Yang-Zhang
        yz_vol_normalized = yz_vol / yz_vol.rolling(window=252).mean()
        
        # Identificar entornos de volatilidad extremadamente baja
        ultra_low_vol = yz_vol < yz_vol.rolling(window=252).quantile(0.15)
        
        # Calcular componentes técnicos
        spy_ema8 = df['SPY_Close'].ewm(span=8, adjust=False).mean()
        spy_ema24 = df['SPY_Close'].ewm(span=24, adjust=False).mean()
        momentum = df['SPY_Close'].pct_change(21)
        cyclical = df[['XLK', 'XLY', 'XLI']].mean(axis=1)
        defensive = df[['XLP', 'XLU', 'XLV']].mean(axis=1)
        cycl_def_ratio = cyclical / defensive
        vix_ratio = df['^VIX'] / df['^VIX3M']

        # Oscillator simplificado
        oscillator = ((spy_ema8 - spy_ema24) / spy_ema24) * momentum + (cycl_def_ratio - vix_ratio)

        # Medias móviles
        ma_rapida = oscillator.ewm(span=5, adjust=False).mean()
        ma_lenta = oscillator.ewm(span=21, adjust=False).mean()

        # Bandas de volatilidad menos restrictivas
        volatility = oscillator.rolling(window=21).std()
        upper_band = ma_lenta + 1.5 * volatility  # Reducido de 2.0
        lower_band = ma_lenta - 1.5 * volatility  # Reducido de 2.0

        # Señales de cruce
        prev_ma_rapida = ma_rapida.shift(1)
        prev_ma_lenta = ma_lenta.shift(1)

        # Señales de compra - MENOS RESTRICTIVAS
        cross_up = (ma_rapida > ma_lenta) & (prev_ma_rapida <= prev_ma_lenta)
        bounce_up = (ma_rapida > lower_band) & (prev_ma_rapida <= lower_band)
        
        # Filtro de volatilidad MENOS RESTRICTIVO (percentil 90 en lugar de 75)
        vol_filter = yz_vol < yz_vol.rolling(window=252).quantile(0.90)
        
        # Señal de compra adicional: retorno a la media en alta volatilidad
        mean_reversion = (momentum < momentum.rolling(window=63).quantile(0.15)) & (yz_vol > yz_vol.rolling(window=252).mean())
        
        buy_signals = (cross_up | bounce_up | mean_reversion).astype(int)

        # Señales de venta - Mantener protección en volatilidad extrema
        cross_down = (ma_rapida < ma_lenta) & (prev_ma_rapida >= prev_ma_lenta)
        bounce_down = (ma_rapida < upper_band) & (prev_ma_rapida >= upper_band)
        extreme_vol = yz_vol > yz_vol.rolling(window=252).quantile(0.95)  # Solo salir en volatilidad EXTREMA
        sell_signals = (cross_down | bounce_down | extreme_vol).astype(int)

        # Filtrar señales muy cercanas - Reducir a 7 días (era 10)
        min_days = 7
        for i in range(min_days, len(buy_signals)):
            if buy_signals.iloc[i] == 1:
                if buy_signals.iloc[i-min_days:i].sum() > 0:
                    buy_signals.iloc[i] = 0

        for i in range(min_days, len(sell_signals)):
            if sell_signals.iloc[i] == 1:
                if sell_signals.iloc[i-min_days:i].sum() > 0:
                    sell_signals.iloc[i] = 0

        results = pd.DataFrame({
            'close_price': df['SPY_Close'],
            'open_price': df['SPY_Open'],
            'yz_volatility': yz_vol,
            'yz_vol_norm': yz_vol_normalized,
            'ultra_low_vol': ultra_low_vol,  # Añadimos indicador de volatilidad ultra baja
            'oscillator': oscillator,
            'ma_rapida': ma_rapida,
            'ma_lenta': ma_lenta,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'buy_signal': buy_signals,
            'sell_signal': sell_signals
        })

        return results

    except Exception as e:
        print(f"Error en calculate_oscillator: {str(e)}")
        raise

def backtest_strategy_with_leverage(df, initial_capital=100000):
    """Realiza backtest con ejecución al siguiente día y apalancamiento variable"""
    # Preparar series de precios y retornos
    close_prices = df['close_price']
    open_prices = df['open_price']
    
    # Un dataframe para el seguimiento de señales y posiciones
    backtest_df = pd.DataFrame(index=df.index)
    backtest_df['Close'] = close_prices
    backtest_df['Open'] = open_prices
    backtest_df['buy_signal'] = df['buy_signal']
    backtest_df['sell_signal'] = df['sell_signal']
    backtest_df['ultra_low_vol'] = df['ultra_low_vol']
    
    # Variables para el seguimiento
    position = 0  # 0 = sin posición, 1 = normal, 3 = apalancado
    entry_price = 0
    leverage = 0
    strategy_returns = []
    equity = initial_capital
    equity_curve = [initial_capital]
    positions = []
    leverages = []
    trades = []
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        prev_date = df.index[i-1]
        
        # Precio de cierre del día anterior y precio de apertura actual
        prev_close = close_prices.iloc[i-1]
        current_open = open_prices.iloc[i]
        
        # Determinar el apalancamiento basado en la volatilidad
        ultra_low_volatility = df['ultra_low_vol'].iloc[i-1]
        
        # Procesar señales generadas al cierre del día anterior
        if position == 0 and df['buy_signal'].iloc[i-1] == 1:
            # Determinar apalancamiento
            if ultra_low_volatility:
                position = 3  # 3x leverage
                leverage = 3.0
            else:
                position = 1  # 1x normal
                leverage = 1.0
                
            entry_price = current_open
            backtest_df.loc[current_date, 'entry'] = current_open
            backtest_df.loc[current_date, 'leverage'] = leverage
            trades.append({
                'type': 'buy',
                'signal_date': prev_date,
                'execution_date': current_date,
                'price': current_open,
                'leverage': leverage
            })
        
        elif position > 0 and df['sell_signal'].iloc[i-1] == 1:
            # Salida al precio de apertura del día actual
            exit_price = current_open
            trade_return = (exit_price / entry_price) - 1
            leveraged_return = trade_return * leverage  # Aplicar apalancamiento al retorno
            
            backtest_df.loc[current_date, 'exit'] = exit_price
            trades.append({
                'type': 'sell',
                'signal_date': prev_date,
                'execution_date': current_date,
                'price': exit_price,
                'return': trade_return,
                'leveraged_return': leveraged_return,
                'leverage': leverage
            })
            position = 0
            leverage = 0
            entry_price = 0
        
        # Calcular retorno diario para la estrategia
        if i > 1:  # Empezamos desde el segundo día
            if position > 0:  # Si estamos en posición, calculamos retorno open-to-open
                if i < len(df) - 1:  # Si no es el último día
                    next_open = open_prices.iloc[i+1]
                    daily_return = (next_open / current_open) - 1
                    leveraged_daily_return = daily_return * leverage  # Aplicar apalancamiento
                else:
                    # Para el último día, usamos close-to-open si es necesario
                    daily_return = (close_prices.iloc[i] / current_open) - 1
                    leveraged_daily_return = daily_return * leverage  # Aplicar apalancamiento
            else:
                daily_return = 0  # Sin posición, sin retorno
                leveraged_daily_return = 0
            
            strategy_returns.append(leveraged_daily_return)
            positions.append(position)
            leverages.append(leverage)
            equity *= (1 + leveraged_daily_return)
            equity_curve.append(equity)
    
    # Convertir a Series
    if len(strategy_returns) > 0:
        strategy_returns = pd.Series(strategy_returns, index=df.index[2:])
        positions = pd.Series(positions, index=df.index[2:])
        leverages = pd.Series(leverages, index=df.index[2:])
        equity_curve = pd.Series(equity_curve, index=df.index[1:])
    else:
        # Manejar caso donde no hay suficientes datos
        strategy_returns = pd.Series(index=df.index[2:])
        positions = pd.Series(index=df.index[2:])
        leverages = pd.Series(index=df.index[2:])
        equity_curve = pd.Series(index=df.index[1:])
    
    # Calcular retornos buy & hold más realistas (open-to-open)
    bh_returns = []
    for i in range(1, len(open_prices) - 1):
        bh_return = (open_prices.iloc[i+1] / open_prices.iloc[i]) - 1
        bh_returns.append(bh_return)
    
    bh_returns = pd.Series(bh_returns, index=open_prices.index[1:-1])
    bh_equity = initial_capital * (1 + bh_returns).cumprod()
    
    return {
        'strategy_returns': strategy_returns,
        'bh_returns': bh_returns,
        'strategy_equity': equity_curve,
        'bh_equity': bh_equity,
        'positions': positions,
        'leverages': leverages,
        'trades': trades,
        'backtest_df': backtest_df
    }

def plot_oscillator_with_trades(df, backtest_df):
    """Plot oscillator con señales y ejecuciones reales"""
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 14))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # Subplot 1: SPY Price con entradas y salidas
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close_price'], label='SPY Close', color='blue', alpha=0.7)
    
    # Marcar las señales
    buy_signals = df[df['buy_signal'] == 1]
    sell_signals = df[df['sell_signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close_price'], marker='^', color='green', 
               alpha=0.7, s=100, label='Buy Signal (Close)')
    ax1.scatter(sell_signals.index, sell_signals['close_price'], marker='v', color='red', 
               alpha=0.7, s=100, label='Sell Signal (Close)')
    
    # Marcar las ejecuciones (entradas/salidas reales)
    entries = backtest_df.dropna(subset=['entry'])
    exits = backtest_df.dropna(subset=['exit'])
    
    if not entries.empty:
        # Diferenciar entradas normales vs apalancadas
        normal_entries = entries[entries['leverage'] == 1.0]
        leveraged_entries = entries[entries['leverage'] == 3.0]
        
        if not normal_entries.empty:
            ax1.scatter(normal_entries.index, normal_entries['entry'], marker='o', color='green', 
                       s=80, label='Entry 1x (Next Open)')
                       
        if not leveraged_entries.empty:
            ax1.scatter(leveraged_entries.index, leveraged_entries['entry'], marker='*', color='green', 
                       s=120, label='Entry 3x (Next Open)')
    
    if not exits.empty:
        ax1.scatter(exits.index, exits['exit'], marker='o', color='red', 
                   s=80, label='Exit (Next Open)')
    
    # Resaltar períodos de volatilidad ultra baja
    ultra_low_vol = df['ultra_low_vol']
    if ultra_low_vol.any():
        ax1.fill_between(df.index, df['close_price'].min(), df['close_price'].max(), 
                       where=ultra_low_vol, color='green', alpha=0.1, label='Ultra Low Volatility (3x)')
    
    ax1.set_title('SPY Price with Signals and Executions', pad=20)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Oscillator with MAs and bands
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df.index, df['oscillator'], label='Oscillator', color='gray', alpha=0.4)
    ax2.plot(df.index, df['ma_rapida'], label='Fast MA (5)', color='blue', linewidth=1.5)
    ax2.plot(df.index, df['ma_lenta'], label='Slow MA (21)', color='red', linewidth=1.5)
    ax2.plot(df.index, df['upper_band'], label='Upper Band', color='lightcoral', linestyle='--')
    ax2.plot(df.index, df['lower_band'], label='Lower Band', color='lightgreen', linestyle='--')

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Oscillator Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Yang-Zhang volatility
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(df.index, df['yz_volatility'], label='YZ Volatility', color='purple', linewidth=1.5)
    
    # Añadir umbrales de volatilidad
    vol_15pct = df['yz_volatility'].rolling(window=252).quantile(0.15)
    vol_95pct = df['yz_volatility'].rolling(window=252).quantile(0.95)
    
    ax3.plot(df.index, vol_15pct, label='15% Quantile (3x Leverage)', color='green', linestyle='--')
    ax3.plot(df.index, vol_95pct, label='95% Quantile (Exit)', color='red', linestyle='--')
    
    # Sombrear periodos de volatilidad extrema
    ax3.fill_between(df.index, 0, vol_15pct, where=df['yz_volatility'] < vol_15pct, 
                   color='green', alpha=0.2, label='Ultra Low Volatility')
    
    ax3.fill_between(df.index, vol_95pct, df['yz_volatility'].max(), where=df['yz_volatility'] > vol_95pct, 
                   color='red', alpha=0.2, label='Extreme Volatility')
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Annualized Volatility')
    ax3.set_title('Yang-Zhang Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_backtest_results(backtest_results, metrics_strategy, metrics_bh):
    """Visualiza los resultados del backtest"""
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(4, 2, height_ratios=[2, 1, 1, 1], width_ratios=[2, 1])

    # Subplot 1: Equity Curves (Log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(backtest_results['strategy_equity'],
             label='Strategy', color='blue', linewidth=1.5)
    ax1.semilogy(backtest_results['bh_equity'],
             label='Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
    ax1.set_title('Equity Curves (Log Scale)')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    strategy_dd = (backtest_results['strategy_equity'] /
                  backtest_results['strategy_equity'].expanding().max() - 1)
    bh_dd = (backtest_results['bh_equity'] /
             backtest_results['bh_equity'].expanding().max() - 1)

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
    ax3.set_title('Position Size (1=Normal, 3=Leveraged)')
    ax3.set_ylim(-0.1, 3.5)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Leverage over time
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(backtest_results['leverages'].index, backtest_results['leverages'], 
            color='orange', label='Leverage')
    ax4.set_title('Leverage Factor Over Time')
    ax4.set_ylim(-0.1, 3.5)
    ax4.set_ylabel('Leverage (x)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Subplot 5: Metrics Comparison
    ax5 = fig.add_subplot(gs[:, 1])
    metrics_comparison = pd.DataFrame({
        'Strategy': [
            f"{metrics_strategy['Total Return']:.2%}",
            f"{metrics_strategy['Annual Return']:.2%}",
            f"{metrics_strategy['Annual Volatility']:.2%}",
            f"{metrics_strategy['Sharpe Ratio']:.2f}",
            f"{metrics_strategy['Sortino Ratio']:.2f}",
            f"{metrics_strategy['Calmar Ratio']:.2f}",
            f"{metrics_strategy['Max Drawdown']:.2%}"
        ],
        'Buy & Hold': [
            f"{metrics_bh['Total Return']:.2%}",
            f"{metrics_bh['Annual Return']:.2%}",
            f"{metrics_bh['Annual Volatility']:.2%}",
            f"{metrics_bh['Sharpe Ratio']:.2f}",
            f"{metrics_bh['Sortino Ratio']:.2f}",
            f"{metrics_bh['Calmar Ratio']:.2f}",
            f"{metrics_bh['Max Drawdown']:.2%}"
        ]
    }, index=[
        'Total Return',
        'Annual Return',
        'Annual Vol',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Calmar Ratio',
        'Max Drawdown'
    ])

    ax5.axis('tight')
    ax5.axis('off')
    table = ax5.table(cellText=metrics_comparison.values,
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

def main():
    try:
        # Fetch data with OHLC prices
        market_data = fetch_data()
        print(f"Datos descargados. Shape: {market_data.shape}")

        # Calculate oscillator with Yang-Zhang volatility
        print("Calculando oscilador con volatilidad Yang-Zhang...")
        results = calculate_oscillator(market_data)
        print(f"Oscilador calculado. Shape: {results.shape}")

        # Realizar backtest con apalancamiento variable
        print("\nRealizando backtest con apalancamiento 3x en volatilidad baja...")
        backtest_results = backtest_strategy_with_leverage(results)

        # Calcular métricas de rendimiento
        strategy_returns = backtest_results['strategy_returns'].dropna()
        bh_returns = backtest_results['bh_returns'].dropna()
        
        if len(strategy_returns) > 0:
            strategy_metrics = calculate_performance_metrics(strategy_returns)
            bh_metrics = calculate_performance_metrics(bh_returns)

            # Mostrar gráficos con señales y ejecuciones
            print("Creando gráficos del oscilador con señales y ejecuciones...")
            fig_osc = plot_oscillator_with_trades(results, backtest_results['backtest_df'])
            plt.show()

            # Mostrar gráficos de backtest
            print("Creando gráficos de backtest...")
            fig_backtest = plot_backtest_results(backtest_results, strategy_metrics, bh_metrics)
            plt.show()

            # Analizar trades
            trades = backtest_results['trades']
            buy_trades = [t for t in trades if t['type'] == 'buy']
            sell_trades = [t for t in trades if t['type'] == 'sell']
            
            # Calcular métricas de apalancamiento
            leveraged_trades = [t for t in buy_trades if t['leverage'] == 3.0]
            normal_trades = [t for t in buy_trades if t['leverage'] == 1.0]
            pct_leveraged = len(leveraged_trades) / len(buy_trades) if len(buy_trades) > 0 else 0
            
            print(f"\nAnálisis de operaciones:")
            print(f"Número total de operaciones: {len(sell_trades)}")
            print(f"Operaciones normales (1x): {len(normal_trades)}")
            print(f"Operaciones apalancadas (3x): {len(leveraged_trades)} ({pct_leveraged:.1%} del total)")
            print(f"Número de señales de compra: {results['buy_signal'].sum()}")
            print(f"Número de señales de venta: {results['sell_signal'].sum()}")
            
            if len(sell_trades) > 0:
                # Analizar rendimientos de operaciones normales vs apalancadas
                normal_returns = []
                leveraged_returns = []
                
                for trade in sell_trades:
                    if 'return' in trade and 'leverage' in trade:
                        if trade['leverage'] == 3.0:
                            leveraged_returns.append(trade['return'])
                        else:
                            normal_returns.append(trade['return'])
                
                # Calcular win rate general
                all_returns = [t['return'] for t in sell_trades if 'return' in t]
                win_rate = sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0
                print(f"Win rate general: {win_rate:.2%}")
                print(f"Retorno promedio por operación (sin apalancamiento): {np.mean(all_returns):.2%}")
                
                if normal_returns:
                    normal_win_rate = sum(1 for r in normal_returns if r > 0) / len(normal_returns)
                    print(f"Win rate operaciones normales (1x): {normal_win_rate:.2%}")
                    print(f"Retorno promedio operaciones normales: {np.mean(normal_returns):.2%}")
                
                if leveraged_returns:
                    leveraged_win_rate = sum(1 for r in leveraged_returns if r > 0) / len(leveraged_returns)
                    print(f"Win rate operaciones apalancadas (3x): {leveraged_win_rate:.2%}")
                    print(f"Retorno promedio operaciones apalancadas: {np.mean(leveraged_returns):.2%}")
                    print(f"Retorno promedio apalancado (3x): {np.mean(leveraged_returns)*3:.2%}")

            # Imprimir métricas principales
            print("\nMétricas de la estrategia (con apalancamiento variable):")
            print(f"Retorno total: {strategy_metrics['Total Return']:.2%}")
            print(f"Retorno anual: {strategy_metrics['Annual Return']:.2%}")
            print(f"Volatilidad anual: {strategy_metrics['Annual Volatility']:.2%}")
            print(f"Ratio de Sharpe: {strategy_metrics['Sharpe Ratio']:.2f}")
            print(f"Ratio de Sortino: {strategy_metrics['Sortino Ratio']:.2f}")
            print(f"Máximo drawdown: {strategy_metrics['Max Drawdown']:.2%}")
            
            return results, backtest_results, strategy_metrics, bh_metrics
        else:
            print("No hay suficientes datos para calcular métricas de rendimiento.")
            return results, backtest_results, None, None

    except Exception as e:
        print(f"Error en main: {str(e)}")
        raise

if __name__ == "__main__":
    results, backtest_results, strategy_metrics, bh_metrics = main()
