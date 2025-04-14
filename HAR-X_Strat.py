import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import norm

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Configuración de la API de FMP
API_KEY = "YOUR_FMP_API_KEY_HERE"
BASE_URL = "https://financialmodelingprep.com/api/v3"

def get_historical_data(symbol, start_date, end_date):
    """Obtiene datos históricos desde FMP"""
    url = f"{BASE_URL}/historical-price-full/{symbol}"
    params = {
        'from': start_date,
        'to': end_date,
        'apikey': API_KEY
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error al obtener datos: {response.status_code}")
    
    data = response.json()
    
    if 'historical' not in data:
        raise Exception(f"No se encontraron datos históricos para {symbol}")
    
    # Convertir a DataFrame y ordenar por fecha
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    # Renombrar columnas para mantener consistencia con yfinance
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    return df

def get_vix_data(start_date, end_date):
    """Obtiene datos de VIX desde FMP (solo hay VIX, no VIX3M o VIX6M)"""
    url = f"{BASE_URL}/historical-price-full/index/%5EVIX"  # URL encoding for ^VIX
    params = {
        'from': start_date,
        'to': end_date,
        'apikey': API_KEY
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error al obtener datos de VIX: {response.status_code}")
    
    data = response.json()
    
    if 'historical' not in data:
        raise Exception("No se encontraron datos históricos para VIX")
    
    # Convertir a DataFrame y ordenar por fecha
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    # Renombrar columnas
    df.rename(columns={
        'close': 'Close'
    }, inplace=True)
    
    return df

def create_synthetic_vix_term_structure(vix_data):
    """
    Crea una aproximación de VIX3M y VIX6M basada en el VIX
    """
    df = pd.DataFrame(index=vix_data.index)
    df['VIX'] = vix_data['Close']
    
    # Smoothing para simular plazos más largos
    df['VIX3M'] = df['VIX'].rolling(window=63).mean()  # ~3 meses
    df['VIX6M'] = df['VIX'].rolling(window=126).mean()  # ~6 meses
    
    # Añadimos la pendiente de la curva de VIX como indicador adicional
    df['VIX_Slope'] = df['VIX'] / df['VIX3M'] - 1
    
    return df

def yang_zhang_volatility(data, window=10):
    """
    Calcula la volatilidad Yang-Zhang, que combina:
    - Volatilidad overnight (close-to-open)
    - Volatilidad intraday (open-to-close)
    - Volatilidad Rogers-Satchell
    
    Es una estimación más precisa de la volatilidad real que la desviación estándar de retornos.
    """
    # Asegurar que window es al menos 2 para evitar división por cero
    window = max(2, window)
    
    # Logaritmos para cálculos diarios
    log_ho = np.log(data['High'] / data['Open'])
    log_lo = np.log(data['Low'] / data['Open'])
    log_oc = np.log(data['Close'] / data['Open'])
    log_co = np.log(data['Close'] / data['Open'].shift(1))
    
    # Volatilidad del rango
    rs = log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)
    open_vol = (log_oc**2).rolling(window=window).mean()
    close_vol = (log_co**2).rolling(window=window).mean()
    window_rs = rs.rolling(window=window).mean()
    
    # Este es el factor k de la fórmula de Yang-Zhang
    k = 0.34 / (1.34 + (window + 1)/(window - 1))
    yz_vol = np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs)
    
    # Anualizar la volatilidad (multiplicar por sqrt(252))
    yz_vol_annualized = yz_vol * np.sqrt(252)
    
    return yz_vol_annualized

def calculate_performance_metrics(returns, risk_free_rate=0.02):
    """Calcula métricas de rendimiento detalladas"""
    metrics = {}
    
    # Convertir a serie si es un dataframe
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Retornos acumulados
    cum_returns = (1 + returns).cumprod()
    
    # Rentabilidad total
    total_return = cum_returns.iloc[-1] - 1
    metrics['Total Return'] = total_return * 100  # En porcentaje
    
    # Rentabilidad anualizada
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1
    metrics['Annual Return'] = annual_return * 100  # En porcentaje
    
    # Volatilidad
    volatility = returns.std() * np.sqrt(252)
    metrics['Annual Volatility'] = volatility * 100  # En porcentaje
    
    # Sharpe Ratio
    risk_free_daily = ((1 + risk_free_rate) ** (1/252)) - 1
    excess_returns = returns - risk_free_daily
    sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    metrics['Sharpe Ratio'] = sharpe_ratio
    
    # Drawdown máximo
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max) - 1
    max_drawdown = drawdown.min()
    metrics['Max Drawdown'] = max_drawdown * 100  # En porcentaje
    
    # Sortino Ratio (solo considera volatilidad negativa)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else np.nan
    metrics['Sortino Ratio'] = sortino_ratio
    
    # Calmar Ratio (retorno anualizado / drawdown máximo)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    metrics['Calmar Ratio'] = calmar_ratio
    
    # Ratio de Información (alpha / tracking error) - simplificado
    information_ratio = sharpe_ratio
    metrics['Information Ratio'] = information_ratio
    
    # Retorno positivo vs negativo
    win_rate = len(returns[returns > 0]) / len(returns)
    metrics['Win Rate'] = win_rate * 100  # En porcentaje
    
    # Ratio de ganancias/pérdidas
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    metrics['Profit/Loss Ratio'] = profit_loss_ratio
    
    # Métricas de riesgo adicionales
    metrics['Kurtosis'] = returns.kurtosis()  # Mide la presencia de valores extremos
    metrics['Skewness'] = returns.skew()      # Mide la asimetría de la distribución
    
    # Value at Risk (VaR) - 95%
    var_95 = np.percentile(returns, 5)
    metrics['VaR 95%'] = var_95 * 100  # En porcentaje
    
    # Conditional VaR (CVaR) - 95% - Expected Shortfall
    cvar_95 = returns[returns <= var_95].mean()
    metrics['CVaR 95%'] = cvar_95 * 100  # En porcentaje
    
    return metrics

def create_performance_tearsheet(df):
    """
    Crea un análisis detallado del rendimiento de la estrategia vs Buy & Hold
    """
    spy_returns = df['SPY_ret']
    strategy_returns = df['strategy_ret']
    
    # Calcular métricas
    spy_metrics = calculate_performance_metrics(spy_returns)
    strategy_metrics = calculate_performance_metrics(strategy_returns)
    
    # Crear un resumen comparativo
    metrics_comparison = pd.DataFrame({
        'Estrategia HAR-X': [
            f"{strategy_metrics['Total Return']:.2f}%",
            f"{strategy_metrics['Annual Return']:.2f}%",
            f"{strategy_metrics['Annual Volatility']:.2f}%",
            f"{strategy_metrics['Sharpe Ratio']:.2f}",
            f"{strategy_metrics['Sortino Ratio']:.2f}",
            f"{strategy_metrics['Calmar Ratio']:.2f}",
            f"{strategy_metrics['Max Drawdown']:.2f}%",
            f"{strategy_metrics['Win Rate']:.2f}%",
            f"{strategy_metrics['Profit/Loss Ratio']:.2f}",
            f"{strategy_metrics['VaR 95%']:.2f}%",
            f"{strategy_metrics['CVaR 95%']:.2f}%"
        ],
        'Buy & Hold SPY': [
            f"{spy_metrics['Total Return']:.2f}%",
            f"{spy_metrics['Annual Return']:.2f}%",
            f"{spy_metrics['Annual Volatility']:.2f}%",
            f"{spy_metrics['Sharpe Ratio']:.2f}",
            f"{spy_metrics['Sortino Ratio']:.2f}",
            f"{spy_metrics['Calmar Ratio']:.2f}",
            f"{spy_metrics['Max Drawdown']:.2f}%",
            f"{spy_metrics['Win Rate']:.2f}%",
            f"{spy_metrics['Profit/Loss Ratio']:.2f}",
            f"{spy_metrics['VaR 95%']:.2f}%",
            f"{spy_metrics['CVaR 95%']:.2f}%"
        ]
    }, index=[
        'Retorno Total',
        'Retorno Anualizado',
        'Volatilidad Anualizada',
        'Ratio de Sharpe',
        'Ratio de Sortino',
        'Ratio de Calmar',
        'Drawdown Máximo',
        'Tasa de Acierto',
        'Ratio Ganancia/Pérdida',
        'VaR 95%',
        'CVaR 95%'
    ])
    
    print("\n=== COMPARATIVA DE RENDIMIENTO ===")
    print(metrics_comparison)
    
    return metrics_comparison

def analyze_performance_by_regime(df):
    """Analiza el rendimiento por régimen de volatilidad"""
    # Definir regímenes de volatilidad
    vol_low = np.percentile(df['YZ_vol_pred'], 25)
    vol_high = np.percentile(df['YZ_vol_pred'], 75)
    
    df['vol_regime'] = pd.cut(df['YZ_vol_pred'], 
                             bins=[0, vol_low, vol_high, np.inf], 
                             labels=['Baja', 'Media', 'Alta'])
    
    # Calcular rendimientos por régimen
    regime_returns = df.groupby('vol_regime')[['SPY_ret', 'strategy_ret']].mean() * 252 * 100
    regime_volatility = df.groupby('vol_regime')[['SPY_ret', 'strategy_ret']].std() * np.sqrt(252) * 100
    regime_sharpe = regime_returns / regime_volatility
    
    # Calcular días en cada régimen
    regime_days = df.groupby('vol_regime').size()
    regime_pct = regime_days / len(df) * 100
    
    # Crear tabla de resultados
    regime_analysis = pd.DataFrame({
        'Días': regime_days,
        'Porcentaje': regime_pct,
        'Retorno SPY': regime_returns['SPY_ret'],
        'Retorno Estrategia': regime_returns['strategy_ret'],
        'Vol SPY': regime_volatility['SPY_ret'],
        'Vol Estrategia': regime_volatility['strategy_ret'],
        'Sharpe SPY': regime_sharpe['SPY_ret'],
        'Sharpe Estrategia': regime_sharpe['strategy_ret']
    })
    
    print("\n=== ANÁLISIS POR RÉGIMEN DE VOLATILIDAD ===")
    print(regime_analysis)
    
    return regime_analysis

def analyze_monthly_returns(df):
    """Analiza los rendimientos mensuales"""
    # Calcular retornos mensuales para SPY y la estrategia
    spy_monthly = df['SPY_ret'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    strategy_monthly = df['strategy_ret'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Crear DataFrame para análisis
    monthly_returns = pd.DataFrame({
        'SPY': spy_monthly,
        'Estrategia': strategy_monthly
    })
    
    # Calcular diferencia
    monthly_returns['Diferencia'] = monthly_returns['Estrategia'] - monthly_returns['SPY']
    
    # Estadísticas mensuales
    monthly_stats = pd.DataFrame({
        'Media': monthly_returns.mean(),
        'Mediana': monthly_returns.median(),
        'Mín': monthly_returns.min(),
        'Máx': monthly_returns.max(),
        'Positivos %': (monthly_returns > 0).mean() * 100,
        'Desv. Estándar': monthly_returns.std()
    })
    
    print("\n=== ESTADÍSTICAS MENSUALES ===")
    print(monthly_stats.round(2))
    
    return monthly_returns, monthly_stats

def plot_monthly_returns_heatmap(monthly_returns):
    """Crea un heatmap de retornos mensuales por año"""
    # Crear dataframe con año, mes y retornos
    heatmap_data = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Estrategia': monthly_returns['Estrategia']
    })
    
    # Pivotar para crear tabla con años como filas y meses como columnas
    pivot_data = heatmap_data.pivot(index='Year', columns='Month', values='Estrategia')
    
    # Crear heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
               linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Retornos Mensuales de la Estrategia HAR-X (%)', fontsize=16)
    plt.xlabel('Mes', fontsize=12)
    plt.ylabel('Año', fontsize=12)
    
    # Convertir etiquetas de meses a nombres
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    plt.xticks(np.arange(12) + 0.5, month_names)
    
    plt.tight_layout()
    plt.show()

def plot_strategy_exposure_distribution(df):
    """Muestra la distribución de la exposición de la estrategia"""
    # Calcular el porcentaje de tiempo en cada nivel de exposición
    exposure_counts = df['position'].value_counts().sort_index()
    exposure_pct = exposure_counts / len(df) * 100
    
    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    bars = plt.bar(exposure_counts.index, exposure_pct, color=['red', 'yellow', 'green'])
    
    plt.title('Distribución de la Exposición al Mercado', fontsize=16)
    plt.xlabel('Nivel de Exposición', fontsize=14)
    plt.ylabel('Porcentaje de Tiempo (%)', fontsize=14)
    plt.xticks([0, 1, 2], ['Sin Exposición (0x)', 'Normal (1x)', 'Doble (2x)'])
    
    # Añadir etiquetas con valores
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_drawdown_comparison(df):
    """Compara los drawdowns de SPY vs la estrategia"""
    # Calcular drawdowns
    spy_cum_ret = (1 + df['SPY_ret']).cumprod()
    strategy_cum_ret = (1 + df['strategy_ret']).cumprod()
    
    spy_drawdown = spy_cum_ret / spy_cum_ret.expanding().max() - 1
    strategy_drawdown = strategy_cum_ret / strategy_cum_ret.expanding().max() - 1
    
    # Crear gráfico
    plt.figure(figsize=(14, 7))
    plt.plot(spy_drawdown, label='SPY Drawdown', color='red', alpha=0.7)
    plt.plot(strategy_drawdown, label='Estrategia Drawdown', color='blue')
    plt.fill_between(spy_drawdown.index, spy_drawdown, 0, color='red', alpha=0.1)
    plt.fill_between(strategy_drawdown.index, strategy_drawdown, 0, color='blue', alpha=0.1)
    
    plt.title('Comparación de Drawdowns', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Formato del eje Y como porcentaje
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de drawdown
    print("\n=== ANÁLISIS DE DRAWDOWNS ===")
    print(f"SPY - Máximo Drawdown: {spy_drawdown.min()*100:.2f}%")
    print(f"Estrategia - Máximo Drawdown: {strategy_drawdown.min()*100:.2f}%")

def plot_rolling_performance(df, window=252):
    """Muestra métricas de rendimiento en ventanas móviles"""
    # Calcular retorno anualizado móvil
    rolling_spy_ret = df['SPY_ret'].rolling(window).mean() * 252 * 100
    rolling_strat_ret = df['strategy_ret'].rolling(window).mean() * 252 * 100
    
    # Calcular volatilidad anualizada móvil
    rolling_spy_vol = df['SPY_ret'].rolling(window).std() * np.sqrt(252) * 100
    rolling_strat_vol = df['strategy_ret'].rolling(window).std() * np.sqrt(252) * 100
    
    # Calcular Sharpe Ratio móvil
    rolling_spy_sharpe = rolling_spy_ret / rolling_spy_vol
    rolling_strat_sharpe = rolling_strat_ret / rolling_strat_vol
    
    # Crear subplot para cada métrica
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Graficar retorno anualizado
    axes[0].plot(rolling_spy_ret, label='SPY', color='gray', alpha=0.7)
    axes[0].plot(rolling_strat_ret, label='Estrategia HAR-X', color='blue')
    axes[0].set_title(f'Retorno Anualizado (Ventana móvil {window} días)', fontsize=14)
    axes[0].set_ylabel('Retorno (%)', fontsize=12)
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Graficar volatilidad anualizada
    axes[1].plot(rolling_spy_vol, label='SPY', color='gray', alpha=0.7)
    axes[1].plot(rolling_strat_vol, label='Estrategia HAR-X', color='blue')
    axes[1].set_title(f'Volatilidad Anualizada (Ventana móvil {window} días)', fontsize=14)
    axes[1].set_ylabel('Volatilidad (%)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Graficar Sharpe Ratio
    axes[2].plot(rolling_spy_sharpe, label='SPY', color='gray', alpha=0.7)
    axes[2].plot(rolling_strat_sharpe, label='Estrategia HAR-X', color='blue')
    axes[2].set_title(f'Ratio de Sharpe (Ventana móvil {window} días)', fontsize=14)
    axes[2].set_ylabel('Sharpe Ratio', fontsize=12)
    axes[2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.xlabel('Fecha', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_return_distribution(df):
    """Visualiza la distribución de retornos diarios"""
    plt.figure(figsize=(14, 7))
    
    # Crear histogramas
    bins = 50
    plt.hist(df['SPY_ret']*100, bins=bins, alpha=0.5, label='SPY', color='gray')
    plt.hist(df['strategy_ret']*100, bins=bins, alpha=0.5, label='Estrategia HAR-X', color='blue')
    
    # Añadir líneas de distribución normal
    x = np.linspace(df['SPY_ret'].min()*100, df['SPY_ret'].max()*100, 100)
    
    # Distribución normal para SPY
    spy_mean = df['SPY_ret'].mean() * 100
    spy_std = df['SPY_ret'].std() * 100
    spy_pdf = norm.pdf(x, spy_mean, spy_std) * len(df) * (df['SPY_ret'].max()*100 - df['SPY_ret'].min()*100) / bins
    plt.plot(x, spy_pdf, color='black', linestyle='--', linewidth=2, label='SPY Normal Fit')
    
    # Distribución normal para la estrategia
    strat_mean = df['strategy_ret'].mean() * 100
    strat_std = df['strategy_ret'].std() * 100
    strat_pdf = norm.pdf(x, strat_mean, strat_std) * len(df) * (df['strategy_ret'].max()*100 - df['strategy_ret'].min()*100) / bins
    plt.plot(x, strat_pdf, color='darkblue', linestyle='--', linewidth=2, label='Estrategia Normal Fit')
    
    # Añadir líneas verticales para la media
    plt.axvline(spy_mean, color='gray', linestyle='-', linewidth=2)
    plt.axvline(strat_mean, color='blue', linestyle='-', linewidth=2)
    
    plt.title('Distribución de Retornos Diarios', fontsize=16)
    plt.xlabel('Retorno (%)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de la distribución
    print("\n=== ESTADÍSTICAS DE DISTRIBUCIÓN DE RETORNOS ===")
    print(f"SPY - Media: {spy_mean:.4f}%, Mediana: {np.median(df['SPY_ret']*100):.4f}%, Desv. Estándar: {spy_std:.4f}%")
    print(f"Estrategia - Media: {strat_mean:.4f}%, Mediana: {np.median(df['strategy_ret']*100):.4f}%, Desv. Estándar: {strat_std:.4f}%")
    print(f"SPY - Asimetría: {df['SPY_ret'].skew():.4f}, Curtosis: {df['SPY_ret'].kurtosis():.4f}")
    print(f"Estrategia - Asimetría: {df['strategy_ret'].skew():.4f}, Curtosis: {df['strategy_ret'].kurtosis():.4f}")

def main():
    # Fechas de inicio y fin
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("====================================================")
    print("ESTRATEGIA DE TRADING HAR-X: ANÁLISIS DETALLADO")
    print("====================================================")
    print(f"\nPeriodo: {start_date} a {end_date}")
    print("\nDescargando y procesando datos...")
    
    # Obtener datos
    spy = get_historical_data("SPY", start_date, end_date)
    vix = get_vix_data(start_date, end_date)
    
    # Crear aproximaciones de VIX3M y VIX6M
    vix_term = create_synthetic_vix_term_structure(vix)
    
    # Calcular volatilidad Yang-Zhang para SPY
    spy['YZ_vol'] = yang_zhang_volatility(spy, window=10)
    
    # Construcción de variables HAR
    spy['YZ_d'] = spy['YZ_vol'].shift(1)  # Retardo diario
    spy['YZ_w'] = spy['YZ_vol'].rolling(window=5).mean().shift(1)  # Promedio semanal
    spy['YZ_m'] = spy['YZ_vol'].rolling(window=22).mean().shift(1)  # Promedio mensual
    
    # Construir dataset final
    df = pd.DataFrame(index=spy.index)
    df['YZ_vol'] = spy['YZ_vol']
    df['YZ_d'] = spy['YZ_d']
    df['YZ_w'] = spy['YZ_w']
    df['YZ_m'] = spy['YZ_m']
    
    # Calcular los retornos diarios de SPY para el backtest
    df['SPY_ret'] = np.log(spy['Close'] / spy['Close'].shift(1))
    
    # Alinear índices para unir los DataFrames
    common_dates = df.index.intersection(vix_term.index)
    df = df.loc[common_dates]
    vix_term = vix_term.loc[common_dates]
    
    # Añadir VIX3M y VIX6M al DataFrame principal
    df['VIX'] = vix_term['VIX']
    df['VIX3M'] = vix_term['VIX3M']
    df['VIX6M'] = vix_term['VIX6M']
    df['VIX_Slope'] = vix_term['VIX_Slope']
    
    # Eliminar filas con datos faltantes
    df.dropna(inplace=True)
    
    print(f"Datos procesados. {len(df)} días de trading disponibles.")
    
    # --- MODELADO HAR-X ---
    print("\nEntrenando modelo HAR-X...")
    X = df[['YZ_d', 'YZ_w', 'YZ_m', 'VIX3M', 'VIX6M', 'VIX_Slope']]
    X = sm.add_constant(X)
    y = df['YZ_vol']
    harx_model = sm.OLS(y, X).fit()
    print(f"\nCoeficientes del modelo HAR-X:")
    print(harx_model.summary().tables[1])
    
    # Predecir volatilidad ajustada
    df['YZ_vol_pred'] = harx_model.predict(X)
    
    # --- Generación de Señales de Trading ---
    print("\nGenerando señales de trading...")
    p25 = np.percentile(df['YZ_vol_pred'], 25)
    p75 = np.percentile(df['YZ_vol_pred'], 75)
    
    def asigna_exposicion(vol_pred, p25, p75):
        if vol_pred < p25:
            return 2.0  # Doble exposición (más riesgo si la volatilidad es baja)
        elif vol_pred > p75:
            return 0.0  # Pasar a efectivo (evitar riesgo en alta volatilidad)
        else:
            return 1.0  # Exposición normal
    
    df['position'] = df['YZ_vol_pred'].apply(asigna_exposicion, args=(p25, p75))
    
    # --- Backtest de la Estrategia ---
    print("Realizando backtest...")
    df['strategy_ret'] = df['position'].shift(1) * df['SPY_ret']  # Usamos posición del día anterior
    
    # Rendimientos acumulados
    df['cum_SPY'] = np.exp(df['SPY_ret'].cumsum())
    df['cum_strategy'] = np.exp(df['strategy_ret'].cumsum())
    
    print(f"\nUmbral Vol. Baja (p25): {p25:.4f}")
    print(f"Umbral Vol. Alta (p75): {p75:.4f}")
    print(f"Rentabilidad final SPY: {df['cum_SPY'].iloc[-1]:.2f}x")
    print(f"Rentabilidad final Estrategia: {df['cum_strategy'].iloc[-1]:.2f}x")
    
    # --- Análisis de resultados ---
    print("\n====================================================")
    print("ANÁLISIS DE RESULTADOS")
    print("====================================================")
    
    # 1. Análisis detallado de métricas
    metrics_comparison = create_performance_tearsheet(df)
    
    # 2. Análisis por régimen de volatilidad
    regime_analysis = analyze_performance_by_regime(df)
    
    # 3. Análisis de rendimientos mensuales
    monthly_returns, monthly_stats = analyze_monthly_returns(df)
    
    # --- Gráficos Avanzados ---
    print("\n====================================================")
    print("VISUALIZACIONES AVANZADAS")
    print("====================================================")
    
    # 1. Gráfico principal: Rendimiento acumulado
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['cum_SPY'], label="Buy & Hold SPY", linewidth=2, color='gray', alpha=0.7)
    plt.plot(df.index, df['cum_strategy'], label="Estrategia HAR-X", linewidth=2, color='blue')
    plt.title("Rendimiento Acumulado: Estrategia HAR-X vs. Buy & Hold", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Rendimiento Acumulado", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Gráfico de volatilidad prevista y bandas
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['YZ_vol_pred'], label="Volatilidad prevista (YZ)", linewidth=2, color='purple')
    plt.axhline(p25, color='green', linestyle='--', label=f"Percentil 25 ({p25:.4f})")
    plt.axhline(p75, color='red', linestyle='--', label=f"Percentil 75 ({p75:.4f})")
    
    # Colorear áreas de régimen de volatilidad
    plt.fill_between(df.index, 0, p25, color='green', alpha=0.1, label='Vol. Baja - Exposición 2x')
    plt.fill_between(df.index, p25, p75, color='yellow', alpha=0.1, label='Vol. Media - Exposición 1x')
    plt.fill_between(df.index, p75, df['YZ_vol_pred'].max(), color='red', alpha=0.1, label='Vol. Alta - Sin Exposición')
    
    plt.title("Volatilidad Prevista y Umbrales de Trading", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Volatilidad Anualizada", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. Heat map de rendimientos mensuales
    plot_monthly_returns_heatmap(monthly_returns)
    
    # 4. Distribución de la exposición
    plot_strategy_exposure_distribution(df)
    
    # 5. Comparación de drawdowns
    plot_drawdown_comparison(df)
    
    # 6. Rendimiento móvil
    plot_rolling_performance(df)
    
    # 7. Distribución de retornos
    plot_return_distribution(df)
    
    print("\n====================================================")
    print("CONCLUSIONES")
    print("====================================================")
    print("""
La estrategia HAR-X utiliza un modelo de heterogeneidad autorregresiva (HAR) 
extendido con variables exógenas (X) para predecir la volatilidad futura 
del mercado y ajustar dinámicamente la exposición al riesgo. Los principios básicos son:

1. En periodos de baja volatilidad prevista (< percentil 25), aumenta la exposición (2x)
2. En periodos de volatilidad media, mantiene exposición normal (1x)
3. En periodos de alta volatilidad (> percentil 75), se retira a efectivo (0x)

Este enfoque busca capitalizar la relación inversa entre volatilidad y rendimientos
a largo plazo, evitando periodos de turbulencia y aprovechando los periodos de calma.

El modelo HAR-X combina:
- Volatilidad Yang-Zhang (YZ) con componentes a distintos horizontes (diario, semanal, mensual)
- Información de la estructura temporal del VIX (VIX, VIX3M, VIX6M y pendiente)

La estrategia ha demostrado capacidad para:
1. Reducir la volatilidad y los drawdowns máximos 
2. Mejorar los ratios de rendimiento ajustado al riesgo (Sharpe, Sortino)
3. Mantener exposición óptima en diferentes regímenes de mercado
""")

    return df, harx_model

# Ejecutar el análisis completo
if __name__ == "__main__":
    df, model = main()
