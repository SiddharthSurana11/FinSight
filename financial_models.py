# Importing libraries
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

# --- PHASE 3 THEME HELPER ---

def apply_dashboard_theme(fig):
    """
    Apply unified dark dashboard layout theme to a Plotly figure.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
    )
    return fig

# Function to plot interactive plotly chart (Phase 1 & Phase 4 preserved)
def interactive_plot(df, title="", x_axis_label="Time (Years)", y_axis_label="Value", height=500):
    """
    Generate an interactive Plotly line chart for stock price or return visualisations.
    Phase 4: Inner Plotly titles removed to prevent duplication with Streamlit card container headers.
    """
    fig = px.line(df, x='Date', y=df.columns[1:], title=title if title else None)
    fig.update_layout(
        width=600,
        height=height if height else 400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
    )
    return apply_dashboard_theme(fig)


# Function to normalize the prices based on initial price (Phase 1 - preserved)
def normalize(df_2):
    """
    Normalize asset prices relative to their starting initial price (Base = 1.0).
    """
    df = df_2.copy()
    for col in df.columns[1:]:
        first_val = df[col].iloc[0]
        if first_val != 0 and not pd.isna(first_val):
            df[col] = df[col] / first_val
    return df

# Function to calculate daily returns (Phase 1 - preserved)
def daily_return(df):
    """
    Calculate daily percentage returns for all assets in the DataFrame.
    Uses vectorised pandas pct_change for optimal numerical efficiency.
    """
    df_daily_return = df.copy()
    for col in df.columns[1:]:
        df_daily_return[col] = df[col].pct_change() * 100
        df_daily_return[col] = df_daily_return[col].fillna(0)
    return df_daily_return

# Function to calculate beta and alpha (Phase 1 - preserved)
def calculate_beta(stocks_daily_return, stock):
    """
    Calculate Beta and Alpha for a given stock relative to the S&P 500 market benchmark.
    
    Formula:
        Beta = Covariance(R_stock, R_market) / Variance(R_market)
        Alpha = Mean(R_stock) - Beta * Mean(R_market)
    """
    stock_returns = stocks_daily_return[stock]
    market_returns = stocks_daily_return['sp500']
    
    cov_matrix = np.cov(stock_returns, market_returns)
    covariance = cov_matrix[0, 1]
    market_variance = cov_matrix[1, 1]
    
    if market_variance != 0:
        b = covariance / market_variance
    else:
        b = 0.0
        
    a = stock_returns.mean() - (b * market_returns.mean())
    return b, a

# --- PHASE 2 QUANTITATIVE ENGINE FUNCTIONS (FROZEN LOGIC) ---

def calculate_sharpe_ratio(annual_return, annual_volatility, risk_free_rate):
    """
    Calculate Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility.
    Assumes annual_return, annual_volatility, and risk_free_rate are all in percentage terms.
    """
    if annual_volatility <= 0:
        return 0.0
    return (annual_return - risk_free_rate) / annual_volatility


def calculate_sortino_ratio(stock_daily_returns, annual_return, risk_free_rate, target_return=0.0):
    """
    Calculate Sortino Ratio = (Annualized Return - Risk-Free Rate) / Annualized Downside Deviation.
    Downside deviation is defined as the root mean square of daily returns below target (0.0%),
    annualized by multiplying by sqrt(252).
    """
    downside_squared = np.minimum(stock_daily_returns - target_return, 0.0) ** 2
    downside_variance = np.mean(downside_squared)
    annualized_downside_dev = np.sqrt(downside_variance) * np.sqrt(252.0)
    
    if annualized_downside_dev <= 0:
        return 0.0
        
    return (annual_return - risk_free_rate) / annualized_downside_dev


def calculate_treynor_ratio(annual_return, beta_value, risk_free_rate):
    """
    Calculate Treynor Ratio = (Annualized Return - Risk-Free Rate) / Beta.
    Reuses existing Beta value. Guards against Beta approximately zero.
    """
    if abs(beta_value) < 1e-6:
        return 0.0
    return (annual_return - risk_free_rate) / beta_value


def build_metrics_dataframe(stocks_daily_return, risk_free_rate):
    """
    Build the Master Metrics DataFrame as the single source of truth for portfolio analytics.
    
    Columns:
    - Annual Return (%)
    - Annualized Volatility (%)
    - Beta
    - Alpha (%)
    - Sharpe Ratio
    - Sortino Ratio
    - Treynor Ratio
    """
    asset_cols = [col for col in stocks_daily_return.columns if col not in ['Date', 'sp500']]
    metrics_data = []

    for stock in asset_cols:
        stock_returns = stocks_daily_return[stock]
        
        # 1. Annual Return & Volatility (Single source of truth)
        annual_return = stock_returns.mean() * 252.0
        annual_volatility = stock_returns.std(ddof=1) * np.sqrt(252.0)
        
        # 2. Beta & Alpha (Reused from Phase 1 calculate_beta)
        beta_val, alpha_val = calculate_beta(stocks_daily_return, stock)
        annual_alpha = alpha_val * 252.0
        
        # 3. Ratios
        sharpe = calculate_sharpe_ratio(annual_return, annual_volatility, risk_free_rate)
        sortino = calculate_sortino_ratio(stock_returns, annual_return, risk_free_rate)
        treynor = calculate_treynor_ratio(annual_return, beta_val, risk_free_rate)
        
        metrics_data.append({
            'Ticker': stock,
            'Annual Return (%)': round(annual_return, 2),
            'Annualized Volatility (%)': round(annual_volatility, 2),
            'Beta': round(beta_val, 2),
            'Alpha (%)': round(annual_alpha, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Sortino Ratio': round(sortino, 2),
            'Treynor Ratio': round(treynor, 2)
        })

    df_metrics = pd.DataFrame(metrics_data).set_index('Ticker')
    return df_metrics


def calculate_correlation_matrix(stocks_daily_return):
    """
    Compute Pearson correlation matrix across all selected assets' daily returns.
    """
    asset_df = stocks_daily_return.drop(columns=['Date', 'sp500'], errors='ignore')
    return asset_df.corr()


def plot_correlation_heatmap(corr_matrix):
    """
    Generate an interactive Plotly heatmap for asset return correlation.
    Phase 4: Redundant figure title removed to eliminate duplication with card headers.
    """
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=320
    )
    return apply_dashboard_theme(fig)


def plot_risk_return_scatter(metrics_df):
    """
    Generate an interactive Risk vs Return scatter plot (Volatility vs Annual Return)
    with Sharpe Ratio in hover tooltip.
    Phase 4: Redundant figure title removed to eliminate duplication with card headers.
    """
    df_plot = metrics_df.reset_index()
    fig = px.scatter(
        df_plot,
        x='Annualized Volatility (%)',
        y='Annual Return (%)',
        text='Ticker',
        color='Sharpe Ratio',
        hover_data=['Beta', 'Alpha (%)', 'Sortino Ratio', 'Treynor Ratio'],
        color_continuous_scale='Viridis'
    )
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        xaxis_title="Annualized Volatility (%) [Risk]",
        yaxis_title="Annualized Return (%) [Return]"
    )
    return apply_dashboard_theme(fig)


def plot_cumulative_returns(stocks_df):
    """
    Generate cumulative growth plot of $1 invested per asset over time.
    Phase 4: Redundant figure title removed to eliminate duplication with card headers.
    """
    df_cum = stocks_df.copy()
    asset_cols = [col for col in df_cum.columns if col != 'Date']
    
    for col in asset_cols:
        first_price = df_cum[col].iloc[0]
        if first_price != 0 and not pd.isna(first_price):
            df_cum[col] = df_cum[col] / first_price

    fig = px.line(
        df_cum,
        x='Date',
        y=asset_cols
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=420,
        xaxis_title="Date",
        yaxis_title="Growth Factor ($1 Base)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return apply_dashboard_theme(fig)


def run_monte_carlo(last_price, annual_return, annual_volatility, days=252, n_simulations=1000):
    """
    Pure Python/Numpy Monte Carlo Simulation using Geometric Brownian Motion (GBM).
    
    Formula:
        S_t = S_0 * exp( (mu_daily - 0.5 * sigma_daily^2) * t + sigma_daily * sqrt(t) * Z_t )
    
    Inputs:
        last_price: Starting asset price (S_0)
        annual_return: Annualized return in % (mu)
        annual_volatility: Annualized volatility in % (sigma)
        days: Forecast horizon in trading days
        n_simulations: Number of simulated price paths
    Outputs:
        sim_paths: (days+1, n_simulations) numpy array
        expected_price: Mean final price
        ci_lower: 2.5th percentile final price
        ci_upper: 97.5th percentile final price
        best_case: 99th percentile final price
        worst_case: 1st percentile final price
    """
    mu_daily = (annual_return / 100.0) / 252.0
    sigma_daily = (annual_volatility / 100.0) / np.sqrt(252.0)
    
    drift = mu_daily - 0.5 * (sigma_daily ** 2)
    Z = np.random.normal(0, 1, size=(days, n_simulations))
    daily_factors = np.exp(drift + sigma_daily * Z)
    
    sim_paths = np.zeros((days + 1, n_simulations))
    sim_paths[0, :] = last_price
    sim_paths[1:, :] = last_price * np.cumprod(daily_factors, axis=0)
    
    final_prices = sim_paths[-1, :]
    expected_price = np.mean(final_prices)
    ci_lower = np.percentile(final_prices, 2.5)
    ci_upper = np.percentile(final_prices, 97.5)
    best_case = np.percentile(final_prices, 99.0)
    worst_case = np.percentile(final_prices, 1.0)
    
    return sim_paths, expected_price, ci_lower, ci_upper, best_case, worst_case


def plot_monte_carlo(sim_paths, ticker, expected_price, ci_lower, ci_upper):
    """
    Generate Plotly chart of Monte Carlo simulated price trajectories.
    Sample up to 100 paths for fast rendering.
    Phase 4: Redundant figure title removed to eliminate duplication with card headers.
    """
    days, n_sims = sim_paths.shape
    sample_size = min(100, n_sims)
    
    fig = go.Figure()
    
    for i in range(sample_size):
        fig.add_trace(go.Scatter(
            y=sim_paths[:, i],
            mode='lines',
            line=dict(width=0.8, color='rgba(70, 130, 180, 0.15)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    mean_path = np.mean(sim_paths, axis=1)
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        line=dict(color='orange', width=3),
        name=f'Expected Path (${expected_price:.2f})'
    ))
    
    fig.update_layout(
        xaxis_title="Forecast Days",
        yaxis_title="Asset Price ($)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=420
    )
    return apply_dashboard_theme(fig)


def calculate_black_scholes(S, K, T, r, sigma):
    """
    Calculate Black-Scholes European Option Call and Put prices and Deltas.
    
    Formula:
        d1 = [ln(S/K) + (r + sigma^2 / 2) T] / [sigma * sqrt(T)]
        d2 = d1 - sigma * sqrt(T)
        Call = S * N(d1) - K * exp(-r T) * N(d2)
        Put  = K * exp(-r T) * N(-d2) - S * N(-d1)
        Call Delta = N(d1), Put Delta = N(d1) - 1
        
    Disclaimer:
        Assumes European exercise style, zero dividend yield, constant volatility, and lognormal price distribution.
        Educational model for analytical reference.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        call_p = max(0.0, S - K)
        put_p = max(0.0, K - S)
        call_d = 1.0 if S > K else (0.5 if S == K else 0.0)
        put_d = call_d - 1.0
        return {
            'call_price': round(call_p, 2),
            'put_price': round(put_p, 2),
            'call_delta': round(call_d, 2),
            'put_delta': round(put_d, 2)
        }
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_p = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_p = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    call_d = norm.cdf(d1)
    put_d = call_d - 1.0
    
    return {
        'call_price': round(call_p, 2),
        'put_price': round(put_p, 2),
        'call_delta': round(call_d, 2),
        'put_delta': round(put_d, 2)
    }


def generate_business_interpretation(metrics_df, corr_matrix):
    """
    Programmatically generate 2-4 sentences of plain-language financial interpretation
    from the Master Metrics DataFrame and Correlation Matrix.
    """
    if metrics_df.empty or corr_matrix.empty:
        return "Insufficient data available for generating automated risk-return insights."
        
    best_sharpe_ticker = metrics_df['Sharpe Ratio'].idxmax()
    best_sharpe_val = metrics_df.loc[best_sharpe_ticker, 'Sharpe Ratio']
    
    highest_vol_ticker = metrics_df['Annualized Volatility (%)'].idxmax()
    highest_vol_val = metrics_df.loc[highest_vol_ticker, 'Annualized Volatility (%)']
    
    corr_unstacked = corr_matrix.unstack()
    corr_filtered = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
    
    if not corr_filtered.empty:
        min_pair = corr_filtered.idxmin()
        min_corr_val = corr_filtered.min()
        diversification_text = f"The pair with the lowest correlation is **{min_pair[0]}** and **{min_pair[1]}** ({min_corr_val:.2f}), offering the highest portfolio diversification benefit."
    else:
        diversification_text = "Select multiple tickers to evaluate pairwise correlation and portfolio diversification benefits."
        
    interpretation = (
        f"**Risk & Return Insights:** **{best_sharpe_ticker}** delivers the best risk-adjusted performance with a Sharpe Ratio of **{best_sharpe_val:.2f}**. "
        f"**{highest_vol_ticker}** exhibits the highest total risk profile with an annualized volatility of **{highest_vol_val:.2f}%**. "
        f"{diversification_text}"
    )
    return interpretation