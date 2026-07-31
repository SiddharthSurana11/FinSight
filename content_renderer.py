# Importing libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from financial_models import (
    interactive_plot, normalize, daily_return, calculate_beta,
    build_metrics_dataframe, calculate_correlation_matrix, plot_correlation_heatmap,
    plot_risk_return_scatter, plot_cumulative_returns, run_monte_carlo,
    plot_monte_carlo, calculate_black_scholes, generate_business_interpretation
)

# PHASE 4: Hero Branding Section (FinSight Version 2)
def render_hero_section():
    """
    Render FinSight Version 2 hero branding section with official logo, product name, subtitle, and platform description.
    """
    col_logo, col_title = st.columns([0.12, 0.88])
    with col_logo:
        st.image("assets/logo/finsight_logo.png", use_container_width=True)
    with col_title:
        st.markdown("<H1 class='finsight-title'>FinSight</H1>", unsafe_allow_html=True)
        st.markdown("<p class='finsight-subtitle'>Professional Financial Risk & Return Analytics Platform</p>", unsafe_allow_html=True)
        st.markdown("<p class='finsight-tagline'>Powered by CAPM • Portfolio Analytics • Monte Carlo Simulation • Black-Scholes Pricing</p>", unsafe_allow_html=True)


# Function to get user input for CAPM (Phase 4 Compact Filter Card)
def get_user_input():
    """
    Render Streamlit inputs for asset selection, evaluation timeframe, and risk-free rate.
    Supports both preset multi-selection and dynamic custom Yahoo Finance ticker entry.
    Wrapped in a compact card container with st.pills control.
    """
    with st.container(border=True):
        st.markdown("#### Portfolio & Asset Selection Filters")
        col1, col2 = st.columns([1.2, 0.8], gap="medium")
        
        with col1:
            input_mode = st.pills(
                "Ticker Selection Method:",
                ["Preset Tickers", "Custom Tickers Input"],
                default="Preset Tickers"
            )
            if input_mode == "Preset Tickers":
                stock_list = st.multiselect(
                    "Choose stocks",
                    ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL', 'META', 'INFY.NS', 'RELIANCE.NS', 'TCS.NS'),
                    ['TSLA', 'AAPL', 'AMZN', 'GOOGL']
                )
            else:
                custom_tickers = st.text_input("Enter Ticker Symbols (comma-separated):", value="AAPL, MSFT, NVDA, INFY.NS")
                stock_list = [s.strip().upper() for s in custom_tickers.split(',') if s.strip()]

        with col2:
            subcol1, subcol2 = st.columns([1, 1], gap="small")
            with subcol1:
                year = st.number_input("Number of years", 1, 10, value=1)
            with subcol2:
                rf = st.number_input("Risk-Free Rate (%)", value=0.0)

    return stock_list, year, rf


# Function to download financial data with Streamlit caching
@st.cache_data(ttl=3600)
def download_data(stock_list, year):
    """
    Download market data for selected assets and S&P 500 benchmark (^GSPC) from Yahoo Finance.
    Implements robust error handling, ticker validation, duplicate removal, ffill/bfill cleaning, and Streamlit caching.
    """
    if isinstance(stock_list, tuple):
        stock_list = list(stock_list)

    cleaned_tickers = [str(s).strip().upper() for s in stock_list if str(s).strip()]
    if not cleaned_tickers:
        st.warning("Please select or enter at least one valid ticker symbol.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        # Download S&P 500 benchmark data (^GSPC)
        sp500_raw = yf.download('^GSPC', period=f'{year}y', progress=False)
        if sp500_raw.empty or 'Close' not in sp500_raw.columns:
            st.error("Unable to download S&P 500 market benchmark (^GSPC). Please check internet connectivity or retry.")
            return pd.DataFrame(), pd.DataFrame()

        sp500_close = sp500_raw['Close']
        if isinstance(sp500_close, pd.DataFrame):
            sp500_close = sp500_close.squeeze()

        SP500 = pd.DataFrame({
            'Date': pd.to_datetime(sp500_close.index).tz_localize(None),
            'sp500': sp500_close.values
        })

        # Download individual asset price series
        stocks_dict = {}
        invalid_tickers = []

        for stock in cleaned_tickers:
            data = yf.download(stock, period=f'{year}y', progress=False)
            if data.empty or 'Close' not in data.columns:
                invalid_tickers.append(stock)
                continue
            
            close_series = data['Close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.squeeze()

            if close_series.dropna().empty:
                invalid_tickers.append(stock)
                continue

            stocks_dict[stock] = close_series

        # Inform user of invalid/delisted tickers
        if invalid_tickers:
            st.warning(f"Could not fetch price data for symbol(s): {', '.join(invalid_tickers)}. They may be invalid, delisted, or missing data.")

        if not stocks_dict:
            st.error("None of the specified tickers returned valid market price data.")
            return pd.DataFrame(), pd.DataFrame()

        # Combine asset price series into unified DataFrame
        stocks_df = pd.DataFrame(stocks_dict)
        stocks_df.reset_index(inplace=True)
        stocks_df.rename(columns={stocks_df.columns[0]: 'Date'}, inplace=True)
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date']).dt.tz_localize(None)

        # Data Cleaning: Sort chronologically and drop duplicate timestamp rows
        stocks_df = stocks_df.sort_values('Date').drop_duplicates(subset=['Date'])
        SP500 = SP500.sort_values('Date').drop_duplicates(subset=['Date'])

        # Normalize Date column formatting for merge key
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date'].dt.strftime('%Y-%m-%d'))
        SP500['Date'] = pd.to_datetime(SP500['Date'].dt.strftime('%Y-%m-%d'))

        # Inner join asset prices with S&P 500 benchmark on Date
        stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

        # Clean missing values via forward fill then backward fill
        valid_cols = [col for col in stocks_df.columns if col != 'Date']
        stocks_df[valid_cols] = stocks_df[valid_cols].ffill().bfill()

        if stocks_df.empty:
            st.error("Data processing yielded an empty dataset after merging. Please check ticker inputs or time frame.")
            return pd.DataFrame(), pd.DataFrame()

        return stocks_df, SP500

    except Exception as e:
        st.error(f"Error executing data download: {e}")
        return pd.DataFrame(), pd.DataFrame()


# PHASE 4: Executive Summary KPI Row (Consolidated Single Source of Truth metrics_df)
def display_executive_kpi_row(metrics_df):
    """
    Render Executive Summary KPI Summary Row immediately after filters & data download.
    Phase 4: Reuses pre-computed single source of truth metrics_df (zero duplicate calculations!).
    """
    if metrics_df.empty:
        return

    best_sharpe_ticker = metrics_df['Sharpe Ratio'].idxmax()
    best_sharpe_val = float(metrics_df.loc[best_sharpe_ticker, 'Sharpe Ratio'])
    
    highest_vol_ticker = metrics_df['Annualized Volatility (%)'].idxmax()
    highest_vol_val = float(metrics_df.loc[highest_vol_ticker, 'Annualized Volatility (%)'])
    
    lowest_vol_ticker = metrics_df['Annualized Volatility (%)'].idxmin()
    lowest_vol_val = float(metrics_df.loc[lowest_vol_ticker, 'Annualized Volatility (%)'])
    
    avg_return_val = float(metrics_df['Annual Return (%)'].mean())

    with st.container(border=True):
        st.markdown("### Executive Summary & Portfolio KPIs")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            label="Best Risk-Adjusted Return",
            value=f"{best_sharpe_ticker}",
            delta=f"{best_sharpe_val:.2f} Sharpe"
        )
        kpi2.metric(
            label="Highest Risk Asset",
            value=f"{highest_vol_ticker}",
            delta=f"{highest_vol_val:.2f}% Vol"
        )
        kpi3.metric(
            label="Lowest Risk Asset",
            value=f"{lowest_vol_ticker}",
            delta=f"{lowest_vol_val:.2f}% Vol"
        )
        kpi4.metric(
            label="Portfolio Avg Return",
            value=f"{avg_return_val:.2f}%"
        )


# Displaying dataframes and plots (Phase 4: Data Preview Expander & Pre-computed Correlation Matrix)
def display_dataframes_and_plots(stocks_df, stocks_daily_return, corr_matrix):
    """
    Render Plotly charts for prices, normalized base values, and pre-computed asset correlation matrix.
    Phase 4: Dataframe Head/Tail moved to collapsed Streamlit expander to eliminate visual clutter.
    Reuses pre-computed stocks_daily_return and corr_matrix.
    """
    if stocks_df.empty:
        return

    # PHASE 4: Collapsed Data Preview Expander
    with st.expander("▶ Data Preview (Expand to Inspect Raw Market Data)"):
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Dataframe Head")
                st.dataframe(stocks_df.head(), use_container_width=True)
            with col2:
                st.markdown("#### Dataframe Tail")
                st.dataframe(stocks_df.tail(), use_container_width=True)

    # Price Charts Cards
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Price of all the Stocks")
            st.plotly_chart(interactive_plot(stocks_df), use_container_width=True)
        with col2:
            st.markdown("### Price of all the Stocks (After Normalizing)")
            st.plotly_chart(interactive_plot(normalize(stocks_df)), use_container_width=True)

    # Correlation Heatmap Card (Reuses pre-computed corr_matrix)
    with st.container(border=True):
        st.markdown("### Asset Return Correlation Matrix")
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)


# Function to calculate and display metrics (Phase 4: Consolidated single source of truth inputs)
def calculate_and_display_metrics(stocks_df, stocks_daily_return, metrics_df, corr_matrix, stock_list, rf):
    """
    Calculate daily returns, Beta, Alpha, and CAPM Expected Return for all valid assets.
    Renders Risk-Return Scatter, Cumulative Growth, Monte Carlo forecast simulation,
    and Black-Scholes option pricing modules using pre-computed single-source-of-truth objects.
    """
    if stocks_df.empty or metrics_df.empty:
        st.warning("No financial data available for metric calculations.")
        return

    # Phase 1 Tables Card (Beta, Alpha, CAPM Expected Return)
    beta = {}
    alpha = {}
    for stock in stocks_daily_return.columns:
        if stock != 'Date' and stock != 'sp500':
            b, a = calculate_beta(stocks_daily_return, stock)
            beta[stock] = b
            alpha[stock] = a

    with st.container(border=True):
        st.markdown("### CAPM Regression Metrics (Beta & Alpha)")
        tab1, tab2, tab3 = st.tabs(["Beta Values", "Alpha Values", "CAPM Expected Return"])
        
        with tab1:
            beta_df = pd.DataFrame({'Stocks': list(beta.keys()), 'Beta Value': [str(round(i, 2)) for i in beta.values()]})
            st.dataframe(beta_df, use_container_width=True)
        with tab2:
            alpha_df = pd.DataFrame({'Stocks': list(alpha.keys()), 'Alpha Value': [str(round(i, 2)) for i in alpha.values()]})
            st.dataframe(alpha_df, use_container_width=True)
        with tab3:
            rm = stocks_daily_return['sp500'].mean() * 252.0
            return_value = [str(round(rf + (b_val * (rm - rf)), 2)) for b_val in beta.values()]
            return_df = pd.DataFrame({'Stock': list(beta.keys()), 'Return Value': return_value})
            st.dataframe(return_df, use_container_width=True)

    # Master Portfolio Risk & Return Metrics Table Card
    with st.container(border=True):
        st.markdown("### Master Portfolio Risk & Return Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Programmatically Generated Business Interpretation (Reuses pre-computed corr_matrix)
        interpretation_text = generate_business_interpretation(metrics_df, corr_matrix)
        st.info(interpretation_text)

    # Risk vs Return Scatter Plot Card
    with st.container(border=True):
        st.markdown("### Risk vs Return Profile")
        st.plotly_chart(plot_risk_return_scatter(metrics_df), use_container_width=True)

    # Cumulative Return Growth Chart Card
    with st.container(border=True):
        st.markdown("### Cumulative Investment Growth ($1 Base)")
        st.plotly_chart(plot_cumulative_returns(stocks_df), use_container_width=True)

    # --- QUANTITATIVE FINANCE MODULES ---
    asset_list = list(metrics_df.index)
    if not asset_list:
        return

    # Monte Carlo Simulation Expander Card
    with st.expander("Monte Carlo Price Path Forecast (Geometric Brownian Motion)"):
        with st.container(border=True):
            st.markdown("**Simulate future asset price trajectories based on historical return drift and volatility.**")
            
            mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 1])
            with mc_col1:
                selected_mc_ticker = st.selectbox("Select Asset for Forecast", asset_list, key="mc_ticker")
            with mc_col2:
                n_sims = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100, key="mc_sims")
            with mc_col3:
                forecast_days = st.number_input("Forecast Horizon (Trading Days)", min_value=10, max_value=756, value=252, key="mc_days")

            last_price = float(stocks_df[selected_mc_ticker].iloc[-1])
            ann_return = float(metrics_df.loc[selected_mc_ticker, 'Annual Return (%)'])
            ann_vol = float(metrics_df.loc[selected_mc_ticker, 'Annualized Volatility (%)'])

            sim_paths, exp_price, ci_low, ci_high, best_c, worst_c = run_monte_carlo(
                last_price=last_price,
                annual_return=ann_return,
                annual_volatility=ann_vol,
                days=forecast_days,
                n_simulations=n_sims
            )

            st.plotly_chart(plot_monte_carlo(sim_paths, selected_mc_ticker, exp_price, ci_low, ci_high), use_container_width=True)

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Current Price", f"${last_price:.2f}")
            res_col2.metric("Expected Price", f"${exp_price:.2f}")
            res_col3.metric("95% Confidence Interval", f"${ci_low:.2f} - ${ci_high:.2f}")
            res_col4.metric("Extreme Range (1% - 99%)", f"${worst_c:.2f} - ${best_c:.2f}")

    # Black-Scholes Options Valuation Expander Card
    with st.expander("Advanced Analytics: Black-Scholes Option Pricing Model"):
        with st.container(border=True):
            st.markdown("**Theoretical European Call & Put option valuation reusing asset volatility and risk-free rate.**")
            
            bs_col1, bs_col2, bs_col3 = st.columns([1, 1, 1])
            with bs_col1:
                selected_bs_ticker = st.selectbox("Select Asset for Options Pricing", asset_list, key="bs_ticker")
                spot_price = float(stocks_df[selected_bs_ticker].iloc[-1])
                st.metric("Current Spot Price (S)", f"${spot_price:.2f}")
                
            with bs_col2:
                strike_price = st.number_input("Strike Price (K)", min_value=0.1, value=round(spot_price, 2), step=1.0, key="bs_strike")
                expiry_days = st.number_input("Time to Expiry (Days)", min_value=1, max_value=1095, value=30, step=1, key="bs_expiry")
                
            with bs_col3:
                asset_vol = float(metrics_df.loc[selected_bs_ticker, 'Annualized Volatility (%)'])
                st.metric("Annualized Volatility (σ)", f"{asset_vol:.2f}%")
                st.metric("Risk-Free Rate (r)", f"{rf:.2f}%")

            T_years = float(expiry_days) / 365.0
            r_decimal = float(rf) / 100.0
            sigma_decimal = float(asset_vol) / 100.0

            bs_results = calculate_black_scholes(
                S=spot_price,
                K=strike_price,
                T=T_years,
                r=r_decimal,
                sigma=sigma_decimal
            )

            val_col1, val_col2, val_col3, val_col4 = st.columns(4)
            val_col1.metric("Call Option Price", f"${bs_results['call_price']:.2f}")
            val_col2.metric("Put Option Price", f"${bs_results['put_price']:.2f}")
            val_col3.metric("Call Delta (Δ)", f"{bs_results['call_delta']:.2f}")
            val_col4.metric("Put Delta (Δ)", f"{bs_results['put_delta']:.2f}")

            st.caption(
                "*Disclaimer: Assumes European exercise style, zero dividend yield, constant volatility, and lognormal price distribution. Educational model.*"
            )


# Apply CSS (Phase 4: FinSight Hero Branding & UI Polish)
def apply_css():
    st.markdown(
        """
            <style>
                    /* FinSight Hero Header Branding */
                    .finsight-title {
                        font-size: 2.8rem;
                        font-weight: 800;
                        font-family: 'Inter', 'Segoe UI', sans-serif;
                        color: #FAFAFA;
                        margin-top: -10px;
                        margin-bottom: 0px;
                        letter-spacing: -0.5px;
                    }
                    
                    .finsight-subtitle {
                        font-size: 1.35rem;
                        font-weight: 600;
                        color: #7C4DFF;
                        margin-top: 0px;
                        margin-bottom: 2px;
                    }

                    .finsight-tagline {
                        font-size: 0.95rem;
                        font-weight: 400;
                        color: #A0AAB8;
                        margin-top: 0px;
                        margin-bottom: 12px;
                    }

                    /* Executive KPI Card Typography & Visual Prominence Enhancement */
                    [data-testid="stMetricValue"] {
                        font-size: 2.1rem !important;
                        font-weight: 700 !important;
                    }
                    [data-testid="stMetricLabel"] {
                        font-size: 1.0rem !important;
                        font-weight: 600 !important;
                        color: #FAFAFA !important;
                    }
                    [data-testid="stMetricDelta"] {
                        font-size: 0.95rem !important;
                        font-weight: 600 !important;
                    }
                    
            </style>
        """,
        unsafe_allow_html=True
    )