# Importing libraries
import streamlit as st
from content_renderer import (
    download_data, calculate_and_display_metrics, display_dataframes_and_plots,
    get_user_input, apply_css, display_executive_kpi_row, render_hero_section
)
from financial_models import daily_return, build_metrics_dataframe, calculate_correlation_matrix
from footer import display_signature

def main():
    # Set Streamlit page configuration for FinSight Analytics
    st.set_page_config(
        page_title="FinSight - Financial Risk & Return Analytics",
        page_icon="assets/logo/finsight_logo.png",
        layout='wide'
    )
    apply_css()

    # 1. Hero Branding Section (FinSight Logo & Product Tagline)
    render_hero_section()

    # 2. Get user input for CAPM & Portfolio Analysis
    stock_list, year, rf = get_user_input()
    
    # 3. Download market price data
    stocks_df, SP500 = download_data(stock_list, year)

    if not stocks_df.empty:
        # Phase 4 Single Source of Truth Analytics Computation (Compute Once, Reuse Everywhere)
        stocks_daily_return = daily_return(stocks_df)
        metrics_df = build_metrics_dataframe(stocks_daily_return, rf)
        corr_matrix = calculate_correlation_matrix(stocks_daily_return)

        # 4. Executive Summary KPI Row (Immediately after Filters, before charts)
        display_executive_kpi_row(metrics_df)
        
        # 5. Display price charts, data preview expander, and correlation heatmap
        display_dataframes_and_plots(stocks_df, stocks_daily_return, corr_matrix)
        
        # 6. Calculate & display quantitative metrics (CAPM, Master Metrics Table, Scatter, Cum Returns, MC, BS)
        calculate_and_display_metrics(stocks_df, stocks_daily_return, metrics_df, corr_matrix, stock_list, rf)
    
    # 7. FinSight Professional Footer
    display_signature()

if __name__ == "__main__":
    main()
