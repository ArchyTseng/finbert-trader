# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %%
"""
FinBERT Trading System - News data analysis for FNSPID Dataset
Analysis pipeline with FNSPID News Dataset for exploring potential news patterns and symbols combination
"""

import sys
import os
import logging
from datetime import datetime

# Generate project root path
if '__file__' in globals():
    project_root = os.path.join(os.path.dirname(__file__), '..')
else:
    # For interactive environments, use current working directory
    project_root = os.path.join(os.getcwd(), '..')

sys.path.insert(0, project_root)

# Import necessary libraries
from src.finbert_trader.config_setup import ConfigSetup
from src.finbert_trader.data_resource import DataResource
from src.finbert_trader.features.feature_engineer import FeatureEngineer
from src.finbert_trader.visualize.visualize_news import VisualizeNews, select_stocks_by_news_coverage
from src.finbert_trader.visualize.visualize_features import generate_standard_feature_visualizations

# Set log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("News Analysis Environment Ready!")
print("=" * 50)

# %%
"""
Step 1: Initialize Configuration
Setup basic configuration for quick experiments
"""

# Set experiment configuration
custom_setup_config = {
    'start': '2000-01-01',
    'end': '2022-12-31',
    'train_start_date': '2000-01-01',
    'train_end_date': '2021-12-31',
    'valid_start_date': '2022-01-01',
    'valid_end_date': '2022-06-30',
    'test_start_date': '2022-07-01',
    'test_end_date': '2022-12-31',
    'use_symbol_name': False,
    'ind_mode': 'long',
}

# Initial config_setup for experiment
config_setup = ConfigSetup(custom_setup_config)
symbols_list = config_setup.nasdaq_100_tickers_july_17_2023
print("Configuration initialized successfully!")
print(f"Symbols: {symbols_list}")
print(f"Time period: {config_setup.start} to {config_setup.end}")

# %%
"""
Step 2: Analyze news data
Setup VisualizeNews, generate news analysis report
"""

config_setup.symbols = symbols_list

# Generate news chunks
dr = DataResource(config_setup)

# Generate cache_path and filtered_cache_path
cache_path, filtered_cache_path = dr.cache_path_config()

# Generate news chunk generator
news_chunk_gen = dr.load_news_data(cache_path, filtered_cache_path, chunksize=config_setup.chunksize)

# Generate news_df
fe = FeatureEngineer(config_setup)
news_df = fe.process_news_chunks(news_chunk_gen)

print("Generate News DataFrame successfully!")
print(f"news data length: {len(news_df)}")

selected_symbols, coverage_stats, plot_paths = select_stocks_by_news_coverage(
    config=config_setup,
    symbols_list=config_setup.symbols,
    top_n=10,  # Select top_n symbols with highest news coverage
    min_news_count=2000,  # minimum news data
    min_coverage_days=2000, # minimum news coverage period
    news_df=news_df,  
)

print("News analysis successfully!")
print(f"Selected symbols: {selected_symbols}")

# %%
"""
Step 3: Explore stock features
Merge stock features and news features, generate stock features analysis report
"""

# Set selected_symbols to ConfigSetup
config_setup.symbols = selected_symbols

# Fetch stock data
stock_data_dict = dr.fetch_stock_data()
if not stock_data_dict:
    raise ValueError("No stock data fetched")
logging.info(f"News_data_analysis - Prepared stock data for {len(stock_data_dict)} symbols")

# Filter target symbols news
filtered_news_df = news_df[news_df["Symbol"].isin(config_setup.symbols)]

# Compute sentiment scores and risk scores
sentiment_score_df = fe.news_engineer.compute_sentiment_risk_score(filtered_news_df.copy(), senti_mode='sentiment')    # Compute sentiment scores using FinBERT or similar
risk_score_df = fe.news_engineer.compute_sentiment_risk_score(filtered_news_df.copy(), senti_mode='risk')  # Compute risk scores if risk mode is active

# Merge stock features 
fused_df = fe.merge_features(stock_data_dict, sentiment_score_df, risk_score_df)

# Generate stock features analysis report
standard_visualize_results = generate_standard_feature_visualizations(fused_df, config_setup)
logging.info(f"FE Module - generate_experiment_data - Successfully generate Standard Visualization Results: {standard_visualize_results}")

# %%
# Test all symbols per group (5 symbols each)

for i in range(0, len(symbols_list), 5):
    symbols = symbols_list[i : i+5]
    print(f"Analyze Symbols: {symbols}")
    config_setup.symbols = symbols

    # Generate news chunks
    dr = DataResource(config_setup)

    # Generate cache_path and filtered_cache_path
    cache_path, filtered_cache_path = dr.cache_path_config()

    # Generate news chunk generator
    news_chunk_gen = dr.load_news_data(cache_path, filtered_cache_path, chunksize=config_setup.chunksize)

    # Generate news_df
    fe = FeatureEngineer(config_setup)
    news_df = fe.process_news_chunks(news_chunk_gen)

    print("Generate News DataFrame successfully!")
    print(f"news data length: {len(news_df)}")

    selected_symbols, coverage_stats, plot_paths = select_stocks_by_news_coverage(
        config=config_setup,
        symbols_list=symbols,
        top_n=3,  # Select top_n symbols with highest news coverage
        min_news_count=100,  # minimum news data
        min_coverage_days=365, # minimum news coverage period
        news_df=news_df,  
    )

print("News analysis successfully!")
    
