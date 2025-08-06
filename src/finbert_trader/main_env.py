# main_env.py (used for Environment Module test)
# Module: Main
# Purpose: Entry point to test the entire pipeline: ConfigSetup → DataResource → FeatureEngineer → ConfigTrading → StockTradingEnv.
# Design: Instantiates configs, fetches/processes data, prepares RL data, sets up trading env, and simulates reset/step for verification.
# Linkage: Calls all developed modules in sequence; upstream rl_data feeds into Env; handles exceptions for robustness.
# Extensibility: Custom configs for overrides; easy to add RL training or backtest modules later.
# Robustness: Try-except blocks; checks for empty outputs; logs detailed progress including shapes/values for debugging.

import warnings
import pandas as pd
import logging
from datetime import datetime
import numpy as np  # For action simulation in Env test

from finbert_trader.config_setup import ConfigSetup  # Updated import: renamed from config
from finbert_trader.config_trading import ConfigTrading  # Trading config for Env/Agent
from finbert_trader.data.data_resource import DataResource
from finbert_trader.preprocessing.feature_engineer import FeatureEngineer
from finbert_trader.environment.stock_trading_env import StockTradingEnv  # RL environment module

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
# Setup logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(custom_setup_config=None, custom_trading_config=None):
    """
    Run the full pipeline for testing, including new trading env.
    Input: custom_setup_config (dict, optional) for upstream overrides; custom_trading_config (dict, optional) for downstream.
    Output: Prints/logs results; returns env instance for further use.
    Logic: Initialize setups → Fetch/process data → Prepare RL data → Init trading config → Setup Env → Simulate reset/step.
    Robustness: Catch exceptions at each step; validate outputs (e.g., non-empty rl_data, state shapes); log key metrics.
    Extensibility: Decoupled configs allow independent tuning; Env simulation can extend to full episodes or agent training.
    """
    try:
        # Step 1: Initialize upstream configuration (data/feature)
        setup_config = ConfigSetup(custom_setup_config)
        logging.info("Upstream configuration initialized with symbols: {}".format(setup_config.symbols))
        
        # Step 2: Fetch data using DataResource
        dr = DataResource(setup_config)
        stock_data_dict = dr.fetch_stock_data()
        if not stock_data_dict:
            raise ValueError("No stock data fetched")
        logging.info("Fetched stock data for {} symbols".format(len(stock_data_dict)))
        
        news_chunks_gen = dr.load_news_data(save_path='nasdaq_exteral_data.csv')  # Use default path
        # Note: Generator for memory efficiency
        
        # Step 3: Feature engineering
        fe = FeatureEngineer(setup_config)
        
        # Process news chunks to get aggregated sentiment
        sentiment_news_df = fe.process_news_chunks(news_chunks_gen)
        if sentiment_news_df.empty:
            logging.warning("No sentiment news data processed")
        
        # Merge stock and news features
        fused_df = fe.merge_features(stock_data_dict, sentiment_news_df)
        if fused_df.empty:
            raise ValueError("Fused DataFrame is empty")
        logging.info("Fused DataFrame shape: {}".format(fused_df.shape))
        
        # Normalize features
        normalized_df = fe.normalize_features(fused_df)
        logging.info("Normalized DataFrame shape: {}".format(normalized_df.shape))
        
        # Prepare RL data (list of windows)
        rl_data = fe.prepare_rl_data(normalized_df)
        if not rl_data:
            raise ValueError("No RL data prepared")
        logging.info("Prepared {} RL data windows".format(len(rl_data)))
        
        # Step 4: Split RL data (use train for Env test, or full if no split)
        splits = fe.split_rl_data(rl_data)
        if setup_config.k_folds and setup_config.k_folds > 1:
            train_data, _ = splits[0]  # Use first fold's train for simplicity
            logging.info("Using first cross-validation fold's train for Env test")
        else:
            train_data, _ = splits
        logging.info("Using {} train windows for Env test".format(len(train_data)))
        
        # Step 5: Initialize downstream trading configuration
        trading_config = ConfigTrading(custom_trading_config, upstream_config=setup_config)
        logging.info("Trading configuration initialized with initial_cash: {}".format(trading_config.initial_cash))
        
        # Step 6: Setup StockTradingEnv with train rl_data and trading config
        env = StockTradingEnv(trading_config, train_data, env_type='train')  # Pass list of rl_data dicts
        logging.info("StockTradingEnv initialized with state_dim: {}".format(env.state_dim))
        
        # Step 7: Simulate Env interaction for test (reset and one step)
        state, _ = env.reset()
        logging.info("Env reset: Initial state shape {}".format(state.shape))
        
        # Simulate a random action (e.g., buy: 0.5)
        action = np.array([0.5])  # Action space: [-1,1] for sell/hold/buy
        next_state, reward, terminated, truncated, info = env.step(action)
        logging.info("Env step: Next state shape {}, Reward: {}, Terminated: {}, Info: {}".format(next_state.shape, reward, terminated, info))
        
        if terminated or truncated:
            logging.warning("Env terminated/truncated after one step; check window_size")
        
        return env
    
    except Exception as e:
        logging.error("Pipeline failed: {}".format(e))
        raise

if __name__ == "__main__":
    # Example custom configs for testing (small scale: one symbol, recent dates)
    custom_setup = {
        'symbols': ['AAPL'],
        'start': '2023-01-01',  # Recent start to reduce data volume
        'end': '2025-07-22',
        'chunksize': 10000,  # Smaller chunks for testing
        'k_folds': None  # Set to 5 for cross-val if needed
    }
    custom_trading = {
        'initial_cash': 100000,  # Override default for test
        'transaction_cost': 0.001
    }
    run_pipeline(custom_setup, custom_trading)
    logging.info("Pipeline test completed successfully!")