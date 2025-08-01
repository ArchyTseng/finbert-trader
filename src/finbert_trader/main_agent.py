# main_agent.py (used for Agent Module test)
# Purpose: Entry point to test the entire pipeline: ConfigSetup → DataResource → FeatureEngineer → ConfigTrading → StockTradingEnv → TradingAgent.
# Design: Instantiates configs, fetches/processes data, prepares RL data, sets up envs, initializes agent, and simulates training for verification.
# Linkage: Upstream rl_data split into train/val for envs; ConfigTrading inherits upstream; Agent consumes envs for training.
# Extensibility: Custom configs for overrides; easy to add backtest after training.
# Robustness: Try-except blocks; checks for empty outputs; logs detailed progress including shapes/values for debugging.

import warnings
import pandas as pd
import logging
from datetime import datetime
import numpy as np  # For potential action simulation if needed

from finbert_trader.config_setup import ConfigSetup  # Upstream config for data/feature
from finbert_trader.config_trading import ConfigTrading  # Downstream config for trading/Env/Agent
from finbert_trader.data.data_resource import DataResource
from finbert_trader.preprocessing.feature_engineer import FeatureEngineer
from finbert_trader.environment.stock_trading_env import StockTradingEnv  # RL environment module
from finbert_trader.agent.trading_agent import TradingAgent  # New: RL agent module

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
# Setup logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(custom_setup_config=None, custom_trading_config=None):
    """
    Run the full pipeline for testing, including agent training.
    Input: custom_setup_config (dict, optional) for upstream overrides; custom_trading_config (dict, optional) for downstream.
    Output: Prints/logs results; returns agent instance for further use.
    Logic: Initialize upstream config → Fetch/process data → Prepare RL data → Split into train/val → Init trading config with upstream inheritance → Setup train/val envs → Init agent → Simulate train.
    Robustness: Catch exceptions at each step; validate outputs (e.g., non-empty rl_data, env states); log key metrics.
    Extensibility: Decoupled configs allow independent tuning; agent train can extend to full hyperparam search.
    """
    try:
        # Step 1: Initialize upstream configuration (data/feature)
        setup_config = ConfigSetup(custom_setup_config)
        logging.info(f"Upstream configuration initialized with symbols: {setup_config.symbols}")
        
        # Step 2: Fetch data using DataResource
        dr = DataResource(setup_config)
        stock_data_dict = dr.fetch_stock_data()
        if not stock_data_dict:
            raise ValueError("No stock data fetched")
        logging.info(f"Fetched stock data for {len(stock_data_dict)} symbols")
        
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
        logging.info(f"Fused DataFrame shape: {fused_df.shape}")
        
        # Normalize features
        normalized_df = fe.normalize_features(fused_df)
        logging.info(f"Normalized DataFrame shape: {normalized_df.shape}")
        
        # Prepare RL data (list of windows)
        rl_data = fe.prepare_rl_data(normalized_df)
        if not rl_data:
            raise ValueError("No RL data prepared")
        logging.info(f"Prepared {len(rl_data)} RL data windows")
        
        # Step 4: Split RL data into train/valid
        splits = fe.split_rl_data(rl_data)
        if setup_config.k_folds and setup_config.k_folds > 1:
            train_data, valid_data = splits[0]  # Use first fold for simplicity
            logging.info("Using first cross-validation fold for train/valid envs")
        else:
            train_data, valid_data = splits
        logging.info(f"Train data: {len(train_data)} windows, Val data: {len(valid_data)} windows")
        
        # Step 5: Initialize downstream trading configuration, inheriting upstream
        trading_config = ConfigTrading(custom_trading_config, upstream_config=setup_config)
        logging.info(f"Trading configuration initialized with initial_cash: {trading_config.initial_cash}")
        
        # Step 6: Setup train and valid envs
        train_env = StockTradingEnv(trading_config, train_data, mode='train')
        valid_env = StockTradingEnv(trading_config, valid_data, mode='valid')
        logging.info(f"Train Env initialized with state_dim: {train_env.state_dim}")
        logging.info(f"Val Env initialized with state_dim: {valid_env.state_dim}")
        
        # Step 7: Initialize TradingAgent and simulate train
        agent = TradingAgent(trading_config, train_env, valid_env)
        agent.train()  # Simulate training; will use SharpeCallback for monitoring
        logging.info("Agent training simulation completed")
        
        # Optional: Test predict after train
        obs, _ = valid_env.reset()
        action, _ = agent.predict(obs)
        logging.info(f"Agent predict test: Action shape {action.shape}")
        
        return agent
    
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
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
        'initial_cash': 100000,  
        'transaction_cost': 0.001,  
        'total_timesteps': 10000  # Smaller for quick test
    }
    run_pipeline(custom_setup, custom_trading)
    logging.info("Pipeline test completed successfully!")