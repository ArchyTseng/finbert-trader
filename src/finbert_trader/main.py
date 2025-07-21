# main.py
# Module: Main
# Purpose: Entry point to test the entire pipeline: ConfigSetup → DataResource → FeatureEngineer.
# Design: Instantiates config, fetches data, processes features, prepares and splits RL data; logs progress and outputs for verification.
# Linkage: Calls all developed modules in sequence; handles exceptions for robustness.
# Extensibility: Custom config can be modified for different tests; easy to add RL training module later.
# Robustness: Try-except blocks; checks for empty outputs; uses logging for traceability.

import warnings
import pandas as pd
import logging
from datetime import datetime
from finbert_trader.config import ConfigSetup  
from finbert_trader.data.data_resource import DataResource
from finbert_trader.preprocessing.feature_engineer import FeatureEngineer  # Adjusted import based on module names

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
# Setup logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(custom_config=None):
    """
    Run the full pipeline for testing.
    Input: custom_config (dict, optional) to override defaults.
    Output: Prints/logs results; returns splits for further use.
    Logic: Initialize config → Fetch stock and news data → Process features → Normalize → Prepare RL data → Split.
    Robustness: Catch exceptions at each step; check data integrity (e.g., non-empty dfs).
    """
    try:
        # Step 1: Initialize configuration
        config = ConfigSetup(custom_config)
        logging.info("Configuration initialized with symbols: {}".format(config.symbols))
        
        # Step 2: Fetch data using DataResource
        dr = DataResource(config)
        stock_data_dict = dr.fetch_stock_data()
        if not stock_data_dict:
            raise ValueError("No stock data fetched")
        logging.info("Fetched stock data for {} symbols".format(len(stock_data_dict)))
        
        news_chunks_gen = dr.load_news_data(save_path='nasdaq_exteral_data.csv')  # Use default path
        # Note: Generator, so we don't load all at once for memory efficiency
        
        # Step 3: Feature engineering
        fe = FeatureEngineer(config)
        
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
        
        # Split RL data (train/val or folds)
        splits = fe.split_rl_data(rl_data)
        if config.k_folds and config.k_folds > 1:
            logging.info("Cross-validation folds created: {}".format(len(splits)))
        else:
            train, val = splits
            logging.info("Train: {} windows, Val: {} windows".format(len(train), len(val)))
        
        return splits
    
    except Exception as e:
        logging.error("Pipeline failed: {}".format(e))
        raise

if __name__ == "__main__":
    # Example custom config for testing (small scale: one symbol, recent dates to speed up)
    custom = {
        'symbols': ['AAPL'],
        'start': '2023-01-01',  # Recent start to reduce data volume for quick test
        'end': datetime.today().strftime('%Y-%m-%d'),
        'chunksize': 10000,  # Smaller chunks for testing
        'k_folds': None  # Set to 5 for cross-val test if needed
    }
    run_pipeline(custom)
    logging.info("Pipeline test completed successfully!")