# Original location in main_backtest.py: Replace the entire file with this multi-stock updated version

# main_backtest.py

# Module: Main
# Purpose: Entry point to test the entire pipeline: ConfigSetup → DataResource → FeatureEngineer → ConfigTrading 
#          → StockTradingEnv → TradingAgent → Backtest.
# Design: Orchestrates all modules; supports experiment modes; runs globally for multi-stock portfolio.
# Linkage: Upstream data split by modes/dates; TradingAgent trains per mode; Backtest evaluates and compares.
# Extensibility: Custom configs for overrides; exper_mode for new comparisons; easy add metrics.
# Robustness: Try-except; checks empty data; logs progress/results; ensures results_cache created.
# Updates: Removed per-symbol loop for true multi-stock processing; global fused_df and env for portfolio optimization, 
#          reference from FinRL_DeepSeek (4.3: aggregate multi-stock); increased total_timesteps for better convergence;
#          force_train=True for complete test; integrated risk_mode=True.

import logging
from datetime import datetime
import numpy as np

from finbert_trader.config_setup import ConfigSetup
from finbert_trader.config_trading import ConfigTrading
from finbert_trader.data.data_resource import DataResource
from finbert_trader.preprocessing.feature_engineer import FeatureEngineer
from finbert_trader.environment.stock_trading_env import StockTradingEnv
from finbert_trader.agent.trading_agent import TradingAgent
from finbert_trader.backtest.trading_backtest import Backtest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(custom_setup_config=None, custom_trading_config=None, force_train=False):
    """
    Run full pipeline: generate data per mode, train agents, backtest, produce results.
    
    Input:
        - custom_setup_config (dict, optional)
        - custom_trading_config (dict, optional)
        - force_train (bool): if True, always retrain models

    Output:
        - Dict of backtest results per mode; generates files in results_cache.

    Logic:
        - Init configs → Fetch data → Generate exper data → Train per mode (with cache check) → Backtest/compare → Produce env.

    Robustness:
        - Catch exceptions; validate data non-empty; log per mode.

    Extensibility:
        - exper_mode dict allows adding new modes; backtest extensible for metrics; force_train for override.

    Updates:
        - Process all symbols globally for multi-stock portfolio;
        - Pass empty symbol to Agent/Backtest for unified paths;
        - Results returned as global summary.
    """
    try:
        # Step 1: Init upstream config
        setup_config = ConfigSetup(custom_setup_config)
        logging.info(f"Main - ConfigSetup initialized with symbols: {setup_config.symbols}")
        
        # Global processing for multi-stock (removed per-symbol loop)
        logging.info(f"Main - Processing all symbols globally: {setup_config.symbols}")
        
        # Step 2: Fetch data for all symbols
        dr = DataResource(setup_config)
        stock_data_dict = dr.fetch_stock_data()
        if not stock_data_dict:
            raise ValueError("No stock data fetched")
        logging.info(f"Main - Prepared stock data for next step")
        
        # Load news data generator
        news_chunks_gen = dr.load_news_data()
        
        # Step 3: Generate experiment data (multi-stock fused, with news switch/modes)
        fe = FeatureEngineer(setup_config)
        exper_data = fe.generate_experiment_data(stock_data_dict, news_chunks_gen)
        logging.info(f"Main - Generated experiment data for modes: {list(exper_data.keys())}")
        
        # Step 4: Init trading config, inheriting upstream config
        trading_config = ConfigTrading(custom_trading_config, upstream_config=setup_config)
        logging.info(f"Main - ConfigTrading initialized with initial_cash: {trading_config.initial_cash}")
        
        # Step 5: Train agents per mode (multi-stock, with model caching support)
        agent = TradingAgent(trading_config, None, None, symbol='')  # Empty symbol for global multi-stock
        models_paths = agent.train_for_experiment(exper_data, force_train=force_train)
        logging.info(f"Main - Trained/loaded models for modes: {list(models_paths.keys())}")
        
        # Step 6: Run backtest and generate results
        backtest = Backtest(
            trading_config, exper_data, models_paths, fe.fused_dfs, stock_data_dict, symbol=''
        )
        results = backtest.run()
        logging.info(f"Main - Backtest completed, results generated in results_cache")
        
        # Optional: Test StockTradingEnv manually for inspection (multi-stock)
        test_data = exper_data.get('PPO', {}).get('test', [])   # Use actual mode like 'PPO' instead of 'benchmark'
        if test_data:
            test_env = StockTradingEnv(trading_config, test_data, mode='test')
            state, _ = test_env.reset()
            action = np.array([0.5] * len(trading_config.symbols))  # Simulate partial buy
            next_state, reward, terminated, truncated, info = test_env.step(action)
            logging.info(
                f"Main - Test Env: State shape {state.shape}, Reward {reward:.2f}, "
                f"Terminated {terminated}, Portfolio {info['portfolio_value']:.2f}"
            )
        else:
            logging.warning(f"Main - No test data available for environment test")
        
        return results  # Global results returned

    except Exception as e:
        logging.error(f"Main - Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Example custom configs for testing
    # Increased total_timesteps to 2e6 for improved convergence (FinRL_DeepSeek reference)
    custom_setup = {
        'symbols': ['GOOGL', 'AAPL', 'MSFT'],  # Multi-stock for portfolio test
        # Optional: ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'AMD', 'TSLA', 'META']
        'start': '2015-01-01',
        'end': '2023-12-31',
        'train_start_date': '2015-01-01',
        'train_end_date': '2021-12-31',
        'valid_start_date': '2022-01-01',
        'valid_end_date': '2022-12-31',
        'test_start_date': '2023-01-01',
        'test_end_date': '2023-12-31',
        'exper_mode': {
            'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Includes CPPO, aligned with FinRL_DeepSeek
        }
    }

    custom_trading = {
        'initial_cash': 100000,
        'transaction_cost': 0.001,
        'total_timesteps': 2000000  # 2M steps for strategy convergence
    }

    results = run_pipeline(custom_setup, custom_trading, force_train=True)
    logging.info(f"Pipeline test completed successfully with results: {results}")