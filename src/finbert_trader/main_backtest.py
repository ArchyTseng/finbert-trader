# Original location in main_backtest.py: Replace the entire file with this optimized version for testing all modules
# main_backtest.py
# Module: Main
# Purpose: Entry point to test the entire pipeline: ConfigSetup → DataResource → FeatureEngineer → ConfigTrading → StockTradingEnv → TradingAgent → Backtest.
# Design: Orchestrates all modules; supports experiment modes; runs per symbol.
# Linkage: Upstream data split by modes/dates; TradingAgent trains per mode; Backtest evaluates and compares.
# Extensibility: Custom configs for overrides; exper_mode for new comparisons; easy add metrics.
# Robustness: Try-except; checks empty data; logs progress/results; ensures results_cache created.
# Updates: Added multi-symbol testing loop in run_pipeline; pass symbol to Agent/Backtest for unique paths; aggregate results per symbol.
# Updates: Updated exper_mode to {'rl_algorithm': ['PPO', 'CPPO', 'A2C']} for testing risk-sensitive models, reference from FinRL_DeepSeek (4.3: CPPO); increased total_timesteps for better convergence; force_train=True for complete test; integrated risk_mode=True.

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
    Input: custom_setup_config (dict, optional), custom_trading_config (dict, optional), force_train (bool: if True, always train models)
    Output: Dict of backtest results per mode; generates files in results_cache.
    Logic: Init configs → Fetch data → Generate exper data → Train per mode (with cache check) → Backtest/compare → Produce env.
    Robustness: Catch exceptions; validate data non-empty; log per mode.
    Extensibility: exper_mode dict allows adding new modes; backtest extensible for metrics; force_train for override.
    Updates: Loop over symbols; pass symbol to Agent/Backtest for unique paths; aggregate results per symbol.
    """
    try:
        # Step 1: Init upstream config
        setup_config = ConfigSetup(custom_setup_config)
        logging.info(f"Main - ConfigSetup initialized with symbols: {setup_config.symbols}")
        
        all_results = {}  # To store results per symbol
        for symbol in setup_config.symbols:
            logging.info(f"Main - Processing symbol: {symbol}")
            # Temporarily set single symbol for per-symbol run (to avoid mixed data)
            single_symbol_config = setup_config
            single_symbol_config.symbols = [symbol]  # Override for this loop
            
            # Step 2: Fetch data for current symbol
            dr = DataResource(single_symbol_config)
            stock_data_dict = dr.fetch_stock_data()
            if not stock_data_dict:
                raise ValueError(f"No stock data fetched for {symbol}")
            logging.info(f"Main - Prepared {symbol} stock data for next step")
            
            news_chunks_gen = dr.load_news_data()
            
            # Step 3: Generate experiment data (per mode, with news switch/modes)
            fe = FeatureEngineer(single_symbol_config)
            exper_data = fe.generate_experiment_data(stock_data_dict, news_chunks_gen)
            logging.info(f"Main - Generated experiment data for modes: {list(exper_data.keys())} for {symbol}")
            
            # Step 4: Init trading config, inheriting upstream
            trading_config = ConfigTrading(custom_trading_config, upstream_config=single_symbol_config)
            logging.info(f"Main - ConfigTrading initialized with initial_cash: {trading_config.initial_cash} for {symbol}")
            
            # Step 5: Train agents per mode (with cache check via force_train)
            agent = TradingAgent(trading_config, None, None, symbol=symbol)  # Pass symbol for unique path
            models_paths = agent.train_for_experiment(exper_data, force_train=force_train)
            logging.info(f"Main - Trained/loaded models for modes: {list(models_paths.keys())} for {symbol}")
            
            # Step 6: Run backtest and generate results
            backtest = Backtest(trading_config, exper_data, models_paths, fe.fused_dfs, stock_data_dict, symbol=symbol)  # Pass symbol for unique dir
            symbol_results = backtest.run()
            all_results[symbol] = symbol_results
            logging.info(f"Main - Backtest completed for {symbol}, results generated in results_cache/{symbol}")
            
            # Added: Test StockTradingEnv explicitly to verify environment
            test_data = exper_data.get('benchmark', {}).get('test', [])  # Use benchmark mode test data if available
            if test_data:
                test_env = StockTradingEnv(trading_config, test_data, mode='test')
                state, _ = test_env.reset()
                action = np.array([0.5])  # Simulate buy action
                next_state, reward, terminated, truncated, info = test_env.step(action)
                logging.info(f"Main - Test Env for {symbol}: State shape {state.shape}, Reward {reward:.2f}, Terminated {terminated}, Portfolio {info['portfolio_value']:.2f}")
            else:
                logging.warning(f"Main - No test data available for environment test on {symbol}")
        
        return all_results
    
    except Exception as e:
        logging.error(f"Main - Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Example custom configs for testing: increased total_timesteps to 100000 for better model training (original 10000 too small, may cause zero-action/sharpe)
    custom_setup = {
        'symbols': ['GOOGL', 'AAPL', 'MSFT'],  # Multi-stock for portfolio test 
        #  ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'AMD', 'TSLA', 'META']
        'start': '2015-01-01',
        'end': '2023-12-31',
        'train_start_date': '2015-01-01',
        'train_end_date': '2021-12-31',
        'valid_start_date': '2022-01-01',
        'valid_end_date': '2022-12-31',
        'test_start_date': '2023-01-01',
        'test_end_date': '2023-12-31',
        'exper_mode': {'rl_algorithm': ['PPO', 'CPPO', 'A2C']}  # Updated to test CPPO, reference from FinRL_DeepSeek
    }
    custom_trading = {
        'initial_cash': 100000,
        'transaction_cost': 0.001,
        'total_timesteps': 2000000  # Increased for better training to ensure non-zero sharpe/returns in backtest
    }
    results = run_pipeline(custom_setup, custom_trading, force_train=True)  # Force retrain to test updated pipeline
    logging.info(f"Pipeline test completed successfully with results: {results}")
# End of modified code in main_backtest.py