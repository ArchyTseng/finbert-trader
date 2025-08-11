# -*- coding: utf-8 -*-
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
# main.py
# Main Pipeline: Complete end-to-end workflow for FinBERT-Driven Multi-Stock RL Trading Agent
# Purpose: Orchestrate the full pipeline from data fetching to backtesting with reproducible results
# Design:
# - Modular pipeline execution with clear separation of concerns
# - Configurable experiment modes and parameters
# - Comprehensive logging and error handling
# - Reproducible results with seed management
# Linkage: Integrates all modules (ConfigSetup -> DataResource -> FeatureEngineer -> TradingAgent -> TradingBacktest)
# Robustness: Graceful error handling, validation, and cleanup
# Extensibility: Easy to modify for different experiments or configurations
import logging
import numpy as np
import pandas as pd
import os
import random
import warnings
from datetime import datetime
import json

# %%
os.chdir('/Users/archy/Projects/finbert_trader/')
# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """
    Set random seeds for reproducible results.

    Parameters
    ----------
    seed : int, optional
        Seed value. Default is 42.

    Returns
    -------
    None
        Sets seeds in place for np, random, and torch (if available).
    """
    # Set numpy seed
    np.random.seed(seed)
    # Set python random seed
    random.seed(seed)
    try:
        import torch
        # Set torch seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # Set cuda seeds for GPU reproducibility
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Enforce deterministic behavior in torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Skip if torch not available

# %%
set_random_seeds(42)
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
# Import project modules
from finbert_trader.config_setup import ConfigSetup
from finbert_trader.data.data_resource import DataResource
from finbert_trader.preprocessing.feature_engineer import FeatureEngineer
from finbert_trader.config_trading import ConfigTrading
from finbert_trader.environment.stock_trading_env import StockTradingEnv
from finbert_trader.agent.trading_agent import TradingAgent
from finbert_trader.backtest.trading_backtest import TradingBacktest
from finbert_trader.exper_tracker import ExperimentTracker
from finbert_trader.visualize.visualize_backtest import VisualizeBacktest
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),  # File handler for persistent logs
        logging.StreamHandler()  # Console handler for real-time output
    ]
)

# %%
def execute_pipeline(custom_setup_config, custom_trading_config):
    """
    Execute the complete pipeline with given configurations.

    Parameters
    ----------
    custom_setup_config : dict
        Configuration for data setup and experiment modes
    custom_trading_config : dict
        Configuration for trading parameters

    Returns
    -------
    dict
        Complete pipeline results including all experiments

    Notes
    -----
    - Orchestrates data flow: config → data → features → envs → agent → backtest.
    - Processes each exper_mode in parallel (sequential here, extensible to multiprocessing).
    """
    try:
        # Step 1: Initialize upstream configuration
        logging.info("Main - Step 1: Initializing configuration setup")
        setup_config = ConfigSetup(custom_setup_config)
        logging.info(f"Main - ConfigSetup initialized with symbols: {setup_config.symbols}")
        logging.info(f"Main - Experiment modes: {setup_config.exper_mode}")
        setup_config.load_or_init_features()
        if setup_config._features_initialized:
            logging.info("Main - Loaded features_* and thresholds from cache.")
        else:
            logging.info("Main - No cached features found, will compute in pipeline.")

        # Step 2: Fetch data for all symbols
        logging.info("Main - Step 2: Fetching stock and news data")
        dr = DataResource(setup_config)
        stock_data_dict = dr.fetch_stock_data()
        if not stock_data_dict:
            raise ValueError("No stock data fetched")
        logging.info(f"Main - Prepared stock data for {len(stock_data_dict)} symbols")

        cache_path, filtered_cache_path = dr.cache_path_config()

        # Load news data generator
        news_chunks_gen = dr.load_news_data(cache_path, filtered_cache_path)
        logging.info("Main - News data loaded successfully")

        # Step 3: Generate experiment data (multi-stock fused, with news switch/modes)
        logging.info("Main - Step 3: Generating experiment data with feature engineering")
        fe = FeatureEngineer(setup_config)
        exper_data_dict = fe.generate_experiment_data(stock_data_dict, news_chunks_gen, exper_mode='rl_algorithm')
        logging.info(f"Main - Generated experiment data for modes: {list(exper_data_dict.keys())}")

        # Step 4: Process each experiment mode
        logging.info("Main - Step 4: Processing experiment modes")
        pipeline_results = {}

        for mode_name, mode_data in exper_data_dict.items():
            logging.info(f"Main - Processing mode: {mode_name}")

            # Create trading configuration for this mode
            trading_config = ConfigTrading(
                custom_config=custom_trading_config,
                upstream_config=setup_config,
                model=mode_data.get('model_type', 'PPO')
            )

            # Process training, validation, and testing
            mode_results = process_mode(mode_name, mode_data, trading_config)
            pipeline_results[mode_name] = mode_results

        return pipeline_results, setup_config

    except Exception as e:
        # Log and re-raise
        logging.error(f"Main - Pipeline execution failed: {e}")
        raise

def process_mode(mode_name, mode_data, trading_config):
    """
    Process a single experiment mode (algorithm).

    Parameters
    ----------
    mode_name : str
        Name of the experiment mode (e.g., 'PPO', 'CPPO', 'A2C')
    mode_data : dict
        Data for this mode including train/valid/test splits
    trading_config : ConfigTrading
        Trading configuration for this mode

    Returns
    -------
    dict
        Results for this mode including trained agent and backtest results

    Notes
    -----
    - Creates envs per split, trains agent, saves model, runs backtest, generates report.
    - Eval_freq dynamic to balance computation (min 10000, ~5% timesteps).
    """
    try:
        logging.info(f"Main - Processing mode {mode_name}")

        # Extract data splits
        train_data = mode_data['train']
        valid_data = mode_data['valid']
        test_data = mode_data['test']

        logging.info(f"Main - Mode {mode_name} - Data splits: "
                    f"Train({len(train_data)}), Valid({len(valid_data)}), Test({len(test_data)})")

        # Create environments
        logging.info(f"Main - Mode {mode_name} - Creating trading environments")
        train_env = StockTradingEnv(trading_config, train_data, env_type='train')
        valid_env = StockTradingEnv(trading_config, valid_data, env_type='valid')
        test_env = StockTradingEnv(trading_config, test_data, env_type='test')

        # Create and train agent
        logging.info(f"Main - Mode {mode_name} - Initializing trading agent")
        agent = TradingAgent(trading_config)

        # Training
        logging.info(f"Main - Mode {mode_name} - Starting training")
        trained_model = agent.train(
            train_env=train_env,
            valid_env=valid_env,
            total_timesteps=trading_config.total_timesteps,
            eval_freq=max(10000, trading_config.total_timesteps // 20),  # Dynamic freq for balanced eval
            n_eval_episodes=5
        )

        # Save trained model
        model_save_path = agent.save_model()
        logging.info(f"Main - Mode {mode_name} - Model saved to: {model_save_path}")

        # Backtesting
        logging.info(f"Main - Mode {mode_name} - Starting backtesting")
        backtester = TradingBacktest(trading_config)
        backtest_results = backtester.run_backtest(agent, test_env, record_trades=True)

        # Generate detailed report
        detailed_report = backtester.generate_detailed_report(backtest_results)

        # Save backtest results
        results_save_path = backtester.save_results(backtest_results)
        logging.info(f"Main - Mode {mode_name} - Backtest results saved to: {results_save_path}")

        # Mode results
        mode_results = {
            'model_path': model_save_path,
            'results_path': results_save_path,
            'metrics': backtest_results['metrics'],
            'detailed_report': detailed_report,
            'backtest_results': backtest_results
        }

        # Log key metrics
        metrics = backtest_results['metrics']
        logging.info(f"Main - Mode {mode_name} - Key Results:")
        logging.info(f" CAGR: {metrics.get('cagr', 0)*100:.2f}%")
        logging.info(f" Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logging.info(f" Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        logging.info(f" Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")

        return mode_results

    except Exception as e:
        # Log and re-raise
        logging.error(f"Main - Error processing mode {mode_name}: {e}")
        raise

def generate_final_report(pipeline_results):
    """
    Generate final comprehensive report comparing all experiment modes.

    Parameters
    ----------
    pipeline_results : dict
        Results from all experiment modes

    Returns
    -------
    None
        Logs report, saves csv/json; no return value.

    Notes
    -----
    - Compares key metrics across modes in df.
    - Identifies best by max CAGR.
    - Saves timestamped files for persistence.
    """
    try:
        # Log report generation
        logging.info("Main - Generating final comparison report")

        # Create comparison data
        comparison_data = []
        for mode_name, results in pipeline_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Algorithm': mode_name,
                'CAGR (%)': f"{metrics.get('cagr', 0)*100:.2f}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.4f}",
                'Max Drawdown (%)': f"{metrics.get('max_drawdown', 0)*100:.2f}",
                'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.4f}",
                'Win Rate (%)': f"{metrics.get('win_rate', 0)*100:.2f}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.4f}",
                'Final Asset ($)': f"${metrics.get('final_asset', 0):,.2f}"
            })

        # Create DataFrame for better presentation
        import pandas as pd
        df = pd.DataFrame(comparison_data)

        # Log table
        logging.info("Main - Final Algorithm Comparison:")
        logging.info("\n" + df.to_string(index=False))

        # Save comparison to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = f'results_cache/final_comparison_{timestamp}.csv'
        os.makedirs('results_cache', exist_ok=True)
        df.to_csv(comparison_file, index=False)
        logging.info(f"Main - Comparison report saved to: {comparison_file}")

        # Identify best performer
        best_cagr = max([results['metrics'].get('cagr', 0) for results in pipeline_results.values()])
        best_algorithm = [name for name, results in pipeline_results.items()
                         if results['metrics'].get('cagr', 0) == best_cagr][0]

        logging.info(f"Main - Best performing algorithm: {best_algorithm} (CAGR: {best_cagr*100:.2f}%)")

        # Save detailed summary
        summary_file = f'results_cache/pipeline_summary_{timestamp}.json'
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'algorithms_tested': list(pipeline_results.keys()),
            'best_algorithm': best_algorithm,
            'best_cagr': best_cagr,
            'comparison_data': comparison_data
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logging.info(f"Main - Summary saved to: {summary_file}")

    except Exception as e:
        # Log error
        logging.error(f"Main - Error generating final report: {e}")

def run_batch_experiment():
    """
    Run batch experiment with different configurations.

    This function demonstrates how to run multiple experiments with different settings.

    Returns
    -------
    dict
        Batch results keyed by config name.

    Notes
    -----
    - Configs vary symbols/timesteps for A/B testing.
    - Extensible: Add parallelism for speed.
    """
    try:
        # Log batch start
        logging.info("Main - Starting batch experiment")

        # Different configurations to test
        configurations = [
            {
                'name': 'baseline',
                'symbols': ['GOOGL', 'AAPL'],
                'total_timesteps': 100000
            },
            {
                'name': 'extended',
                'symbols': ['GOOGL', 'AAPL', 'MSFT', 'AMZN'],
                'total_timesteps': 200000
            }
        ]

        batch_results = {}

        for config in configurations:
            # Log current config
            logging.info(f"Main - Running batch experiment: {config['name']}")

            custom_setup_config = {
                'symbols': config['symbols'],
                'start': '2015-01-01',
                'end': '2023-12-31',
                'train_start_date': '2015-01-01',
                'train_end_date': '2021-12-31',
                'valid_start_date': '2022-01-01',
                'valid_end_date': '2022-12-31',
                'test_start_date': '2023-01-01',
                'test_end_date': '2023-12-31',
                'exper_mode': {
                    'rl_algorithm': ['PPO', 'CPPO']  # Subset for faster testing
                }
            }

            custom_trading_config = {
                'initial_cash': 100000,
                'total_timesteps': config['total_timesteps']
            }

            # Run pipeline for this configuration
            results = execute_pipeline(custom_setup_config, custom_trading_config)
            batch_results[config['name']] = results

        # Log completion
        logging.info("Main - Batch experiment completed")
        return batch_results

    except Exception as e:
        # Log and re-raise
        logging.error(f"Main - Batch experiment failed: {e}")
        raise

# Utility functions for common operations
def load_and_evaluate_saved_model(model_path, config_trading, test_env):
    """
    Load a saved model and evaluate it on test environment.

    Parameters
    ----------
    model_path : str
        Path to saved model
    config_trading : ConfigTrading
        Trading configuration
    test_env : StockTradingEnv
        Test environment

    Returns
    -------
    dict
        Evaluation results

    Notes
    -----
    - Uses agent.load_model and backtester.run_backtest for quick eval.
    """
    try:
        # Initialize agent and load model
        agent = TradingAgent(config_trading)
        agent.load_model(model_path)

        # Run backtest
        backtester = TradingBacktest(config_trading)
        results = backtester.run_backtest(agent, test_env)

        return results
    except Exception as e:
        # Log and re-raise
        logging.error(f"Main - Error loading and evaluating model: {e}")
        raise

def run_cross_validation_experiment(setup_config, trading_config, exper_data_dict):
    """
    Run cross-validation experiment.

    Parameters
    ----------
    setup_config : ConfigSetup
        Setup configuration
    trading_config : ConfigTrading
        Trading configuration
    exper_data_dict : dict
        Experiment data

    Returns
    -------
    dict
        Cross-validation results

    Notes
    -----
    - Placeholder: Implement k-fold splits, train per fold, aggregate.
    """
    try:
        # This would implement k-fold cross-validation logic
        # For each fold, create train/valid splits and train models
        # Then aggregate results across folds
        pass
    except Exception as e:
        # Log and re-raise
        logging.error(f"Main - Error in cross-validation experiment: {e}")
        raise

# %%
def main():
    """
    Main pipeline execution function for FinBERT-Driven Multi-Stock RL Trading Agent.

    This function orchestrates the complete end-to-end workflow:
    1. Configuration setup
    2. Data fetching and preprocessing
    3. Feature engineering with FinBERT integration
    4. RL agent training and validation
    5. Backtesting and performance evaluation
    6. Results analysis and reporting

    Returns
    -------
    dict
        Complete pipeline results including all experiments.
    """
    try:
        logging.info("Main - Starting FinBERT-Driven Multi-Stock RL Trading Pipeline")
        logging.info("=" * 80)
        
        # Configuration setup
        custom_setup_config = {
            'symbols': ['GOOGL', 'AAPL'],
            'start': '2015-01-01',
            'end': '2023-12-31',
            'train_start_date': '2015-01-01',
            'train_end_date': '2021-12-31',
            'valid_start_date': '2022-01-01',
            'valid_end_date': '2022-12-31',
            'test_start_date': '2023-01-01',
            'test_end_date': '2023-12-31',
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO', 'A2C']
            }
        }

        custom_trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 200000
        }
        
        # Pipeline execution
        pipeline_results, setup_config = execute_pipeline(custom_setup_config, custom_trading_config)

        # Initialize experiment tracker and visualizer
        et = ExperimentTracker(setup_config)
        vb = VisualizeBacktest(setup_config)
        
        # Generate final report
        generate_final_report(pipeline_results)
        
        # Generate visualizations using new visualizer
        logging.info("Main - Generating visualizations")
        asset_curve_plot = vb.generate_asset_curve_comparison(pipeline_results)
        heatmap_plot = vb.generate_performance_heatmap(pipeline_results)
        
        if asset_curve_plot:
            logging.info(f"Main - Asset curve plot generated: {asset_curve_plot}")
        if heatmap_plot:
            logging.info(f"Main - Performance heatmap generated: {heatmap_plot}")
        
        # Log experiment
        experiment_config = {
            'setup_config': custom_setup_config,
            'trading_config': custom_trading_config,
            'description': 'Initial pipeline execution with baseline parameters'
        }
        
        et.log_experiment(
            experiment_id='baseline_run',
            config_params=experiment_config,
            results=pipeline_results,
            notes='Baseline execution with default parameters'
        )
        
        # Generate experiment report
        report_path = et.generate_experiment_report()
        logging.info(f"Main - Experiment report generated: {report_path}")
        
        logging.info("Main - Pipeline execution completed successfully")
        logging.info("=" * 80)
        
        return pipeline_results
        
    except Exception as e:
        logging.error(f"Main - Pipeline execution failed: {e}")
        raise


# %%
# Main execution
if __name__ == "__main__":
    try:
        # Run main pipeline
        results = main()

        # Optionally run batch experiment
        # batch_results = run_batch_experiment()

        # Log overall success
        logging.info("Main - All experiments completed successfully!")

    except KeyboardInterrupt:
        # Handle user interrupt gracefully
        logging.info("Main - Pipeline interrupted by user")
    except Exception as e:
        # Log fatal error
        logging.error(f"Main - Pipeline failed with error: {e}")
        raise

# %%
