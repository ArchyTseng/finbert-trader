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
import sys
import random
import warnings
from datetime import datetime
import json

# %%
# Generate project root path
if '__file__' in globals():
    project_root = os.path.join(os.path.dirname(__file__), '..')
else:
    # For interactive environments, use current working directory
    project_root = os.path.join(os.getcwd(), '..')

sys.path.insert(0, project_root)

# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
# %%
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
from src.finbert_trader.config_setup import ConfigSetup
from src.finbert_trader.data_resource import DataResource
from src.finbert_trader.features.feature_engineer import FeatureEngineer
from src.finbert_trader.config_trading import ConfigTrading
from src.finbert_trader.stock_trading_env import StockTradingEnv
from src.finbert_trader.trading_agent import TradingAgent
from src.finbert_trader.trading_backtest import TradingBacktest
from src.finbert_trader.exper_tracker import ExperimentTracker
from src.finbert_trader.trading_analysis import analyze_trade_history
from src.finbert_trader.visualize.visualize_backtest import VisualizeBacktest, generate_all_visualizations_with_benchmark

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
        agent.train(
            train_env=train_env,
            valid_env=valid_env,
            total_timesteps=trading_config.total_timesteps,
            eval_freq=max(10000, trading_config.total_timesteps // 20),  # Dynamic freq for balanced eval
            n_eval_episodes=5
        )

        # Save trained model
        model_save_path = agent.save_model()
        logging.info(f"Main - Mode {mode_name} - Model saved to: {model_save_path}")

        # Backtesting with benchmark support - 使用模块中已有的功能
        logging.info(f"Main - Mode {mode_name} - Starting backtesting with benchmark")
        backtester = TradingBacktest(trading_config)
        backtest_results = backtester.run_backtest(agent, test_env, record_trades=True, use_benchmark=True)

        # Generate detailed report
        detailed_report = backtester.generate_detailed_report(backtest_results)

        # Generate trading history analysis
        symbols_list = getattr(trading_config, 'symbols', None)
        initial_value = backtest_results.get('asset_history', [1.0])[0] if backtest_results.get('asset_history') else 1.0
        trading_analy_dict = analyze_trade_history(
            backtest_results.get('trade_history', []), 
            initial_asset_value=initial_value,
            symbols=symbols_list
        )
        # Add trading analysis to detailed report
        detailed_report['trading_analysis'] = trading_analy_dict

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
        logging.info(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
        logging.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logging.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        logging.info(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        
        # Log benchmark metrics if available
        if 'benchmark_cagr' in metrics:
            logging.info(f"  Benchmark CAGR: {metrics.get('benchmark_cagr', 0)*100:.2f}%")
            logging.info(f"  Information Ratio: {metrics.get('information_ratio', 0):.4f}")
            logging.info(f"  Alpha: {metrics.get('alpha', 0)*100:.2f}%")

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
            comparison_row = {
                'Algorithm': mode_name,
                'CAGR (%)': f"{metrics.get('cagr', 0)*100:.2f}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.4f}",
                'Max Drawdown (%)': f"{metrics.get('max_drawdown', 0)*100:.2f}",
                'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.4f}",
                'Win Rate (%)': f"{metrics.get('win_rate', 0)*100:.2f}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.4f}",
                'Final Asset ($)': f"${metrics.get('final_asset', 0):,.2f}"
            }
            
            # 添加基准比较指标
            if 'benchmark_cagr' in metrics:
                comparison_row.update({
                    'Benchmark CAGR (%)': f"{metrics.get('benchmark_cagr', 0)*100:.2f}",
                    'Information Ratio': f"{metrics.get('information_ratio', 0):.4f}",
                    'Alpha (%)': f"{metrics.get('alpha', 0)*100:.2f}"
                })
            
            comparison_data.append(comparison_row)

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

def generate_comprehensive_visualizations(pipeline_results, setup_config):
    """
    Generate comprehensive visualizations including benchmark comparisons using existing modules.

    Parameters
    ----------
    pipeline_results : dict
        Results from all experiment modes
    setup_config : ConfigSetup
        Configuration setup instance

    Returns
    -------
    dict
        Paths to generated visualization files
    """
    try:
        logging.info("Main - Generating comprehensive visualizations with benchmark support")
        
        # 使用模块中已有的增强可视化功能
        visualization_results = generate_all_visualizations_with_benchmark(
            pipeline_results=pipeline_results,
            config_trading=setup_config,
            benchmark_name='Nasdaq-100'
        )
        
        # Log generated visualizations
        for viz_name, viz_path in visualization_results.items():
            if viz_path and os.path.exists(viz_path):
                logging.info(f"Main - {viz_name.replace('_', ' ').title()} generated: {viz_path}")
        
        return visualization_results
        
    except Exception as e:
        logging.error(f"Main - Error generating comprehensive visualizations: {e}")
        # Fallback to basic visualizations
        try:
            from src.finbert_trader.visualize.visualize_backtest import generate_all_visualizations
            basic_viz = generate_all_visualizations(pipeline_results, setup_config)
            logging.info("Main - Generated basic visualizations as fallback")
            return basic_viz
        except Exception as fallback_error:
            logging.error(f"Main - Error generating fallback visualizations: {fallback_error}")
            return {}

def run_performance_analysis(pipeline_results):
    """
    Run detailed performance analysis including benchmark comparisons.

    Parameters
    ----------
    pipeline_results : dict
        Results from all experiment modes

    Returns
    -------
    dict
        Performance analysis results
    """
    try:
        logging.info("Main - Running detailed performance analysis")
        
        analysis_results = {}
        
        for mode_name, results in pipeline_results.items():
            metrics = results.get('metrics', {})
            
            # 基本性能指标
            basic_performance = {
                'cagr': metrics.get('cagr', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0)
            }
            
            # 基准比较指标
            benchmark_performance = {}
            if 'benchmark_cagr' in metrics:
                benchmark_performance = {
                    'benchmark_cagr': metrics.get('benchmark_cagr', 0),
                    'alpha': metrics.get('alpha', 0),
                    'beta': metrics.get('beta', 0),
                    'information_ratio': metrics.get('information_ratio', 0),
                    'tracking_error': metrics.get('tracking_error', 0),
                    'excess_return': metrics.get('excess_return', 0)
                }
            
            # 风险调整后收益
            risk_adjusted = {
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'volatility': metrics.get('volatility', 0)
            }
            
            analysis_results[mode_name] = {
                'basic_performance': basic_performance,
                'benchmark_performance': benchmark_performance,
                'risk_adjusted': risk_adjusted
            }
            
            # Log key findings
            logging.info(f"Main - {mode_name} Performance Analysis:")
            logging.info(f"  Absolute Performance - CAGR: {basic_performance['cagr']*100:.2f}%, Sharpe: {basic_performance['sharpe_ratio']:.4f}")
            if benchmark_performance:
                logging.info(f"  Relative Performance - Alpha: {benchmark_performance['alpha']*100:.2f}%, IR: {benchmark_performance['information_ratio']:.4f}")
        
        return analysis_results
        
    except Exception as e:
        logging.error(f"Main - Error running performance analysis: {e}")
        return {}

# %%
def main():
    """
    Main pipeline execution function for FinBERT-Driven Multi-Stock RL Trading Agent.

    This function orchestrates the complete end-to-end workflow:
    1. Configuration setup
    2. Data fetching and preprocessing
    3. Feature engineering with FinBERT integration
    4. RL agent training and validation
    5. Backtesting and performance evaluation with benchmark comparison
    6. Results analysis and reporting with professional visualizations

    Returns
    -------
    dict
        Complete pipeline results including all experiments.
    """
    try:
        logging.info("Main - Starting FinBERT-Driven Multi-Stock RL Trading Pipeline with Benchmark Comparison")
        logging.info("=" * 80)
        
        symbols = [
            "AMD",
            "SBUX",
            "PYPL",
            "GILD",
            "COST",
            "MU",
            "CMCSA",
            "QCOM"
        ]   # Selected by news data analysis
        
       # Set configuration
        custom_setup_config = {
            'symbols': symbols,
            'start': '2015-01-01',
            'end': '2023-12-31',
            'train_start_date': '2015-01-01',
            'train_end_date': '2019-12-31',
            'valid_start_date': '2020-01-01',
            'valid_end_date': '2021-12-31',
            'test_start_date': '2022-01-01',
            'test_end_date': '2023-12-31',
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Test single algorithm
            },
            'window_size': 50,  # Initial small window size
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': 5,
            'smooth_window_size': 10,
            'plot_feature_visualization': False,
            'save_npz': True,  # Disable saving for quick experiments
            'force_process_news': False,
            'force_fuse_data': False,
            'force_normalize_features': True,    # Ensure normalize target columns
            'use_senti_factor': False,
            'use_risk_factor': False,
            'use_senti_features': False,
            'use_risk_features': False,
            'use_senti_threshold': False,
            'use_risk_threshold': False,
            'use_dynamic_infusion': False,
            'use_dynamic_ind_threshold': True,
            'use_symbol_name': True,
            'filter_ind': [],
            # Cache path config
            'CONFIG_CACHE_DIR': 'config_cache',
            'RAW_DATA_DIR': 'raw_data_cache',
            'PROCESSED_NEWS_DIR': 'processed_news_cache',
            'FUSED_DATA_DIR': 'fused_data_cache',
            'EXPER_DATA_DIR': 'exper_data_cache',
            'PLOT_FEATURES_DIR': 'plot_features_cache',
            'PLOT_NEWS_DIR': 'plot_news_cache',
            'PLOT_EXPER_DIR': 'plot_exper_cache',
            'RESULTS_CACHE_DIR': 'results_cache',
            'EXPERIMENT_CACHE_DIR': 'exper_cache',
            'SCALER_CACHE_DIR': 'scaler_cache',
            'LOG_SAVE_DIR': 'logs',
        }

        custom_trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 100000,  # 减少时间步以加快测试
            'reward_scaling': 1e-3
        }
        
        # Pipeline execution
        pipeline_results, setup_config = execute_pipeline(custom_setup_config, custom_trading_config)

        # Initialize experiment tracker
        et = ExperimentTracker(setup_config)
        
        # Generate final report
        generate_final_report(pipeline_results)
        
        # Run detailed performance analysis
        performance_analysis = run_performance_analysis(pipeline_results)
        
        # Generate comprehensive visualizations with benchmark support - 使用模块功能
        logging.info("Main - Generating visualizations")
        visualization_results = generate_comprehensive_visualizations(pipeline_results, setup_config)
        
        # Log experiment
        experiment_config = {
            'setup_config': custom_setup_config,
            'trading_config': custom_trading_config,
            'description': 'Complete pipeline execution with benchmark comparison'
        }
        
        et.log_experiment(
            experiment_id='complete_pipeline_run',
            config_params=experiment_config,
            results=pipeline_results,
            notes='Complete execution with benchmark comparison and enhanced visualizations'
        )
        
        # Generate experiment report
        report_path = et.generate_experiment_report()
        logging.info(f"Main - Experiment report generated: {report_path}")
        
        # Summary of key results
        logging.info("Main - Pipeline Execution Summary:")
        logging.info("=" * 50)
        for mode_name, results in pipeline_results.items():
            metrics = results['metrics']
            logging.info(f"{mode_name} Results:")
            logging.info(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            logging.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logging.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            if 'benchmark_cagr' in metrics:
                logging.info(f"  Benchmark CAGR: {metrics.get('benchmark_cagr', 0)*100:.2f}%")
                logging.info(f"  Information Ratio: {metrics.get('information_ratio', 0):.4f}")
                logging.info(f"  Alpha: {metrics.get('alpha', 0)*100:.2f}%")
        
        logging.info("Main - Generated Visualizations:")
        for viz_name, viz_path in visualization_results.items():
            if viz_path:
                logging.info(f"  {viz_name.replace('_', ' ').title()}: {viz_path}")
        
        logging.info("Main - Pipeline execution completed successfully")
        logging.info("=" * 80)
        
        return {
            'pipeline_results': pipeline_results,
            'performance_analysis': performance_analysis,
            'visualization_results': visualization_results
        }
        
    except Exception as e:
        logging.error(f"Main - Pipeline execution failed: {e}")
        raise

# %%
def run_extended_experiment():
    """
    Run extended experiment with multiple algorithms and full benchmark comparison.
    
    This function demonstrates the complete workflow with all features enabled.
    """
    try:
        logging.info("Main - Starting Extended Experiment with Full Benchmark Comparison")
        logging.info("=" * 80)

        symbols = [
            "AMD",
            "SBUX",
            "PYPL",
            "GILD",
            "COST",
            "MU",
            "CMCSA",
            "QCOM"
        ]   # Selected by news data analysis
        
       # Set configuration
        custom_setup_config = {
            'symbols': symbols,
            'start': '2015-01-01',
            'end': '2023-12-31',
            'train_start_date': '2015-01-01',
            'train_end_date': '2019-12-31',
            'valid_start_date': '2020-01-01',
            'valid_end_date': '2021-12-31',
            'test_start_date': '2022-01-01',
            'test_end_date': '2023-12-31',
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Test single algorithm
            },
            'window_size': 50,  # Initial small window size
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': 5,
            'smooth_window_size': 10,
            'plot_feature_visualization': False,
            'save_npz': True,  # Disable saving for quick experiments
            'force_process_news': False,
            'force_fuse_data': False,
            'force_normalize_features': True,    # Ensure normalize target columns
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            'use_senti_threshold': True,
            'use_risk_threshold': True,
            'use_dynamic_infusion': True,
            'use_dynamic_ind_threshold': True,
            'use_symbol_name': True,
            'filter_ind': [],
            # Cache path config
            'CONFIG_CACHE_DIR': 'config_cache',
            'RAW_DATA_DIR': 'raw_data_cache',
            'PROCESSED_NEWS_DIR': 'processed_news_cache',
            'FUSED_DATA_DIR': 'fused_data_cache',
            'EXPER_DATA_DIR': 'exper_data_cache',
            'PLOT_FEATURES_DIR': 'plot_features_cache',
            'PLOT_NEWS_DIR': 'plot_news_cache',
            'PLOT_EXPER_DIR': 'plot_exper_cache',
            'RESULTS_CACHE_DIR': 'results_cache',
            'EXPERIMENT_CACHE_DIR': 'exper_cache',
            'SCALER_CACHE_DIR': 'scaler_cache',
            'LOG_SAVE_DIR': 'logs',
        }

        custom_trading_config = {
            'initial_cash': 1000000,  # 更大初始资金
            'total_timesteps': 500000,  # 更长训练时间
            'reward_scaling': 1e-2,
            'cash_penalty_proportion': 0.001,
            'commission_rate': 0.0001
        }
        
        # Pipeline execution
        pipeline_results, setup_config = execute_pipeline(custom_setup_config, custom_trading_config)

        # Initialize experiment tracker
        et = ExperimentTracker(setup_config)
        
        # Generate final report
        generate_final_report(pipeline_results)
        
        # Run detailed performance analysis
        performance_analysis = run_performance_analysis(pipeline_results)
        
        # Generate comprehensive visualizations with benchmark support - 使用模块功能
        logging.info("Main - Generating comprehensive visualizations")
        visualization_results = generate_comprehensive_visualizations(pipeline_results, setup_config)
        
        # Log experiment
        experiment_config = {
            'setup_config': custom_setup_config,
            'trading_config': custom_trading_config,
            'description': 'Extended experiment with full benchmark comparison'
        }
        
        et.log_experiment(
            experiment_id='extended_experiment',
            config_params=experiment_config,
            results=pipeline_results,
            notes='Extended execution with all algorithms and full benchmark analysis'
        )
        
        # Generate experiment report
        report_path = et.generate_experiment_report()
        logging.info(f"Main - Experiment report generated: {report_path}")
        
        # Summary of key results
        logging.info("Main - Extended Experiment Summary:")
        logging.info("=" * 50)
        for mode_name, results in pipeline_results.items():
            metrics = results['metrics']
            logging.info(f"{mode_name} Results:")
            logging.info(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            logging.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logging.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            if 'benchmark_cagr' in metrics:
                logging.info(f"  Benchmark CAGR: {metrics.get('benchmark_cagr', 0)*100:.2f}%")
                logging.info(f"  Information Ratio: {metrics.get('information_ratio', 0):.4f}")
                logging.info(f"  Alpha: {metrics.get('alpha', 0)*100:.2f}%")
                logging.info(f"  Beta: {metrics.get('beta', 0):.4f}")
        
        logging.info("Main - Generated Visualizations:")
        for viz_name, viz_path in visualization_results.items():
            if viz_path:
                logging.info(f"  {viz_name.replace('_', ' ').title()}: {viz_path}")
        
        logging.info("Main - Extended experiment completed successfully")
        logging.info("=" * 80)
        
        return {
            'pipeline_results': pipeline_results,
            'performance_analysis': performance_analysis,
            'visualization_results': visualization_results
        }
        
    except Exception as e:
        logging.error(f"Main - Extended experiment failed: {e}")
        raise

# %%
def benchmark_only_test():
    """
    Quick test to verify benchmark functionality only.
    
    This function tests the Nasdaq-100 benchmark integration without full pipeline.
    """
    try:
        logging.info("Main - Starting Benchmark Functionality Test")
        logging.info("=" * 80)
        
        setup_config = ConfigSetup()

        # Test date range (recent 2 years)
        test_config = {
            'start': '2022-01-01',
            'end': '2023-12-31',
        }

        # Create a simple ConfigTrading instance for testing
        config_trading = ConfigTrading(custom_config=test_config, upstream_config=setup_config)

        # Initialize TradingBacktest
        backtester = TradingBacktest(config_trading)
        
        print("Testing Nasdaq-100 benchmark data fetching...")

        print(f"Fetching Nasdaq-100 data from {config_trading.start} to {config_trading.end}...")

        # Fetch benchmark data
        benchmark_data = backtester._get_nasdaq100_benchmark()
        if benchmark_data is not None and not benchmark_data.empty:
            logging.info("Nasdaq-100 benchmark data fetched successfully!")
            logging.info(f"Data shape: {benchmark_data.shape}")
            logging.info(f"Date range: {benchmark_data.index[0]} to {benchmark_data.index[-1]}")
            logging.info(f"Price range: ${benchmark_data.min():.2f} to ${benchmark_data.max():.2f}")
            
            # 使用模块中已有的功能计算基准收益率
            benchmark_returns = backtester._calculate_benchmark_returns(benchmark_data)
            if len(benchmark_returns) > 0:
                total_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[0] - 1) * 100
                annualized_return = ((1 + total_return/100) ** (252/len(benchmark_returns)) - 1) * 100
                volatility = np.std(benchmark_returns) * np.sqrt(252) * 100
                
                logging.info(f"\nNasdaq-100 Benchmark Performance Metrics:")
                logging.info(f"  Total Return: {total_return:.2f}%")
                logging.info(f"  Annualized Return: {annualized_return:.2f}%")
                logging.info(f"  Annualized Volatility: {volatility:.2f}%")
                logging.info(f"  Sharpe Ratio: {annualized_return/volatility:.4f}")
                
                # 使用模块中已有的可视化功能
                backtester.plot_performance_comparison(
                    results={'asset_history': [benchmark_data.iloc[0]] + list(benchmark_data.values)}, 
                    benchmark_prices=benchmark_data
                )
                
                logging.info("✅ Benchmark functionality test completed successfully!")
                
                return {
                    'benchmark_data': benchmark_data,
                    'benchmark_returns': benchmark_returns
                }
            else:
                logging.error("Could not calculate benchmark returns")
        else:
            logging.error("Failed to fetch Nasdaq-100 benchmark data")
            
    except Exception as e:
        logging.error(f"Main - Benchmark test failed: {e}")
        raise

# %%
# Main execution
if __name__ == "__main__":
    try:
        logging.info("Main - Starting Complete Trading Pipeline with Benchmark Comparison")
        logging.info("=" * 80)
        
        # Option 1: Quick benchmark test
        print("\nChoose execution mode:")
        print("1. Quick Benchmark Test Only")
        print("2. Basic Pipeline with Benchmark")
        print("3. Extended Experiment with Full Analysis")
        
        choice = input("Enter your choice (1-3, default 2): ").strip()
        
        if choice == "1":
            # Run quick benchmark test
            benchmark_results = benchmark_only_test()
            if benchmark_results:
                logging.info("Quick benchmark test completed successfully!")
        elif choice == "3":
            # Run extended experiment
            extended_results = run_extended_experiment()
            logging.info("Extended experiment completed successfully!")
        else:
            # Run basic pipeline (default)
            results = main()
            logging.info("Basic pipeline completed successfully!")
        
        # Log overall success
        logging.info("Main - All experiments completed successfully!")
        logging.info("=" * 80)

    except KeyboardInterrupt:
        # Handle user interrupt gracefully
        logging.info("Main - Pipeline interrupted by user")
    except Exception as e:
        # Log fatal error
        logging.error(f"Main - Pipeline failed with error: {e}")
        raise

# %%
