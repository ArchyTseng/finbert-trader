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
"""
FinBERT Trading System - Quick Experiment Test
Test pipeline with quick experiments to validate current implementation
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
from src.finbert_trader.exper_scheme import ExperimentScheme, run_quick_experiment_sequence
from src.finbert_trader.exper_tracker import ExperimentTracker
from src.finbert_trader.visualize.visualize_experiment import VisualizeExperiment

def setup_logging():
    # Config logging level
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('quick_experiment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()

print("Quick Experiment Test Environment Ready!")
print("=" * 50)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Step 1: Initialize Configuration
Setup basic configuration for quick experiments
"""

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

# Set experiment configuration
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
        'rl_algorithm': ['PPO']  # Test single algorithm
    },
    'window_size': 50,  # Initial small window size
    'window_factor': 2,
    'window_extend': 50,
    'smooth_window_size': 10,
    'plot_feature_visualization': False,
    'save_npz': True,  # Disable saving for quick experiments
    'use_symbol_name': True,
    'force_process_news': False,
    'force_fuse_data': False,
    'force_normalize_features': True,    # Ensure normalize target columns
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

# Initial config_setup for experiment
config_setup = ConfigSetup(custom_setup_config)
print("Configuration initialized successfully!")
print(f"Symbols: {config_setup.symbols}")
print(f"Time period: {config_setup.start} to {config_setup.end}")

# %%
"""
Step 2: Run Quick Experiment 1 - Basic Parameter Validation
Test basic pipeline functionality with minimal parameters
"""

# Initial experiment_scheme
experiment_scheme = ExperimentScheme(config_setup)

print("Running Quick Experiment 1: Basic Parameter Validation")
print("This will test basic pipeline functionality...")
print(f"Experiment symbols: {experiment_scheme.symbols}")
print(f"Filter indicators: {experiment_scheme.filter_ind}")

# Run quick experiment 1
quick_exper_1_results = experiment_scheme.quick_exper_1()

print("Quick Experiment 1 completed!")
print(f"Results keys: {list(quick_exper_1_results.keys())}")

# View experiment results metrics
if 'PPO' in quick_exper_1_results:
    metrics = quick_exper_1_results['PPO'].get('metrics', {})
    print("\nKey Metrics from Quick Experiment 1:")
    print(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
    print(f"  Final Asset: ${metrics.get('final_asset', 0):,.2f}")
    
    # Benchmark comparison metrics
    if 'benchmark_cagr' in metrics:
        print(f"  Benchmark CAGR: {metrics.get('benchmark_cagr', 0)*100:.2f}%")
        print(f"  Information Ratio: {metrics.get('information_ratio', 0):.4f}")
        print(f"  Alpha: {metrics.get('alpha', 0)*100:.2f}%")

# %%
"""
Step 3: Run Quick Experiment 2 - Reward Function Optimization
Test enhanced reward signals and learning efficiency
"""

print("\n" + "="*50)
print("Running Quick Experiment 2: Reward Function Optimization")
print("This will test enhanced reward function parameters...")

# Run quick experiment 2
quick_exper_2_results = experiment_scheme.quick_exper_2()

print("Quick Experiment 2 completed!")

# Compare two experiment results
if 'PPO' in quick_exper_1_results and 'PPO' in quick_exper_2_results:
    metrics_1 = quick_exper_1_results['PPO'].get('metrics', {})
    metrics_2 = quick_exper_2_results['PPO'].get('metrics', {})
    
    print("\nComparison between Quick Experiment 1 and 2:")
    print(f"  CAGR: {metrics_1.get('cagr', 0)*100:.2f}% -> {metrics_2.get('cagr', 0)*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics_1.get('sharpe_ratio', 0):.4f} -> {metrics_2.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {metrics_1.get('max_drawdown', 0)*100:.2f}% -> {metrics_2.get('max_drawdown', 0)*100:.2f}%")
    
    # Benchmark comparison
    if 'benchmark_cagr' in metrics_1 and 'benchmark_cagr' in metrics_2:
        print(f"  Benchmark CAGR: {metrics_1.get('benchmark_cagr', 0)*100:.2f}% -> {metrics_2.get('benchmark_cagr', 0)*100:.2f}%")
        print(f"  Information Ratio: {metrics_1.get('information_ratio', 0):.4f} -> {metrics_2.get('information_ratio', 0):.4f}")

# %%
"""
Step 4: Run Quick Experiment 3 - RL Hyperparameter Tuning
Test optimized RL algorithm parameters
"""

print("\n" + "="*50)
print("Running Quick Experiment 3: RL Hyperparameter Tuning")
print("This will test optimized PPO parameters...")

# Run Experiment 3
quick_exper_3_results = experiment_scheme.quick_exper_3()

print("Quick Experiment 3 completed!")

# Compare Three experiment results
print("\nComparison of all three quick experiments:")
experiments = [
    ('Quick Exp 1', quick_exper_1_results),
    ('Quick Exp 2', quick_exper_2_results),
    ('Quick Exp 3', quick_exper_3_results)
]

for exp_name, exp_results in experiments:
    if 'PPO' in exp_results:
        metrics = exp_results['PPO'].get('metrics', {})
        print(f"{exp_name}:")
        print(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        
        # Benchmark metrics
        if 'benchmark_cagr' in metrics:
            print(f"  Benchmark CAGR: {metrics.get('benchmark_cagr', 0)*100:.2f}%")
            print(f"  Information Ratio: {metrics.get('information_ratio', 0):.4f}")

# %% [markdown]
# """
# Step 5: Run All Quick Experiments in Sequence
# Test the sequence execution functionality
# """
#
# print("\n" + "="*50)
# print("Running All Quick Experiments in Sequence")
# print("This will test the experiment sequence functionality...")
#
# # Run all quick experiments
# all_quick_results = run_quick_experiment_sequence(config_setup, ['GOOGL', 'AAPL'])
#
# print("All quick experiments completed!")
# print(f"Number of experiments run: {len(all_quick_results)}")
# print(f"Experiment names: {list(all_quick_results.keys())}")

# %% [markdown]
# """
# Step 6: Generate Experiment Report and Visualizations
# Create comprehensive analysis of the experiments
# """
#
# print("\n" + "="*50)
# print("Generating Experiment Report and Visualizations")
#
# # Initial experiment_tracker and experiment_visualizer
# experiment_tracker = ExperimentTracker(config_setup.EXPERIMENT_CACHE_DIR)
# experiment_visualizer = VisualizeExperiment(config_setup)
#
# # Generate experiment report
# try:
#     report_path = experiment_tracker.generate_experiment_report()
#     print(f"Experiment report generated: {report_path}")
# except Exception as e:
#     print(f"Could not generate experiment report: {e}")
#
# # Generate visualization figs
# try:
#     # Collect experiment generation files
#     experiment_files = []
#     for file in os.listdir(config_setup.EXPERIMENT_CACHE_DIR):
#         if file.startswith('experiment_log_') and file.endswith('.json'):
#             experiment_files.append(os.path.join(config_setup.EXPERIMENT_CACHE_DIR, file))
#     
#     if experiment_files:
#         print(f"Found {len(experiment_files)} experiment records for visualization")
#         
#         # Generate integrated report
#         comparison_report = experiment_visualizer.generate_experiment_comparison_report(experiment_files)
#         print(f"Comparison report generated: {comparison_report}")
#         
#         # Generate benchmark comparison report
#         benchmark_report = experiment_visualizer.generate_experiment_comparison_with_benchmark(
#             experiment_files, 
#             benchmark_name='Nasdaq-100'
#         )
#         print(f"Benchmark comparison report generated: {benchmark_report}")
#         
#         # Generate optimization scheme
#         optimization_path = experiment_visualizer.generate_optimization_path_visualization(experiment_files)
#         print(f"Optimization path visualization generated: {optimization_path}")
#         
#         # Generate parameter impact analysis
#         parameter_analysis = experiment_visualizer.generate_parameter_impact_analysis(
#             experiment_files, 
#             ['trading_config.reward_scaling', 'trading_config.cash_penalty_proportion']
#         )
#         print(f"Parameter impact analysis generated: {parameter_analysis}")
#         
#     else:
#         print("No experiment records found for visualization")
#         
# except Exception as e:
#     print(f"Could not generate visualizations: {e}")

# %% [markdown]
# """
# Step 7: Summary and Analysis
# Provide final summary of the quick experiments
# """
#
# print("\n" + "="*60)
# print("QUICK EXPERIMENT TEST SUMMARY")
# print("="*60)
#
# print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Test symbols: {config_setup.symbols}")
# print(f"Test period: {config_setup.start} to {config_setup.end}")
#
# # Conclusion of all experiment results
# best_performance = None
# best_experiment = None
# best_metrics = None
#
# experiments = [
#     ('Quick Experiment 1', quick_exper_1_results),
#     ('Quick Experiment 2', quick_exper_2_results),
#     ('Quick Experiment 3', quick_exper_3_results)
# ]
#
# for exp_name, exp_results in experiments:
#     if 'PPO' in exp_results:
#         metrics = exp_results['PPO'].get('metrics', {})
#         cagr = metrics.get('cagr', 0)
#         sharpe = metrics.get('sharpe_ratio', 0)
#         max_dd = abs(metrics.get('max_drawdown', 0))
#         win_rate = metrics.get('win_rate', 0)
#         
#         # Benchmark related metrics
#         benchmark_cagr = metrics.get('benchmark_cagr', 0)
#         information_ratio = metrics.get('information_ratio', 0)
#         alpha = metrics.get('alpha', 0)
#         
#         print(f"\n{exp_name}:")
#         print(f"  Strategy CAGR: {cagr*100:.2f}%")
#         print(f"  Benchmark CAGR: {benchmark_cagr*100:.2f}%")
#         print(f"  Relative Performance: {(cagr - benchmark_cagr)*100:.2f}%")
#         print(f"  Sharpe Ratio: {sharpe:.4f}")
#         print(f"  Information Ratio: {information_ratio:.4f}")
#         print(f"  Alpha (Annualized): {alpha*100:.2f}%")
#         print(f"  Max Drawdown: {max_dd*100:.2f}%")
#         print(f"  Win Rate: {win_rate*100:.2f}%")
#         
#         # Use more comprehensive scoring considering benchmark performance
#         performance_score = sharpe * (1 - max_dd) + information_ratio  # Enhanced scoring
#         if best_performance is None or performance_score > best_performance:
#             best_performance = performance_score
#             best_experiment = exp_name
#             best_metrics = metrics
#
# if best_experiment:
#     print(f"\n🏆 Best performing experiment: {best_experiment}")
#     print(f"   Performance Score: {best_performance:.4f}")
#
# print("\n" + "="*60)
# print("TEST COMPLETED SUCCESSFULLY!")
# print("Check the following directories for results:")
# print(f"  - Experiment logs: {config_setup.EXPERIMENT_CACHE_DIR}")
# print(f"  - Plots: {config_setup.PLOT_FEATURES_DIR}")
# print(f"  - Results: {config_setup.RESULTS_CACHE_DIR}")
#
# # Enhanced visualization summary
# print("\nGenerated Visualizations:")
# print("  - Comprehensive Experiment Comparison Report")
# print("  - Benchmark Performance Comparison Report")
# print("  - Optimization Path Visualization")
# print("  - Parameter Sensitivity Analysis")
# print("  -  Risk-Return Tradeoff Analysis")
# print("  - Drawdown Comparison Analysis")  
# print("  - Relative Performance Analysis") 
#
# print("="*60)

# %% [markdown]
# """
# Optional: Detailed Analysis of Best Performing Experiment
# """
#
# if best_experiment and best_metrics:
#     print(f"\nDetailed analysis of {best_experiment}:")
#     
#     print(f"\nPerformance Metrics:")
#     print(f"  Total Return: {best_metrics.get('total_return', 0)*100:.2f}%")
#     print(f"  CAGR: {best_metrics.get('cagr', 0)*100:.2f}%")
#     print(f"  Annual Return: {best_metrics.get('annual_return', 0)*100:.2f}%")
#     
#     print(f"\nRisk Metrics:")
#     print(f"  Volatility: {best_metrics.get('volatility', 0)*100:.2f}%")
#     print(f"  Maximum Drawdown: {best_metrics.get('max_drawdown', 0)*100:.2f}%")
#     print(f"  CVaR (5%): {best_metrics.get('cvar_5_percent', 0)*100:.2f}%")
#     
#     print(f"\nRisk-Adjusted Metrics:")
#     print(f"  Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.4f}")
#     print(f"  Sortino Ratio: {best_metrics.get('sortino_ratio', 0):.4f}")
#     print(f"  Calmar Ratio: {best_metrics.get('calmar_ratio', 0):.4f}")
#     
#     # Benchmark comparison metrics
#     print(f"\nBenchmark Comparison Metrics:")
#     print(f"  Benchmark CAGR: {best_metrics.get('benchmark_cagr', 0)*100:.2f}%")
#     print(f"  Alpha (Annualized): {best_metrics.get('alpha', 0)*100:.2f}%")
#     print(f"  Beta: {best_metrics.get('beta', 0):.4f}")
#     print(f"  Information Ratio: {best_metrics.get('information_ratio', 0):.4f}")
#     print(f"  Tracking Error: {best_metrics.get('tracking_error', 0)*100:.2f}%")
#     print(f"  Excess Return: {best_metrics.get('excess_return', 0)*100:.2f}%")
#     
#     print(f"\nTrade Metrics:")
#     print(f"  Profit Factor: {best_metrics.get('profit_factor', 0):.4f}")
#     print(f"  Win Rate: {best_metrics.get('win_rate', 0)*100:.2f}%")
#     print(f"  Max Consecutive Wins: {best_metrics.get('max_consecutive_wins', 0)}")
#     print(f"  Max Consecutive Losses: {best_metrics.get('max_consecutive_losses', 0)}")

# %% [markdown]
# """
# Final Recommendations and Next Steps
# """
#
# print(f"\nRECOMMENDATIONS AND NEXT STEPS:")
# print(f"="*50)
#
# if best_experiment and best_metrics:
#     cagr = best_metrics.get('cagr', 0)
#     sharpe = best_metrics.get('sharpe_ratio', 0)
#     max_dd = abs(best_metrics.get('max_drawdown', 0))
#     information_ratio = best_metrics.get('information_ratio', 0)
#     
#     # Performance assessment
#     if cagr > 0.15:
#         performance_level = "Excellent"
#     elif cagr > 0.10:
#         performance_level = "Good"
#     elif cagr > 0.05:
#         performance_level = "Moderate"
#     else:
#         performance_level = "Needs Improvement"
#     
#     # Risk assessment
#     if max_dd < 0.15:
#         risk_level = "Low"
#     elif max_dd < 0.25:
#         risk_level = "Moderate"
#     else:
#         risk_level = "High"
#     
#     # Risk-adjusted return assessment
#     if sharpe > 1.5:
#         risk_return_level = "Excellent"
#     elif sharpe > 1.0:
#         risk_return_level = "Good"
#     elif sharpe > 0.5:
#         risk_return_level = "Moderate"
#     else:
#         risk_return_level = "Poor"
#     
#     print(f"\nPerformance Assessment: {performance_level}")
#     print(f"  Risk Assessment: {risk_level}")
#     print(f"  Risk-Adjusted Return: {risk_return_level}")
#     
#     # Benchmark comparison
#     if information_ratio > 0.5:
#         benchmark_performance = "Outperforms Benchmark Significantly"
#     elif information_ratio > 0.2:
#         benchmark_performance = "Outperforms Benchmark"
#     elif information_ratio > -0.2:
#         benchmark_performance = "Comparable to Benchmark"
#     else:
#         benchmark_performance = "Underperforms Benchmark"
#     
#     print(f"Benchmark Comparison: {benchmark_performance}")
#     
#     print(f"\nNext Steps:")
#     if performance_level in ["Excellent", "Good"] and risk_level in ["Low", "Moderate"]:
#         print(f"  • Proceed to full experiments with this configuration")
#         print(f"  • Consider expanding to more symbols")
#         print(f"  • Test with additional algorithms")
#     elif performance_level in ["Moderate"] or risk_level in ["High"]:
#         print(f"  • Refine reward function parameters")
#         print(f"  • Adjust risk management settings")
#         print(f"  • Consider different algorithm configurations")
#     else:
#         print(f"  • Revisit feature engineering approach")
#         print(f"  • Review data quality and preprocessing")
#         print(f"  • Consider alternative algorithm selection")
#     
#     print(f"  • Analyze generated visualizations for insights")
#     print(f"  • Review detailed trading analysis reports")
#     print(f"  • Document findings and parameter configurations")
#
# print(f"\nTest execution completed successfully!")
# print(f"Ready for advanced experimentation and analysis!")
