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

# Set log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("Quick Experiment Test Environment Ready!")
print("=" * 50)

# %%
"""
Step 1: Initialize Configuration
Setup basic configuration for quick experiments
"""

# Set experiment configuration
custom_setup_config = {
    'symbols': ['GOOGL', 'AAPL', ''],  # Use few symbols
    'start': '2020-01-01',
    'end': '2022-12-31',
    'train_start_date': '2020-01-01',
    'train_end_date': '2021-12-31',
    'valid_start_date': '2022-01-01',
    'valid_end_date': '2022-06-30',
    'test_start_date': '2022-07-01',
    'test_end_date': '2022-12-31',
    'exper_mode': {
        'rl_algorithm': ['PPO']  # Test single algorithm
    },
    'ind_mode': 'long', # Set indicator time period
    'window_size': 20,  # Initial small window size
    'window_factor': 3,
    'window_extend': 50,
    'save_npz': False,  # Disable saving for quick experiments
    # Cache path config
    'RAW_DATA_DIR': 'raw_data_cache',
    'EXPER_DATA_DIR': 'exper_data_cache',
    'PLOT_CACHE_DIR': 'plot_cache',
    'RESULTS_CACHE_DIR': 'results_cache',
    'EXPERIMENT_CACHE_DIR': 'exper_cache',
    'LOG_SAVE_DIR': 'logs'
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

# Run quick experiment 1
quick_exper_1_results = experiment_scheme.quick_exper_1(['GOOGL', 'AAPL'])

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

# %%
"""
Step 3: Run Quick Experiment 2 - Reward Function Optimization
Test enhanced reward signals and learning efficiency
"""

print("\n" + "="*50)
print("Running Quick Experiment 2: Reward Function Optimization")
print("This will test enhanced reward function parameters...")

# Run quick experiment 2
quick_exper_2_results = experiment_scheme.quick_exper_2(['GOOGL', 'AAPL'])

print("Quick Experiment 2 completed!")

# Compare two experiment results
if 'PPO' in quick_exper_1_results and 'PPO' in quick_exper_2_results:
    metrics_1 = quick_exper_1_results['PPO'].get('metrics', {})
    metrics_2 = quick_exper_2_results['PPO'].get('metrics', {})
    
    print("\nComparison between Quick Experiment 1 and 2:")
    print(f"  CAGR: {metrics_1.get('cagr', 0)*100:.2f}% -> {metrics_2.get('cagr', 0)*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics_1.get('sharpe_ratio', 0):.4f} -> {metrics_2.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {metrics_1.get('max_drawdown', 0)*100:.2f}% -> {metrics_2.get('max_drawdown', 0)*100:.2f}%")

# %%
"""
Step 4: Run Quick Experiment 3 - RL Hyperparameter Tuning
Test optimized RL algorithm parameters
"""

print("\n" + "="*50)
print("Running Quick Experiment 3: RL Hyperparameter Tuning")
print("This will test optimized PPO parameters...")

# Run Experiment 3
quick_exper_3_results = experiment_scheme.quick_exper_3(['GOOGL', 'AAPL'])

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

# %%
"""
Step 5: Run All Quick Experiments in Sequence
Test the sequence execution functionality
"""

print("\n" + "="*50)
print("Running All Quick Experiments in Sequence")
print("This will test the experiment sequence functionality...")

# Run all quick experiments
all_quick_results = run_quick_experiment_sequence(config_setup, ['GOOGL', 'AAPL'])

print("All quick experiments completed!")
print(f"Number of experiments run: {len(all_quick_results)}")
print(f"Experiment names: {list(all_quick_results.keys())}")

# %%
"""
Step 6: Generate Experiment Report and Visualizations
Create comprehensive analysis of the experiments
"""

print("\n" + "="*50)
print("Generating Experiment Report and Visualizations")

# Initial experiment_tracker and experiment_visualizer
experiment_tracker = ExperimentTracker(config_setup.EXPERIMENT_CACHE_DIR)
experiment_visualizer = VisualizeExperiment(config_setup)

# Generate experiment report
try:
    report_path = experiment_tracker.generate_experiment_report()
    print(f"Experiment report generated: {report_path}")
except Exception as e:
    print(f"Could not generate experiment report: {e}")

# Generate visualization figs
try:
    # Collect experiment generation files
    experiment_files = []
    for file in os.listdir(config_setup.EXPERIMENT_CACHE_DIR):
        if file.startswith('experiment_log_') and file.endswith('.json'):
            experiment_files.append(os.path.join(config_setup.EXPERIMENT_CACHE_DIR, file))
    
    if experiment_files:
        print(f"Found {len(experiment_files)} experiment records for visualization")
        
        # Generate integrated report
        comparison_report = experiment_visualizer.generate_experiment_comparison_report(experiment_files)
        print(f"Comparison report generated: {comparison_report}")
        
        # Generate optimization scheme
        optimization_path = experiment_visualizer.generate_optimization_path_visualization(experiment_files)
        print(f"Optimization path visualization generated: {optimization_path}")
        
    else:
        print("No experiment records found for visualization")
        
except Exception as e:
    print(f"Could not generate visualizations: {e}")

# %%
"""
Step 7: Summary and Analysis
Provide final summary of the quick experiments
"""

print("\n" + "="*60)
print("QUICK EXPERIMENT TEST SUMMARY")
print("="*60)

print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Test symbols: {config_setup.symbols}")
print(f"Test period: {config_setup.start} to {config_setup.end}")

# Conclusion of all experiment results
best_performance = None
best_experiment = None

for exp_name, exp_results in [
    ('Quick Experiment 1', quick_exper_1_results),
    ('Quick Experiment 2', quick_exper_2_results),
    ('Quick Experiment 3', quick_exper_3_results)
]:
    if 'PPO' in exp_results:
        metrics = exp_results['PPO'].get('metrics', {})
        cagr = metrics.get('cagr', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        print(f"\n{exp_name}:")
        print(f"  CAGR: {cagr*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        
        # Record optimal performance
        performance_score = cagr * sharpe  # Simple score
        if best_performance is None or performance_score > best_performance:
            best_performance = performance_score
            best_experiment = exp_name

if best_experiment:
    print(f"\nBest performing experiment: {best_experiment}")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY!")
print("Check the following directories for results:")
print(f"  - Experiment logs: {config_setup.EXPERIMENT_CACHE_DIR}")
print(f"  - Plots: {config_setup.PLOT_CACHE_DIR}")
print(f"  - Results: {config_setup.RESULTS_CACHE_DIR}")
print("="*60)

# %%
"""
Optional: Detailed Analysis of Best Performing Experiment
"""

if best_experiment:
    print(f"\nDetailed analysis of {best_experiment}:")
    
    exp_results = None
    if best_experiment == 'Quick Experiment 1':
        exp_results = quick_exper_1_results
    elif best_experiment == 'Quick Experiment 2':
        exp_results = quick_exper_2_results
    elif best_experiment == 'Quick Experiment 3':
        exp_results = quick_exper_3_results
    
    if exp_results and 'PPO' in exp_results:
        detailed_report = exp_results['PPO'].get('detailed_report', {})
        if detailed_report:
            formatted_metrics = detailed_report.get('formatted_metrics', {})
            for category, metrics in formatted_metrics.items():
                print(f"\n{category}:")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value}")
