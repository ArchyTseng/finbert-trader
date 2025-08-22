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

# %% [markdown]
# # FinBERT-Driven Trading System - Experiment Main Pipeline
# This script provides a consolidated and efficient way to run experiments,
# generate visualizations, and perform robustness analysis.

# %%
# --- Step 1: Necessary Import and Logging Configuation ---

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
# --- Step 2: Configuration and Setup ---

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
    'prediction_days': 5,
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

# Initialize Experiment Scheme
try:
    experiment_scheme = ExperimentScheme(config_setup)
    print("ExperimentScheme initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize ExperimentScheme: {e}")
    raise

# %% [markdown]
# ---
# # Quick Experiments Pipeline
# Run all quick experiments (1-4) in sequence, then generate a comprehensive analysis report.

# %%
# --- Step 2: Run All Quick Experiments ---
print("\n" + "="*60)
print("Starting Quick Experiments Pipeline")
print("="*60)

try:
    # This single call replaces the need to run quick_exper_1/2/3/4 individually
    # and then calling run_quick_experiment_sequence.
    quick_experiment_results_and_visualizations = experiment_scheme.run_and_visualize_quick_experiments_sequence(
        symbols=config_setup.symbols
    )
    
    # Extract results
    quick_results = quick_experiment_results_and_visualizations.get('experiment_results', {})
    quick_visualizations = quick_experiment_results_and_visualizations.get('visualizations', {})
    
    if 'error' in quick_experiment_results_and_visualizations:
        print(f"Error during quick experiments: {quick_experiment_results_and_visualizations['error']}")
    else:
        print("\n--- Quick Experiments Completed ---")
        print(f"Number of experiments run: {len(quick_results)}")
        print(f"Experiment names: {list(quick_results.keys())}")
        print("\n--- Generated Quick Experiment Visualizations ---")
        for viz_name, viz_path in quick_visualizations.items():
            print(f"  - {viz_name.replace('_', ' ').title()}: {viz_path}")

except Exception as e:
    logging.error(f"Error in Quick Experiments Pipeline: {e}", exc_info=True)
    print(f"Quick Experiments Pipeline failed: {e}")

# %% [markdown]
# ---
# # Full Experiments Pipeline (Optional)
# Run all full experiments (1-3) in sequence, then generate a comprehensive analysis report.
# This section can be run independently or after the Quick Experiments.

# %%
# --- Step 3: (Optional) Run All Full Experiments ---
run_full_experiments = input("\nDo you want to run the Full Experiments (1-3) now? (y/n): ").strip().lower()

if run_full_experiments in ['y', 'yes']:
    print("\n" + "="*60)
    print("Starting Full Experiments Pipeline")
    print("="*60)

    try:
        # Run all full experiments
        full_experiment_results = experiment_scheme.run_experiment_sequence(
            experiment_names=['full_exper_1', 'full_exper_2', 'full_exper_3'],
            symbols=config_setup.symbols
        )
        
        print("\n--- Full Experiments Completed ---")
        print(f"Number of experiments run: {len(full_experiment_results)}")
        print(f"Experiment names: {list(full_experiment_results.keys())}")

        # --- Generate Visualizations for Full Experiments ---
        print("\nGenerating visualizations for Full Experiments...")
        full_visualizations = experiment_scheme.generate_experiment_visualizations_from_cache()
        
        if 'error' in full_visualizations:
            print(f"Error during full experiment visualization: {full_visualizations['error']}")
        else:
            print("\n--- Generated Full Experiment Visualizations ---")
            for viz_name, viz_path in full_visualizations.items():
                print(f"  - {viz_name.replace('_', ' ').title()}: {viz_path}")

    except Exception as e:
        logging.error(f"Error in Full Experiments Pipeline: {e}", exc_info=True)
        print(f"Full Experiments Pipeline failed: {e}")
else:
    print("\nSkipping Full Experiments Pipeline.")

# %% [markdown]
# ---
# # Robustness Test (Optional)
# Perform a robustness test on a specific experiment to assess its stability.

# %%
# --- Step 4: (Optional) Run Robustness Test ---
run_robustness_test = input("\nDo you want to run a Robustness Test on an experiment? (y/n): ").strip().lower()

if run_robustness_test in ['y', 'yes']:
    print("\n" + "="*60)
    print("Starting Robustness Test")
    print("="*60)
    
    # Let user choose the experiment
    print("Available experiments for robustness test:")
    print("  1. quick_exper_1")
    print("  2. quick_exper_2")
    print("  3. quick_exper_3")
    print("  4. quick_exper_4")
    print("  5. full_exper_1")
    print("  6. full_exper_2")
    print("  7. full_exper_3")
    
    choice_map = {
        '1': ('quick_exper_1', experiment_scheme.quick_exper_1),
        '2': ('quick_exper_2', experiment_scheme.quick_exper_2),
        '3': ('quick_exper_3', experiment_scheme.quick_exper_3),
        '4': ('quick_exper_4', experiment_scheme.quick_exper_4),
        '5': ('full_exper_1', experiment_scheme.full_exper_1),
        '6': ('full_exper_2', experiment_scheme.full_exper_2),
        '7': ('full_exper_3', experiment_scheme.full_exper_3),
    }
    
    choice = input("Enter the number of the experiment to test (e.g., 1): ").strip()
    if choice in choice_map:
        experiment_name, experiment_method = choice_map[choice]
        
        try:
            num_runs_input = input("Enter number of runs for robustness test (default 5): ").strip()
            num_runs = int(num_runs_input) if num_runs_input.isdigit() else 5
            
            print(f"\nRunning robustness test for {experiment_name} ({num_runs} runs)...")
            
            # Run robustness test
            robustness_results = experiment_scheme.run_robustness_test(
                experiment_method=experiment_method,
                num_runs=num_runs,
                run_prefix=f"{experiment_name}_robustness_run"
            )
            
            # Display results
            print("\n" + "-" * 40)
            print("Robustness Test Completed!")
            print("-" * 40)
            
            aggregated_metrics = robustness_results.get('aggregated_metrics', {})
            report_path = robustness_results.get('robustness_report_path', 'N/A')
            viz_path = robustness_results.get('robustness_visualization_path', 'N/A')
            
            print("\n--- Aggregated Robustness Metrics ---")
            import pprint
            pprint.pprint(aggregated_metrics)
            
            print(f"\n--- Generated Files ---")
            print(f"Robustness Report: {report_path}")
            print(f"Robustness Visualization: {viz_path}")
            
            print("\nRobustness test analysis complete. Please review the generated report and visualization.")

        except Exception as e:
            logging.error(f"Error running robustness test for {experiment_name}: {e}", exc_info=True)
            print(f"Robustness test for {experiment_name} failed: {e}")
    else:
        print("Invalid choice. Skipping robustness test.")

else:
    print("\nSkipping Robustness Test.")

# %% [markdown]
# ---
# # Summary
# This script provides a streamlined workflow for your experimentation.
# 1. Run and analyze Quick Experiments.
# 2. Optionally run and analyze Full Experiments.
# 3. Optionally perform a Robustness Test on any experiment.
# All results and visualizations are saved to their respective cache directories.

# %%
print("\n" + "="*60)
print("EXPERIMENT MAIN PIPELINE COMPLETED")
print("="*60)
print(f"Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nPlease check the following directories for outputs:")
print(f"  - Experiment logs: {config_setup.EXPERIMENT_CACHE_DIR}")
print(f"  - Plots: {config_setup.PLOT_EXPER_DIR}")
print(f"  - Results: {config_setup.RESULTS_CACHE_DIR}")
print("="*60)
