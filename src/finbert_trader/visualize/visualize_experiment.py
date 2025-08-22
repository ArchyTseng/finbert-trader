# finbert_trader/visualize/visualize_experiment.py
"""
Visualization Module for FinBERT-Driven Trading System Experiment Analysis

Purpose:
    Generate a single, comprehensive experiment comparison report.
    This module is the central hub for comparing multiple experiments.
    It leverages `visualize_backtest.py` for detailed backtest-specific visualizations.

Notes:
    - This module only generates one report: `experiment_comprehensive_report`.
    - Other reports (`benchmark_comparison`, `optimization_path`, etc.) are considered redundant and have been removed.
    - The core functionality of `visualize_backtest.py` is reused to avoid code duplication.
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import project modules
from .visualize_backtest import VisualizeBacktest, generate_all_visualizations_with_benchmark
from ..config_setup import ConfigSetup


class VisualizeExperiment:
    """
    Class for generating the comprehensive experiment comparison report.

    This class is designed to be simple and focused. Its primary responsibility is to
    orchestrate the creation of a single, unified report that compares multiple experiments.
    It delegates complex visualization tasks to `VisualizeBacktest`.
    """

    def __init__(self, config: ConfigSetup):
        """
        Initialize VisualizeExperiment with unified configuration.

        Parameters
        ----------
        config : ConfigSetup
            Configuration object containing directory paths and other settings.
        """
        self.config = config
        self.plot_exper_dir = getattr(self.config, 'PLOT_EXPER_DIR', 'plot_exper_cache')
        os.makedirs(self.plot_exper_dir, exist_ok=True)
        logging.info(f"VE Module - Initialized VisualizeExperiment with plot directory: {self.plot_exper_dir}")

    def _extract_metrics_from_results(self, results_section: Dict) -> Dict[str, Dict]:
        """
        Extracts metrics for each mode from the 'results' section of an experiment log.

        Parameters
        ----------
        results_section : dict
            The `results` dictionary from an experiment log file.

        Returns
        -------
        dict
            A dictionary {mode_name: metrics_dict}.
        """
        metrics_summary = {}
        reserved_keys = ['model_path', 'results_path', 'detailed_report']
        if isinstance(results_section, dict):
            for mode_name, mode_data in results_section.items():
                if mode_name in reserved_keys or not isinstance(mode_data, dict):
                    continue
                metrics_summary[mode_name] = mode_data.get('metrics', {})
        return metrics_summary

    def _get_nested_value(self, data_dict: Dict, key_path: str, default=None):
        """
        Helper function to safely retrieve a nested value from a dictionary using a dot-separated key path.

        Parameters
        ----------
        data_dict : dict
            The dictionary to search.
        key_path : str
            Dot-separated string representing the path to the desired value.
        default : any, optional
            The default value to return if the key path is not found.

        Returns
        -------
        any
            The value found at the key path, or the default value.
        """
        keys = key_path.split('.')
        current_data = data_dict
        try:
            for key in keys:
                current_data = current_data[key]
            return current_data
        except (KeyError, TypeError):
            return default

    def _plot_performance_comparison_heatmap(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot performance comparison heatmap based on experiment metrics."""
        try:
            metrics_data = []
            # Define metrics to display in the heatmap
            metric_keys = ['cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate']

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                # Read from 'results' instead of 'pipeline_results'
                results_section = record_data.get('results', {})
                metrics_summary = self._extract_metrics_from_results(results_section)

                for mode_name, metrics in metrics_summary.items():
                    row_data = {'Experiment': exp_id, 'Algorithm': mode_name}
                    for key in metric_keys:
                        value = metrics.get(key, np.nan)
                        # Convert percentage-based metrics for better readability in heatmap
                        if key in ['cagr', 'max_drawdown', 'win_rate']:
                            value = value * 100 if not np.isnan(value) else value
                        row_data[key] = value
                    metrics_data.append(row_data)

            if not metrics_data:
                ax.text(0.5, 0.5, 'No Metrics Data Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Comparison Heatmap')
                return

            df = pd.DataFrame(metrics_data)
            
            # Create subplots for each metric in the heatmap
            num_metrics = len(metric_keys)
            cols = 2
            rows = (num_metrics + cols - 1) // cols
            fig_heatmap, axes_heatmap = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
            axes_heatmap = axes_heatmap.flatten() if num_metrics > 1 else [axes_heatmap]

            for i, metric in enumerate(metric_keys):
                if metric not in df.columns:
                    axes_heatmap[i].text(0.5, 0.5, f'Metric {metric} Missing', ha='center', va='center',
                                         transform=axes_heatmap[i].transAxes)
                    axes_heatmap[i].set_title(f'{metric.replace("_", " ").title()}')
                    continue

                # Pivot data for heatmap
                pivot_df = df.pivot(index='Experiment', columns='Algorithm', values=metric)
                
                # Choose colormap based on metric type for better visual interpretation
                cmap = 'viridis' if metric != 'max_drawdown' else 'viridis_r' # Lower DD is better
                if metric == 'sharpe_ratio':
                    cmap = 'RdYlGn' # Higher Sharpe is better

                sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap=cmap, ax=axes_heatmap[i], cbar_kws={'shrink': 0.8})
                axes_heatmap[i].set_title(f'{metric.replace("_", " ").title()}')
                axes_heatmap[i].tick_params(axis='x', rotation=45)
                axes_heatmap[i].tick_params(axis='y', rotation=0)

            # Hide any unused subplots
            for j in range(i + 1, len(axes_heatmap)):
                fig_heatmap.delaxes(axes_heatmap[j])

            plt.tight_layout()
            
            # Save the separate heatmap figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            heatmap_filename = f"performance_heatmap_detailed_{timestamp}.png"
            heatmap_path = os.path.join(self.plot_exper_dir, heatmap_filename)
            fig_heatmap.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close(fig_heatmap) # Close the separate figure

            # Indicate on the main report subplot where to find the heatmap
            ax.text(0.5, 0.5, f'Heatmap Generated\nSee: {os.path.basename(heatmap_path)}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Performance Comparison Heatmap')

        except Exception as e:
            logging.error(f"VE Module - Error in _plot_performance_comparison_heatmap: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error Generating Heatmap:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Performance Comparison Heatmap')

    def _plot_asset_curves_comparison(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot asset curves comparison by leveraging visualize_backtest.py."""
        try:
            # Try use visualize_backtest.py class function
            pipeline_results_for_viz = {}

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                # Get  results_path from .json files
                results_path = record_data.get('results', {}).get('PPO', {}).get('results_path')  # 假设 PPO 是主要模式

                if results_path and os.path.exists(results_path):
                    # Load backtest_results from .pkl files
                    with open(results_path, 'rb') as f:
                        backtest_data = pickle.load(f) 

                    # Add to pipeline_results 
                    pipeline_results_for_viz[exp_id] = backtest_data

            # Try use visualize_backtest.py class function
            vb = VisualizeBacktest(self.config)
            benchmark_data = None
            if pipeline_results_for_viz:
                first_exp_data = next(iter(pipeline_results_for_viz.values()))
                benchmark_data = first_exp_data.get('benchmark_prices_with_date', None)

            # Generate asset curve comparison and save
            plot_path = vb.generate_asset_curve_comparison(pipeline_results_for_viz, benchmark_data=benchmark_data)

            # Set axis
            ax.text(0.5, 0.5, f'Asset Curves Plot\nGenerated at: {os.path.basename(plot_path)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Normalized Asset Curves Comparison')
            logging.info(f"VE Module - Asset curves comparison plot generated: {plot_path}")
        except Exception as e:
            logging.error(f"VE Module - Error in _plot_asset_curves_comparison: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error Generating Asset Curves:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Normalized Asset Curves Comparison')

    def _plot_cumulative_excess_return_vs_benchmark(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot cumulative excess return vs benchmark by leveraging visualize_backtest.py."""
        try:
            # Try use visualize_backtest.py class function
            # Build pipeline_results_for_viz dict and fetch benchmark_returns
            pipeline_results_for_viz = {}
            benchmark_returns_array = None

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                results_path = record_data.get('results', {}).get('PPO', {}).get('results_path')

                if results_path and os.path.exists(results_path):
                    with open(results_path, 'rb') as f:
                        backtest_data = pickle.load(f)
                    pipeline_results_for_viz[exp_id] = backtest_data
                    if benchmark_returns_array is None:
                        benchmark_returns_array = backtest_data.get('benchmark_returns', None)

            # Try use visualize_backtest.py class function
            vb = VisualizeBacktest(self.config)
            plot_path = vb.generate_benchmark_relative_performance(pipeline_results_for_viz, benchmark_returns=benchmark_returns_array)

            ax.text(0.5, 0.5, f'Cumulative Excess Return Plot\nGenerated at: {os.path.basename(plot_path)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cumulative Excess Return vs QQQ')
            logging.info(f"VE Module - Cumulative excess return plot generated: {plot_path}")
        except Exception as e:
            logging.error(f"VE Module - Error in _plot_cumulative_excess_return_vs_benchmark: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error Generating Excess Return:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Cumulative Excess Return vs QQQ')

    def _plot_parameter_sensitivity_2d(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot parameter sensitivity analysis for two parameters (reward_scaling, cash_penalty_proportion)."""
        try:
            param_values = []
            metric_values = []  # Sharpe Ratio
            labels = []

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                # Get target parameters
                reward_scaling = self._get_nested_value(record_data, 'config_params.trading_config.reward_scaling')
                cash_penalty = self._get_nested_value(record_data, 'config_params.trading_config.cash_penalty_proportion')
                
                # Get Sharpe Ratio
                metrics_summary = self._extract_metrics_from_results(record_data.get('results', {}))
                sharpe_ratios = [metrics.get('sharpe_ratio', np.nan) for metrics in metrics_summary.values()]
                valid_sharpes = [s for s in sharpe_ratios if not np.isnan(s)]
                avg_sharpe = np.mean(valid_sharpes) if valid_sharpes else np.nan

                if not np.isnan(reward_scaling) and not np.isnan(cash_penalty) and not np.isnan(avg_sharpe):
                    param_values.append((reward_scaling, cash_penalty))
                    metric_values.append(avg_sharpe)
                    labels.append(exp_id)

            if param_values and metric_values:
                # Plot scatter
                scatter = ax.scatter([p[0] for p in param_values], [p[1] for p in param_values], c=metric_values, cmap='viridis', alpha=0.7)
                ax.set_xlabel('Reward Scaling')
                ax.set_ylabel('Cash Penalty Proportion')
                ax.set_title('Parameter Sensitivity: reward_scaling vs cash_penalty_proportion')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
                
                # Add label
                for i, txt in enumerate(labels):
                    ax.annotate(txt, (param_values[i][0], param_values[i][1]), fontsize=8, ha='right')
            else:
                ax.text(0.5, 0.5, 'Insufficient Data for Parameter Sensitivity', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title('Parameter Sensitivity')
        except Exception as e:
            logging.error(f"VE Module - Error in _plot_parameter_sensitivity_2d: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error in Parameter Sensitivity:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Parameter Sensitivity')

    def _plot_algorithm_radar_chart(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot radar chart for algorithm performance comparison.
        This chart compares the average performance of different algorithms (modes)
        across multiple experiments on a set of key metrics.
        """
        try:
            # Define the target metrics and label for radar chart
            # Metrics names
            categories = ['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
            # Metrics keys
            metric_keys = ['cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            # Build metrics func
            transformations = [
                lambda x: x * 100,           # CAGR: 转换为百分比
                lambda x: x,                 # Sharpe: 保持不变
                lambda x: -x * 100,          # Max Drawdown: 取负值并转为百分比 (风险越小，负值越大)
                lambda x: x * 100            # Win Rate: 转换为百分比
            ]

            # Collect target data for each algorithm
            # Structure: {algo_name: {'values': [[exp1_metrics], [exp2_metrics], ...], 'labels': [exp_id1, exp_id2, ...]}}
            algo_data = {}

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                # Get data from 'results' 
                results_section = record_data.get('results', {})
                metrics_summary = self._extract_metrics_from_results(results_section)

                # Traverse all algorithm mode
                for mode_name, metrics in metrics_summary.items():
                    # Get and convert data
                    normalized_values = []
                    for key, transform in zip(metric_keys, transformations):
                        raw_value = metrics.get(key, np.nan)
                        # Ensure data type
                        if not isinstance(raw_value, (int, float)) or np.isnan(raw_value):
                            normalized_value = np.nan
                        else:
                            try:
                                normalized_value = transform(raw_value)
                            except Exception:
                                normalized_value = np.nan
                        normalized_values.append(normalized_value)
                    
                    # Add to algo_data
                    if mode_name not in algo_data:
                        algo_data[mode_name] = {'values': [], 'labels': []}
                    algo_data[mode_name]['values'].append(normalized_values)
                    algo_data[mode_name]['labels'].append(exp_id)

            # Check data
            if not algo_data:
                ax.text(0.5, 0.5, 'No Algorithm Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Algorithm Performance Radar Chart')
                return

            # Calculate average performance for each algorithm
            # Structure: {algo_name: [avg_cagr, avg_sharpe, avg_drawdown, avg_win_rate]}
            avg_algo_data = {}
            all_valid_values = [] # For global Y axis
            for algo_name, data in algo_data.items():
                values_list = data['values'] # List of lists
                if not values_list:
                    continue
                
                # Convert to numpy
                try:
                    values_array = np.array(values_list, dtype=float) # Ensure float dtype
                    # Calculate mean value
                    avg_values = np.nanmean(values_array, axis=0)
                    avg_algo_data[algo_name] = avg_values.tolist()
                    
                    # Collect valid values
                    valid_values = values_array[~np.isnan(values_array)]
                    all_valid_values.extend(valid_values)
                    
                except (ValueError, TypeError) as e:
                    logging.warning(f"VE Module - _plot_algorithm_radar_chart - Error processing data for {algo_name}: {e}")
                    continue 

            if not avg_algo_data:
                ax.text(0.5, 0.5, 'No Valid Algorithm Data to Plot', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Algorithm Performance Radar Chart')
                return

            # Set angles for radar chart
            num_vars = len(categories)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1] 

            # Plot radar chart
            # Plot polygon for each algorithm
            for algo_name, avg_values in avg_algo_data.items():
                # Append the first value to the tail
                values_to_plot = avg_values + avg_values[:1]
                ax.plot(angles, values_to_plot, linewidth=2, linestyle='solid', label=algo_name)
                ax.fill(angles, values_to_plot, alpha=0.25)

            # Plot configuration
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Set the range of Y axis
            # Calculate the upper limit
            if all_valid_values:
                max_val = np.max(all_valid_values)
                min_val = np.min(all_valid_values)
                y_range = max_val - min_val
                margin = y_range * 0.1 if y_range > 0 else 1
                # Ensure Y axis range valid
                y_min = min_val - margin
                y_max = max_val + margin
                # Ensure range safe
                y_min_final = min(y_min, 0)
                y_max_final = max(y_max, 0)
                
                try:
                    ax.set_ylim(y_min_final, y_max_final)
                except ValueError as ve:
                    logging.warning(f"VE Module - _plot_algorithm_radar_chart - Failed to set ylim: {ve}. Using default.")
                    ax.set_ylim(-10, 10)
            else:
                # Fallback to a default range
                ax.set_ylim(-1, 1)

            ax.set_title('Algorithm Performance Radar Chart')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

            logging.info("VE Module - Algorithm performance radar chart plotted successfully.")

        except Exception as e:
            logging.error(f"VE Module - Error in _plot_algorithm_radar_chart: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error in Algorithm Radar Chart:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Algorithm Performance Radar Chart')

    def _plot_experiment_timeline(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot experiment execution timeline."""
        try:
            timestamps = []
            experiment_ids = []

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                # Use the timestamp from the log file
                timestamp_str = record_data.get('timestamp', '')
                
                # Attempt to parse the timestamp string into a datetime object
                # Adjust the format string if your timestamp format is different
                try:
                    # Assuming timestamp is in format "YYYYMMDD_HHMMSS"
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    try:
                        # Fallback, if it's a full datetime string like "2025-08-21 20:47:02"
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        # If parsing fails, use file modification time or current time
                        if isinstance(record, str) and os.path.exists(record):
                            file_mtime = os.path.getmtime(record)
                            timestamp = datetime.fromtimestamp(file_mtime)
                        else:
                            timestamp = datetime.now()
                
                timestamps.append(timestamp)
                experiment_ids.append(exp_id)

            if timestamps:
                # Sort by timestamp to ensure correct order on the plot
                sorted_data = sorted(zip(timestamps, experiment_ids))
                sorted_timestamps, sorted_ids = zip(*sorted_data)
                
                ax.plot(sorted_timestamps, range(len(sorted_timestamps)), 'o-')
                ax.set_yticks(range(len(sorted_ids)))
                ax.set_yticklabels(sorted_ids)
                ax.set_xlabel('Execution Time')
                ax.set_title('Experiment Execution Timeline')
                ax.grid(True, alpha=0.3)
                fig = ax.get_figure()
                fig.autofmt_xdate() # Rotate x-axis labels for better readability
            else:
                ax.text(0.5, 0.5, 'No Timeline Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Experiment Execution Timeline')

        except Exception as e:
            logging.error(f"VE Module - Error in _plot_experiment_timeline: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error in Timeline Plot:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Experiment Execution Timeline')

    def _plot_metrics_summary(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot a summary table of key metrics."""
        try:
            data = []
            # Define columns for the summary table
            columns = ['Experiment', 'Mode', 'CAGR (%)', 'Sharpe', 'Max DD (%)', 'Win Rate (%)']

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                
                # --- Key Fix: Read results from 'results' ---
                results_section = record_data.get('results', {})
                metrics_summary = self._extract_metrics_from_results(results_section)

                # Populate table rows with metrics for each mode
                for mode_name, metrics in metrics_summary.items():
                    row = [
                        exp_id,
                        mode_name,
                        f"{metrics.get('cagr', np.nan) * 100:.2f}" if not np.isnan(metrics.get('cagr', np.nan)) else 'N/A',
                        f"{metrics.get('sharpe_ratio', np.nan):.2f}" if not np.isnan(metrics.get('sharpe_ratio', np.nan)) else 'N/A',
                        f"{metrics.get('max_drawdown', np.nan) * 100:.2f}" if not np.isnan(metrics.get('max_drawdown', np.nan)) else 'N/A',
                        f"{metrics.get('win_rate', np.nan) * 100:.2f}" if not np.isnan(metrics.get('win_rate', np.nan)) else 'N/A',
                    ]
                    data.append(row)

            if data:
                # Create a pandas DataFrame and display it as a table in the plot
                df_table = pd.DataFrame(data, columns=columns)
                ax.axis('off') # Turn off axis for table display
                table = ax.table(cellText=df_table.values, colLabels=df_table.columns, cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                ax.set_title('Key Metrics Summary')
            else:
                ax.text(0.5, 0.5, 'No Metrics Data for Summary', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Key Metrics Summary')

        except Exception as e:
            logging.error(f"VE Module - Error in _plot_metrics_summary: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error in Metrics Summary:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Key Metrics Summary')

    def _plot_average_cagr_comparison(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot bar chart comparison of average CAGR across experiments."""
        try:
            experiments = []
            avg_cagrs = [] # Using CAGR as the comparison metric

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                
                # Get data from 'results'
                results_section = record_data.get('results', {})
                metrics_summary = self._extract_metrics_from_results(results_section)

                # Aggregate CAGR across modes for each experiment
                cagrs = [metrics.get('cagr', np.nan) * 100 for metrics in metrics_summary.values()] # Convert to %
                valid_cagrs = [c for c in cagrs if not np.isnan(c)]
                if valid_cagrs:
                    avg_cagr = np.mean(valid_cagrs)
                    experiments.append(exp_id)
                    avg_cagrs.append(avg_cagr)

            if experiments and avg_cagrs:
                # Create bar chart
                bars = ax.bar(experiments, avg_cagrs, color='skyblue')
                ax.set_xlabel('Experiment')
                ax.set_ylabel('Average CAGR (%)')
                ax.set_title('Average CAGR Comparison')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on top of bars for clarity
                for bar, value in zip(bars, avg_cagrs):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f'{value:.2f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'Insufficient Data for Performance Comparison', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title('Average CAGR Comparison')

        except Exception as e:
            logging.error(f"VE Module - Error in _plot_average_cagr_comparison: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error in Performance Comparison Plot:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Average CAGR Comparison')

    def _plot_risk_return_tradeoff(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot risk-return tradeoff analysis."""
        try:
            returns = [] # CAGR
            risks = []   # Volatility
            labels = []
            sharpe_ratios = []

            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record

                exp_id = record_data.get('experiment_id', 'Unknown')
                
                # Get results from 'results'
                results_section = record_data.get('results', {})
                metrics_summary = self._extract_metrics_from_results(results_section)

                for mode_name, metrics in metrics_summary.items():
                    # Get return and risk metrics, converting to percentages for plot
                    ret = metrics.get('cagr', np.nan) * 100 # Convert to %
                    risk = metrics.get('volatility', np.nan) * 100 # Convert to %
                    sharpe = metrics.get('sharpe_ratio', np.nan)

                    # Only plot if both return and risk are valid numbers
                    if not (np.isnan(ret) or np.isnan(risk)):
                        returns.append(ret)
                        risks.append(risk)
                        labels.append(f"{exp_id}-{mode_name}")
                        sharpe_ratios.append(sharpe)

            if returns and risks:
                # Create scatter plot, color-coded by Sharpe Ratio
                scatter = ax.scatter(risks, returns, c=sharpe_ratios, cmap='viridis', alpha=0.7)
                ax.set_xlabel('Risk (Volatility %)')
                ax.set_ylabel('Return (CAGR %)')
                ax.set_title('Risk-Return Tradeoff')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
                
                # Annotate points for identification
                for i, txt in enumerate(labels):
                    ax.annotate(txt, (risks[i], returns[i]), fontsize=8, ha='right')
            else:
                ax.text(0.5, 0.5, 'Insufficient Data for Risk-Return Analysis', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title('Risk-Return Tradeoff')

        except Exception as e:
            logging.error(f"VE Module - Error in _plot_risk_return_tradeoff: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Error in Risk-Return Plot:\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Risk-Return Tradeoff')

    def generate_experiment_comparison_report(self, experiment_records: List[Union[str, Dict]]) -> str:
        """
        Generate the comprehensive experiment comparison report.

        This is the ONLY public method in this module. It creates a single, unified report
        that combines all relevant analyses into one figure.

        Parameters
        ----------
        experiment_records : list
            A list of paths to experiment log JSON files or the loaded dictionaries themselves.

        Returns
        -------
        str
            Path to the saved comprehensive report PNG file.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.figure(figsize=(20, 15))

            # 1. Performance Comparison Heatmap
            ax1 = plt.subplot(3, 3, 1)
            self._plot_performance_comparison_heatmap(experiment_records, ax1)

            # 2. Asset Curves Comparison (using visualize_backtest)
            ax2 = plt.subplot(3, 3, 2)
            self._plot_asset_curves_comparison(experiment_records, ax2)

            # 3. Risk-Return Tradeoff
            ax3 = plt.subplot(3, 3, 3)
            self._plot_risk_return_tradeoff(experiment_records, ax3)

            # 4. Parameter Sensitivity (2D)
            ax4 = plt.subplot(3, 3, 4)
            self._plot_parameter_sensitivity_2d(experiment_records, ax4)

            # 5. Average CAGR Comparison
            ax5 = plt.subplot(3, 3, 5)
            self._plot_average_cagr_comparison(experiment_records, ax5)

            # 6. Experiment Timeline
            ax6 = plt.subplot(3, 3, 6)
            self._plot_experiment_timeline(experiment_records, ax6)

            # 7. Key Metrics Summary
            ax7 = plt.subplot(3, 3, 7)
            self._plot_metrics_summary(experiment_records, ax7)

            # 8. Cumulative Excess Return vs Benchmark (using visualize_backtest)
            ax8 = plt.subplot(3, 3, 8)
            self._plot_cumulative_excess_return_vs_benchmark(experiment_records, ax8)

            # 9. Algorithm Performance Radar Chart
            ax9 = plt.subplot(3, 3, 9)
            self._plot_algorithm_radar_chart(experiment_records, ax9)

            plt.tight_layout()

            report_filename = f"experiment_comprehensive_report_{timestamp}.png"
            report_path = os.path.join(self.plot_exper_dir, report_filename)
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"VE Module - Comprehensive experiment report saved: {report_path}")
            return report_path

        except Exception as e:
            logging.error(f"VE Module - Error generating comprehensive experiment report: {e}", exc_info=True)
            plt.close('all')
            raise


# --- Utility Functions ---

def create_experiment_visualizer(config: ConfigSetup) -> VisualizeExperiment:
    """
    Create and return a VisualizeExperiment instance.

    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance.

    Returns
    -------
    VisualizeExperiment
        New VisualizeExperiment instance.
    """
    return VisualizeExperiment(config)


def generate_comprehensive_experiment_report(config: ConfigSetup, experiment_records: List[Union[str, Dict]]) -> str:
    """
    Generate comprehensive experiment report.

    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance.
    experiment_records : list
        List of experiment records.

    Returns
    -------
    str
        Path to saved comprehensive report.
    """
    try:
        visualizer = VisualizeExperiment(config)
        return visualizer.generate_experiment_comparison_report(experiment_records)
    except Exception as e:
        logging.error(f"VE Module - Error in generate_comprehensive_experiment_report: {e}", exc_info=True)
        raise