# finbert_trader/visualize/visualize_backtest.py
"""
Visualization Module for FinBERT-Driven Trading System Backtesting
Purpose: Generate comprehensive visualizations for backtest results and experiment analysis
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizeBacktest:
    """
    Class for generating visualizations for backtest results and experiment analysis.
    
    This class provides methods to create various plots including asset curves,
    performance comparisons, parameter sensitivity analysis, and detailed metrics visualizations.
    """
    
    def __init__(self, config):
        """
        Initialize VisualizeBacktest.
        
        Parameters
        ----------
        config_trading: Class instance
            Inherit config_trading, globally config DIR path, default 'plot_exper_cache'.
        """
        self.config = config
        self.plot_exper_dir = getattr(self.config, 'PLOT_EXPER_DIR', 'plot_exper_cache')

        os.makedirs(self.plot_exper_dir, exist_ok=True)

        logging.info(f"VB Module - Initialized VisualizeBacktest with plot cache: {self.plot_exper_dir}")

    def generate_asset_curve_comparison(self, pipeline_results: Dict[str, Any],
                                        benchmark_data: Optional[Union[List[float], pd.Series]] = None,
                                        benchmark_name: str = 'Nasdaq-100',
                                        show_performance_metrics: bool = False) -> str:
        """
        Generate a comparison plot of asset curves with performance metrics.
        This function plots the cumulative return curves for multiple backtest results.
        It attempts to align them on a date axis, prioritizing the benchmark's dates if provided
        as a pandas Series with a DatetimeIndex.

        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing results for all algorithms/modes.
            Expected structure: {'mode_name': {'asset_history': [...], 'trade_history': [...], ...}}
        benchmark_data : list or pd.Series, optional
            Benchmark data for comparison. If pd.Series with DatetimeIndex, it will be used
            for date-aligned plotting. If list/ndarray, it's assumed to align with the first strategy.
        benchmark_name : str, optional
            Name of the benchmark for display in the legend (default is 'Nasdaq-100').
        show_performance_metrics : bool, optional
            Whether to add performance metrics to the legend labels (default is True).

        Returns
        -------
        str
            File path to the saved plot.
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(25, 12))

            # --- Process Benchmark Data ---
            benchmark_df = None
            if benchmark_data is not None:
                if isinstance(benchmark_data, pd.Series) and isinstance(benchmark_data.index, pd.DatetimeIndex):
                    # Assume benchmark_data is a price series with DatetimeIndex
                    benchmark_prices = benchmark_data
                    benchmark_initial_price = benchmark_prices.iloc[0] if len(benchmark_prices) > 0 else np.nan
                    if not np.isnan(benchmark_initial_price) and benchmark_initial_price > 0:
                        # Calculate cumulative return for benchmark (normalized to start at 1)
                        benchmark_cumulative = (benchmark_prices / benchmark_initial_price)
                        benchmark_df = pd.DataFrame({'Benchmark': benchmark_cumulative})
                        benchmark_df.index = benchmark_prices.index # Ensure DatetimeIndex is preserved
                        logging.debug(f"VB Module - Benchmark data processed with {len(benchmark_df)} points and date index.")
                    else:
                        logging.warning("VB Module - Benchmark initial price is invalid (zero/negative/NaN), skipping benchmark plot.")
                elif isinstance(benchmark_data, (list, np.ndarray)):
                    # If list/ndarray, cannot align by date easily without explicit date info
                    logging.info("VB Module - Benchmark data is list/ndarray. Date alignment might be approximate.")
                    # We will attempt to align later using strategy dates or a default index
                else:
                    logging.warning(f"VB Module - Unsupported benchmark_data type: {type(benchmark_data)}. Skipping benchmark.")

            # --- Collect Strategy Data ---
            strategy_dfs = {}
            strategy_dates_dict = {} # Store dates for potential fallback alignment
            for idx, (mode_name, results) in enumerate(pipeline_results.items()):
                asset_history = results.get('asset_history', [])
                # Get initial cash from self.config or use a default
                initial_cash = getattr(self.config, 'initial_cash', 1e6)

                if len(asset_history) == 0:
                    logging.warning(f"VB Module - No asset history for {mode_name}, skipping in plot")
                    continue

                # Convert absolute asset value to cumulative return (normalized by initial cash)
                strategy_cumulative = np.array(asset_history) / initial_cash

                # --- Attempt to get strategy dates ---
                strategy_dates = None
                # Try to get dates from trade_history if available and populated with 'date' key
                trade_history = results.get('trade_history', [])
                if trade_history and all(isinstance(record, dict) and 'date' in record and record['date'] for record in trade_history):
                    try:
                        # Assumes 'date' in trade_history is a string like 'YYYY-MM-DD'
                        strategy_dates = pd.to_datetime([record['date'] for record in trade_history])
                        logging.debug(f"VB Module - Extracted {len(strategy_dates)} dates for strategy {mode_name}.")
                    except (ValueError, TypeError) as e:
                        logging.warning(f"VB Module - Error parsing strategy dates for {mode_name}: {e}. Using integer index.")
                        strategy_dates = None # Reset on error

                # Create DataFrame for the strategy
                if strategy_dates is not None and len(strategy_dates) == len(strategy_cumulative):
                    strategy_df = pd.DataFrame({mode_name: strategy_cumulative}, index=strategy_dates)
                    strategy_dates_dict[mode_name] = strategy_dates # Store for potential use
                else:
                    # Fallback to integer index if dates are missing or mismatched
                    strategy_df = pd.DataFrame({mode_name: strategy_cumulative})
                    logging.info(f"VB Module - Using integer index for strategy {mode_name} curve (len={len(strategy_cumulative)}).")
                    # Store integer index as well for fallback alignment
                    strategy_dates_dict[mode_name] = pd.RangeIndex(start=0, stop=len(strategy_cumulative))

                strategy_dfs[mode_name] = strategy_df

            # --- Determine Common Index for Plotting ---
            plot_index = None
            # Priority 1: Use benchmark's DatetimeIndex if available
            if benchmark_df is not None and not benchmark_df.empty:
                plot_index = benchmark_df.index
                logging.debug("VB Module - Using benchmark's DatetimeIndex for main plot X-axis.")
            # Priority 2: Use the first strategy's DatetimeIndex if available
            elif strategy_dfs:
                first_mode_name = next(iter(strategy_dfs))
                first_strategy_df = strategy_dfs[first_mode_name]
                # Check if the first strategy's index is datetime-like
                if isinstance(first_strategy_df.index, (pd.DatetimeIndex, pd.RangeIndex)):
                    plot_index = first_strategy_df.index
                    logging.debug(f"VB Module - Using first strategy's ({first_mode_name}) index for main plot X-axis.")
                # If not, we might need to create a default one later

            # If no suitable index found, create a default one based on max length
            if plot_index is None:
                max_len = max((len(df) for df in strategy_dfs.values()), default=0)
                if benchmark_df is not None:
                    max_len = max(max_len, len(benchmark_df))
                if max_len > 0:
                    plot_index = pd.RangeIndex(start=0, stop=max_len)
                    logging.info(f"VB Module - No common date index found. Using default integer index (0 to {max_len-1}).")
                else:
                    error_msg = "VB Module - No valid data length found to determine plot index."
                    logging.error(error_msg)
                    raise ValueError(error_msg)

            # --- Reindex and Plot Data ---
            # Reindex benchmark data to the common plot index
            if benchmark_df is not None and not benchmark_df.empty:
                # Use 'pad' or 'nearest' for forward-filling or matching if indices don't align perfectly
                # 'pad' is good if plot_index covers a wider range than benchmark
                benchmark_df_plot = benchmark_df.reindex(plot_index, method='pad')
                ax.plot(benchmark_df_plot.index, benchmark_df_plot['Benchmark'],
                        label=benchmark_name, color='black', linewidth=2, linestyle='--')

            # Use a consistent color palette
            colors = sns.color_palette("husl", len(strategy_dfs))
            # Reindex strategy data and plot
            for idx, (mode_name, strategy_df) in enumerate(strategy_dfs.items()):
                # Reindex strategy data to the common plot index
                strategy_df_plot = strategy_df.reindex(plot_index, method='pad')
                ax.plot(strategy_df_plot.index, strategy_df_plot[mode_name],
                        label=mode_name, color=colors[idx % len(colors)], linewidth=2)

            # --- Formatting ---
            ax.set_title('Asset Curve Comparison (Cumulative Return)', fontsize=16)
            # Dynamically set xlabel based on index type
            if isinstance(plot_index, pd.DatetimeIndex):
                ax.set_xlabel('Date', fontsize=12)
                # Improve date formatting on x-axis if needed
                fig.autofmt_xdate() # Rotates and aligns the tick labels
            else:
                ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Cumulative Return (Normalized to Initial Cash)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            # plt.xticks(rotation=45) # Handled by fig.autofmt_xdate if dates
            plt.tight_layout()

            # --- Save Plot ---
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"asset_curve_comparison_{timestamp}.png"
            # Use self.config.plot_exper_dir or self.plot_exper_dir (depending on your __init__)
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            os.makedirs(self.plot_exper_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"VB Module - Asset curve comparison plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            logging.error(f"VB Module - Error generating asset curve comparison: {e}")
            # Ensure plot is closed even if an error occurs
            plt.close()
            raise # Re-raise the exception
    
    def generate_experiment_comparison_plot(self, experiment_records: List[Union[str, Dict]]) -> str:
        """
        Generate comparison plot across multiple experiments.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records (file paths or dictionaries)
            
        Returns
        -------
        str
            Path to saved comparison plot
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract data for comparison
            comparison_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    # Load from file
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                # Handle different metrics formats
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            comparison_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'CAGR': algo_metrics.get('cagr', 0) * 100,
                                'Sharpe_Ratio': algo_metrics.get('sharpe_ratio', 0),
                                'Max_Drawdown': algo_metrics.get('max_drawdown', 0) * 100,
                                'Win_Rate': algo_metrics.get('win_rate', 0) * 100,
                                'Final_Asset': algo_metrics.get('final_asset', 0)
                            })
            
            if not comparison_data:
                logging.warning("VB Module - No comparison data available")
                return ""
            
            # Create DataFrame
            df = pd.DataFrame(comparison_data)
            
            # Generate multiple comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle('Experiment Comparison Analysis', fontsize=16, fontweight='bold')
            
            # CAGR Comparison
            if 'CAGR' in df.columns:
                pivot_cagr = df.pivot(index='Experiment', columns='Algorithm', values='CAGR')
                pivot_cagr.plot(kind='bar', ax=axes[0, 0])
                axes[0, 0].set_title('CAGR Comparison (%)')
                axes[0, 0].set_ylabel('CAGR (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe Ratio Comparison
            if 'Sharpe_Ratio' in df.columns:
                pivot_sharpe = df.pivot(index='Experiment', columns='Algorithm', values='Sharpe_Ratio')
                pivot_sharpe.plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Sharpe Ratio Comparison')
                axes[0, 1].set_ylabel('Sharpe Ratio')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # Max Drawdown Comparison
            if 'Max_Drawdown' in df.columns:
                pivot_dd = df.pivot(index='Experiment', columns='Algorithm', values='Max_Drawdown')
                pivot_dd.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('Max Drawdown Comparison (%)')
                axes[1, 0].set_ylabel('Max Drawdown (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # Win Rate Comparison
            if 'Win_Rate' in df.columns:
                pivot_win = df.pivot(index='Experiment', columns='Algorithm', values='Win_Rate')
                pivot_win.plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Win Rate Comparison (%)')
                axes[1, 1].set_ylabel('Win Rate (%)')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"experiment_comparison_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Experiment comparison plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            logging.error(f"VB Module - Error generating experiment comparison plot: {e}")
            plt.close()
            raise
    
    def generate_parameter_sensitivity_plot(self, experiment_records: List[Union[str, Dict]], 
                                          parameter_name: str) -> str:
        """
        Generate parameter sensitivity analysis plot.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        parameter_name : str
            Name of parameter to analyze (e.g., 'trading_config.reward_scaling')
            
        Returns
        -------
        str
            Path to saved sensitivity plot
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract parameter values and corresponding metrics
            param_values = []
            metric_values = {'CAGR': [], 'Sharpe': [], 'Drawdown': []}
            
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                # Extract parameter value
                config_params = record_data.get('config_params', {})
                param_value = self._get_nested_value(config_params, parameter_name)
                if param_value is not None:
                    param_values.append(param_value)
                    
                    # Extract metrics (assuming single algorithm for simplicity)
                    metrics = record_data.get('metrics_summary', {})
                    if metrics:
                        # Get first algorithm metrics
                        first_algo_metrics = next(iter(metrics.values())) if isinstance(metrics, dict) else metrics
                        if isinstance(first_algo_metrics, dict):
                            metric_values['CAGR'].append(first_algo_metrics.get('cagr', 0) * 100)
                            metric_values['Sharpe'].append(first_algo_metrics.get('sharpe_ratio', 0))
                            metric_values['Drawdown'].append(first_algo_metrics.get('max_drawdown', 0) * 100)
            
            if not param_values:
                logging.warning(f"VB Module - No data found for parameter: {parameter_name}")
                return ""
            
            # Create sensitivity plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Parameter Sensitivity Analysis: {parameter_name}', fontsize=16, fontweight='bold')
            
            # Sort data by parameter values
            sorted_indices = np.argsort(param_values)
            sorted_params = np.array(param_values)[sorted_indices]
            
            # CAGR vs Parameter
            if metric_values['CAGR']:
                axes[0].plot(sorted_params, np.array(metric_values['CAGR'])[sorted_indices], 'o-', linewidth=2)
                axes[0].set_xlabel(parameter_name)
                axes[0].set_ylabel('CAGR (%)')
                axes[0].set_title('CAGR vs Parameter')
                axes[0].grid(True, alpha=0.3)
            
            # Sharpe vs Parameter
            if metric_values['Sharpe']:
                axes[1].plot(sorted_params, np.array(metric_values['Sharpe'])[sorted_indices], 'o-', linewidth=2)
                axes[1].set_xlabel(parameter_name)
                axes[1].set_ylabel('Sharpe Ratio')
                axes[1].set_title('Sharpe Ratio vs Parameter')
                axes[1].grid(True, alpha=0.3)
            
            # Drawdown vs Parameter
            if metric_values['Drawdown']:
                axes[2].plot(sorted_params, np.array(metric_values['Drawdown'])[sorted_indices], 'o-', linewidth=2)
                axes[2].set_xlabel(parameter_name)
                axes[2].set_ylabel('Max Drawdown (%)')
                axes[2].set_title('Max Drawdown vs Parameter')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Generate filename and save
            plot_filename = f"parameter_sensitivity_{parameter_name.replace('.', '_')}_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Parameter sensitivity plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            logging.error(f"VB Module - Error generating parameter sensitivity plot: {e}")
            plt.close()
            raise
    
    def _get_nested_value(self, dictionary: Dict, key_path: str) -> Any:
        """
        Get value from nested dictionary using dot notation.
        
        Parameters
        ----------
        dictionary : dict
            Dictionary to search
        key_path : str
            Dot-separated path to value (e.g., 'model_params.learning_rate')
            
        Returns
        -------
        any
            Value if found, None otherwise
        """
        try:
            keys = key_path.split('.')
            current = dictionary
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
                    
            return current
        except Exception:
            return None
    
    def generate_performance_heatmap(self, pipeline_results: Dict[str, Any]) -> str:
        """
        Generate performance comparison heatmap.
        
        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing results for all algorithms
            
        Returns
        -------
        str
            Path to saved heatmap plot file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract key metrics for heatmap
            metrics_data = []
            algorithms = []
            
            for mode_name, results in pipeline_results.items():
                if 'metrics' in results:
                    algorithms.append(mode_name)
                    metrics = results['metrics']
                    metrics_data.append([
                        metrics.get('cagr', 0) * 100,
                        metrics.get('sharpe_ratio', 0),
                        metrics.get('max_drawdown', 0) * 100,
                        metrics.get('calmar_ratio', 0),
                        metrics.get('win_rate', 0) * 100,
                        metrics.get('profit_factor', 0)
                    ])
            
            if not metrics_data:
                logging.warning("VB Module - No metrics data available for heatmap")
                return ""
            
            # Create DataFrame
            metrics_df = pd.DataFrame(metrics_data, 
                                    columns=['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                                           'Calmar Ratio', 'Win Rate (%)', 'Profit Factor'],
                                    index=algorithms)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(metrics_df.T, annot=True, cmap='RdYlGn', center=0, 
                       fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Performance Metrics Comparison Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('Algorithms')
            plt.ylabel('Metrics')
            plt.tight_layout()
            
            # Generate dynamic filename
            plot_filename = f"performance_heatmap_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            
            # Save plot
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Performance heatmap saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            logging.error(f"VB Module - Error generating performance heatmap: {e}")
            plt.close()
            raise

    def generate_benchmark_relative_performance(self, pipeline_results: Dict[str, Any], 
                                              benchmark_returns: np.ndarray) -> str:
        """
        Generate relative performance comparison against benchmark.
        
        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing results for all algorithms
        benchmark_returns : np.ndarray
            Benchmark returns for comparison
            
        Returns
        -------
        str
            Path to saved relative performance plot
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if benchmark_returns is None or len(benchmark_returns) == 0:
                logging.warning("VB Module - No benchmark returns provided for relative performance plot")
                return ""
            
            # Calculate relative performance for each algorithm
            relative_performance_data = {}
            strategy_dates = {}
            
            for mode_name, results in pipeline_results.items():
                asset_history = None
                if 'backtest_results' in results and 'asset_history' in results['backtest_results']:
                    asset_history = results['backtest_results']['asset_history']
                elif 'asset_history' in results:
                    asset_history = results['asset_history']
                
                if asset_history is not None and len(asset_history) > 1:
                    # Calculate strategy returns
                    strategy_returns = np.array(asset_history[1:]) / np.array(asset_history[:-1]) - 1
                    
                    # Align lengths
                    min_length = min(len(strategy_returns), len(benchmark_returns))
                    aligned_strategy_returns = strategy_returns[:min_length]
                    aligned_benchmark_returns = benchmark_returns[:min_length]
                    
                    # Calculate excess returns (strategy - benchmark)
                    excess_returns = aligned_strategy_returns - aligned_benchmark_returns
                    # Calculate cumulative excess returns
                    cumulative_excess = np.cumsum(excess_returns)
                    relative_performance_data[mode_name] = cumulative_excess

                    # Try fetching dates from results
                    trade_history = results.get('trade_history', [])
                    if trade_history and all('date' in record and record['date'] for record in trade_history):
                        try:
                            # Convert dates to datetime 
                            dates_list = [pd.to_datetime(record['date']) for record in trade_history]
                            # Save date with same range of cumulative_excess 
                            strategy_dates[mode_name] = dates_list[:len(cumulative_excess)]
                        except Exception as e:
                            logging.warning(f"VB Module - Error parsing dates for {mode_name}: {e}")
                            strategy_dates[mode_name] = range(len(cumulative_excess))
            if not relative_performance_data:
                logging.warning("VB Module - No valid strategy data for relative performance calculation")
                return ""
            
            # Plot relative performance
            plt.figure(figsize=(25, 12))
            sns.set_style("whitegrid")
            palette = sns.color_palette("Set2", len(relative_performance_data))
            
            for i, (strategy_name, cumulative_excess) in enumerate(relative_performance_data.items()):
                # Set X_axis
                x_axis = strategy_dates.get(strategy_name, range(len(cumulative_excess)))
                plt.plot(x_axis, cumulative_excess,
                     label=f"{strategy_name} vs Benchmark", linewidth=2, color=palette[i])
            
            plt.title("Relative Performance vs Benchmark (Cumulative Excess Returns)", fontsize=16, fontweight='bold')
            # Set x_axis automatically
            if isinstance(x_axis[0], (datetime, pd.Timestamp)):
                plt.xlabel("Date", fontsize=12)
                fig.autofmt_xdate()  # Automatically set xdate
            else:
                plt.xlabel("Trading Days", fontsize=12)
            plt.ylabel("Cumulative Excess Return", fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            # Generate filename and save
            plot_filename = f"relative_performance_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Relative performance plot saved to: {plot_path}")
            return plot_path
        except Exception as e:
            logging.error(f"VB Module - Error generating relative performance plot: {e}")
            plt.close()
            raise

    def generate_drawdown_comparison(self, pipeline_results: Dict[str, Any], 
                                   benchmark_data: Optional[Union[List[float], pd.Series]] = None,
                                   benchmark_name: str = 'Nasdaq-100') -> str:
        """
        Generate drawdown comparison plot for strategies and benchmark.
        
        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing results for all algorithms
        benchmark_data : list or pd.Series, optional
            Benchmark data for comparison
        benchmark_name : str, optional
            Name of the benchmark for legend display
            
        Returns
        -------
        str
            Path to saved drawdown comparison plot
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            drawdown_data = {}
            # Create a DataFrame to store asset curves
            asset_curve_df = pd.DataFrame()
            strategy_dates = {}  

            # Extract asset history and dates for each algorithm
            for mode_name, results in pipeline_results.items():
                asset_history = None
                if 'backtest_results' in results and 'asset_history' in results['backtest_results']:
                    asset_history = results['backtest_results']['asset_history']
                elif 'asset_history' in results:
                    asset_history = results['asset_history']
                
                if asset_history is not None:
                    assets = np.array(asset_history)
                    rolling_max = np.maximum.accumulate(assets)
                    drawdown = (assets - rolling_max) / (rolling_max + 1e-8) * 100  # Percentage
                    asset_curve_df[mode_name] = drawdown

                    # Try fetching dates from results
                    trade_history = results.get('trade_history', [])
                    if trade_history and all('date' in record and record['date'] for record in trade_history):
                        try:
                            dates_list = [pd.to_datetime(record['date']) for record in trade_history]
                            strategy_dates[mode_name] = dates_list
                        except Exception as e:
                            logging.warning(f"VB Module - Error parsing dates for {mode_name}: {e}")
                            strategy_dates[mode_name] = range(len(drawdown))

            # Add benchmark drawdown if provided
            if benchmark_data is not None:
                if isinstance(benchmark_data, (list, np.ndarray)):
                    benchmark_prices = np.array(benchmark_data)
                else:
                    benchmark_prices = benchmark_data.values
                
                if len(benchmark_prices) > 1:
                    rolling_max = np.maximum.accumulate(benchmark_prices)
                    benchmark_drawdown = (benchmark_prices - rolling_max) / (rolling_max + 1e-8) * 100
                    drawdown_data[benchmark_name] = benchmark_drawdown
            
            if not drawdown_data:
                logging.warning("VB Module - No data available for drawdown comparison")
                return ""
            
            # Plot drawdowns
            plt.figure(figsize=(25, 12))
            sns.set_style("whitegrid")
            colors = plt.cm.Set1(np.linspace(0, 1, len(drawdown_data)))

            for i, (strategy_name, drawdown_series) in enumerate(drawdown_data.items()):
                # Set X_axis
                x_axis = strategy_dates.get(strategy_name, range(len(drawdown_series)))
                plt.plot(x_axis, drawdown_series,
                        label=strategy_name, linewidth=2, color=colors[i])
            
            plt.title("Drawdown Comparison", fontsize=16, fontweight='bold')
            # Set x_axis automatically
            if isinstance(x_axis[0], (datetime, pd.Timestamp)):
                plt.xlabel("Date", fontsize=12)
                fig.autofmt_xdate()  # Automatically set xdate
            else:
                plt.xlabel("Trading Days", fontsize=12)
            plt.ylabel("Drawdown (%)", fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.tight_layout()

            # Generate filename and save
            plot_filename = f"drawdown_comparison_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"VB Module - Drawdown comparison plot saved to: {plot_path}")
            return plot_path
        except Exception as e:
            logging.error(f"VB Module - Error generating drawdown comparison plot: {e}")
            plt.close()
            raise

# Utility functions
def generate_all_visualizations(pipeline_results: Dict[str, Any], 
                              config_trading: Any) -> Dict[str, str]:
    """
    Generate all standard visualizations for pipeline results.
    
    Parameters
    ----------
    pipeline_results : dict
        Dictionary containing results for all algorithms
    config_trading : ConfigTrading
        Trading configuration
        
    Returns
    -------
    dict
        Dictionary containing paths to all generated plots
    """
    try:
        visualizer = VisualizeBacktest(config_trading)
        
        # Generate all visualizations
        asset_curve_plot = visualizer.generate_asset_curve_comparison(pipeline_results)
        heatmap_plot = visualizer.generate_performance_heatmap(pipeline_results)
        
        visualization_results = {
            'asset_curve_comparison': asset_curve_plot,
            'performance_heatmap': heatmap_plot
        }
        
        logging.info("VB Module - All standard visualizations generated successfully")
        return visualization_results
        
    except Exception as e:
        logging.error(f"VB Module - Error generating standard visualizations: {e}")
        raise

def generate_all_visualizations_with_benchmark(pipeline_results: Dict[str, Any], 
                                             config_trading: Any,
                                             benchmark_data: Optional[Union[List[float], pd.Series]] = None,
                                             benchmark_returns: Optional[np.ndarray] = None,
                                             benchmark_name: str = 'Nasdaq-100') -> Dict[str, str]:
    """
    Generate all standard visualizations for pipeline results with benchmark support.
    
    Parameters
    ----------
    pipeline_results : dict
        Dictionary containing results for all algorithms
    config_trading : ConfigTrading
        Trading configuration
    benchmark_data : list or pd.Series, optional
        Benchmark price/asset data for comparison
    benchmark_returns : np.ndarray, optional
        Benchmark returns for relative performance analysis
    benchmark_name : str, optional
        Name of the benchmark for display
        
    Returns
    -------
    dict
        Dictionary containing paths to all generated plots
    """
    try:
        visualizer = VisualizeBacktest(config_trading)
        
        # Generate all visualizations
        asset_curve_plot = visualizer.generate_asset_curve_comparison(
            pipeline_results,
            benchmark_data=benchmark_data,
            benchmark_name=benchmark_name)
        
        heatmap_plot = visualizer.generate_performance_heatmap(pipeline_results)
        
        drawdown_plot = visualizer.generate_drawdown_comparison(
            pipeline_results, benchmark_data, benchmark_name)
        
        relative_performance_plot = ""
        if benchmark_returns is not None:
            relative_performance_plot = visualizer.generate_benchmark_relative_performance(
                pipeline_results, benchmark_returns)
        
        visualization_results = {
            'asset_curve_comparison': asset_curve_plot,
            'performance_heatmap': heatmap_plot,
            'drawdown_comparison': drawdown_plot,
            'relative_performance': relative_performance_plot
        }
        logging.info("VB Module - All visualizations with benchmark generated successfully")
        return visualization_results
        
    except Exception as e:
        logging.error(f"VB Module - Error generating visualizations with benchmark: {e}")
        raise