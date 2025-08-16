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
                                      show_performance_metrics: bool = True) -> str:
        """
        Generate a comparison plot of asset curves with performance metrics.
        
        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing results for all algorithms
        benchmark_data : list or pd.Series, optional
            Benchmark data for comparison
        benchmark_name : str, optional
            Name of the benchmark for legend display
        show_performance_metrics : bool, optional
            Whether to include performance metrics in legend
            
        Returns
        -------
        str
            Path to saved plot file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a DataFrame to store asset curves
            asset_curve_df = pd.DataFrame()
            
            # Extract asset history and dates for each algorithm
            strategy_dates = {}
            
            for mode_name, results in pipeline_results.items():
                if 'backtest_results' in results and 'asset_history' in results['backtest_results']:
                    asset_history = results['backtest_results']['asset_history']
                    # Get Date information from environmeng
                    if hasattr(results['backtest_results'], 'dates') and results['backtest_results'].dates:
                        dates = results['backtest_results'].dates
                    else:
                        dates = range(len(asset_history))
                    
                    asset_curve = pd.Series(asset_history, index=dates, name=mode_name)
                    asset_curve_df = pd.concat([asset_curve_df, asset_curve], axis=1)
                    strategy_dates[mode_name] = dates
                elif 'asset_history' in results:
                    asset_history = results['asset_history']
                    dates = range(len(asset_history))
                    asset_curve = pd.Series(asset_history, index=dates, name=mode_name)
                    asset_curve_df = pd.concat([asset_curve_df, asset_curve], axis=1)
                    strategy_dates[mode_name] = dates
            
            # Add benchmark if provided
            if benchmark_data is not None:
                if isinstance(benchmark_data, (list, np.ndarray)):
                    benchmark_series = pd.Series(benchmark_data, name=benchmark_name)
                else:
                    benchmark_series = benchmark_data.rename(benchmark_name)
                
                # Align benchmark length with strategy data if needed
                if len(asset_curve_df) > 0:
                    min_length = min(len(asset_curve_df), len(benchmark_series))
                    benchmark_series = benchmark_series.iloc[:min_length]
                    asset_curve_df = asset_curve_df.iloc[:min_length]
                
                asset_curve_df = pd.concat([asset_curve_df, benchmark_series], axis=1)
            
            # Normalize all curves to start from 1.0
            if len(asset_curve_df) > 0:
                for column in asset_curve_df.columns:
                    if len(asset_curve_df[column].dropna()) > 0:
                        initial_value = asset_curve_df[column].dropna().iloc[0]
                        if initial_value != 0:
                            asset_curve_df[column] = asset_curve_df[column] / initial_value
            
            # Plot the asset curves
            plt.figure(figsize=(15, 8))
            sns.set_style("whitegrid")
            palette = sns.color_palette("husl", len(asset_curve_df.columns))
            
            # Prepare legend labels with performance metrics
            legend_labels = []
            
            for i, column in enumerate(asset_curve_df.columns):
                label = column
                
                # Add metrics to labels
                if show_performance_metrics and column != benchmark_name:
                    # Get strategy information ratio from pipeline_results
                    if column in pipeline_results:
                        results = pipeline_results[column]
                        if 'backtest_results' in results:
                            metrics = results['backtest_results'].get('metrics', {})
                        else:
                            metrics = results.get('metrics', {})
                        
                        ir = metrics.get('information_ratio', 0)
                        if abs(ir) > 0.001:  # Threshold to filter information ratio
                            label += f" (IR={ir:.4f})"
                
                legend_labels.append(label)
            
            # Plot asset curve
            for i, column in enumerate(asset_curve_df.columns):
                plt.plot(asset_curve_df.index, asset_curve_df[column], 
                        label=legend_labels[i], linewidth=2, color=palette[i])
            
            # Set title and label
            plt.title("Cumulative Returns with Performance Metrics", fontsize=16, fontweight='bold')
            plt.xlabel("Trading Days", fontsize=12)
            plt.ylabel("Cumulative Return (Normalized to 1.0)", fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Add benchmark line
            if benchmark_name in asset_curve_df.columns:
                plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Generate filename and save
            plot_filename = f"cumulative_returns_with_metrics_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Cumulative returns with metrics plot saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            logging.error(f"VB Module - Error generating asset curve comparison: {e}")
            plt.close()
            raise
    
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
            
            for mode_name, results in pipeline_results.items():
                asset_history = None
                if 'backtest_results' in results and 'asset_history' in results['backtest_results']:
                    asset_history = results['backtest_results']['asset_history']
                elif 'asset_history' in results:
                    asset_history = results['asset_history']
                
                if asset_history is not None and len(asset_history) > 1:
                    # Calculate strategy returns
                    strategy_returns = np.diff(asset_history) / asset_history[:-1]
                    
                    # Align lengths
                    min_length = min(len(strategy_returns), len(benchmark_returns))
                    aligned_strategy_returns = strategy_returns[:min_length]
                    aligned_benchmark_returns = benchmark_returns[:min_length]
                    
                    # Calculate excess returns (strategy - benchmark)
                    excess_returns = aligned_strategy_returns - aligned_benchmark_returns
                    
                    # Calculate cumulative excess returns
                    cumulative_excess = np.cumsum(excess_returns)
                    
                    relative_performance_data[mode_name] = cumulative_excess
            if not relative_performance_data:
                logging.warning("VB Module - No valid strategy data for relative performance calculation")
                return ""
            
            # Plot relative performance
            plt.figure(figsize=(15, 8))
            sns.set_style("whitegrid")
            palette = sns.color_palette("Set2", len(relative_performance_data))
            
            for i, (strategy_name, cumulative_excess) in enumerate(relative_performance_data.items()):
                plt.plot(range(len(cumulative_excess)), cumulative_excess, 
                        label=f"{strategy_name} vs Benchmark", linewidth=2, color=palette[i])
            
            plt.title("Relative Performance vs Benchmark (Cumulative Excess Returns)", fontsize=16, fontweight='bold')
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
            
            # Calculate drawdowns for all strategies
            drawdown_data = {}
            
            for mode_name, results in pipeline_results.items():
                asset_history = None
                if 'backtest_results' in results and 'asset_history' in results['backtest_results']:
                    asset_history = results['backtest_results']['asset_history']
                elif 'asset_history' in results:
                    asset_history = results['asset_history']
                
                if asset_history is not None and len(asset_history) > 1:
                    assets = np.array(asset_history)
                    rolling_max = np.maximum.accumulate(assets)
                    drawdown = (assets - rolling_max) / (rolling_max + 1e-8) * 100  # Percentage
                    drawdown_data[mode_name] = drawdown

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
            plt.figure(figsize=(15, 8))
            sns.set_style("whitegrid")
            palette = sns.color_palette("Set1", len(drawdown_data))

            for i, (name, drawdown) in enumerate(drawdown_data.items()):
                plt.plot(range(len(drawdown)), drawdown, 
                        label=name, linewidth=2, color=palette[i])
                plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color=palette[i])
            
            plt.title("Drawdown Comparison", fontsize=16, fontweight='bold')
            plt.xlabel("Trading Days", fontsize=12)
            plt.ylabel("Drawdown (%)", fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
            pipeline_results, benchmark_data, benchmark_name, show_performance_metrics=True)
        
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