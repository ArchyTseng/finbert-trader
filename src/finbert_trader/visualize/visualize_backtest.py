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
            Inherit config_trading, globally config DIR path, default 'plot_cache'.
        """
        self.config = config
        self.plot_cache_dir = getattr(self.config, 'PLOT_CACHE_DIR', 'plot_cache')

        os.makedirs(self.plot_cache_dir, exist_ok=True)

        logging.info(f"VB Module - Initialized VisualizeBacktest with plot cache: {self.plot_cache_dir}")
    
    def generate_asset_curve_comparison(self, pipeline_results: Dict[str, Any], 
                                      benchmark_data: Optional[List[float]] = None) -> str:
        """
        Generate a comparison plot of asset curves for all algorithms.
        
        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing results for all algorithms
        benchmark_data : list, optional
            Benchmark data for comparison
            
        Returns
        -------
        str
            Path to saved plot file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a DataFrame to store asset curves
            asset_curve_df = pd.DataFrame()
            
            # Extract asset history for each algorithm
            for mode_name, results in pipeline_results.items():
                if 'backtest_results' in results and 'asset_history' in results['backtest_results']:
                    asset_curve = pd.Series(results['backtest_results']['asset_history'], name=mode_name)
                    asset_curve_df = pd.concat([asset_curve_df, asset_curve], axis=1)
            
            # Add benchmark if provided
            if benchmark_data is not None:
                benchmark_series = pd.Series(benchmark_data, name='Benchmark')
                asset_curve_df = pd.concat([asset_curve_df, benchmark_series], axis=1)
            else:
                # Use initial cash as simple benchmark
                if len(asset_curve_df) > 0 and 'backtest_results' in list(pipeline_results.values())[0]:
                    first_results = list(pipeline_results.values())[0]['backtest_results']
                    if 'asset_history' in first_results and len(first_results['asset_history']) > 0:
                        initial_cash = first_results['asset_history'][0]
                        benchmark = pd.Series([initial_cash] * len(asset_curve_df), name='Initial_Cash')
                        asset_curve_df = pd.concat([asset_curve_df, benchmark], axis=1)
            
            # Plot the asset curves
            plt.figure(figsize=(15, 8))
            sns.set_style("whitegrid")
            palette = sns.color_palette("husl", len(asset_curve_df.columns))
            
            for i, column in enumerate(asset_curve_df.columns):
                plt.plot(asset_curve_df.index, asset_curve_df[column], 
                        label=column, linewidth=2, color=palette[i])
            
            plt.title("Asset Curve Comparison", fontsize=16, fontweight='bold')
            plt.xlabel("Trading Steps", fontsize=12)
            plt.ylabel("Portfolio Value ($)", fontsize=12)
            plt.legend(title="Strategies", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Generate dynamic filename
            plot_filename = f"asset_curve_comparison_{timestamp}.png"
            plot_path = os.path.join(self.plot_cache_dir, plot_filename)
            
            # Save the plot
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Asset curve comparison plot saved to: {plot_path}")
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
            
            # 1. CAGR Comparison
            if 'CAGR' in df.columns:
                pivot_cagr = df.pivot(index='Experiment', columns='Algorithm', values='CAGR')
                pivot_cagr.plot(kind='bar', ax=axes[0, 0])
                axes[0, 0].set_title('CAGR Comparison (%)')
                axes[0, 0].set_ylabel('CAGR (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Sharpe Ratio Comparison
            if 'Sharpe_Ratio' in df.columns:
                pivot_sharpe = df.pivot(index='Experiment', columns='Algorithm', values='Sharpe_Ratio')
                pivot_sharpe.plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Sharpe Ratio Comparison')
                axes[0, 1].set_ylabel('Sharpe Ratio')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Max Drawdown Comparison
            if 'Max_Drawdown' in df.columns:
                pivot_dd = df.pivot(index='Experiment', columns='Algorithm', values='Max_Drawdown')
                pivot_dd.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('Max Drawdown Comparison (%)')
                axes[1, 0].set_ylabel('Max Drawdown (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Win Rate Comparison
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
            plot_path = os.path.join(self.plot_cache_dir, plot_filename)
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
            
            # Save plot
            plot_filename = f"parameter_sensitivity_{parameter_name.replace('.', '_')}_{timestamp}.png"
            plot_path = os.path.join(self.plot_cache_dir, plot_filename)
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
            plot_path = os.path.join(self.plot_cache_dir, plot_filename)
            
            # Save the plot
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VB Module - Performance heatmap saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            logging.error(f"VB Module - Error generating performance heatmap: {e}")
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