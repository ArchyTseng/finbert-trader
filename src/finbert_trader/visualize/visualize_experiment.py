# src/finbert_trader/visualize/visualize_experiment.py
"""
Experiment Visualization Module for FinBERT-Driven Trading System
Purpose: Generate comprehensive visualizations for experiment analysis and comparison
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import project modules
from .visualize_backtest import VisualizeBacktest
from ..config_setup import ConfigSetup

class VisualizeExperiment:
    """
    Class for generating visualizations specific to experiment analysis and comparison.
    
    This class extends the basic backtest visualization with experiment-specific
    analysis including parameter sensitivity, experiment comparison, and optimization paths.
    It leverages enhanced backtest visualization capabilities for comprehensive analysis.
    """
    
    def __init__(self, config: ConfigSetup):
        """
        Initialize VisualizeExperiment with unified configuration.
        
        Parameters
        ----------
        config : ConfigSetup
            Configuration setup instance containing cache directory paths
        """
        self.config = config
        self.plot_exper_dir = getattr(config, 'PLOT_EXPER_DIR', 'plot_exper_cache')
        self.experiment_cache_dir = getattr(config, 'EXPERIMENT_CACHE_DIR', 'exper_cache')
        
        # Ensure directories exist
        os.makedirs(self.plot_exper_dir, exist_ok=True)
        os.makedirs(self.experiment_cache_dir, exist_ok=True)
        
        # Initialize base visualizer
        self.base_visualizer = VisualizeBacktest(self.config)
        
        logging.info("VE Module - Initialized VisualizeExperiment")
        logging.info(f"VE Module - Plot cache directory: {self.plot_exper_dir}")
        logging.info(f"VE Module - Experiment cache directory: {self.experiment_cache_dir}")
    
    def generate_experiment_comparison_report(self, experiment_records: List[Union[str, Dict]]) -> str:
        """
        Generate comprehensive experiment comparison report with multiple visualization types.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records (file paths or dictionaries)
            
        Returns
        -------
        str
            Path to saved comprehensive report
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create comprehensive report with multiple plots
            fig = plt.figure(figsize=(20, 25))
            fig.suptitle('Comprehensive Experiment Analysis Report', fontsize=20, fontweight='bold')
            
            # Performance Comparison (Top algorithms across experiments)
            ax1 = plt.subplot(4, 2, 1)
            self._plot_performance_comparison(experiment_records, ax1)
            
            # Parameter Sensitivity Analysis
            ax2 = plt.subplot(4, 2, 2)
            self._plot_parameter_sensitivity(experiment_records, 'trading_config.reward_scaling', ax2)
            
            # Convergence Analysis
            ax3 = plt.subplot(4, 2, 3)
            self._plot_convergence_analysis(experiment_records, ax3)
            
            # Risk-Return Tradeoff
            ax4 = plt.subplot(4, 2, 4)
            self._plot_risk_return_tradeoff(experiment_records, ax4)
            
            # Algorithm Comparison Heatmap
            ax5 = plt.subplot(4, 2, (5, 6))
            self._plot_algorithm_comparison_heatmap(experiment_records, ax5)
            
            # Experiment Timeline
            ax6 = plt.subplot(4, 2, 7)
            self._plot_experiment_timeline(experiment_records, ax6)
            
            # Key Metrics Summary
            ax7 = plt.subplot(4, 2, 8)
            self._plot_metrics_summary(experiment_records, ax7)
            
            plt.tight_layout()
            
            # Save comprehensive report
            report_filename = f"experiment_comprehensive_report_{timestamp}.png"
            report_path = os.path.join(self.plot_exper_dir, report_filename)
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VE Module - Comprehensive experiment report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"VE Module - Error generating comprehensive report: {e}")
            plt.close()
            raise
    
    def _plot_performance_comparison(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot performance comparison across experiments.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            comparison_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            comparison_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'CAGR': algo_metrics.get('cagr', 0) * 100,
                                'Sharpe_Ratio': algo_metrics.get('sharpe_ratio', 0),
                                'Max_Drawdown': abs(algo_metrics.get('max_drawdown', 0)) * 100
                            })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                # Pivot for better visualization
                pivot_cagr = df.pivot_table(index='Experiment', columns='Algorithm', values='CAGR', fill_value=0)
                pivot_cagr.plot(kind='bar', ax=ax)
                ax.set_title('CAGR Comparison Across Experiments')
                ax.set_ylabel('CAGR (%)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('CAGR Comparison Across Experiments')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot performance comparison: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('CAGR Comparison Across Experiments')
    
    def _plot_parameter_sensitivity(self, experiment_records: List[Union[str, Dict]], 
                                  parameter_name: str, ax):
        """
        Plot parameter sensitivity analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        parameter_name : str
            Name of parameter to analyze
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            param_values = []
            cagr_values = []
            
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                config_params = record_data.get('config_params', {})
                param_value = self._get_nested_value(config_params, parameter_name)
                if param_value is not None:
                    param_values.append(param_value)
                    
                    metrics = record_data.get('metrics_summary', {})
                    if metrics:
                        first_algo_metrics = next(iter(metrics.values()))
                        if isinstance(first_algo_metrics, dict):
                            cagr_values.append(first_algo_metrics.get('cagr', 0) * 100)
            
            if param_values and cagr_values:
                # Sort by parameter values
                sorted_indices = np.argsort(param_values)
                sorted_params = np.array(param_values)[sorted_indices]
                sorted_cagr = np.array(cagr_values)[sorted_indices]
                
                ax.plot(sorted_params, sorted_cagr, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel(parameter_name)
                ax.set_ylabel('CAGR (%)')
                ax.set_title(f'Parameter Sensitivity: {parameter_name}')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Parameter Sensitivity: {parameter_name}')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot parameter sensitivity: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Parameter Sensitivity: {parameter_name}')
    
    def _plot_convergence_analysis(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot convergence analysis based on training steps.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            # This would require training metrics data
            ax.text(0.5, 0.5, 'Convergence Analysis\n(Training Data Required)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Convergence Analysis')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot convergence analysis: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Convergence Analysis')
    
    def _plot_risk_return_tradeoff(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot risk-return tradeoff analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            risk_return_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                metrics = record_data.get('metrics_summary', {})
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            risk_return_data.append({
                                'Algorithm': algo_name,
                                'Sharpe_Ratio': algo_metrics.get('sharpe_ratio', 0),
                                'Max_Drawdown': abs(algo_metrics.get('max_drawdown', 0)) * 100,
                                'CAGR': algo_metrics.get('cagr', 0) * 100
                            })
            
            if risk_return_data:
                df = pd.DataFrame(risk_return_data)
                for algo in df['Algorithm'].unique():
                    algo_data = df[df['Algorithm'] == algo]
                    ax.scatter(algo_data['Max_Drawdown'], algo_data['CAGR'], 
                              label=algo, s=100, alpha=0.7)
                
                ax.set_xlabel('Max Drawdown (%)')
                ax.set_ylabel('CAGR (%)')
                ax.set_title('Risk-Return Tradeoff Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Risk-Return Tradeoff Analysis')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot risk-return tradeoff: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk-Return Tradeoff Analysis')
    
    def _plot_algorithm_comparison_heatmap(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot algorithm comparison heatmap.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            # Extract metrics for heatmap
            metrics_data = []
            algorithms = []
            experiments = []
            
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                experiments.append(exp_id)
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if algo_name not in algorithms:
                            algorithms.append(algo_name)
                        if isinstance(algo_metrics, dict):
                            metrics_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'CAGR': algo_metrics.get('cagr', 0) * 100,
                                'Sharpe_Ratio': algo_metrics.get('sharpe_ratio', 0)
                            })
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                # Create pivot table for heatmap
                pivot_table = df.pivot_table(index='Algorithm', columns='Experiment', 
                                           values='CAGR', fill_value=0)
                
                sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0, 
                           fmt='.1f', cbar_kws={'shrink': 0.8}, ax=ax)
                ax.set_title('Algorithm Performance Heatmap (CAGR %)')
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Algorithm Performance Heatmap (CAGR %)')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot algorithm comparison heatmap: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Algorithm Performance Heatmap (CAGR %)')
    
    def _plot_experiment_timeline(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot experiment timeline and progression.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            timeline_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                timestamp = record_data.get('timestamp', '')
                exp_id = record_data.get('experiment_id', 'Unknown')
                
                # Try to parse timestamp
                try:
                    if len(timestamp) >= 8:
                        # Extract date part for simple timeline
                        date_part = timestamp[:8]  # YYYYMMDD
                        timeline_data.append({
                            'Experiment': exp_id,
                            'Date': date_part,
                            'Timestamp': timestamp
                        })
                except Exception as parse_error:
                    logging.debug(f"VE Module - Could not parse timestamp {timestamp}: {parse_error}")
                    pass
            
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                ax.barh(range(len(df)), [1] * len(df), height=0.5)
                ax.set_yticks(range(len(df)))
                ax.set_yticklabels([f"{row['Experiment']}\n{row['Date']}" for _, row in df.iterrows()])
                ax.set_title('Experiment Timeline')
                ax.set_xlabel('Execution Order')
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Experiment Timeline')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot experiment timeline: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Experiment Timeline')
    
    def _plot_metrics_summary(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot key metrics summary.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            summary_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            summary_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Metric': 'CAGR',
                                'Value': algo_metrics.get('cagr', 0) * 100
                            })
                            summary_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Metric': 'Sharpe',
                                'Value': algo_metrics.get('sharpe_ratio', 0)
                            })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                # Group by metric and plot
                metrics_list = df['Metric'].unique()
                x_pos = np.arange(len(metrics_list))
                width = 0.35
                
                for i, exp in enumerate(df['Experiment'].unique()[:2]):  # Limit to first 2 for clarity
                    exp_data = df[df['Experiment'] == exp]
                    if not exp_data.empty:
                        values = [exp_data[exp_data['Metric'] == metric]['Value'].mean() 
                                if not exp_data[exp_data['Metric'] == metric].empty else 0 
                                for metric in metrics_list]
                        ax.bar(x_pos + i*width, values, width, label=exp)
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Values')
                ax.set_title('Key Metrics Summary')
                ax.set_xticks(x_pos + width/2)
                ax.set_xticklabels(metrics_list)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Key Metrics Summary')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot metrics summary: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Key Metrics Summary')
    
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
        except Exception as e:
            logging.debug(f"VE Module - Error getting nested value for {key_path}: {e}")
            return None
    
    def generate_optimization_path_visualization(self, experiment_records: List[Union[str, Dict]]) -> str:
        """
        Generate visualization showing the optimization path across experiments.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
            
        Returns
        -------
        str
            Path to saved optimization path visualization
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Extract optimization path data
            path_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                timestamp_str = record_data.get('timestamp', '')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict) and metrics:
                    # Get best performing algorithm in this experiment
                    best_algo = max(metrics.items(), 
                                  key=lambda x: x[1].get('cagr', 0) if isinstance(x[1], dict) else 0)
                    if isinstance(best_algo[1], dict):
                        path_data.append({
                            'Experiment': exp_id,
                            'Timestamp': timestamp_str,
                            'CAGR': best_algo[1].get('cagr', 0) * 100,
                            'Sharpe': best_algo[1].get('sharpe_ratio', 0),
                            'Drawdown': abs(best_algo[1].get('max_drawdown', 0)) * 100
                        })
            
            if path_data:
                df = pd.DataFrame(path_data)
                df = df.sort_values('Timestamp')  # Sort by execution order
                
                # Plot optimization path
                ax.plot(range(len(df)), df['CAGR'], 'o-', linewidth=2, markersize=8, label='CAGR (%)')
                ax.set_xlabel('Experiment Sequence')
                ax.set_ylabel('CAGR (%)', color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                
                # Add secondary y-axis for Sharpe ratio
                ax2 = ax.twinx()
                ax2.plot(range(len(df)), df['Sharpe'], 's-', linewidth=2, markersize=8, 
                        color='red', label='Sharpe Ratio')
                ax2.set_ylabel('Sharpe Ratio', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Add experiment labels
                for i, (idx, row) in enumerate(df.iterrows()):
                    ax.annotate(row['Experiment'], (i, row['CAGR']), 
                              textcoords="offset points", xytext=(0,10), ha='center')
                
                ax.set_title('Optimization Path Across Experiments')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Optimization Path Across Experiments')
            
            plt.tight_layout()
            
            # Save optimization path visualization
            filename = f"optimization_path_{timestamp}.png"
            filepath = os.path.join(self.plot_exper_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VE Module - Optimization path visualization saved: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"VE Module - Error generating optimization path visualization: {e}")
            plt.close()
            raise
    
    def generate_parameter_impact_analysis(self, experiment_records: List[Union[str, Dict]], 
                                        parameter_names: List[str]) -> str:
        """
        Generate comprehensive parameter impact analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        parameter_names : list
            List of parameter names to analyze
            
        Returns
        -------
        str
            Path to saved parameter impact analysis
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            fig, axes = plt.subplots(len(parameter_names), 2, figsize=(20, 6*len(parameter_names)))
            if len(parameter_names) == 1:
                axes = [axes]  # Ensure axes is always a list
            elif len(parameter_names) == 0:
                logging.warning("VE Module - No parameter names provided for impact analysis")
                return ""
            
            for i, param_name in enumerate(parameter_names):
                ax_cagr = axes[i][0]
                ax_sharpe = axes[i][1]
                
                # Extract parameter values and metrics
                param_values = []
                cagr_values = []
                sharpe_values = []
                
                for record in experiment_records:
                    if isinstance(record, str):
                        with open(record, 'r') as f:
                            record_data = json.load(f)
                    else:
                        record_data = record
                    
                    config_params = record_data.get('config_params', {})
                    param_value = self._get_nested_value(config_params, param_name)
                    if param_value is not None:
                        param_values.append(param_value)
                        
                        metrics = record_data.get('metrics_summary', {})
                        if metrics:
                            first_algo_metrics = next(iter(metrics.values()))
                            if isinstance(first_algo_metrics, dict):
                                cagr_values.append(first_algo_metrics.get('cagr', 0) * 100)
                                sharpe_values.append(first_algo_metrics.get('sharpe_ratio', 0))
                
                if param_values and cagr_values:
                    # Sort by parameter values
                    sorted_indices = np.argsort(param_values)
                    sorted_params = np.array(param_values)[sorted_indices]
                    sorted_cagr = np.array(cagr_values)[sorted_indices]
                    sorted_sharpe = np.array(sharpe_values)[sorted_indices]
                    
                    # CAGR vs Parameter
                    ax_cagr.plot(sorted_params, sorted_cagr, 'o-', linewidth=2, markersize=8)
                    ax_cagr.set_xlabel(param_name)
                    ax_cagr.set_ylabel('CAGR (%)')
                    ax_cagr.set_title(f'CAGR vs {param_name}')
                    ax_cagr.grid(True, alpha=0.3)
                    
                    # Sharpe vs Parameter
                    ax_sharpe.plot(sorted_params, sorted_sharpe, 's-', linewidth=2, markersize=8, color='red')
                    ax_sharpe.set_xlabel(param_name)
                    ax_sharpe.set_ylabel('Sharpe Ratio')
                    ax_sharpe.set_title(f'Sharpe Ratio vs {param_name}')
                    ax_sharpe.grid(True, alpha=0.3)
                else:
                    ax_cagr.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_cagr.transAxes)
                    ax_cagr.set_title(f'CAGR vs {param_name}')
                    ax_sharpe.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_sharpe.transAxes)
                    ax_sharpe.set_title(f'Sharpe Ratio vs {param_name}')
            
            plt.tight_layout()
            
            # Save parameter impact analysis
            filename = f"parameter_impact_analysis_{timestamp}.png"
            filepath = os.path.join(self.plot_exper_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VE Module - Parameter impact analysis saved: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"VE Module - Error generating parameter impact analysis: {e}")
            plt.close()
            raise

    def generate_experiment_comparison_with_benchmark(self, experiment_records: List[Union[str, Dict]],
                                                    benchmark_name: str = 'Nasdaq-100') -> str:
        """
        Generate experiment comparison report with benchmark analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        benchmark_name : str, optional
            Name of benchmark for display
            
        Returns
        -------
        str
            Path to saved benchmark comparison report
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create benchmark comparison report
            fig = plt.figure(figsize=(20, 20))
            fig.suptitle(f'Experiment Analysis with {benchmark_name} Benchmark', 
                        fontsize=20, fontweight='bold')
            
            # Relative Performance vs Benchmark
            ax1 = plt.subplot(3, 3, 1)
            self._plot_relative_performance_vs_benchmark(experiment_records, benchmark_name, ax1)
            
            # Information Ratio Analysis
            ax2 = plt.subplot(3, 3, 2)
            self._plot_information_ratio_analysis(experiment_records, ax2)
            
            # Alpha Analysis
            ax3 = plt.subplot(3, 3, 3)
            self._plot_alpha_analysis(experiment_records, ax3)
            
            # Benchmark Correlation Analysis
            ax4 = plt.subplot(3, 3, 4)
            self._plot_benchmark_correlation(experiment_records, ax4)
            
            # Tracking Error Analysis
            ax5 = plt.subplot(3, 3, 5)
            self._plot_tracking_error_analysis(experiment_records, ax5)
            
            # Excess Return Distribution
            ax6 = plt.subplot(3, 3, 6)
            self._plot_excess_return_distribution(experiment_records, ax6)
            
            # Performance vs Benchmark Scatter
            ax7 = plt.subplot(3, 3, (7, 9))
            self._plot_performance_vs_benchmark_scatter(experiment_records, benchmark_name, ax7)
            
            plt.tight_layout()
            
            # Save benchmark comparison report
            filename = f"experiment_benchmark_comparison_{timestamp}.png"
            filepath = os.path.join(self.plot_exper_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VE Module - Experiment benchmark comparison report saved: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"VE Module - Error generating benchmark comparison report: {e}")
            plt.close()
            raise

    def _plot_relative_performance_vs_benchmark(self, experiment_records: List[Union[str, Dict]], 
                                              benchmark_name: str, ax):
        """
        Plot relative performance analysis vs benchmark.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        benchmark_name : str
            Name of benchmark
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            relative_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            strategy_cagr = algo_metrics.get('cagr', 0)
                            benchmark_cagr = algo_metrics.get('benchmark_cagr', 0)
                            relative_performance = strategy_cagr - benchmark_cagr if benchmark_cagr != 0 else 0
                            
                            relative_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Relative_CAGR': relative_performance * 100
                            })
            
            if relative_data:
                df = pd.DataFrame(relative_data)
                pivot_data = df.pivot_table(index='Experiment', columns='Algorithm', 
                                          values='Relative_CAGR', fill_value=0)
                pivot_data.plot(kind='bar', ax=ax)
                ax.set_title(f'Relative Performance vs {benchmark_name}')
                ax.set_ylabel('Relative CAGR (%)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Relative Performance vs {benchmark_name}')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot relative performance: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Relative Performance vs {benchmark_name}')

    def _plot_information_ratio_analysis(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot information ratio analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            ir_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            ir_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Information_Ratio': algo_metrics.get('information_ratio', 0)
                            })
            
            if ir_data:
                df = pd.DataFrame(ir_data)
                pivot_data = df.pivot_table(index='Experiment', columns='Algorithm', 
                                          values='Information_Ratio', fill_value=0)
                pivot_data.plot(kind='bar', ax=ax)
                ax.set_title('Information Ratio Analysis')
                ax.set_ylabel('Information Ratio')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Information Ratio Analysis')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot information ratio analysis: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Information Ratio Analysis')

    def _plot_alpha_analysis(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot alpha analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            alpha_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            alpha_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Alpha': algo_metrics.get('alpha', 0) * 100  # Convert to percentage
                            })
            
            if alpha_data:
                df = pd.DataFrame(alpha_data)
                pivot_data = df.pivot_table(index='Experiment', columns='Algorithm', 
                                          values='Alpha', fill_value=0)
                pivot_data.plot(kind='bar', ax=ax)
                ax.set_title('Alpha Analysis (Annualized)')
                ax.set_ylabel('Alpha (%)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Alpha Analysis (Annualized)')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot alpha analysis: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Alpha Analysis (Annualized)')

    def _plot_benchmark_correlation(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot benchmark correlation analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            correlation_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            correlation_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Beta': algo_metrics.get('beta', 0)
                            })
            
            if correlation_data:
                df = pd.DataFrame(correlation_data)
                pivot_data = df.pivot_table(index='Experiment', columns='Algorithm', 
                                          values='Beta', fill_value=0)
                pivot_data.plot(kind='bar', ax=ax)
                ax.set_title('Benchmark Beta (Correlation)')
                ax.set_ylabel('Beta')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Beta')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Benchmark Beta (Correlation)')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot benchmark correlation: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Benchmark Beta (Correlation)')

    def _plot_tracking_error_analysis(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot tracking error analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            te_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                exp_id = record_data.get('experiment_id', 'Unknown')
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            te_data.append({
                                'Experiment': exp_id,
                                'Algorithm': algo_name,
                                'Tracking_Error': algo_metrics.get('tracking_error', 0) * 100
                            })
            
            if te_data:
                df = pd.DataFrame(te_data)
                pivot_data = df.pivot_table(index='Experiment', columns='Algorithm', 
                                          values='Tracking_Error', fill_value=0)
                pivot_data.plot(kind='bar', ax=ax)
                ax.set_title('Tracking Error Analysis')
                ax.set_ylabel('Tracking Error (%)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Tracking Error Analysis')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot tracking error analysis: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Tracking Error Analysis')

    def _plot_excess_return_distribution(self, experiment_records: List[Union[str, Dict]], ax):
        """
        Plot excess return distribution analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            excess_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            excess_return = algo_metrics.get('excess_return', 0) * 100
                            excess_data.append(excess_return)
            
            if excess_data:
                ax.hist(excess_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title('Excess Return Distribution')
                ax.set_xlabel('Excess Return (%)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                ax.axvline(np.mean(excess_data), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(excess_data):.2f}%')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Excess Return Distribution')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot excess return distribution: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Excess Return Distribution')

    def _plot_performance_vs_benchmark_scatter(self, experiment_records: List[Union[str, Dict]], 
                                             benchmark_name: str, ax):
        """
        Plot performance vs benchmark scatter analysis.
        
        Parameters
        ----------
        experiment_records : list
            List of experiment records
        benchmark_name : str
            Name of benchmark
        ax : matplotlib.axes.Axes
            Axes object for plotting
        """
        try:
            scatter_data = []
            for record in experiment_records:
                if isinstance(record, str):
                    with open(record, 'r') as f:
                        record_data = json.load(f)
                else:
                    record_data = record
                
                metrics = record_data.get('metrics_summary', {})
                
                if isinstance(metrics, dict):
                    for algo_name, algo_metrics in metrics.items():
                        if isinstance(algo_metrics, dict):
                            strategy_cagr = algo_metrics.get('cagr', 0) * 100
                            benchmark_cagr = algo_metrics.get('benchmark_cagr', 0) * 100
                            
                            scatter_data.append({
                                'Algorithm': algo_name,
                                'Strategy_CAGR': strategy_cagr,
                                'Benchmark_CAGR': benchmark_cagr
                            })
            
            if scatter_data:
                df = pd.DataFrame(scatter_data)
                for algo in df['Algorithm'].unique():
                    algo_data = df[df['Algorithm'] == algo]
                    ax.scatter(algo_data['Benchmark_CAGR'], algo_data['Strategy_CAGR'], 
                              label=algo, s=100, alpha=0.7)
                
                # Add 45-degree line (perfect correlation)
                min_val = min(df['Benchmark_CAGR'].min(), df['Strategy_CAGR'].min())
                max_val = max(df['Benchmark_CAGR'].max(), df['Strategy_CAGR'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
                
                ax.set_xlabel(f'{benchmark_name} CAGR (%)')
                ax.set_ylabel('Strategy CAGR (%)')
                ax.set_title('Strategy vs Benchmark Performance')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Strategy vs Benchmark Performance')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot performance vs benchmark scatter: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Strategy vs Benchmark Performance')

# Utility functions
def create_experiment_visualizer(config: ConfigSetup) -> VisualizeExperiment:
    """
    Create and return a VisualizeExperiment instance.
    
    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance
        
    Returns
    -------
    VisualizeExperiment
        New VisualizeExperiment instance
    """
    return VisualizeExperiment(config)

def generate_comprehensive_experiment_report(config: ConfigSetup, 
                                          experiment_records: List[Union[str, Dict]]) -> str:
    """
    Generate comprehensive experiment report.
    
    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance
    experiment_records : list
        List of experiment records
        
    Returns
    -------
    str
        Path to saved comprehensive report
    """
    try:
        visualizer = VisualizeExperiment(config)
        return visualizer.generate_experiment_comparison_report(experiment_records)
    except Exception as e:
        logging.error(f"VE Module - Error generating comprehensive experiment report: {e}")
        raise

def generate_experiment_benchmark_report(config: ConfigSetup, 
                                       experiment_records: List[Union[str, Dict]],
                                       benchmark_name: str = 'Nasdaq-100') -> str:
    """
    Generate experiment report with benchmark comparison.
    
    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance
    experiment_records : list
        List of experiment records
    benchmark_name : str, optional
        Name of benchmark for comparison
        
    Returns
    -------
    str
        Path to saved benchmark comparison report
    """
    try:
        visualizer = VisualizeExperiment(config)
        return visualizer.generate_experiment_comparison_with_benchmark(
            experiment_records, benchmark_name
        )
    except Exception as e:
        logging.error(f"VE Module - Error generating experiment benchmark report: {e}")
        raise