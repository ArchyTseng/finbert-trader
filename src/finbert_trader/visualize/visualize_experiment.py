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
        self.plot_cache_dir = getattr(config, 'PLOT_CACHE_DIR', 'plot_cache')
        self.experiment_cache_dir = getattr(config, 'EXPERIMENT_CACHE_DIR', 'exper_cache')
        
        # Ensure directories exist
        os.makedirs(self.plot_cache_dir, exist_ok=True)
        os.makedirs(self.experiment_cache_dir, exist_ok=True)
        
        # Initialize base visualizer
        self.base_visualizer = VisualizeBacktest(self.config)
        
        logging.info("VE Module - Initialized VisualizeExperiment")
        logging.info(f"VE Module - Plot cache directory: {self.plot_cache_dir}")
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
            
            # 1. Performance Comparison (Top algorithms across experiments)
            ax1 = plt.subplot(4, 2, 1)
            self._plot_performance_comparison(experiment_records, ax1)
            
            # 2. Parameter Sensitivity Analysis
            ax2 = plt.subplot(4, 2, 2)
            self._plot_parameter_sensitivity(experiment_records, 'trading_config.reward_scaling', ax2)
            
            # 3. Convergence Analysis
            ax3 = plt.subplot(4, 2, 3)
            self._plot_convergence_analysis(experiment_records, ax3)
            
            # 4. Risk-Return Tradeoff
            ax4 = plt.subplot(4, 2, 4)
            self._plot_risk_return_tradeoff(experiment_records, ax4)
            
            # 5. Algorithm Comparison Heatmap
            ax5 = plt.subplot(4, 2, (5, 6))
            self._plot_algorithm_comparison_heatmap(experiment_records, ax5)
            
            # 6. Experiment Timeline
            ax6 = plt.subplot(4, 2, 7)
            self._plot_experiment_timeline(experiment_records, ax6)
            
            # 7. Key Metrics Summary
            ax7 = plt.subplot(4, 2, 8)
            self._plot_metrics_summary(experiment_records, ax7)
            
            plt.tight_layout()
            
            # Save comprehensive report
            report_filename = f"experiment_comprehensive_report_{timestamp}.png"
            report_path = os.path.join(self.plot_cache_dir, report_filename)
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VE Module - Comprehensive experiment report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"VE Module - Error generating comprehensive report: {e}")
            plt.close()
            raise
    
    def _plot_performance_comparison(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot performance comparison across experiments."""
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
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot performance comparison: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_parameter_sensitivity(self, experiment_records: List[Union[str, Dict]], 
                                  parameter_name: str, ax):
        """Plot parameter sensitivity analysis."""
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
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot parameter sensitivity: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_convergence_analysis(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot convergence analysis based on training steps."""
        try:
            # This would require training metrics data
            ax.text(0.5, 0.5, 'Convergence Analysis\n(Training Data Required)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Convergence Analysis')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot convergence analysis: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_risk_return_tradeoff(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot risk-return tradeoff analysis."""
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
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot risk-return tradeoff: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_algorithm_comparison_heatmap(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot algorithm comparison heatmap."""
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
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot algorithm comparison heatmap: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_experiment_timeline(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot experiment timeline and progression."""
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
                except:
                    pass
            
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                ax.barh(range(len(df)), [1] * len(df), height=0.5)
                ax.set_yticks(range(len(df)))
                ax.set_yticklabels([f"{row['Experiment']}\n{row['Date']}" for _, row in df.iterrows()])
                ax.set_title('Experiment Timeline')
                ax.set_xlabel('Execution Order')
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot experiment timeline: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_metrics_summary(self, experiment_records: List[Union[str, Dict]], ax):
        """Plot key metrics summary."""
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
            
        except Exception as e:
            logging.warning(f"VE Module - Could not plot metrics summary: {e}")
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes)
    
    def _get_nested_value(self, dictionary: Dict, key_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
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
            
            plt.tight_layout()
            
            # Save optimization path visualization
            filename = f"optimization_path_{timestamp}.png"
            filepath = os.path.join(self.plot_cache_dir, filename)
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
            
            plt.tight_layout()
            
            # Save parameter impact analysis
            filename = f"parameter_impact_analysis_{timestamp}.png"
            filepath = os.path.join(self.plot_cache_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"VE Module - Parameter impact analysis saved: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"VE Module - Error generating parameter impact analysis: {e}")
            plt.close()
            raise

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
    visualizer = VisualizeExperiment(config)
    return visualizer.generate_experiment_comparison_report(experiment_records)