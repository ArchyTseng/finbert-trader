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

    def generate_asset_curve_comparison(self, pipeline_results: Dict[str, Any], benchmark_data: Optional[Union[List[float], pd.Series]] = None, benchmark_name: str = 'Nasdaq-100', show_performance_metrics: bool = False) -> str:
        """Generate a comparison plot of asset curves with performance metrics.

        This function plots the cumulative return curves for the backtest result.
        It prioritizes using the pre-aligned strategy and benchmark data from pipeline_results
        for accurate comparison. It expects pipeline_results to be a dict of {mode_name: backtest_data_dict}.

        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing backtest results. Expected structure:
            {
                'PPO': { # mode_name
                    'strategy_assets_with_date': pd.Series (index=dates, values=assets),
                    'benchmark_prices_with_date': pd.Series or None (index=dates, values=prices),
                    'metrics': dict, ...
                },
                # ... potentially other modes like 'CPPO', 'A2C', etc.
            }
        benchmark_data : list or pd.Series, optional
            Fallback benchmark data (e.g., prices). If provided and the relevant backtest_data lacks
            'benchmark_prices_with_date', this will be used. Default is None.
        benchmark_name : str, optional
            Name of the benchmark for legend display. Default is 'Nasdaq-100'.
        show_performance_metrics : bool, optional
            Whether to display performance metrics on the plot (currently not implemented
            in this version but kept for API compatibility/signature). Default is False.

        Returns
        -------
        str
            File path to the saved plot, or an empty string if plotting failed or data missing.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.figure(figsize=(25, 12))
            sns.set_style("whitegrid")

            # --- 关键修改 1: 正确从嵌套的 pipeline_results 中提取数据 ---
            # pipeline_results 的结构是 {mode_name: backtest_data_dict}
            if not pipeline_results:
                logging.warning("VB Module - generate_asset_curve_comparison - pipeline_results is empty. Nothing to plot.")
                plt.close()
                return ""

            # --- 策略：为第一个找到的有效模式绘制图表 ---
            # 你可以修改此逻辑以绘制所有模式或特定模式
            target_mode_name = None
            target_backtest_data = None
            strategy_assets_series = None
            benchmark_prices_series_from_data = None # 从 backtest_data 中获取的基准

            for mode_name, backtest_data in pipeline_results.items():
                # backtest_data is the dict returned by TradingBacktest.run_backtest()
                # 例如 {'metrics': ..., 'strategy_assets_with_date': pd.Series, ...}
                target_mode_name = mode_name
                target_backtest_data = backtest_data
                strategy_assets_series = backtest_data.get('strategy_assets_with_date')
                
                if strategy_assets_series is not None and not strategy_assets_series.empty:
                    # 找到第一个有效的策略数据，就跳出循环
                    logging.debug(f"VB Module - generate_asset_curve_comparison - Found valid strategy data for mode {mode_name}.")
                    # 同时尝试获取该模式的基准数据
                    benchmark_prices_series_from_data = backtest_data.get('benchmark_prices_with_date')
                    break
                else:
                    logging.warning(f"VB Module - generate_asset_curve_comparison - 'strategy_assets_with_date' not found or empty for mode {mode_name}. Checking next mode.")
                    target_mode_name = None
                    target_backtest_data = None
                    strategy_assets_series = None

            # 检查是否成功提取到了策略数据
            if strategy_assets_series is None or strategy_assets_series.empty:
                logging.error("VB Module - generate_asset_curve_comparison - Critical: Pre-aligned 'strategy_assets_with_date' (pandas Series) is missing or empty for all modes. Cannot plot accurately.")
                plt.close()
                return ""
            # --- 关键修改 1 结束 ---

            # --- 关键修改 2: 处理策略资产数据 ---
            # Ensure strategy asset series index dtype and sorted
            strategy_assets_series.index = pd.to_datetime(strategy_assets_series.index)
            strategy_assets_series.sort_index(inplace=True)

            # Normalize cumulative strategy asset
            initial_strategy_asset = strategy_assets_series.iloc[0] if len(strategy_assets_series) > 0 else 1.0
            if initial_strategy_asset > 0:
                strategy_cumulative = (strategy_assets_series / initial_strategy_asset)
            else:
                logging.warning("VB Module - generate_asset_curve_comparison - Initial strategy asset is non-positive. Plotting raw asset values.")
                strategy_cumulative = strategy_assets_series
            # --- 关键修改 2 结束 ---

            # --- 关键修改 3: 处理基准数据 ---
            # 优先使用从 backtest_data 中获取的基准价格数据
            final_benchmark_series = None
            if benchmark_prices_series_from_data is not None and not benchmark_prices_series_from_data.empty:
                logging.debug(f"VB Module - generate_asset_curve_comparison - Using benchmark data from backtest results for mode {target_mode_name}.")
                final_benchmark_series = benchmark_prices_series_from_data
            # 如果 backtest_data 中没有，且传入了 benchmark_data 参数，则使用它
            elif benchmark_data is not None:
                logging.info("VB Module - generate_asset_curve_comparison - Using provided benchmark_data as fallback.")
                if isinstance(benchmark_data, pd.Series):
                    final_benchmark_series = benchmark_data
                elif isinstance(benchmark_data, (list, np.ndarray)):
                    # 如果只提供了价格列表，需要为其创建日期索引
                    # 最好的方式是它与策略日期对齐，但这在 fallback 中难以保证
                    # 简单处理：如果长度匹配，使用策略的日期索引
                    if len(benchmark_data) == len(strategy_cumulative):
                        final_benchmark_series = pd.Series(benchmark_data, index=strategy_cumulative.index)
                        logging.warning("VB Module - generate_asset_curve_comparison - Created benchmark Series using strategy dates (lengths matched). "
                                        "Ensure this is logically correct.")
                    else:
                        logging.warning("VB Module - generate_asset_curve_comparison - benchmark_data list length mismatch with strategy. "
                                        "Plotting without benchmark or using integer index might be inaccurate.")
                        # 可以创建一个同长度的 Series，但用整数索引，但这不理想
                        # 这里我们选择不绘制基准，因为对齐不准确
                        final_benchmark_series = None 
                # Ensure index dtype and sorted if we created/finalized a series
                if final_benchmark_series is not None:
                    final_benchmark_series.index = pd.to_datetime(final_benchmark_series.index)
                    final_benchmark_series.sort_index(inplace=True)
            else:
                logging.info("VB Module - generate_asset_curve_comparison - No benchmark data available or provided.")

            plot_strategy_cum = strategy_cumulative
            plot_benchmark_cum = None

            # 如果最终获取到了基准数据 Series
            if final_benchmark_series is not None and not final_benchmark_series.empty:
                # Ensure index dtype and sorted
                final_benchmark_series.index = pd.to_datetime(final_benchmark_series.index)
                final_benchmark_series.sort_index(inplace=True)

                # --- 与策略数据进行精确对齐 ---
                # 使用 Pandas 的 join 或 concat 来确保两者在相同的日期点上绘图
                # inner join 会取交集日期，确保两者都有数据
                combined_df = pd.concat([strategy_cumulative, final_benchmark_series], axis=1, join='inner', keys=['strategy', 'benchmark'])
                if not combined_df.empty:
                    plot_strategy_cum = combined_df['strategy']
                    aligned_benchmark_prices = combined_df['benchmark']
                    
                    # 计算基准的累积收益 (归一化)
                    initial_benchmark_price = aligned_benchmark_prices.iloc[0] if len(aligned_benchmark_prices) > 0 else 1.0
                    if initial_benchmark_price > 0:
                        plot_benchmark_cum = (aligned_benchmark_prices / initial_benchmark_price)
                    else:
                        logging.warning("VB Module - generate_asset_curve_comparison - Initial benchmark price is non-positive. Plotting raw benchmark values.")
                        plot_benchmark_cum = aligned_benchmark_prices

                    # --- 绘制基准曲线 ---
                    plt.plot(plot_benchmark_cum.index, plot_benchmark_cum.values, label=benchmark_name, linewidth=2)
                    logging.info(f"VB Module - generate_asset_curve_comparison - Plotted aligned benchmark curve: {benchmark_name}")
                else:
                    logging.warning("VB Module - generate_asset_curve_comparison - No overlapping dates found between strategy and benchmark after alignment. Benchmark not plotted.")
            else:
                logging.info("VB Module - generate_asset_curve_comparison - No valid benchmark data available for plotting.")
            # --- 关键修改 3 结束 ---

            # --- 绘制策略曲线 ---
            plt.plot(plot_strategy_cum.index, plot_strategy_cum.values, label=f'{target_mode_name} Strategy', linewidth=2)
            logging.info(f"VB Module - generate_asset_curve_comparison - Plotted aligned strategy curve for mode {target_mode_name}.")

            # --- 配置图表 ---
            plt.title("Asset Curve Comparison", fontsize=16, fontweight='bold')
            
            # --- 智能设置 X 轴 ---
            x_axis = strategy_cumulative.index # 或 aligned_strategy_cum.index，如果用了对齐
            if len(x_axis) > 0 and isinstance(x_axis[0], (pd.Timestamp, datetime)):
                plt.xlabel("Date", fontsize=12)
                # 自动格式化日期，防止重叠
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                # 可以设置更智能的定位器，例如每隔 N 个点或每隔几个月
                # plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
                plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # 如果索引不是日期（例如 fallback 到了整数索引）
                plt.xlabel("Trading Days", fontsize=12)

            plt.ylabel("Cumulative Return (Initial Value = 1.0)", fontsize=12) # 或 "Normalized Portfolio Value"
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- 保存图表 ---
            plot_filename = f"asset_curve_comparison_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            os.makedirs(self.plot_exper_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"VB Module - generate_asset_curve_comparison - Asset curve comparison plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            logging.error(f"VB Module - generate_asset_curve_comparison - Error generating asset curve comparison: {e}", exc_info=True)
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

    def generate_benchmark_relative_performance(self, pipeline_results: Dict[str, Any], benchmark_returns: Optional[np.ndarray] = None, benchmark_name: str = 'Nasdaq-100') -> str:
        """Generate relative performance vs benchmark (Cumulative Excess Returns).

        This function plots the cumulative excess return (strategy return - benchmark return)
        over time. It prioritizes using the pre-aligned 'strategy_assets_with_date' Series from pipeline_results
        for accurate date-aligned comparison.

        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing backtest results. Expected structure:
            {
                'PPO': { # mode_name
                    'strategy_assets_with_date': pd.Series (index=dates, values=assets),
                    'benchmark_returns': np.ndarray,
                    'metrics': dict, ...
                },
                # ... potentially other modes
            }
        benchmark_returns : np.ndarray, optional
            Fallback benchmark returns. If provided and the relevant backtest_data lacks
            'benchmark_returns', this will be used. Default is None.
        benchmark_name : str, optional
            Name of the benchmark for plot title/display. Default is 'Nasdaq-100'.

        Returns
        -------
        str
            File path to the saved plot, or an empty string if plotting failed or data missing.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.figure(figsize=(25, 12))
            sns.set_style("whitegrid")

            # --- 关键修改 1: 正确从嵌套的 pipeline_results 中提取数据 ---
            # 策略：为第一个找到的有效模式绘制图表
            target_mode_name = None
            target_backtest_data = None
            strategy_assets_series = None
            benchmark_returns_array_from_data = None

            for mode_name, backtest_data in pipeline_results.items():
                # backtest_data is the dict returned by TradingBacktest.run_backtest()
                target_mode_name = mode_name
                target_backtest_data = backtest_data
                strategy_assets_series = backtest_data.get('strategy_assets_with_date')

                if strategy_assets_series is not None and not strategy_assets_series.empty:
                    logging.debug(f"VB Module - generate_benchmark_relative_performance - Found valid strategy data for mode {mode_name}.")
                    # 同时尝试获取该模式的基准回报率数组
                    benchmark_returns_array_from_data = backtest_data.get('benchmark_returns')
                    break
                else:
                    logging.warning(f"VB Module - generate_benchmark_relative_performance - 'strategy_assets_with_date' not found or empty for mode {mode_name}. Checking next mode.")
                    target_mode_name = None
                    target_backtest_data = None
                    strategy_assets_series = None

            # 检查是否成功提取到了策略数据
            if strategy_assets_series is None or strategy_assets_series.empty:
                logging.error("VB Module - generate_benchmark_relative_performance - Critical: Pre-aligned 'strategy_assets_with_date' (pandas Series) is missing or empty. Cannot plot accurately.")
                plt.close()
                return ""

            # --- 关键修改 2: 计算策略回报率 ---
            # Ensure index dtype and sorted
            strategy_assets_series.index = pd.to_datetime(strategy_assets_series.index)
            strategy_assets_series.sort_index(inplace=True)

            # Calculate strategy returns based on the aligned Series
            strategy_returns_series = strategy_assets_series.pct_change().fillna(0)
            # --- 关键修改 2 结束 ---

            # --- 关键修改 3: 获取并处理基准回报率 ---
            # 优先使用从 backtest_data 中获取的基准回报率数组
            final_benchmark_returns_array = None
            if benchmark_returns_array_from_data is not None and len(benchmark_returns_array_from_data) > 0:
                logging.debug(f"VB Module - generate_benchmark_relative_performance - Using benchmark returns array from backtest results for mode {target_mode_name}.")
                final_benchmark_returns_array = benchmark_returns_array_from_data
            elif benchmark_returns is not None and len(benchmark_returns) > 0:
                logging.info("VB Module - generate_benchmark_relative_performance - Using provided benchmark_returns array as fallback.")
                final_benchmark_returns_array = benchmark_returns
            else:
                logging.error("VB Module - generate_benchmark_relative_performance - No benchmark returns data available (either in pipeline_results or as function argument). Cannot calculate relative performance.")
                plt.close()
                return ""

            # --- 关键修改 4: 对齐策略和基准回报率 ---
            # 策略回报率是一个 Series (有日期索引)
            # 基准回报率是一个 numpy array (假设与策略日期对齐)
            # 我们需要确保它们在相同的日期点上对齐
            min_len = min(len(strategy_returns_series), len(final_benchmark_returns_array))
            aligned_strategy_returns = strategy_returns_series.iloc[:min_len]
            aligned_benchmark_returns = final_benchmark_returns_array[:min_len]

            # 创建基准回报率的 Series，以确保索引一致
            bench_returns_series = pd.Series(aligned_benchmark_returns, index=aligned_strategy_returns.index)

            # --- 关键修改 5: 计算累计超额回报 ---
            # 计算超额回报 (策略 - 基准)
            excess_returns_series = aligned_strategy_returns - bench_returns_series
            # 计算累计超额回报 (从0开始累加)
            cumulative_excess_series = excess_returns_series.cumsum()
            # --- 关键修改 5 结束 ---

            # --- 绘图 ---
            plt.plot(cumulative_excess_series.index, cumulative_excess_series.values, 
                    label=f"{target_mode_name} vs {benchmark_name}", linewidth=2, color='purple')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            logging.info(f"VB Module - generate_benchmark_relative_performance - Plotted relative performance (cumulative excess returns).")

            # --- 配置图表 ---
            plt.title("Relative Performance vs Benchmark (Cumulative Excess Returns)", fontsize=16, fontweight='bold')
            
            # --- 智能设置 X 轴 ---
            x_axis = cumulative_excess_series.index
            if len(x_axis) > 0 and isinstance(x_axis[0], (pd.Timestamp, datetime)):
                plt.xlabel("Date", fontsize=12)
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                plt.xlabel("Trading Days", fontsize=12)

            plt.ylabel("Cumulative Excess Return", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- 保存图表 ---
            plot_filename = f"relative_performance_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            os.makedirs(self.plot_exper_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"VB Module - generate_benchmark_relative_performance - Relative performance plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            logging.error(f"VB Module - generate_benchmark_relative_performance - Error generating relative performance plot: {e}", exc_info=True)
            plt.close() # Ensure plot is closed on error
            raise # Re-raise to notify calling function


    def generate_drawdown_comparison(self, pipeline_results: Dict[str, Any], benchmark_data: Optional[Union[List[float], pd.Series]] = None, benchmark_name: str = 'Nasdaq-100') -> str:
        """Generate drawdown comparison plot for strategies and benchmark.

        This function plots the drawdown curves for multiple backtest results.
        It prioritizes using the pre-aligned 'strategy_assets_with_date' Series from pipeline_results
        for accurate date-aligned comparison.

        Parameters
        ----------
        pipeline_results : dict
            Dictionary containing backtest results. Expected structure:
            {
                'PPO': { # mode_name
                    'strategy_assets_with_date': pd.Series (index=dates, values=assets),
                    'benchmark_prices_with_date': pd.Series or None (index=dates, values=prices),
                    'metrics': dict, ...
                },
                # ... potentially other modes
            }
        benchmark_data : list or pd.Series, optional
            Fallback benchmark data (e.g., prices). If provided and the relevant backtest_data lacks
            'benchmark_prices_with_date', this will be used. Default is None.
        benchmark_name : str, optional
            Name of the benchmark for legend display. Default is 'Nasdaq-100'.

        Returns
        -------
        str
            File path to the saved plot, or an empty string if plotting failed or data missing.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.figure(figsize=(25, 12))
            sns.set_style("whitegrid")

            # --- 关键修改 1: 正确从嵌套的 pipeline_results 中提取数据 ---
            # 策略：为第一个找到的有效模式绘制图表
            target_mode_name = None
            target_backtest_data = None
            strategy_assets_series = None
            benchmark_prices_series_from_data = None

            for mode_name, backtest_data in pipeline_results.items():
                # backtest_data is the dict returned by TradingBacktest.run_backtest()
                target_mode_name = mode_name
                target_backtest_data = backtest_data
                strategy_assets_series = backtest_data.get('strategy_assets_with_date')

                if strategy_assets_series is not None and not strategy_assets_series.empty:
                    logging.debug(f"VB Module - generate_drawdown_comparison - Found valid strategy data for mode {mode_name}.")
                    # 同时尝试获取该模式的基准价格数据
                    benchmark_prices_series_from_data = backtest_data.get('benchmark_prices_with_date')
                    break
                else:
                    logging.warning(f"VB Module - generate_drawdown_comparison - 'strategy_assets_with_date' not found or empty for mode {mode_name}. Checking next mode.")
                    target_mode_name = None
                    target_backtest_data = None
                    strategy_assets_series = None

            # 检查是否成功提取到了策略数据
            if strategy_assets_series is None or strategy_assets_series.empty:
                logging.error("VB Module - generate_drawdown_comparison - Critical: Pre-aligned 'strategy_assets_with_date' (pandas Series) is missing or empty. Cannot plot accurately.")
                plt.close()
                return ""

            # --- 关键修改 2: 计算策略回撤 ---
            # Ensure index dtype and sorted
            strategy_assets_series.index = pd.to_datetime(strategy_assets_series.index)
            strategy_assets_series.sort_index(inplace=True)

            # Calculate drawdown based on the aligned Series
            assets = strategy_assets_series.values
            rolling_max = np.maximum.accumulate(assets)
            # Avoid division by zero
            drawdown_denominator = np.where(rolling_max > 0, rolling_max, 1e-8)
            strategy_drawdown = (assets - rolling_max) / drawdown_denominator * 100 # Percentage
            # Create a Series for drawdown with the same index as the asset series
            strategy_drawdown_series = pd.Series(strategy_drawdown, index=strategy_assets_series.index)
            # --- 关键修改 2 结束 ---

            # --- 关键修改 3: 处理基准数据 ---
            # 优先使用从 backtest_data 中获取的基准价格数据
            final_benchmark_series = None
            if benchmark_prices_series_from_data is not None and not benchmark_prices_series_from_data.empty:
                logging.debug(f"VB Module - generate_drawdown_comparison - Using benchmark data from backtest results for mode {target_mode_name}.")
                final_benchmark_series = benchmark_prices_series_from_data
            elif benchmark_data is not None:
                logging.info("VB Module - generate_drawdown_comparison - Using provided benchmark_data as fallback.")
                if isinstance(benchmark_data, pd.Series):
                    final_benchmark_series = benchmark_data
                elif isinstance(benchmark_data, (list, np.ndarray)):
                    # 如果只提供了价格列表，需要为其创建日期索引
                    # 最好的方式是它与策略日期对齐，但这在 fallback 中难以保证
                    # 简单处理：如果长度匹配，使用策略的日期索引
                    if len(benchmark_data) == len(strategy_drawdown_series):
                        final_benchmark_series = pd.Series(benchmark_data, index=strategy_drawdown_series.index)
                        logging.warning("VB Module - generate_drawdown_comparison - Created benchmark Series using strategy dates (lengths matched). "
                                        "Ensure this is logically correct.")
                    else:
                        logging.warning("VB Module - generate_drawdown_comparison - benchmark_data list length mismatch with strategy. "
                                        "Plotting without benchmark or using integer index might be inaccurate.")
                        final_benchmark_series = None
                # Ensure index dtype and sorted if we created/finalized a series
                if final_benchmark_series is not None:
                    final_benchmark_series.index = pd.to_datetime(final_benchmark_series.index)
                    final_benchmark_series.sort_index(inplace=True)
            else:
                logging.info("VB Module - generate_drawdown_comparison - No benchmark data available or provided.")

            # 如果最终获取到了基准数据 Series
            if final_benchmark_series is not None and not final_benchmark_series.empty:
                # Ensure index dtype and sorted
                final_benchmark_series.index = pd.to_datetime(final_benchmark_series.index)
                final_benchmark_series.sort_index(inplace=True)

                # --- 与策略数据进行精确对齐 ---
                # 使用 Pandas 的 join 或 concat 来确保两者在相同的日期点上绘图
                combined_df = pd.concat([strategy_drawdown_series, final_benchmark_series], axis=1, join='inner', keys=['strategy', 'benchmark'])
                if not combined_df.empty:
                    aligned_strategy_drawdown = combined_df['strategy']
                    aligned_benchmark_prices = combined_df['benchmark']

                    # Calculate benchmark drawdown
                    benchmark_assets = aligned_benchmark_prices.values
                    benchmark_rolling_max = np.maximum.accumulate(benchmark_assets)
                    benchmark_drawdown_denominator = np.where(benchmark_rolling_max > 0, benchmark_rolling_max, 1e-8)
                    benchmark_drawdown = (benchmark_assets - benchmark_rolling_max) / benchmark_drawdown_denominator * 100
                    benchmark_drawdown_series = pd.Series(benchmark_drawdown, index=aligned_benchmark_prices.index)

                    # --- 绘制基准回撤曲线 ---
                    plt.plot(benchmark_drawdown_series.index, benchmark_drawdown_series.values, label=benchmark_name, linewidth=2)
                    logging.info(f"VB Module - generate_drawdown_comparison - Plotted benchmark drawdown curve: {benchmark_name}")
                else:
                    logging.warning("VB Module - generate_drawdown_comparison - No overlapping dates found between strategy and benchmark after alignment. Benchmark not plotted.")
            else:
                logging.info("VB Module - generate_drawdown_comparison - No valid benchmark data available for plotting.")
            # --- 关键修改 3 结束 ---

            # --- 绘制策略回撤曲线 ---
            plt.plot(strategy_drawdown_series.index, strategy_drawdown_series.values, label=f'{target_mode_name} Strategy', linewidth=2)
            logging.info(f"VB Module - generate_drawdown_comparison - Plotted strategy drawdown curve for mode {target_mode_name}.")

            # --- 配置图表 ---
            plt.title("Drawdown Comparison", fontsize=16, fontweight='bold')
            
            # --- 智能设置 X 轴 ---
            x_axis = strategy_drawdown_series.index
            if len(x_axis) > 0 and isinstance(x_axis[0], (pd.Timestamp, datetime)):
                plt.xlabel("Date", fontsize=12)
                # 自动格式化日期，防止重叠
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                plt.xlabel("Trading Days", fontsize=12)

            plt.ylabel("Drawdown (%)", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- 保存图表 ---
            plot_filename = f"drawdown_comparison_{timestamp}.png"
            plot_path = os.path.join(self.plot_exper_dir, plot_filename)
            os.makedirs(self.plot_exper_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"VB Module - generate_drawdown_comparison - Drawdown comparison plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            logging.error(f"VB Module - generate_drawdown_comparison - Error generating drawdown comparison: {e}", exc_info=True)
            plt.close() # Ensure plot is closed on error
            raise # Re-raise to notify calling function


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