# visualize/visualize_news.py
"""
Module: VisualizeNews
Purpose: Analyze and visualize news data coverage for stock selection.
Design:
- Supports chunked loading of large news datasets.
- Integrates with existing news processing pipeline (load_news_data, compute_sentiment_risk_score).
- Provides detailed statistics and visualizations for news coverage per stock.
Linkage: Uses ConfigSetup/ConfigTrading for symbols, dates, and cache paths.
Robustness: Handles missing data/columns gracefully; uses standard error handling.
Extensibility: Easy to add new visualization types or modify existing ones.
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizeNews:
    def __init__(self, config):
        """
        Initialize VisualizeNews.

        Parameters
        ----------
        config : object
            Configuration object containing paths, dates, and symbols.
        """
        self.config = config
        self.raw_data_path = getattr(self.config, 'RAW_DATA_DIR', 'raw_data_cache')
        self.plot_news_dir = getattr(self.config, 'PLOT_NEWS_DIR', 'plot_news_cache')
        os.makedirs(self.plot_news_dir, exist_ok=True)
        self.symbols = getattr(self.config, 'symbols', [])
        self.start_date = pd.to_datetime(getattr(self.config, 'start', '2020-01-01'))
        self.end_date = pd.to_datetime(getattr(self.config, 'end', '2022-12-31'))

        self.use_symbol_name = getattr(self.config, 'use_symbol_name', True)

        logging.info(f"VN Module - Initialized VisualizeNews with raw data path: {self.raw_data_path}")
        logging.info(f"VN Module - Analysis period: {self.start_date} to {self.end_date}")
        logging.info(f"VN Module - Symbols to analyze: {self.symbols}")

    def _generate_filename(self, base_name: str, extension: str = ".png") -> str:
        """Generates a timestamped filename."""
        if self.use_symbol_name:
            symbols_name = '_'.join(self.symbols if self.symbols else "all_symbols" )
        else:
            symbols_name = "all_symbols"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{symbols_name}_{timestamp}{extension}"

    def _save_plot(self, fig: plt.Figure, filename: str) -> str:
        """Saves a matplotlib figure."""
        filepath = os.path.join(self.plot_news_dir, filename)
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logging.info(f"VN Module - Saved plot to: {filepath}")
            plt.close(fig)
            return filepath
        except Exception as e:
            logging.error(f"VN Module - Failed to save plot {filename}: {e}")
            plt.close(fig)
            return ""

    def analyze_news_coverage(self, news_df: pd.DataFrame, filter_cols: List =None) -> pd.DataFrame:
        """
        Analyzes news coverage for each symbol within the specified date range.

        Parameters
        ----------
        news_df : pd.DataFrame
            DataFrame containing news data with columns ['Date', 'Symbol'].

        Returns
        -------
        pd.DataFrame
            A DataFrame with statistics for each symbol's news coverage.
            Columns: ['Symbol', 'Total_News_Count', 'Unique_Dates', 'Coverage_Days',
                      'Start_Date', 'End_Date', 'News_Per_Day']
        """
        if news_df.empty:
            logging.warning("VN Module - analyze_news_coverage - No news data provided.")
            return pd.DataFrame()

        filter_cols = filter_cols if filter_cols is not None else self.config.text_cols

        mask = ((news_df[filter_cols].notna()) & (news_df[filter_cols] != '')).all(axis=1)
        filtered_news_df = news_df[mask]
        logging.info(f"VN Module - analyze_news_coverage - Raw news data length: {len(news_df)}, after filtered length: {len(filtered_news_df)}")

        if filtered_news_df.empty:
            logging.warning(f"VN Module - analyze_news_coverage - No valid news data after filtering empty {filter_cols}")
            return pd.DataFrame()

        stats_list = []
        for symbol in self.symbols:
            df_symbol = filtered_news_df[filtered_news_df['Symbol'] == symbol]
            total_count = len(df_symbol)
            unique_dates = df_symbol['Date'].nunique()
            start_date = df_symbol['Date'].min() if not df_symbol.empty else pd.NaT
            end_date = df_symbol['Date'].max() if not df_symbol.empty else pd.NaT
            coverage_days = (end_date - start_date).days + 1 if pd.notna(start_date) and pd.notna(end_date) else 0
            news_per_day = total_count / coverage_days if coverage_days > 0 else 0.0

            stats_list.append({
                'Symbol': symbol,
                'Total_News_Count': total_count,
                'Unique_Dates': unique_dates,
                'Coverage_Days': coverage_days,
                'Start_Date': start_date,
                'End_Date': end_date,
                'News_Per_Day': news_per_day
            })

        stats_df = pd.DataFrame(stats_list)
        stats_df = stats_df.sort_values(by='Total_News_Count', ascending=False).reset_index(drop=True)

        logging.info(f"VN Module - analyze_news_coverage - Completed news coverage analysis.")
        return stats_df

    def plot_news_coverage(self, coverage_stats_df: pd.DataFrame, figsize: Tuple[int, int] = (14, 10)) -> Dict[str, str]:
        """
        Plots visualizations for the news coverage statistics.

        Creates:
        1. Bar chart of Total News Count per Symbol.
        2. Heatmap/calendar view of news distribution over time (if data allows).

        Parameters
        ----------
        coverage_stats_df : pd.DataFrame
            DataFrame returned by analyze_news_coverage.
        figsize : Tuple[int, int], optional
            Figure size for the plots.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping plot descriptions to file paths.
        """
        if coverage_stats_df.empty:
            logging.warning("VN Module - plot_news_coverage - Coverage stats DataFrame is empty.")
            return {}

        generated_plots = {}
        symbols = coverage_stats_df['Symbol'].tolist()

        try:
            # --- 1. Bar Chart: Total News Count ---
            fig, ax = plt.subplots(figsize=figsize)
            bars = ax.bar(coverage_stats_df['Symbol'], coverage_stats_df['Total_News_Count'], color='skyblue')
            ax.set_title('Total News Count per Stock Symbol')
            ax.set_xlabel('Stock Symbol')
            ax.set_ylabel('Number of News Articles')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.75)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

            fig.tight_layout()
            filename = self._generate_filename("news_coverage_count")
            path = self._save_plot(fig, filename)
            if path:
                generated_plots['news_count_bar_chart'] = path

            # --- 2. Scatter Plot: News Count vs. Coverage Days ---
            # This can help identify symbols with high count but low duration or vice versa.
            if not coverage_stats_df[['Total_News_Count', 'Coverage_Days']].isnull().any().any():
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    coverage_stats_df['Total_News_Count'],
                    coverage_stats_df['Coverage_Days'],
                    c=coverage_stats_df['News_Per_Day'],
                    cmap='viridis',
                    s=100,
                    alpha=0.7,
                    edgecolors='black'
                )
                ax.set_xlabel('Total News Count')
                ax.set_ylabel('Coverage Days')
                ax.set_title('News Count vs. Coverage Duration (Color: News/Day)')

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('News Articles per Day')

                # Annotate points with symbol names
                for i, txt in enumerate(coverage_stats_df['Symbol']):
                    ax.annotate(txt, (coverage_stats_df['Total_News_Count'].iloc[i], coverage_stats_df['Coverage_Days'].iloc[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.9)

                ax.grid(True, alpha=0.5)
                fig.tight_layout()
                filename = self._generate_filename("news_count_vs_duration")
                path = self._save_plot(fig, filename)
                if path:
                    generated_plots['news_count_vs_duration_scatter'] = path

            logging.info(f"VN Module - plot_news_coverage - Generated plots: {list(generated_plots.keys())}")
            return generated_plots

        except Exception as e:
            logging.error(f"VN Module - plot_news_coverage - Error during plotting: {e}")
            plt.close('all')
            return generated_plots
        
    def plot_news_frequency_per_stock(self, news_df: pd.DataFrame, figsize: Tuple[int, int] = (32, 8)) -> str:
        """
        Plots the number of news articles per stock per day.

        Parameters
        ----------
        news_df : pd.DataFrame
            DataFrame containing news data with columns ['Date', 'Symbol'].
        figsize : Tuple[int, int], optional
            Figure size for the plot.

        Returns
        -------
        str
            Path to the saved plot.
        """
        if news_df.empty:
            logging.warning("VN Module - plot_news_frequency_per_stock - No news data provided.")
            return ""

        try:
            # Step 1: Group by Date and Symbol, count news articles
            grouped = news_df.groupby(['Date', 'Symbol']).size().reset_index(name='News_Count')
            grouped['Date'] = pd.to_datetime(grouped['Date'])

            # Step 2: Pivot the data for easier plotting
            pivoted = grouped.pivot(index='Date', columns='Symbol', values='News_Count').fillna(0)

            # Step 3: Plot the time series
            fig, ax = plt.subplots(figsize=figsize)
            sns.lineplot(data=pivoted, ax=ax, marker='o', markersize=5)
            ax.set_title('Daily News Frequency per Stock')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of News Articles')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.75)

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Stock Symbol', bbox_to_anchor=(1.05, 1), loc='upper left')

            # Save the plot
            filename = self._generate_filename("news_frequency_per_stock")
            return self._save_plot(fig, filename)

        except Exception as e:
            logging.error(f"VN Module - plot_news_frequency_per_stock - Error during plotting: {e}")
            plt.close('all')
            return ""

    def generate_news_analysis(self, news_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Generates comprehensive news analysis and visualizations.

        Parameters
        ----------
        news_df : pd.DataFrame
            DataFrame containing news data with columns ['Date', 'Symbol'].

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, str]]
            - A DataFrame with news coverage statistics.
            - A dictionary of paths to generated plots.
        """
        if news_df.empty:
            logging.warning("VN Module - generate_news_analysis - No news data provided.")
            return pd.DataFrame(), {}

        try:
            # Analyze news coverage
            coverage_stats = self.analyze_news_coverage(news_df)

            # Step 2: Generate visualizations
            plot_paths = self.plot_news_coverage(coverage_stats)

            # Step 3: Add daily news frequency per stock plot
            news_freq_path = self.plot_news_frequency_per_stock(news_df)
            if news_freq_path:
                plot_paths['news_frequency_per_stock'] = news_freq_path

            return coverage_stats, plot_paths

        except Exception as e:
            logging.error(f"VN Module - generate_news_analysis - Error during analysis or visualization: {e}")
            return pd.DataFrame(), {}

# --- Utility Function for Pipeline Integration ---

def select_stocks_by_news_coverage(config: Any,
                                  symbols_list: List[str],
                                  top_n: Optional[int] = None,
                                  min_news_count: int = None,
                                  min_coverage_days: int = None,
                                  news_data_path: Optional[str] = None,
                                  news_df: Optional[pd.DataFrame] = None) -> Tuple[List[str], pd.DataFrame, Dict[str, str]]:
    """
    Utility function to select stocks based on news coverage quality.

    Parameters
    ----------
    config : Any
        Configuration object.
    symbols_list : List[str]
        Initial list of stock symbols to evaluate.
    top_n : int, optional
        Number of top stocks to select based on news count. If provided, overrides min_* thresholds.
    min_news_count : int, optional
        Minimum number of news articles required for a stock to be considered.
    min_coverage_days : int, optional
        Minimum number of unique days covered by news for a stock to be considered.
    news_data_path : str, optional
        Path to the FNSPID CSV file.

    Returns
    -------
    Tuple[List[str], pd.DataFrame, Dict[str, str]]
        A tuple containing:
        1. The list of selected stock symbols.
        2. The full coverage statistics DataFrame.
        3. A dictionary of paths to generated plots.
    """
    try:
        analyzer = VisualizeNews(config)

        # Analyze news coverage
        coverage_stats, plot_paths = analyzer.generate_news_analysis(news_df)

        # Select stocks based on criteria
        selected_symbols = []
        # Get max news count and max coverage days if not provided
        max_news_count = coverage_stats['Total_News_Count'].max()
        max_coverage_days = coverage_stats['Coverage_Days'].max()
        # Set news selection threshold
        min_news_count = min_news_count if min_news_count is not None else int(max_news_count * 0.8)
        min_coverage_days = min_coverage_days if min_coverage_days is not None else int(max_coverage_days * 0.8)
        # Select target symbols
        mask = (coverage_stats['Total_News_Count'] >= min_news_count) & (coverage_stats['Coverage_Days'] >= min_coverage_days)
        selected_symbols_temp = coverage_stats[mask]['Symbol'].tolist()
        logging.info(f"VN Module - select_stocks_by_news_coverage - Selected stocks meeting criteria (count>={min_news_count}, days>={min_coverage_days}): {selected_symbols}")

        if top_n is not None and top_n < len(selected_symbols_temp):
            # Select top N 
            selected_symbols = selected_symbols_temp[:top_n]
            logging.info(f"VN Module - select_stocks_by_news_coverage - Selected top {top_n} stocks by news count: {selected_symbols}")
        else:
            # Select all stocks meeting criteria
            selected_symbols = selected_symbols_temp
        if not selected_symbols:
            logging.warning("VN Module - select_stocks_by_news_coverage - No stocks met the selection criteria.")

        analyzer.config._update_selected_symbols(selected_symbols, min_count=min_news_count, min_days=min_coverage_days)
        logging.debug(f"VN Module - select_stocks_by_news_coverage - Updated config symbols with selected symbols: {analyzer.config.symbols}")
        return selected_symbols, coverage_stats, plot_paths

    except Exception as e:
        logging.error(f"VN Module - select_stocks_by_news_coverage - Error: {e}")
        return [], pd.DataFrame(), {}