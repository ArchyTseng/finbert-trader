# visualize/visualize_features.py
"""
Module: VisualizeFeatures
Purpose: Generate visualizations for input features used in the FinBERT-Driven Trading System.
Design:
- Provides methods to visualize feature time series, distributions, and correlations.
- Integrates with the system's configuration to use standard cache directories.
- Generates dynamic filenames with timestamps for unique outputs.
Linkage: Uses ConfigTrading/ConfigSetup for cache paths and symbol lists.
Robustness: Handles missing data/columns gracefully; uses standard error handling.
Extensibility: Easy to add new visualization types or modify existing ones.
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Suppress specific warnings for cleaner output if needed
# warnings.filterwarnings('ignore', category=UserWarning) # Example

# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizeFeatures:
    """
    Class for generating visualizations for input features like prices, indicators, sentiment, and risk scores.
    """

    def __init__(self, config, prefix: str = "", suffix: str = ""):
        """
        Initialize VisualizeFeatures.

        Parameters
        ----------
        config : object
            Configuration object (e.g., ConfigTrading or ConfigSetup) containing paths and settings.
            Expected attributes: PLOT_CACHE_DIR, symbols.
        prefix : str, optional
            Prefix to add to generated filenames (e.g., "pre_normalization", "post_normalization_train").
        suffix : str, optional
            Suffix to add to generated filenames.
        """
        self.config = config
        self.prefix = prefix
        self.suffix = suffix
        self.plot_cache_dir = getattr(self.config, 'PLOT_CACHE_DIR', 'plot_cache')
        os.makedirs(self.plot_cache_dir, exist_ok=True)
        self.symbols = getattr(self.config, 'symbols', [])
        logging.info(f"VF Module - Initialized VisualizeFeatures with plot cache: {self.plot_cache_dir}")

        self.use_symbol_name = getattr(self.config, 'use_symbol_name', True)

    def _generate_filename(self, base_name: str, extension: str = ".png") -> str:
        """Generates a timestamped filename with optional prefix/suffix."""
        # Build filename components
        components = []
        if self.prefix:
            components.append(self.prefix)
        components.append(base_name)
        if self.use_symbol_name:
            symbols_name = '_'.join(self.symbols if self.symbols else "all_symbols")
            components.append(symbols_name)
        if self.suffix:
            components.append(self.suffix)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        components.append(timestamp)
        
        filename = "_".join(components) + extension
        return filename

    def _save_plot(self, fig: plt.Figure, filename: str) -> str:
        """Saves a matplotlib figure."""
        filepath = os.path.join(self.plot_cache_dir, filename)
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logging.info(f"VF Module - Saved plot to: {filepath}")
            plt.close(fig) # Important to close the figure
            return filepath
        except Exception as e:
            logging.error(f"VF Module - Failed to save plot {filename}: {e}")
            plt.close(fig)
            return ""

    def plot_feature_timeseries(self, fused_df: pd.DataFrame, symbols: Optional[List[str]] = None,
                                features_to_plot: Optional[Dict[str, int]] = None,
                                figsize: Tuple[int, int] = (25, 10)) -> Dict[str, str]:
        """
        Plots time series for key features (prices, indicators, sentiment, risk).

        Parameters
        ----------
        fused_df : pd.DataFrame
            The DataFrame containing features, typically from FeatureEngineer.merge_features.
            Expected to have a 'Date' column or DatetimeIndex.
        symbols : List[str], optional
            List of symbols to plot. Defaults to self.config.symbols.
        features_to_plot : Dict[str, int], optional
            A dictionary specifying features and how many top variance ones to plot per symbol.
            Keys: 'price', 'indicator', 'sentiment', 'risk'.
            Values: Number of top variance features to plot (e.g., {'indicator': 3}).
            If None, defaults are used.
        figsize : Tuple[int, int], optional
            Figure size for the plots (width, height).

        Returns
        -------
        Dict[str, str]
            A dictionary mapping plot descriptions to file paths.
            E.g., {'price_plot': 'plot_cache/feature_price_20231027_120000.png', ...}
        """
        symbols = symbols or self.symbols
        if not symbols:
            logging.warning("VF Module - No symbols provided or found in config for plotting.")
            return {}

        # Ensure Date is the index for time series plotting
        plot_df = fused_df.copy()
        if 'Date' in plot_df.columns:
            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
            plot_df.set_index('Date', inplace=True)
        elif not isinstance(plot_df.index, pd.DatetimeIndex):
            logging.warning("VF Module - fused_df index is not DatetimeIndex and no 'Date' column found. Plotting might be incorrect.")

        default_features = {'price': 1, 'indicator': len(self.config.indicators), 'sentiment': 1, 'risk': 1}
        features_to_plot = features_to_plot or default_features
        generated_plots = {}

        try:
            # --- 1. Price Time Series ---
            if features_to_plot.get('price', 0) > 0:
                price_cols = [col for col in plot_df.columns if "Adj_Close" in col and any(symbol in col for symbol in symbols)]
                if price_cols:
                    fig, ax = plt.subplots(figsize=figsize)
                    for col in price_cols:
                        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1) # Thinner lines for clarity
                    ax.set_title("Adjusted Close Price Time Series")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.5)
                    fig.tight_layout()
                    filename = self._generate_filename("feature_price_timeseries")
                    path = self._save_plot(fig, filename)
                    if path:
                        generated_plots['price_plot'] = path

            # --- 2. Technical Indicators (Top Variance per Symbol) ---
            n_ind = features_to_plot.get('indicator', 3)
            if n_ind > 0:
                 for symbol in symbols:
                    # Define columns that are *not* indicators for this symbol
                    non_ind_cols = (
                        [col for col in plot_df.columns if f"Adj_" in col and col.endswith(f"_{symbol}")] +
                        [col for col in plot_df.columns if f"Volume_" in col and col.endswith(f"_{symbol}")] +
                        [col for col in plot_df.columns if "sentiment_score" in col and col.endswith(f"_{symbol}")] +
                        [col for col in plot_df.columns if "risk_score" in col and col.endswith(f"_{symbol}")]
                    )
                    # Get indicator columns for this symbol
                    ind_cols = [col for col in plot_df.columns if col.endswith(f"_{symbol}") and col not in non_ind_cols]

                    if ind_cols:
                        # Calculate variance for ranking
                        variances = plot_df[ind_cols].var()
                        top_ind_cols = variances.nlargest(n_ind).index.tolist()

                        if top_ind_cols:
                            fig, ax = plt.subplots(figsize=figsize)
                            for col in top_ind_cols:
                                ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1)
                            ax.set_title(f"Top {n_ind} Technical Indicators Variance for {symbol}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Indicator Value")
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.5)
                            fig.tight_layout()
                            filename = self._generate_filename(f"feature_indicators_{symbol}")
                            path = self._save_plot(fig, filename)
                            if path:
                                generated_plots[f'indicators_plot_{symbol}'] = path

            # --- 3. Sentiment Score Time Series ---
            n_senti = features_to_plot.get('sentiment', 1)
            if n_senti > 0:
                senti_cols = [col for col in plot_df.columns if "sentiment_score" in col and any(sym in col for sym in symbols)]
                if senti_cols:
                     # Optionally, plot top N by variance or just all
                    plot_senti_cols = senti_cols[:n_senti] if len(senti_cols) > n_senti else senti_cols
                    fig, ax = plt.subplots(figsize=figsize)
                    for col in plot_senti_cols:
                        ax.plot(plot_df.index, plot_df[col], label=col, marker='o', markersize=2, linewidth=1)
                    ax.set_title("Sentiment Score Time Series")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Sentiment Score")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.5)
                    fig.tight_layout()
                    filename = self._generate_filename("feature_sentiment_timeseries")
                    path = self._save_plot(fig, filename)
                    if path:
                        generated_plots['sentiment_plot'] = path

            # --- 4. Risk Score Time Series (if enabled) ---
            n_risk = features_to_plot.get('risk', 1)
            if getattr(self.config, 'risk_mode', False) and n_risk > 0:
                risk_cols = [col for col in plot_df.columns if "risk_score" in col and any(sym in col for sym in symbols)]
                if risk_cols:
                    # Optionally, plot top N by variance or just all
                    plot_risk_cols = risk_cols[:n_risk] if len(risk_cols) > n_risk else risk_cols
                    fig, ax = plt.subplots(figsize=figsize)
                    for col in plot_risk_cols:
                        ax.plot(plot_df.index, plot_df[col], label=col, marker='x', markersize=2, linewidth=1)
                    ax.set_title("Risk Score Time Series")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Risk Score")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.5)
                    fig.tight_layout()
                    filename = self._generate_filename("feature_risk_timeseries")
                    path = self._save_plot(fig, filename)
                    if path:
                        generated_plots['risk_plot'] = path

            logging.info(f"VF Module - Feature time series visualization completed. Generated plots: {list(generated_plots.keys())}")
            return generated_plots

        except Exception as e:
            logging.error(f"VF Module - Error in plot_feature_timeseries: {e}")
            # Ensure any open figures are closed on error
            plt.close('all')
            return generated_plots # Return what was generated before error

    def plot_feature_distributions(self, fused_df: pd.DataFrame, symbols: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (12, 10)) -> Dict[str, str]:
        """
        Plots distributions (histograms/boxplots) for key feature types.

        Parameters
        ----------
        fused_df : pd.DataFrame
            The DataFrame containing features.
        symbols : List[str], optional
            List of symbols to consider. Defaults to self.config.symbols.
        figsize : Tuple[int, int], optional
            Figure size for the plots.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping plot descriptions to file paths.
        """
        symbols = symbols or self.symbols
        if not symbols:
            logging.warning("VF Module - No symbols provided or found in config for distribution plotting.")
            return {}

        plot_df = fused_df.copy()
        # No need to set index for distribution plots
        generated_plots = {}

        try:
             # --- 1. Price Distributions ---
            price_cols = [col for col in plot_df.columns if "Adj_Close" in col and any(symbol in col for symbol in symbols)]
            if price_cols:
                fig, ax = plt.subplots(figsize=figsize)
                # Melt for easier plotting with seaborn
                price_melt = plot_df[price_cols].melt(var_name='Symbol_Price', value_name='Value')
                sns.boxplot(data=price_melt, x='Symbol_Price', y='Value', ax=ax)
                ax.set_title("Adjusted Close Price Distribution (Boxplot)")
                ax.set_xlabel("Symbol_Price")
                ax.set_ylabel("Price")
                ax.tick_params(axis='x', rotation=45)
                fig.tight_layout()
                filename = self._generate_filename("feature_price_distribution")
                path = self._save_plot(fig, filename)
                if path:
                    generated_plots['price_dist_plot'] = path

            # --- 2. Sentiment Score Distributions ---
            senti_cols = [col for col in plot_df.columns if "sentiment_score" in col and any(sym in col for sym in symbols)]
            if senti_cols:
                fig, ax = plt.subplots(figsize=figsize)
                senti_melt = plot_df[senti_cols].melt(var_name='Symbol_Sentiment', value_name='Score')
                sns.histplot(data=senti_melt, x='Score', hue='Symbol_Sentiment', kde=True, ax=ax, alpha=0.6)
                ax.set_title("Sentiment Score Distribution (Histogram)")
                ax.set_xlabel("Sentiment Score")
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                filename = self._generate_filename("feature_sentiment_distribution")
                path = self._save_plot(fig, filename)
                if path:
                    generated_plots['sentiment_dist_plot'] = path

            # --- 3. Risk Score Distributions (if enabled) ---
            if getattr(self.config, 'risk_mode', False):
                risk_cols = [col for col in plot_df.columns if "risk_score" in col and any(sym in col for sym in symbols)]
                if risk_cols:
                    fig, ax = plt.subplots(figsize=figsize)
                    risk_melt = plot_df[risk_cols].melt(var_name='Symbol_Risk', value_name='Score')
                    sns.histplot(data=risk_melt, x='Score', hue='Symbol_Risk', kde=True, ax=ax, alpha=0.6)
                    ax.set_title("Risk Score Distribution (Histogram)")
                    ax.set_xlabel("Risk Score")
                    ax.set_ylabel("Frequency")
                    fig.tight_layout()
                    filename = self._generate_filename("feature_risk_distribution")
                    path = self._save_plot(fig, filename)
                    if path:
                        generated_plots['risk_dist_plot'] = path

            # --- 4. Sample Technical Indicator Distributions ---
            # Plotting distributions for a few sample indicators from the first symbol
            if symbols:
                sample_symbol = symbols[0]
                 # Define non-indicator columns for this symbol
                non_ind_cols_sym = (
                    [col for col in plot_df.columns if f"Adj_" in col and col.endswith(f"_{sample_symbol}")] +
                    [col for col in plot_df.columns if f"Volume_" in col and col.endswith(f"_{sample_symbol}")] +
                    [col for col in plot_df.columns if "sentiment_score" in col and col.endswith(f"_{sample_symbol}")] +
                    [col for col in plot_df.columns if "risk_score" in col and col.endswith(f"_{sample_symbol}")]
                )
                # Get indicator columns for this symbol
                ind_cols_sym = [col for col in plot_df.columns if col.endswith(f"_{sample_symbol}") and col not in non_ind_cols_sym]
                sample_ind_cols = ind_cols_sym[:3] # Take first 3 as sample

                if sample_ind_cols:
                    fig, axes = plt.subplots(1, len(sample_ind_cols), figsize=(5*len(sample_ind_cols), 5), sharey=False)
                    if len(sample_ind_cols) == 1:
                        axes = [axes] # Make it iterable
                    for i, col in enumerate(sample_ind_cols):
                        sns.histplot(plot_df[col].dropna(), kde=True, ax=axes[i], alpha=0.7) # Drop NaNs for hist
                        axes[i].set_title(f"Distribution of {col}")
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("Frequency")
                    fig.suptitle(f"Sample Technical Indicator Distributions for {sample_symbol}")
                    fig.tight_layout()
                    filename = self._generate_filename(f"feature_sample_indicators_dist_{sample_symbol}")
                    path = self._save_plot(fig, filename)
                    if path:
                        generated_plots[f'sample_ind_dist_plot_{sample_symbol}'] = path

            logging.info(f"VF Module - Feature distribution visualization completed. Generated plots: {list(generated_plots.keys())}")
            return generated_plots

        except Exception as e:
            logging.error(f"VF Module - Error in plot_feature_distributions: {e}")
            plt.close('all')
            return generated_plots

    def plot_feature_correlation(self, fused_df: pd.DataFrame, symbols: Optional[List[str]] = None,
                                 sample_size: Optional[int] = 1000,
                                 figsize: Tuple[int, int] = (20, 18)) -> str:
        """
        Plots a correlation matrix heatmap for selected features.

        Parameters
        ----------
        fused_df : pd.DataFrame
            The DataFrame containing features.
        symbols : List[str], optional
            List of symbols to consider. Defaults to self.config.symbols.
        sample_size : int, optional
            Number of rows to sample for correlation calculation (for performance). If None, uses all data.
        figsize : Tuple[int, int], optional
            Figure size for the heatmap.

        Returns
        -------
        str
            File path to the saved correlation heatmap, or empty string on failure.
        """
        symbols = symbols or self.symbols
        if not symbols:
            logging.warning("VF Module - No symbols provided or found in config for correlation plotting.")
            return ""

        # Select relevant columns
        relevant_cols = [col for col in fused_df.columns
                         if any(sym in col for sym in symbols) or col == 'Date']
        if 'Date' in relevant_cols:
            relevant_cols.remove('Date')

        df_plot = fused_df[relevant_cols].copy()

        # Sampling for performance if requested
        if sample_size and len(df_plot) > sample_size:
            df_plot = df_plot.sample(n=sample_size, random_state=42)
            logging.info(f"VF Module - Correlation plot: Sampled {sample_size} rows from {len(fused_df)}.")

        if df_plot.empty:
            logging.warning("VF Module - No data available for correlation plot.")
            return ""

        try:
            # Calculate correlation matrix
            corr_matrix = df_plot.corr()

            # Plot heatmap
            fig, ax = plt.subplots(figsize=figsize)
            # Use a mask to show only lower triangle or full matrix
            # mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Uncomment for upper triangle
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                        cbar_kws={"shrink": .8}, ax=ax, fmt=".2f") # fmt for potential future annot=True
            ax.set_title("Feature Correlation Matrix")
            # Rotate labels for better readability if many features
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()

            filename = self._generate_filename("feature_correlation_heatmap")
            path = self._save_plot(fig, filename)
            if path:
                logging.info(f"VF Module - Saved feature correlation heatmap to: {path}")
            return path

        except Exception as e:
            logging.error(f"VF Module - Error in plot_feature_correlation: {e}")
            plt.close('all')
            return ""


# --- Utility Functions (Optional, for direct script usage) ---

def generate_standard_feature_visualizations(fused_df: pd.DataFrame, config: Any, prefix: str = "", suffix: str = "") -> Dict[str, Any]:
    """
    Generates a standard set of feature visualizations.

    Parameters
    ----------
    fused_df : pd.DataFrame
        The fused feature DataFrame.
    config : Any
        Configuration object.
    prefix : str, optional
        Prefix to add to generated filenames (e.g., "pre_normalization", "post_normalization_train").
    suffix : str, optional
        Suffix to add to generated filenames.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing paths to generated plots.
        Keys: 'timeseries_plots', 'distribution_plots', 'correlation_plot'
    """
    try:
        visualizer = VisualizeFeatures(config, prefix=prefix, suffix=suffix)

        ts_plots = visualizer.plot_feature_timeseries(fused_df)
        dist_plots = visualizer.plot_feature_distributions(fused_df)
        corr_plot_path = visualizer.plot_feature_correlation(fused_df)

        results = {
            'timeseries_plots': ts_plots,
            'distribution_plots': dist_plots,
            'correlation_plot': corr_plot_path
        }
        logging.info(f"VF Module - Standard feature visualizations generated successfully with prefix='{prefix}', suffix='{suffix}'.")
        return results

    except Exception as e:
        logging.error(f"VF Module - Error generating standard visualizations: {e}")
        return {
            'timeseries_plots': {},
            'distribution_plots': {},
            'correlation_plot': ""
        }