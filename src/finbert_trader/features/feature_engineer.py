# -*- coding: utf-8 -*-
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
# feature_engineer.py
# Module: FeatureEngineer
# Purpose: Orchestrator for feature engineering; delegates to StockFeatureEngineer and NewsFeatureEngineer.
# Design: Manages merge, normalize, prepare, split; supports experiment modes.
# Linkage: Inputs from DataResource; outputs split RL data for Environment/Agent.
# Extensibility: Supports 'rl_algorithm' group in exper_mode; fixed news processing to 'title_fulltext'.
# Robustness: Checks sentiment variance; adds mode-specific noise; validates splits.
# Updates: Added 'rl_algorithm' group handling; fixed news to 'title_fulltext'; added 'model_type' in exper_data_dict.
# Updates: Added risk_score computation if config.risk_mode; merged with sentiment in fused_df; adjusted _check_and_adjust_sentiment for both scores (var<0.1 add noise); extended feature_cols to include 'risk_score'; prepare_rl_data includes risk in states, reference from FinRL_DeepSeek (4.3: aggregate R_f for returns adjustment).

# %%
# import os
# os.chdir('/Users/archy/Projects/finbert_trader/')

# %%
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import logging
import hashlib
import os
from datetime import datetime
import joblib

# %%
from .stock_features import StockFeatureEngineer
from .news_features import NewsFeatureEngineer
from ..visualize.visualize_features import generate_standard_feature_visualizations

# %%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
class FeatureEngineer:
    def __init__(self, config):
        """
        Introduction
        ------------
        Initialize the FeatureEngineer with configuration parameters.
        Sets up decay lambda, window sizes, split options, experiment modes, and instantiates stock/news engineers.
        Configures experiment data path and train/valid/test date ranges.

        Parameters
        ----------
        config : object
            Configuration object containing all necessary parameters.

        Notes
        -----
        - Extracts parameters like decay_lambda, window_size, etc., from config for centralized management.
        - Instantiates StockFeatureEngineer and NewsFeatureEngineer for modular feature computation.
        - Initializes fused_dfs dict for caching fused DataFrames across methods.
        - Date ranges reference FinRL conventions for consistent time-series splitting.
        - Experiment data saved to config.EXPER_DATA_DIR.
        """
        self.config = config  
        self.decay_lambda = self.config.decay_lambda  # Decay factor for potential time-weighted features

        self.window_size = getattr(self.config, 'window_size', 50)  # Window size for RL observation states
        self.W_f = getattr(self.config, 'window_factor', 2)    # For scale window size
        self.W_e = getattr(self.config, 'window_extend', 10)     # For extend window size
        self.prediction_days = getattr(self.config, 'prediction_days', 3)  # Number of future days for target prediction

        self.split_ratio = self.config.split_ratio  # Ratio for data splitting if split_mode='ratio'
        self.k_folds = self.config.k_folds  # Number of folds for cross-validation
        self.split_mode = self.config.split_mode  # Mode for data splitting ('date' or 'ratio')
        self.cross_valid_mode = self.config.cross_valid_mode  # Cross-validation type ('time_series' or 'kfold')
        self.exper_mode = self.config.exper_mode  # Dictionary of experiment modes/groups
        self.risk_mode = self.config.risk_mode  # Flag to enable risk score computation

        self.stock_engineer = StockFeatureEngineer(config)  # Instantiate stock feature engineer for technical indicators
        self.news_engineer = NewsFeatureEngineer(config)  # Instantiate news feature engineer for sentiment/risk
        self.fused_dfs = {}  # Initialize dict to cache fused DataFrames per mode for reuse

        # Config exper_data_dict save dir
        self.exper_data_path = self.config.EXPER_DATA_DIR  # Path for saving/loading experiment data (e.g., NPZ files)
        self.scaler_cache_path = self.config.SCALER_CACHE_DIR   # Path for saving/loading scaler data (e.g., PKL files)

        # Config train/valid/test date , reference from FinRL
        self.train_start_date = self.config.train_start_date  # Start date for training data
        self.train_end_date = self.config.train_end_date  # End date for training data
        self.valid_start_date = self.config.valid_start_date  # Start date for validation data
        self.valid_end_date = self.config.valid_end_date  # End date for validation data
        self.test_start_date = self.config.test_start_date  # Start date for test data
        self.test_end_date = self.config.test_end_date  # End date for test data

        self.save_npz = getattr(self.config, 'save_npz', True)  # Flag to save NPZ files)
        self.load_npz = getattr(self.config, 'load_npz', False)  # Flag to load NPZ files)

        self.smooth_window = getattr(self.config, 'smooth_window_size', 5)

        self.plot_feature_visualization = getattr(self.config, 'plot_feature_visualization', False)
        self.force_process_news = getattr(self.config, 'force_process_news', False)
        self.force_fuse_data = getattr(self.config, 'force_fuse_data', False)
        self.force_normalize_features = getattr(self.config, 'force_normalize_features', True)
        self.filter_ind = getattr(self.config, 'filter_ind', [])

        self.use_symbol_name = getattr(self.config, 'use_symbol_name', True)  # Flag to use symbol name in cache file)

        self.fused_data_dir = getattr(self.config, 'FUSED_DATA_DIR', 'fused_data_cache')    # Config for saving fused_df after merge_features
        self.processed_news_dir = getattr(self.config, 'PROCESSED_NEWS_DIR', 'processed_news_cache')

    def process_news_chunks(self, news_chunks_gen):
        """
        Introduction
        ------------
        Process news data chunks from a generator, clean each chunk, and aggregate into a single DataFrame.
        Skips empty chunks; returns empty DF with predefined columns if no valid data.

        Parameters
        ----------
        news_chunks_gen : generator
            Generator yielding news DataFrame chunks.

        Returns
        -------
        pd.DataFrame
            Aggregated cleaned news DataFrame, or empty with columns matching FNSPID dataset.

        Notes
        -----
        - Cleans each chunk via news_engineer.clean_news_data (drops useless columns, cleans text).
        - Optional random filtering per symbol/day is commented out.
        - Aggregates via pd.concat; logs row count for monitoring.
        - Empty return maintains column structure for downstream compatibility.
        """
        processed_news_list = os.listdir(self.processed_news_dir)
        if len(processed_news_list) > 0 and not self.force_process_news:
            logging.info(f"FE Module - _news_chunks - Exist {len(processed_news_list)} processed news cache : {processed_news_list}")
            path_suffix = self._generate_path_suffix(extension='.csv')
            
            # Check cache file
            for filename in processed_news_list:
                if filename.endswith(path_suffix):
                    self.config.news_cache_path = os.path.join(self.processed_news_dir, filename)
                    logging.info(f"FE Module - _news_chunks - Target cache path : {self.config.news_cache_path}")
                    
                    # Load cache
                    logging.info(f"FE Module - _news_chunks - Load news_df for {self.config.symbols}")
                    news_df = pd.read_csv(self.config.news_cache_path, parse_dates=['Date'])
                    logging.info(f"FE Module - _news_chunks - Loaded news_df: {len(news_df)} rows")
                    return news_df
                
        # If no such cache, reprocess
        logging.info("FE Module - _news_chunks - No symbols or cache_path provided, processing all news chunks")
        processed_chunks = []  # List to collect cleaned chunks for aggregation
        for chunk in news_chunks_gen:
            # Iterate over generator chunks to process streaming data efficiently
            if chunk.empty:
                continue  # Skip empty chunks to avoid unnecessary processing
            cleaned_chunk = self.news_engineer.clean_news_data(chunk)   # Drop useless columns and clean text for each column
            # filtered_chunk = self.news_engineer.filter_random_news(cleaned_chunk)   # Filter one random news for each symbol per day (commented out; optional for reducing duplicates)
            if not cleaned_chunk.empty:
                processed_chunks.append(cleaned_chunk)  # Append only non-empty cleaned chunks
        if processed_chunks:
            aggregated_df = pd.concat(processed_chunks, ignore_index=True)  # Concatenate all chunks into a single DF, reset index for clean aggregation
            logging.info(f"FE Module - _news_chunks - Aggregated cleaned news: {len(aggregated_df)} rows")  # Log aggregated row count for data volume tracking
            processed_path = self.save_target_data_csv(aggregated_df, prefix="Processed", save_path=self.processed_news_dir)
            self.config.processed_cache_path = processed_path
            logging.info(f"FE Module - _news_chunks - Saved processed news to {processed_path}")
            return aggregated_df  # Return the full aggregated DataFrame
        logging.info("FE Module - _news_chunks - No valid news chunks, returning empty DataFrame")  # Log if no data processed
        return pd.DataFrame(columns=['Date', 'Symbol', 'Article_title', 'Full_Text', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']) # Match original columns in FNSPID dataset for consistency

    def _fill_score_columns(self, df, prefix, fill_value=3.0):
        """
        Introduction
        ------------
        Fill NaN values in columns starting with a given prefix (e.g., 'sentiment_score_').
        Computes and logs the number of NaNs filled for monitoring data quality post-merge.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing columns to fill.
        prefix : str
            Prefix to match columns for filling (e.g., 'sentiment_score_').
        fill_value : float, optional
            Value to fill NaNs with (default: 3.0, neutral for scores).

        Returns
        -------
        pd.DataFrame
            The DataFrame with NaNs filled in matched columns.

        Notes
        -----
        - Filters columns by prefix; skips if none match.
        - Calculates NaNs before/after to log filled count.
        - Useful for sentiment/risk columns after pivoting and merging.
        """
        score_cols = [col for col in df.columns if col.startswith(prefix)]  # Filter columns matching the prefix
        # Sum NaN values before filling
        nulls_before = df[score_cols].isna().sum().sum()  # Count total NaNs in matched columns pre-fill
        df[score_cols] = df[score_cols].fillna(fill_value)  # Batch fill NaNs with specified value
        # Sum Nan values after filling
        nulls_after = df[score_cols].isna().sum().sum()  # Count post-fill to verify
        logging.info(f"FE Module - _fill_score_columns - Filled {nulls_before - nulls_after} NaNs in {prefix} columns")  # Log filled count for data quality tracking
        return df  # Return the updated DataFrame
    
    def _fill_nan_after_merge(self, df):
        """
        Introduction
        ------------
        Fill NaN values in the fused DataFrame after merging features.
        Applies column-specific filling: 3.0 for sentiment/risk scores (neutral value), 0 for technical indicators.

        Parameters
        ----------
        df : pd.DataFrame
            The fused DataFrame with potential NaNs post-merge.

        Returns
        -------
        pd.DataFrame
            The DataFrame with NaNs filled according to rules.

        Notes
        -----
        - Checks for NaNs per column; fills only if any present.
        - Sentiment/risk: fill with 3.0 (mid-range neutral).
        - Indicators (macd, rsi, cci, sma, dx, boll, close): fill with 0.
        - Logs each filling operation for traceability.
        """
        for col in df.columns:
            # Loop through all columns to check and fill NaNs based on type
            if 'sentiment_score_' in col or 'risk_score_' in col:
                # Handle sentiment/risk columns: Fill with neutral 3.0 if NaNs exist
                if df[col].isna().any():
                    nulls_before = df[col].isna().sum().sum() 
                    df[col] = df[col].fillna(3.0)  # Neutral value for scores (e.g., 1-5 scale)
                    nulls_after = df[col].isna().sum().sum() 
                    logging.info(f"FE Module - _after_merge - Fillna {nulls_before - nulls_after} NaNs with value 3.0 in {col}")
            elif any(keyword in col for keyword in ['macd', 'rsi', 'cci', 'sma', 'dx', 'boll', 'close_sma']):
                # Handle technical indicator columns: Fill with 0 if NaNs exist
                if df[col].isna().any():
                    nulls_before = df[col].isna().sum().sum() 
                    df[col] = df[col].fillna(0)  # Zero for indicators to represent no signal
                    nulls_after = df[col].isna().sum().sum() 
                    logging.info(f"FE Module - _after_merge - Fillna {nulls_before - nulls_after} NaNs with value 0.0 in {col}")
        return df  # Return the filled DataFrame

    def merge_features(self, stock_data_dict, sentiment_df, risk_df=None, prefix='Fused_Data'):
        """
        Introduction
        ------------
        Merge stock price data with computed features, sentiment scores, and optional risk scores into a single wide DataFrame.
        Processes each symbol's features, concatenates horizontally, pivots scores by symbol, fills NaNs, filters positive Adj_Close, and reorders columns.

        Parameters
        ----------
        stock_data_dict : dict
            Dictionary of stock DataFrames keyed by symbol.
        sentiment_df : pd.DataFrame
            DataFrame with sentiment scores, columns: 'Date', 'Symbol', 'sentiment_score'.
        risk_df : pd.DataFrame, optional
            DataFrame with risk scores, similar structure (enabled if risk_mode=True).

        Returns
        -------
        pd.DataFrame
            Fused wide DataFrame with 'Date' and symbol-suffixed columns, sorted chronologically.

        Notes
        -----
        - Computes features per symbol using stock_engineer.
        - Pivots sentiment/risk to wide format with symbol prefixes.
        - Fills NaNs post-merge via _fill_score_columns and _fill_nan_after_merge.
        - Filters rows where any Adj_Close_{symbol} <= 0.
        - Reorders columns grouping by symbol for consistency.
        - Logs final shape and risk_mode status.
        """
        processed_stocks = []  # List to hold processed DataFrames per symbol
        for symbol, df in stock_data_dict.items():
            # Compute technical indicators and features for each symbol's stock data
            processed_df = self.stock_engineer.compute_features(df, symbol)
            processed_stocks.append(processed_df.set_index('Date'))  # Set Date as index for concat
        all_stock_df = pd.concat(processed_stocks, axis=1, join='outer').reset_index()  # Concat horizontally to wide table, reset index to bring back Date

        for symbol in self.config.symbols:
            # Fallback: Rename Volume if not suffixed, to ensure consistency across symbols
            if f'Volume_{symbol}' not in all_stock_df.columns and 'Volume' in all_stock_df.columns:
                all_stock_df.rename(columns={'Volume': f'Volume_{symbol}'}, inplace=True)  # Fallback suffix if missed

        if not sentiment_df.empty:
            # Pivot sentiment to wide format with symbol prefixes for merging
            sentiment_df = sentiment_df.pivot(index='Date',
                                            columns='Symbol',
                                            values='sentiment_score').add_prefix('sentiment_score_')
            all_stock_df = pd.merge(all_stock_df, sentiment_df, left_on='Date', right_index=True, how='left')  # Left merge to keep all stock dates
            # Fill NaN value after merge
            all_stock_df = self._fill_score_columns(all_stock_df, 'sentiment_score_')  # Custom fill for sentiment columns

        if self.risk_mode and risk_df is not None and not risk_df.empty:
            # Similar pivot and merge for risk scores if enabled
            risk_df = risk_df.pivot(index='Date',
                                    columns='Symbol',
                                    values='risk_score').add_prefix('risk_score_')
            all_stock_df = pd.merge(all_stock_df, risk_df, left_on='Date', right_index=True, how='left')  # Left merge for risk
            # Fill NaN value after merge
            all_stock_df = self._fill_score_columns(all_stock_df, 'risk_score_')  # Custom fill for risk columns

        fused_df = all_stock_df.sort_values('Date').reset_index(drop=True)  # Sort by Date and reset index for clean DF

        # Global filter for positive Adj_Close per-symbol after merge (reference FinRL processor_yahoofinance.py)
        # Ensures prices positive without altering core merge/normalize
        for symbol in self.config.symbols:
            fused_df = fused_df[fused_df[f'Adj_Close_{symbol}'] > 0]  # Filter rows where Adj_Close > 0 for each symbol
        logging.info(f"FE Module - merge_features - Filtered fused_df to positive Adj_Close: {fused_df.shape} rows")  # Log post-filter shape

        # Get columns without 'Date'
        cols = fused_df.columns.tolist()
        cols.remove('Date')  # Exclude Date for reordering
        # Reorder columns by field-type across symbols (group by symbol)
        symbols = self.config.symbols
        ordered_cols = ['Date']  # Start with Date
        for symbol in symbols:
            ordered_cols += [col for col in cols if col.endswith(f'_{symbol}')]  # Append columns per symbol

        fused_df = fused_df[ordered_cols]  # Reorder DF for symbol-grouped columns
        logging.info(f"FE Module - merge_features - Fused features: {fused_df.shape} rows, with risk_mode={self.risk_mode}")  # Log final fused shape and mode

        fused_df = self._fill_nan_after_merge(fused_df)  # Final NaN fill after all operations
        logging.info(f"FE Module - merge_features - Prepared fused_df , Columns : {fused_df.columns.tolist()}")
        # Save fused_df
        self.config.fused_cache_path = self.save_target_data_csv(fused_df, prefix=prefix, save_path=self.fused_data_dir)
        logging.info(f"FE Module - generate_experiment_data - Saved fused_df to {self.config.fused_cache_path}")
        return fused_df  # Return the fully fused and cleaned DataFrame

    def _generate_scaler_path(self, base_dir, group='rl_algorithm', mode='PPO'):
        """Generate scaler path dynamiclly"""
        filename = f"scaler_{group}_{mode}_train.pkl"
        scaler_path = os.path.join(base_dir, filename)
        logging.info(f"FE Module - _generate_scaler_path - Generated scaler path: {scaler_path}")
        return scaler_path
    
    def smooth_features(self, df, smooth_window=None):
        """
        Apply smoothing to features using a rolling mean.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing features.
        window_size : int, optional
            Size of the rolling window for smoothing.

        Returns
        -------
        pd.DataFrame
            DataFrame with smoothed features.
        """
        window_size = smooth_window or self.smooth_window
        # Ensure 'Date' is the index
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        # Apply rolling mean smoothing to all numeric columns
        smoothed_df = df.rolling(window=window_size, min_periods=1).mean()
        # Reset index to maintain original structure
        smoothed_df.reset_index(inplace=True)
        
        return smoothed_df

    def normalize_features(self, df, fit=False, means_stds=None, scaler_path=None, data_type='train'):
        """
        Introduction
        ------------
        Normalize selected feature columns in the DataFrame using mean-std standardization.
        Supports fit mode (compute and save scaler) and transform mode (apply existing scaler).
        Applies clipping to normalized indicator columns based on quantiles to limit outlier impact.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing features to normalize.
        fit : bool, optional
            If True, compute means and stds (default: False).
        means_stds : dict, optional
            Pre-computed means and stds for columns (used in transform mode).
        scaler_path : str, optional
            Path to load/save scaler with joblib.
        data_type : str, optional
            Type of data being processed ('train', 'valid', 'test'). Used for logging.

        Returns
        -------
        pd.DataFrame or tuple
            - If fit=True: (normalized_df, means_stds_dict)
            - Else: normalized_df

        Notes
        -----
        - Normalizes indicators + 'sentiment_score', 'risk_score'; excludes open/high/low/close/volume.
        - Filters for numeric columns only; sets 'Date' as index if present.
        - Uses min std=1e-6 to avoid division by zero.
        - Caches scaler to disk for reuse; logs loading/saving.
        - Returns empty dict if no columns to normalize.
        - Applies clipping based on quantiles (e.g., 0.5% and 99.5%) to normalized indicator columns post-normalization.
        - In 'fit' mode, quantile clipping bounds are calculated and saved.
        - In 'transform' mode, saved quantile bounds are loaded and applied.
        """
        logging.info(f"FE Module - normalize_features - {data_type} - Full Data Columns : {df.columns.tolist()}")
        if self.filter_ind and all(any(col.startswith(ind) for col in df.columns) for ind in self.filter_ind):
            logging.info(f"FE Module - normalize_features - {data_type} - Filter columns to normalize: {self.filter_ind}")
            filtered_ind_cols = [col for col in df.columns if any(col.startswith(ind) for ind in self.filter_ind)]
            to_normalize = filtered_ind_cols + ['sentiment_score', 'risk_score']  # List of filtered columns to target for normalization
            logging.info(f"FE Module - normalize_features - {data_type} - Target columns to normalize: {to_normalize}")
        else:
            to_normalize = self.stock_engineer.indicators + ['sentiment_score', 'risk_score']  # List of base columns to target for normalization
            logging.info(f"FE Module - normalize_features - {data_type} - Target columns to normalize: {to_normalize}")
        filter_normalize_cols = [col for col in df.columns 
                        if any(ind in col for ind in to_normalize) 
                        and not any(x in col for x in ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume']) 
                        and pd.api.types.is_numeric_dtype(df[col])]  # Filter columns: match targets, exclude raw prices/volumes, ensure numeric
        logging.info(f"FE Module - normalize_features - {data_type} - Filtered Normalizing columns: {filter_normalize_cols}")
        # Prepare target columns
        final_target_cols = [col for col in df.columns if 'Adj_Close' in col] + filter_normalize_cols

        if 'Date' in df.columns:
            df = df.set_index('Date')  # Set Date as index to exclude from features if present

        if not filter_normalize_cols:
            # Early exit if no valid columns found; avoid unnecessary processing
            logging.warning(f"FE Module - normalize_features - {data_type} - No valid columns to normalize.")
            return (df, {}) if fit else df # Always return tuple for consistency in fit mode

        # Define Clipping Parameters
        # Quantile thresholds for clipping (can be made configurable if needed)
        LOWER_QUANTILE = 0.005  # 0.5%
        UPPER_QUANTILE = 0.995  # 99.5%
        # Keys for saving/loading clipping bounds in the scaler file
        CLIP_BOUNDS_KEY = "_clip_bounds_"

        # Internal Helper Function for Clipping 
        def _apply_clipping(dataframe, columns_to_clip, clipping_bounds_dict=None, calculate_bounds=False, operation_desc=""):
            """
            Applies clipping to specified columns in a DataFrame.
            If calculate_bounds is True, it calculates and returns the bounds.
            If calculate_bounds is False, it uses the provided clipping_bounds_dict.
            """
            if not columns_to_clip:
                logging.info(f"FE Module - _apply_clipping - No columns provided for clipping ({operation_desc}).")
                return dataframe, {} # Return empty dict for bounds if none to clip

            clipped_df = dataframe.copy() # Work on a copy to avoid modifying the original passed df immediately
            calculated_bounds_dict = {}
            
            if calculate_bounds:
                logging.info(f"FE Module - _apply_clipping - Calculating and applying clipping based on current data's quantiles [{LOWER_QUANTILE}, {UPPER_QUANTILE}] ({operation_desc}).")
                for col in columns_to_clip:
                    if col in clipped_df.columns:
                        lower_bound = clipped_df[col].quantile(LOWER_QUANTILE)
                        upper_bound = clipped_df[col].quantile(UPPER_QUANTILE)
                        calculated_bounds_dict[col] = (lower_bound, upper_bound)
                        clipped_df[col] = clipped_df[col].clip(lower=lower_bound, upper=upper_bound)
                        logging.debug(f"FE Module - _apply_clipping - Clipping bounds for {col} ({operation_desc}): [{lower_bound:.4f}, {upper_bound:.4f}]")
                    else:
                        logging.warning(f"FE Module - _apply_clipping - Column '{col}' not found in DataFrame for clipping ({operation_desc}).")
            else:
                if not clipping_bounds_dict:
                    logging.info(f"FE Module - _apply_clipping - No clipping bounds provided ({operation_desc}). Applying default [min, max].")
                else:
                    logging.info(f"FE Module - _apply_clipping - Applying provided clipping bounds ({operation_desc}).")
                
                for col in columns_to_clip:
                    if col not in clipped_df.columns:
                        logging.warning(f"FE Module - _apply_clipping - Column '{col}' not found for clipping ({operation_desc}).")
                        continue
                    
                    if clipping_bounds_dict and col in clipping_bounds_dict:
                        lower_bound, upper_bound = clipping_bounds_dict[col]
                        log_msg_suffix = "provided"
                    else: # Fallback or no bounds dict
                        lower_bound = clipped_df[col].min()
                        upper_bound = clipped_df[col].max()
                        log_msg_suffix = "default (min/max)"
                    
                    clipped_df[col] = clipped_df[col].clip(lower=lower_bound, upper=upper_bound)
                    logging.debug(f"FE Module - _apply_clipping - Applied {log_msg_suffix} clipping bounds for {col} ({operation_desc}): [{lower_bound:.4f}, {upper_bound:.4f}]")

            return clipped_df, calculated_bounds_dict

        # Main Logic 
        if fit:
            # --- Fit Mode: Calculate statistics and save scaler ---
            scaler_data_to_save = None
            # Handle force recompute or loading existing scaler
            if scaler_path and os.path.exists(scaler_path):
                if self.force_normalize_features:
                    logging.info(f"FE Module - normalize_features - {data_type} - Removed old scaler at {scaler_path} to recompute.")
                    os.remove(scaler_path)
                else:
                    logging.info(f"FE Module - normalize_features - {data_type} - Loading existing scaler from {scaler_path} (fit mode)")
                    loaded_data = joblib.load(scaler_path)
                    if isinstance(loaded_data, dict) and 'means_stds' in loaded_data:
                        # Use loaded scaler if it's the new format
                        means_stds = loaded_data['means_stds']
                        clip_bounds = loaded_data.get(CLIP_BOUNDS_KEY, {})
                        logging.info(f"FE Module - normalize_features - {data_type} - Using loaded scaler (new format).")
                    else:
                        # Backward compatibility
                        means_stds = loaded_data
                        clip_bounds = {}
                        logging.info(f"FE Module - normalize_features - {data_type} - Using loaded scaler (old format).")

            # If no valid scaler loaded or forced to recompute
            if not means_stds or self.force_normalize_features:
                logging.info(f"FE Module - normalize_features - {data_type} - Calculating new scaler for {data_type} data.")
                # Calculate means and stds
                means_stds = {}
                for col in filter_normalize_cols:
                    logging.debug(f"FE Module - normalize_features - {data_type} - Calculating mean/std for {col}")
                    mean = df[col].mean()
                    std = df[col].std()
                    std = max(std, 1e-6)
                    means_stds[col] = (mean, std)
                    df[col] = (df[col] - mean) / std

                # Calculate and apply clipping bounds
                df, clip_bounds = _apply_clipping(df, filter_normalize_cols, calculate_bounds=True, operation_desc=f"for {data_type} data (fit - calculate)")

                # Save scaler
                if scaler_path:
                    scaler_data_to_save = {'means_stds': means_stds, CLIP_BOUNDS_KEY: clip_bounds}
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                    joblib.dump(scaler_data_to_save, scaler_path)
                    logging.info(f"FE Module - normalize_features - {data_type} - Saved new scaler to {scaler_path}")
            else:
                # Use loaded scaler
                logging.info(f"FE Module - normalize_features - {data_type} - Using loaded scaler for normalization of {data_type} data.")
                # Apply normalization
                for col in filter_normalize_cols:
                    mean, std = means_stds.get(col, (0, 1))
                    std = max(std, 1e-6)
                    df[col] = (df[col] - mean) / std
                
                # Apply clipping (loaded bounds)
                df, _ = _apply_clipping(df, filter_normalize_cols, clipping_bounds_dict=clip_bounds, calculate_bounds=False, operation_desc=f"for {data_type} data (fit - using loaded)")

            # Filter final columns
            df = df[final_target_cols]
            
            logging.info(f"FE Module - normalize_features - {data_type} - Fit mode completed for {data_type}. Returning df and means_stds.")
            # Return the means_stds part for consistency with the original signature
            # If we saved new data, return the means_stds from it, otherwise return the one we used/created.
            if scaler_data_to_save and 'means_stds' in scaler_data_to_save:
                 return df, scaler_data_to_save['means_stds']
            else:
                 return df, means_stds # This should be the one we created/calculated

        else: # Transform Mode
            # --- Transform Mode: Apply existing statistics ---
            # Determine source of means_stds and clip_bounds
            # Priority: 1. Load from `scaler_path` 2. Function argument `means_stds`
            local_means_stds = None # Initialize
            local_clip_bounds = {} # Initialize 
            
            # Try loading from `scaler_path` first
            if scaler_path and os.path.exists(scaler_path):
                logging.info(f"FE Module - normalize_features - {data_type} - Loading scaler from {scaler_path} for {data_type} data (transform mode)")
                try: # Add try-except for robustness
                    loaded_data = joblib.load(scaler_path)
                    if isinstance(loaded_data, dict) and 'means_stds' in loaded_data:
                        local_means_stds = loaded_data['means_stds']
                        local_clip_bounds = loaded_data.get(CLIP_BOUNDS_KEY, {})
                        logging.info(f"FE Module - normalize_features - {data_type} - Loaded scaler (new format) for transform.")
                    else:
                        # Backward compatibility - if file contains only old means_stds dict
                        local_means_stds = loaded_data 
                        local_clip_bounds = {} # No clip bounds in old format
                        logging.info(f"FE Module - normalize_features - {data_type} - Loaded scaler (old format) for transform.")
                except Exception as e:
                    logging.error(f"FE Module - normalize_features - {data_type} - Failed to load scaler from {scaler_path}: {e}")

            #  Get parameters from means_stds
            if not local_means_stds and means_stds:
                 logging.info(f"FE Module - normalize_features - {data_type} - Using provided means_stds argument for {data_type} data (transform mode). Clip bounds will be default.")
                 local_means_stds = means_stds
                 # Use default clip bouns when local_clip_bounds is empty
                 logging.warning(f"FE Module - normalize_features - Provided means_stds but no clip_bounds for {data_type}. Using default clipping.")
            
            # Check again and set safe return
            if not local_means_stds:
                logging.warning(f"FE Module - normalize_features - {data_type} - No scaler (means_stds) available for {data_type} data (transform mode). Returning unnormalized df.")
                df = df[final_target_cols]
                return df

            # Apply normalization
            logging.info(f"FE Module - normalize_features - {data_type} - Applying normalization to {data_type} data.")
            for col in filter_normalize_cols:
                mean, std = local_means_stds.get(col, (0, 1))
                std = max(std, 1e-6)
                df[col] = (df[col] - mean) / std

            # Apply clipping
            logging.info(f"FE Module - normalize_features - {data_type} - Applying clipping based on {data_type} data's own quantiles.")
            df, _ = _apply_clipping(df, filter_normalize_cols, clipping_bounds_dict=None, calculate_bounds=True, operation_desc=f"for {data_type} data (transform mode)")

            # Filter final columns
            df = df[final_target_cols]
            
            logging.info(f"FE Module - normalize_features - {data_type} - Transform mode completed for {data_type}. Returning df.")
            return df

    def prepare_rl_data(self, fused_df, symbols=None, data_type='train'):
        """
        Introduction
        ------------
        Prepare RL-compatible data from fused DataFrame by creating sliding windows as 2D states and future prices as targets.
        States shaped as (window_size, features_dim_per_symbol) via horizontal concat of per-symbol features; handles NaN filling and index setting.
        Updates feature categories for trading env inheritance; yields list of dicts for downstream splitting and environment use.

        Parameters
        ----------
        fused_df : pd.DataFrame
            Fused DataFrame with features, indexed by Date if not already.
        symbols : list of str, optional
            List of symbols to process; defaults to self.config.symbols.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'start_date': pd.Timestamp - Start date of the window.
            - 'states': np.ndarray - 2D array (window_size, n_symbols * features_dim_per_symbol) of concatenated features.
            - 'targets': np.ndarray - 2D array (prediction_days, n_symbols) of future Adj_Close values.

        Notes
        -----
        - Asserts input is DataFrame; fills NaNs with 0 for completeness.
        - Sets 'Date' as index if present; excludes it from features.
        - Calls _update_features_categories to update config for ConfigTrading inheritance.
        - Concatenates per-symbol windows along axis=1 for unified 2D states.
        - Targets extracted as matrix for multi-symbol prediction.
        - Logs prepared RL data count; assumes symbols ordered for consistent concat.
        """
        logging.debug(f"FE Module - prepare_rl_data - Debug info:")
        logging.debug(f"  fused_df shape: {fused_df.shape}")
        logging.debug(f"  symbols: {symbols or self.config.symbols}")
        logging.debug(f"  window_size: {self.window_size}")
        logging.debug(f"  W_f: {self.W_f}, W_e: {self.W_e}")
        logging.debug(f"  min_episode_length: {self.W_f * self.window_size + self.W_e}")

        symbols = symbols or self.config.symbols  # Default to config symbols if not provided
        logging.info(f"FE Module - prepare_rl_data - Begin preparing RL data with window={self.window_size}, prediction_days={self.prediction_days}, symbols={symbols}")  # Log preparation params
        assert isinstance(fused_df, pd.DataFrame), f"FE Module - Expected DataFrame, got {type(fused_df)}"  # Assert input type to prevent invalid data processing

        if fused_df.isna().sum().sum() > 0:
            # Check for NaNs; if found, fill with 0 to maintain data integrity for RL
            logging.warning(f"FE Module - prepare_rl_data - NaN values found in fused_df: {fused_df.isna().sum().sum()} before RL window generation")
            fused_df = fused_df.fillna(0)  # Fill NaN with 0 to avoid NaN propagation in windows
        # Ensure excluded Date column
        if 'Date' in fused_df.columns:
            fused_df = fused_df.set_index('Date')  # Set Date as index to exclude from features
        logging.info(f"FE Module - prepare_rl_data - Feature Columns: {fused_df.columns.tolist()}")  # Log feature columns for debugging

        logging.info(f"FE Module - prepare_rl_data - Columns before update features_categories: {fused_df.columns.tolist()}")
        # Ensure fused_df columns order follows Adj_Close_symbol1, indicators_symbol1, senti/risk_score_symbol1, Adj_Close_symbol2, ......
        fused_df = fused_df[[col for symbol in symbols for col in fused_df.columns if symbol in col]]
        logging.info(f"FE Module - prepare_rl_data - Columns after reorder: {fused_df.columns.tolist()}")
        
        if self.config._features_initialized:
            logging.info(f"FE Module - prepare_rl_data - Skipping features/threshold update "
                        f"data_type={data_type} because they are loaded from cache.")
        else:
            self._update_features_categories(fused_df)  # Update features_* attributes to self.config for inheriting by ConfigTrading
            logging.info(f"FE Module - prepare_rl_data - Features updated for {data_type} ")
            self._update_senti_risk_threshold(fused_df, data_type)  # Update senti/risk thresholds to self.config for inheriting by ConfigTrading
            logging.info(f"FE Module - prepare_rl_data - Senti/Risk updated for {data_type} ")
            if data_type == 'test': # After first test , stop update ConfigSetup
                self.config.save_config_cache()
                logging.info(f"FE Module - prepare_rl_data - Finished save config to cache file.")
                self.config._features_initialized = True
                logging.info(f"FE Module - prepare_rl_data - Set features_initialized -> {self.config._features_initialized}")

        rl_data = []  # Initialize list to store RL data dicts
        dates = fused_df.index  # Extract dates for start_date assignment

        min_required_steps = 30    # Set min step for each episode
        min_window_size = max(self.W_f * self.window_size + self.W_e, self.window_size + min_required_steps) # More winsow for explore
        logging.info(f"FE Module - prepare_rl_data - Data length: {len(fused_df)}")
        logging.info(f"FE Module - prepare_rl_data - Window size: {self.window_size}")
        logging.info(f"FE Module - prepare_rl_data - Configured episode length: {self.W_f * self.window_size + self.W_e}")
        logging.info(f"FE Module - prepare_rl_data - Final min episode length: {min_window_size}")

        if len(fused_df) < min_window_size:
            # valid/test data with small length，implement dynamic episode length
            actual_window_size = self.window_size if len(fused_df) - self.window_size > min_required_steps else len(fused_df) // 2
            logging.warning(f"FE Module - prepare_rl_data - Insufficient data, adjusting episode length: {min_window_size} -> {actual_window_size}")
        else:
            actual_window_size = min_window_size if len(fused_df) - min_window_size > min_required_steps else self.window_size
        
        # Get prediction days from config (default to 3 days)
        prediction_days = self.prediction_days
        logging.info(f"FE Module - prepare_rl_data - Prediction days: {prediction_days}")
        max_trading_steps = len(fused_df) - actual_window_size - prediction_days + 1
        
        if max_trading_steps <= 0:
            logging.error(f"FE Module - prepare_rl_data - No valid episodes can be created")
            return []
        
        # Limit episodes to prevent excessive memory usage
        num_episodes = min(max_trading_steps, 200)
        for i in range(num_episodes):    # in case too much step
            try:
                # # Boundary check for current episode
                if i + actual_window_size + prediction_days > len(fused_df):
                    logging.debug(f"FE Module - prepare_rl_data - Skipping episode {i} due to boundary issues")
                    break

                # Sliding window loop: Generate windows leaving room for prediction_days
                window_parts = []  # List to collect per-symbol window arrays for concat
                for symbol in symbols:
                    symbol_cols = self.config.features_all[symbol]  # Get all features for this symbol from config
                    symbol_window = fused_df.iloc[i:i+actual_window_size][symbol_cols].values  # Extract (window_size, features_dim_per_symbol) array
                    window_parts.append(symbol_window)  # Append for later concat

                # Concatenate all symbol windows along axis=1 to form 2D states (window_size, n_symbols * features_dim_per_symbol)
                states = np.concatenate(window_parts, axis=1)  # (actual_window_size, total_features)

                # Validate states shape
                if states.shape[0] != actual_window_size:
                    logging.warning(f"FE Module - Episode {i} has inconsistent shape: {states.shape} vs {actual_window_size}")
                    continue
                
                # Targets - Generate target for each episode
                target_cols = [f"Adj_Close_{symbol}" for symbol in symbols]
                # Construct targets: future prediction_days' adjusted close prices
                episode_end = i + actual_window_size
                targets = fused_df.iloc[episode_end:episode_end + prediction_days][target_cols].values  # shape: (prediction_days, n_symbols)

                # Validate targets shape
                if targets.shape[0] != prediction_days:
                    logging.warning(f"FE Module - Episode {i} has insufficient targets: {targets.shape[0]} vs {prediction_days}")
                    continue

                rl_data.append({
                    'start_date': dates[i] if i < len(dates) else None,
                    'states': states,   # shape: (actual_window_size, total_features)
                    'targets': targets  # shape: (prediction_days, n_symbols)
                })
            
            except Exception as e:
                logging.error(f"FE Module - prepare_rl_data - Error processing episode {i}: {e}")

        logging.info(f"FE Module - prepare_rl_data - Prepared {len(rl_data)} RL episodes")
        if rl_data:
            logging.info(f"FE Module - prepare_rl_data - Sample episode shapes - states: {rl_data[0]['states'].shape}, targets: {rl_data[0]['targets'].shape}")

        return rl_data  # Return list of RL data dicts

    def split_rl_data(self, rl_data):
        """
        Introduction
        ------------
        Split RL data into train, valid, and test sets based on mode (date or ratio).
        Supports fallback to full data if splits are empty, and optional k-fold cross-validation on train set.

        Parameters
        ----------
        rl_data : list
            List of RL data dictionaries, each with 'start_date' key for sorting/splitting.

        Returns
        -------
        dict
            - If k-folds enabled: {'train_folds': [(train_fold, valid_fold), ...], 'valid': [...], 'test': [...]}
            - Else: {'train': [...], 'valid': [...], 'test': [...]}
            - Fallback if empty splits: {'train': rl_data, 'valid': [], 'test': []}

        Notes
        -----
        - Sorts data by 'start_date' to ensure chronological order.
        - 'date' mode uses predefined date ranges; 'ratio' uses split_ratio (e.g., 0.8) + 0.1 for valid.
        - k-folds applies only to train set, using TimeSeriesSplit or KFold (no shuffle).
        - Logs split sizes; raises ValueError for invalid modes.
        """
        assert isinstance(rl_data[0]['start_date'], pd.Timestamp), "FE Module - split_rl_data - 'start_date' must be pandas.Timestamp"

        rl_data = sorted(rl_data, key=lambda x: x['start_date'])  # Sort RL data by start_date to maintain chronological order
        if self.split_mode == 'date':
            # Date-based split: Filter data within predefined date ranges for train/valid/test
            train_rl_data = [data for data in rl_data if pd.to_datetime(self.train_start_date) <= data['start_date'] <= pd.to_datetime(self.train_end_date)]
            valid_rl_data = [data for data in rl_data if pd.to_datetime(self.valid_start_date) <= data['start_date'] <= pd.to_datetime(self.valid_end_date)]
            test_rl_data = [data for data in rl_data if pd.to_datetime(self.test_start_date) <= data['start_date'] <= pd.to_datetime(self.test_end_date)]
            logging.info(f"FE Module - split_rl_data - Split RL data: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")  # Log split sizes for monitoring
        elif self.split_mode == 'ratio':
            # Ratio-based split: Divide data sequentially using split_ratio for train, then 0.1 for valid, rest for test
            n = len(rl_data)
            train_end_idx = int(n * self.split_ratio)  # Calculate train end index
            valid_end_idx = int(n * (self.split_ratio + 0.1))  # Valid end: additional 10% after train
            train_rl_data = rl_data[:train_end_idx]
            valid_rl_data = rl_data[train_end_idx:valid_end_idx]
            test_rl_data = rl_data[valid_end_idx:]
            logging.info(f"FE Module - split_rl_data - Split RL data: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")  # Log split sizes for monitoring
        else:
            raise ValueError(f"FE Module - split_rl_data - Invalid split mode: {self.split_mode}")  # Error for unsupported split mode

        if len(train_rl_data) == 0 or len(valid_rl_data) == 0 or len(test_rl_data) == 0:
            # Fallback: If any split is empty, use all data as train to prevent training failure
            logging.warning("FE Module - split_rl_data - Empty data split. Falling back to all data as train")
            return {'train': rl_data, 'valid': [], 'test': []}

        if self.k_folds and self.k_folds > 1:
            # k-folds enabled: Apply cross-validation only on train data
            if self.cross_valid_mode == 'time_series':
                split = TimeSeriesSplit(n_splits=self.k_folds)  # TimeSeriesSplit to preserve temporal order
            elif self.cross_valid_mode == 'kfold':
                split = KFold(n_splits=self.k_folds, shuffle=False)  # KFold without shuffle for consistency
            else:
                raise ValueError(f"Invalid cross validation mode: {self.cross_valid_mode}")  # Error for invalid CV mode
            train_folds = []
            indices = np.arange(len(train_rl_data))  # Generate indices for splitting train data
            for train_idx, valid_idx in split.split(indices):
                train_fold = [train_rl_data[i] for i in train_idx]  # Subset train fold
                valid_fold = [train_rl_data[i] for i in valid_idx]  # Subset valid fold from train
                train_folds.append((train_fold, valid_fold))  # Append fold pair
            return {'train_folds': train_folds, 'valid': valid_rl_data, 'test': test_rl_data}  # Return with folds for CV
        logging.info(f"FE Module - split_rl_data - Successfully split rl data")
        return {'train': train_rl_data, 'valid': valid_rl_data, 'test': test_rl_data}  # Standard return without folds

    def _check_and_adjust_sentiment(self, score_df, mode, col='sentiment_score'):
        """
        Introduction
        ------------
        Check the variance of sentiment or risk scores in the DataFrame.
        If variance is low, adjust by adding mode-specific Gaussian noise to increase variability for better RL training.

        Parameters
        ----------
        score_df : pd.DataFrame
            DataFrame containing sentiment or risk scores.
        mode : str
            Experiment mode name, used for generating reproducible noise seed.
        col : str, optional
            Column name for the score to check/adjust (default: 'sentiment_score').

        Returns
        -------
        pd.DataFrame
            The original or adjusted score DataFrame.

        Notes
        -----
        - Skips adjustment if DataFrame is empty or column missing.
        - Uses SHA256 hash of mode for reproducible random seed.
        - Adds Gaussian noise (mean=0, std=0.3) and clips values to [1.0, 5.0].
        - Threshold for low variance is 0.1; logs stats before/after adjustment.
        """
        if score_df.empty or col not in score_df.columns:
            # Early exit if DataFrame is empty or specified column is missing to avoid errors
            logging.warning(f"FE Module - _adjust_sentiment - No {col} for mode {mode}; skipping check")
            return score_df
        
        score_var = score_df[col].var()  # Calculate variance of the score column
        score_mean = score_df[col].mean()  # Calculate mean for logging stats
        logging.info(f"FE Module - _adjust_sentiment - {col} stats for mode {mode}: var={score_var:.4f}, mean={score_mean:.4f}")  # Log initial statistics for debugging
        
        if score_var < 0.1:    # Check if variance is below threshold (increased from 0.05 to 0.1 for leniency)
            # Low variance detected; proceed to add noise to enhance data variability
            logging.warning(f"FE Module - _adjust_sentiment - Low var ({score_var:.4f}) for {col} in mode {mode}; adding mode-specific noise")
            seed = int(hashlib.sha256(mode.encode()).hexdigest(), 16) % (2**32) # Generate reproducible seed using hash of mode for consistency across runs
            np.random.seed(seed)  # Set seed for numpy random to ensure reproducibility
            # Add Gaussian noise and clip to maintain score range [1.0, 5.0] to prevent invalid values
            score_df[col] = np.clip(score_df[col] + np.random.normal(0, 0.3, len(score_df)), 1.0, 5.0)
            new_var = score_df[col].var()  # Recalculate variance after adjustment
            logging.info(f"FE Module - _adjust_sentiment - Adjusted var for {col} in mode {mode}: {new_var:.4f}")  # Log new variance to verify adjustment
        
        return score_df  # Return the potentially adjusted DataFrame

    def _generate_path_suffix(self, extension='.npz'):
        """
        Generate a unique path suffix for file caching, based on configuration settings.

        Parameters
        ----------
        file_format : str, optional
            The file extension to append to the suffix. Default is '.npz'.

        Returns
        -------
        str or None
            A string suffix in the format 'symbols_start_end.file_format' if configuration is available,
            otherwise None (with a warning logged).

        Notes
        -----
        This method relies on self.config being properly initialized. It is typically used internally
        for generating cache file names to ensure uniqueness based on symbols and date range.
        """
        try:
            if self.config:
                # Extract key configuration values for suffix generation
                start = self.config.start
                end = self.config.end
                if self.use_symbol_name:
                    symbols = "_".join(self.config.symbols)  # Join symbols with underscore for compact representation
                else:
                    symbols = "all_symbols"
                # Construct the suffix by combining symbols, dates, and file format for unique identification
                path_suffix = f"{symbols}_{start}_{end}{extension}"
                logging.info(f"FE Module - _generate_path_suffix - Successfully generate path suffix: {path_suffix}")
                return path_suffix
            else:
                # Log warning if config is missing to alert for potential setup issues
                logging.warning(f"FE Module - _generate_path_suffix - No ConfigSetup to generate path suffix")
                return None  # Explicitly return None to handle missing config gracefully
        except Exception as e:
            # Catch any unexpected errors during suffix generation and log for debugging
            logging.warning(f"FE Module - _generate_path_suffix - Fail to generate path suffix : {e}")
            return None  # Return None on failure to prevent downstream errors
    
    def load_fused_df_cache(self, prefix='Fused_Data'):
        """
        Load the fused DataFrame from a cache file if available.

        Returns
        -------
        pd.DataFrame or None
            The loaded fused DataFrame if a valid cache file is found, otherwise None.

        Notes
        -----
        - The cache file is identified by a unique suffix based on the current configuration.
        - If a valid cache file is found, it is loaded and returned.
        - If no valid cache file is found, None is returned, and a new cache file will be generated.
        """
        logging.info("FE Module - generate_experiment_data - Trying to load fused_df cache")
        fused_data_list = os.listdir(self.fused_data_dir)

        # Trying to load cache fused_df
        fused_df = None
        if len(fused_data_list) > 0 and not self.force_fuse_data:
            logging.debug(f"FE Module - generate_experiment_data - Exist {len(fused_data_list)} processed news cache : {fused_data_list}")
            fused_data_path = f"{prefix}_{self._generate_path_suffix(extension='.csv')}"
            
            # Check target cache
            for filename in fused_data_list:
                if filename.endswith(fused_data_path):
                    self.config.fused_cache_path = os.path.join(self.fused_data_dir, filename)
                    logging.info(f"FE Module - generate_experiment_data - Target cache path : {self.config.fused_cache_path}")
                    
                    # Load cache
                    logging.info(f"FE Module - generate_experiment_data - Load fused_df for {self.config.symbols}")
                    fused_df = pd.read_csv(self.config.fused_cache_path, parse_dates=['Date'])
                    logging.debug(f"FE Module - generate_experiment_data - Loaded fused_df: {len(fused_df)} rows")
                    return fused_df
        logging.warning(f"FE Module - load_fused_df_cache - No cache fused_df exist. Return Empty DataFrame")
        return fused_df

    def save_target_data_csv(self, target_df, prefix=None, save_path=None, extension='.csv'):
        """
        Save the fused DataFrame to a CSV file.

        Parameters
        ----------
        fused_df : pd.DataFrame
            The fused DataFrame to be saved.

        save_dir : str, optional
            Directory where the CSV file will be saved. Default is 'fused_data_csv'.

        Returns
        -------
        None
            This function saves the DataFrame to a CSV file and does not return anything.

        Notes
        -----
        The CSV file is named using the symbols and date range from the configuration.
        """
        os.makedirs(save_path if save_path else self.processed_news_dir, exist_ok=True)
        logging.info("FE Module - save_fused_data_csv - Initial fused_data_csv save path")
        try:
            path_suffix = self._generate_path_suffix(extension=extension)
            filename = f"{prefix}_{path_suffix}"
            if path_suffix:
                file_path = os.path.join(save_path if save_path else self.processed_news_dir, filename)
                target_df.to_csv(file_path, index=False)
                logging.info(f"FE Module - save_fused_data_csv - Successfully saved fused data to {file_path}")
                return file_path
            else:
                logging.warning("FE Module - save_fused_data_csv - Failed to generate path suffix for CSV file")
                return None
        except Exception as e:
            logging.error(f"FE Module - save_fused_data_csv - Error saving fused data: {e}")

    def save_exper_data_dict_npz(self, exper_data_dict):
        """
        Save experiment data dictionary to .npz files (one per mode).

        Parameters
        ----------
        exper_data_dict : dict
            Dictionary of format:
            {
                'PPO': {
                    'train': [dict_list],
                    'valid': [dict_list],
                    'test': [dict_list],
                    'model_type': 'PPO'
                },
                ...
            }
            Each dict_list contains dicts with keys: 'start_date', 'states', 'targets'.

        save_dir : str, optional
            Directory where .npz files will be saved. Default is 'exper_data_npz'.

        Returns
        -------
        None
            This function saves .npz files to disk and does not return anything.

        Notes
        -----
        Each mode (PPO, CPPO, etc.) will be saved as a single .npz file.
        Inside each file, train/valid/test will each be a numpy structured array.
        """
        os.makedirs(self.exper_data_path, exist_ok=True)
        logging.info("FE Module - save_exper_data_dict_npz - Initial exper_data_dict save path")
        try:
            path_suffix = self._generate_path_suffix()
            for mode, data_dict in exper_data_dict.items():
                mode_path = os.path.join(self.exper_data_path, f"{mode}_{path_suffix}")
                logging.info(f"FE Module - save_exper_data_dict_npz - Initial exper_data_dict save path {mode_path} for mode {mode}")
                npz_data = {}

                for target in ['train', 'valid', 'test']:
                    records = data_dict.get(target, [])
                    # Convert list of dicts to structured arrays
                    if records:
                        # Verify shape 
                        states_shapes = [data['states'].shape for data in records]
                        targets_shapes = [data['targets'].shape for data in records]
                        logging.info(f"FE Module - save_exper_data_dict_npz - {target} shapes: "
                                f"states {set(states_shapes)}, targets {set(targets_shapes)}")
                        # If same shape, implement normal array.Otherwise Object
                        if len(set(states_shapes)) == 1 and len(set(targets_shapes)) == 1:
                            start_dates = np.array([str(data['start_date']) for data in records])
                            states = np.array([data['states'] for data in records])
                            targets = np.array([data['targets'] for data in records])
                            logging.info(f"FE Module - save_exper_data_dict_npz - Using regular arrays for {target}")
                        else:
                            # If not same shape, implement Object
                            start_dates = np.array([str(data['start_date']) for data in records], dtype=object)
                            states = np.array([data['states'] for data in records], dtype=object)
                            targets = np.array([data['targets'] for data in records], dtype=object)
                            logging.info(f"FE Module - save_exper_data_dict_npz - Using object arrays for {target} due to shape inconsistency")

                        npz_data[f"{target}_dates"] = start_dates
                        npz_data[f"{target}_states"] = states
                        npz_data[f"{target}_targets"] = targets

                np.savez_compressed(mode_path, **npz_data)
                logging.info(f"FE Module - save_exper_data_dict_npz - Saved {mode} data to {mode_path}")
            logging.info(f"FE Module - save_exper_data_dict_npz - Save exper_data_dict successfully")
        except Exception as e:
            logging.warning(f"FE Module - save_exper_data_dict_npz - Save exper_data_dict failed: {e}")
            raise

    def load_exper_data_dict_npz(self):
        """
        Load experiment data dictionary from .npz files (one per mode).

        Parameters
        ----------
        load_dir : str, optional
            Directory where .npz files are stored. Default is 'exper_data_npz'.

        Returns
        -------
        exper_data_dict : dict
            Dictionary in the same format as used during saving:
            {
                'PPO': {
                    'train': [dict_list],
                    'valid': [dict_list],
                    'test': [dict_list],
                    'model_type': 'PPO'
                },
                ...
            }

        Notes
        -----
        Assumes each .npz file contains arrays for dates, states, and targets.
        Reconstructs the original list-of-dicts structure.
        """
        exper_data_dict = {}
        try:
            path_suffix = self._generate_path_suffix()
            for filename in os.listdir(self.exper_data_path):
                logging.info(f"FE Module - load_exper_data_dict_npz - Loading exper_data_dict from {filename}")
                if filename.endswith('.npz'):
                    mode = filename.replace(f'{path_suffix}', '')
                    file_path = os.path.join(self.exper_data_path, filename)
                    data = np.load(file_path, allow_pickle=True)

                    mode_dict = {}
                    for target in ['train', 'valid', 'test']:
                        logging.info(f"FE Module - load_exper_data_dict_npz - Loading exper_data_dict from {target} for mode {mode}")
                        dates_key = f"{target}_dates"
                        states_key = f"{target}_states"
                        targets_key = f"{target}_targets"
                        if dates_key in data:
                            target_list = []
                            for i in range(len(data[dates_key])):
                                item = {
                                    'start_date': data[dates_key][i],
                                    'states': data[states_key][i],
                                    'targets': data[targets_key][i]
                                }
                                target_list.append(item)
                            mode_dict[target] = target_list

                    mode_dict['model_type'] = mode
                    exper_data_dict[mode] = mode_dict
                    logging.info(f"FE Module - load_exper_data_dict_npz - Loaded {mode} data from {filename} completely")
            logging.info(f"FE Module - load_exper_data_dict_npz - Load exper_data_dict successfully")
        except Exception as e:
            logging.warning(f"FE Module - load_exper_data_dict_npz - Load exper_data_dict failed: {e}")
        return exper_data_dict

    def _update_senti_risk_threshold(self, fused_df, data_type=None):
        """
        Introduction
        ------------
        Update sentiment / risk features threshold in config dynamiclly and prepare to be inherited by ConfigTrading in downstream pipeline.
        Compute sentiment / risk mean and std values for introducing Senti_facor and Risk_facor dynamically.
        Prepares global config for stock_trading_env state tracking after feature merging.

        Parameters
        ----------
        fused_df : pd.DataFrame
            Fused DataFrame with symbol-suffixed columns.

        data_type : str
            Type of data to process, expected: 'train', 'valid' or 'test'.

        Notes
        -----
        - Asserts input type; filters columns with 'sentiment' or 'risk'.
        - Updates config dicts: senti_threshold, risk_threshold.
        - Logs updates and warnings on exceptions for traceability.
        """
        assert isinstance(fused_df, pd.DataFrame), f"FE Module - _update_senti_risk_threshold - Unexpected data type : {type(fused_df)}"  # Assert input is DataFrame to prevent invalid processing
        
        logging.info(f"FE Module - _update_senti_risk_threshold - Start to update senti / risk threshold to ConfigSetup")  # Log start of update process
        logging.info(f"FE Module - _update_senti_risk_threshold - Symbols to process: {self.config.symbols}")  # Log symbols for context
        try:
            senti_cols = [col for col in fused_df.columns if 'sentiment' in col]
            risk_cols = [col for col in fused_df.columns if 'risk' in col]
            logging.info(f"FE Module - _update_senti_risk_threshold - Senti columns: {senti_cols}, Risk columns: {risk_cols}")

            senti_values = fused_df[senti_cols].values.flatten()
            risk_values = fused_df[risk_cols].values.flatten()
            logging.info(f"FE Module - _update_senti_risk_threshold - Senti values: {len(senti_values)}, Risk values: {len(risk_values)}")
            if len(senti_values) == 0 or len(risk_values) == 0:
                # Early warning and return if DF is empty to avoid unnecessary grouping
                logging.warning(f"FE Module - _update_senti_risk_threshold - Empty senti_values or risk_values at updating senti / risk threshold step ")
                return
            mean_senti = np.mean(senti_values)
            std_senti = np.std(senti_values)
            logging.info(f"FE Module - _update_senti_risk_threshold - Senti mean: {mean_senti}, Senti std: {std_senti}")
            mean_risk = np.mean(risk_values)
            std_risk = np.std(risk_values)
            logging.info(f"FE Module - _update_senti_risk_threshold - Risk mean: {mean_risk}, Risk std: {std_risk}")
            
            T_f = getattr(self.config, 'threshold_factor', 0.5)
            self.config.senti_threshold[data_type] = {'mean': mean_senti,
                                                      'std': std_senti,
                                                      'pos_threshold': mean_senti + T_f * std_senti,
                                                      'neg_threshold': mean_senti - T_f * std_senti}
            self.config.risk_threshold[data_type] = {'mean': mean_risk,
                                                      'std': std_risk,
                                                      'pos_threshold': mean_risk + T_f * std_risk,
                                                      'neg_threshold': mean_risk - T_f * std_risk}
            logging.info(f"FE Module - _update_senti_risk_threshold - Successfully Update senti / risk threshold to ConfigSetup")
        except Exception as e:
            logging.error(f"FE Module - _update_senti_risk_threshold - Fail to update senti / risk threshold : {e}")
            raise ValueError(f"FE Module - _update_senti_risk_threshold - Fail to update senti / risk threshold")

    
    def _update_features_categories(self, fused_df):
        """
        Introduction
        ------------
        Update feature categories in config based on fused DataFrame columns per symbol.
        Categorizes into price, indicators, sentiment, risk, and all; computes dim per symbol for trading env monitoring.
        Prepares global config for stock_trading_env state tracking after feature merging.

        Parameters
        ----------
        fused_df : pd.DataFrame
            Fused DataFrame with symbol-suffixed columns.

        Notes
        -----
        - Asserts input type; filters columns ending with '_{symbol}'.
        - Excludes price/senti/risk to derive indicators; logs counts per category.
        - Updates config dicts: features_price, features_ind, features_senti, features_risk, features_all.
        - Sets features_dim_per_symbol as len of first symbol's all features (assumes symmetry).
        - Logs updates and warnings on exceptions for traceability.
        """
        assert isinstance(fused_df, pd.DataFrame), f"FE Module - _update_features_categories - Unexpected data type : {type(fused_df)}"  # Assert input is DataFrame to prevent invalid processing
        
        logging.info(f"FE Module - _update_features_categories - Start to update feature categories to ConfigSetup")  # Log start of update process
        logging.info(f"FE Module - _update_features_categories - Symbols to process: {self.config.symbols}")  # Log symbols for context
        try:
            for symbol in self.config.symbols:
                # Extract all columns for this symbol
                symbol_cols = [col for col in fused_df.columns if col.endswith(f"_{symbol}")]  # Filter cols ending with _sym, e.g., 'macd_AAPL'

                # Categorize: Price (Adj_Close), sentiment, risk
                price_cols = [col for col in symbol_cols if "Adj_Close" in col]  # Price-related columns
                senti_cols = [col for col in symbol_cols if "sentiment_score" in col]  # Sentiment columns
                risk_cols = [col for col in symbol_cols if "risk_score" in col]  # Risk columns

                # Derive indicators by exclusion
                full_price_cols = [col for col in symbol_cols if "Adj_" in col or "Volume_" in col]
                exclude_cols = set(full_price_cols + senti_cols + risk_cols)  # Set of excluded categories
                ind_cols = [col for col in symbol_cols if col not in exclude_cols]  # Remaining as indicators

                # Merge all features per symbol
                features_cols_per_symbol = price_cols + ind_cols + senti_cols + risk_cols

                # Debug log category counts
                logging.debug(f"FE Module - _update_features_categories - {symbol} - price:{len(price_cols)}, ind:{len(ind_cols)}, senti:{len(senti_cols)}, risk:{len(risk_cols)}")
                
                # Update config with categorized lists per symbol
                self.config.features_price[symbol] = price_cols  # Store price cols
                self.config.features_ind[symbol] = ind_cols  # Store indicator cols
                self.config.features_senti[symbol] = senti_cols  # Store sentiment cols
                self.config.features_risk[symbol] = risk_cols  # Store risk cols
                self.config.features_all[symbol] = features_cols_per_symbol  # Store all combined
                logging.info(f"FE Module - _update_features_categories - Update feature categories for symbol: {symbol}")  # Log per-symbol update
                logging.debug(f"FE Module - _update_features_categories - Features dim : features_price {price_cols}, features_ind {ind_cols}, features_senti {senti_cols}, features_risk {risk_cols}, features_all:{len(price_cols + ind_cols + senti_cols + risk_cols)}")

                if self.config.features_dim_per_symbol is None:
                    # Compute and update dimension per symbol
                    self.config.features_dim_per_symbol = len(features_cols_per_symbol)
                    logging.debug(f"FE Module - _update_features_categories - Update Features dim per symbol : {self.config.features_dim_per_symbol}")  # Log computed dim
            
            logging.info(f"FE Module - _update_features_categories - Successfully update all feature categories to ConfigSetup")  # Log overall success
        except Exception as e:
            logging.warning(f"FE Module - _update_features_categories - Failed to Update feature categories : {e} ")  # Log any exceptions without crashing

    def generate_experiment_data(self, stock_data_dict, news_chunks_gen, exper_mode=None, single_mode=None):
        """
        Introduction
        ------------
        Generate and return RL experiment data for multiple or single experiment modes.
        Handles both full-mode group processing and single-mode testing.

        Parameters
        ----------
        stock_data_dict : dict
            Stock price data for all symbols.
        news_chunks_gen : generator
            Generator that yields news data chunks.
        exper_mode : str, optional
            Name of the experiment group, e.g., 'rl_algorithm'.
        single_mode : str, optional
            Name of a single mode to run, used for one-off testing.

        Returns
        -------
        dict or tuple
            - If `single_mode` is specified, returns a tuple:
            (train_data, valid_data, test_data)
            - Else, returns a dict:
            {
                'PPO': {'train': [...], 'valid': [...], 'test': [...], 'model_type': 'PPO'},
                'CPPO': {...},
                ...
            }

        Notes
        -----
        - Saves and loads from disk using compressed `.npz` format.
        - Automatically reuses cached files if available.
        - Supports risk score injection and FinBERT sentiment adjustment.
        """
        os.makedirs(self.exper_data_path, exist_ok=True)
        exper_data_path_suffix = self._generate_path_suffix()
        exper_data_list = os.listdir(self.exper_data_path)
        if exper_data_list and exper_data_list[0].endswith(exper_data_path_suffix) and self.load_npz:
            # Check if experiment data directory is not empty; if so, load existing data to avoid regeneration
            logging.info("=========== Start to load experiment data dict ===========")
            logging.info(f"FE Module - generate_experiment_data - Loading exper_data_dict from {self.exper_data_path}")
            exper_data_dict = self.load_exper_data_dict_npz()  # Load pre-generated data from NPZ files for efficiency
            return exper_data_dict
        else:
            # Directory is empty; proceed to generate new experiment data
            logging.info("=========== Start to generate experiment data dict ===========")
            exper_data_dict = {}    # Initialize dictionary to store data for each mode
            exper_news_cols = getattr(self.config, 'exper_mode', {
                                        'benchmark': [],
                                        'title_only': ['Article_title'],
                                        'title_textrank': ['Article_title', 'Textrank_summary'],
                                        'title_fulltext': ['Article_title', 'Full_Text']
                                    })   # Define news column configurations for different modes
            
            # Flag to control feature analysis execution (execute only once)
            pre_feature_analysis_completed = False
            pro_feature_analysis_completed = False
            
            if single_mode:
                # Single mode: Run only one specific mode for testing
                logging.info(f"FE Module - generate_experiment_data - Running Single mode: {single_mode}")
                exper_modes = [single_mode]
                group = next((g for g, modes in self.exper_mode.items() if single_mode in modes), None) # Find group containing the single mode
                if not group:
                    raise ValueError(f"Unknown single_mode: {single_mode}") # Error if mode not found in any group
            elif exper_mode:
                # Experiment mode: Run all modes in a specified group
                logging.info(f"FE Module - generate_experiment_data - Running Experiment mode: {exper_mode}")
                exper_modes = self.exper_mode.get(exper_mode, [])
                group = exper_mode
                if not exper_modes:
                    raise ValueError(f"Unknown exper_mode group: {exper_mode}") # Error if group not defined
            else:
                # No mode specified: Run all modes across all groups
                logging.info("FE Module - generate_experiment_data - Running All experiment modes")
                exper_modes = sum(self.exper_mode.values(), [])
                group = None

            logging.info(f"FE Module - generate_experiment_data - Experiment modes: {exper_modes} from group: {group}")    # Log the modes to be processed

            cleaned_news = self.process_news_chunks(news_chunks_gen)    # Process news generator into cleaned DataFrame
            logging.info(f"FE Module - generate_experiment_data - Loaded and cleaned news: {len(cleaned_news)} rows")  # Log news data size after cleaning

            for mode in exper_modes:
                original_exper_news_cols = self.news_engineer.text_cols # Backup original text columns to restore later
                # Determine model_type and news_cols based on group
                group = next((group for group, modes in self.exper_mode.items() if mode in modes), None)    # Re-fetch group if needed
                if group == 'rl_algorithm':
                    self.news_engineer.text_cols = original_exper_news_cols  # Dynamic changed by Config
                    model_type = mode  # PPO, CPPO, etc.
                else:
                    self.news_engineer.text_cols = exper_news_cols.get(mode, ['Article_title', 'Textrank_summary'])    # Use mode-specific news columns, default ['Article_title', 'Textrank_summary']
                    model_type = 'PPO'  # Default for indicator/news group
                logging.info(f"FE Module - generate_experiment_data - Mode {mode} in group {group}, model_type={model_type}, news_cols={self.news_engineer.text_cols}")    # Log configuration for this mode

                # Trying to load cache fused_df
                fused_df = self.load_fused_df_cache()

                # If no cache fused_df, recompute
                if fused_df is None or fused_df.empty:
                    logging.info("FE Module - generate_experiment_data - Computing senti/risk score features and merge all features")
                    logging.info("FE Module - generate_experiment_data - No symbols or cache_path provided, computing senti/risk score features and merge all features")
                    # Compute sentiment
                    if group == 'rl_algorithm' or mode != 'benchmark':
                        sentiment_score_df = self.news_engineer.compute_sentiment_risk_score(cleaned_news.copy(), senti_mode='sentiment')    # Compute sentiment scores using FinBERT or similar
                        sentiment_score_df = self._check_and_adjust_sentiment(sentiment_score_df, mode, col='sentiment_score')  # Adjust sentiment if needed (e.g., FinBERT adjustment)
                        # Compute risk if enabled
                        if self.risk_mode:
                            risk_score_df = self.news_engineer.compute_sentiment_risk_score(cleaned_news.copy(), senti_mode='risk')  # Compute risk scores if risk mode is active
                            risk_score_df = self._check_and_adjust_sentiment(risk_score_df, mode, col='risk_score') # Adjust risk scores similarly
                    else:
                        sentiment_score_df = pd.DataFrame(columns=['Date', 'Symbol', 'sentiment_score'])    # Empty DF for benchmark mode
                        if self.risk_mode:
                            risk_score_df = pd.DataFrame(columns=['Date', 'Symbol', 'risk_score'])    # Empty DF for benchmark mode
                        logging.info("FE Module - generate_experiment_data - Benchmark mode: no sentiment")    # Log skipping sentiment for benchmark

                    # Merge features and split train/valid/test data by date
                    fused_df = self.merge_features(stock_data_dict, sentiment_score_df, risk_score_df)    # Fuse stock data with sentiment/risk features
                    if fused_df.empty:
                        logging.warning(f"FE Module - generate_experiment_data - Empty DataFrame for mode {mode}")
                        raise ValueError(f"Fused DataFrame empty for mode {mode}")  # Error if fusion results in empty DF
                
                # Generate pre-normalization feature analysis (only once)
                if not pre_feature_analysis_completed and self.plot_feature_visualization:
                    logging.info(f"FE Module - generate_experiment_data - Generating pre-normalization feature analysis for mode {mode}")
                    pre_normalize_results = generate_standard_feature_visualizations(
                        fused_df, self.config, prefix="pre_normalization"
                    )
                    logging.info(f"FE Module - generate_experiment_data - Successfully generated pre-normalization visualization results")
                    pre_feature_analysis_completed = True  # Mark as completed to avoid repetition

                # Time split data
                train_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.train_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.train_end_date))]
                valid_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.valid_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.valid_end_date))]
                test_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.test_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.test_end_date))]
                if train_df.empty or valid_df.empty or test_df.empty:
                    raise ValueError(f"Empty split for mode {mode}")    
                logging.info(f"FE Module - generate_experiment_data - Split for mode {mode}: train {len(train_df)}, valid {len(valid_df)}, test {len(test_df)}")

                # Smooth features before normalization
                logging.info(f"FE Module - generate_experiment_data - Applying feature smoothing with window size {self.smooth_window}")
                train_df = self.smooth_features(train_df)
                valid_df = self.smooth_features(valid_df)
                test_df = self.smooth_features(test_df)

                # Normalize indicators + sentiment + risk columns for RL training
                train_scaler_path = self._generate_scaler_path(self.scaler_cache_path, group=group, mode=mode)   # Dynamic per-group/mode
                train_df, means_stds = self.normalize_features(train_df, fit=True, scaler_path=train_scaler_path, data_type='train') # Load cache scaler if existed
                valid_df = self.normalize_features(valid_df, fit=False, means_stds=means_stds, scaler_path=train_scaler_path, data_type='valid') # Load cache scaler if existed
                test_df = self.normalize_features(test_df, fit=False, means_stds=means_stds, scaler_path=train_scaler_path, data_type='test')   # Load cache scaler if existed
                logging.info(f"FE Module - generate_experiment_data - Normalized features for mode {mode}")

                if not pro_feature_analysis_completed and self.plot_feature_visualization:
                    # Generate post-normalization feature analysis for each dataset split
                    logging.info(f"FE Module - generate_experiment_data - Generating post-normalization feature analysis")
                    # Train set analysis
                    train_analysis_results = generate_standard_feature_visualizations(
                        train_df, self.config, prefix=f"post_normalization_train"
                    )
                    # Valid set analysis
                    valid_analysis_results = generate_standard_feature_visualizations(
                        valid_df, self.config, prefix=f"post_normalization_valid"
                    )
                    # Test set analysis
                    test_analysis_results = generate_standard_feature_visualizations(
                        test_df, self.config, prefix=f"post_normalization_test"
                    )
                    logging.info(f"FE Module - generate_experiment_data - Successfully generated post-normalization visualization results")
                    pro_feature_analysis_completed = True   # Mark as completed to avoid repetition

                # Prepare window for RL observation
                train_rl_data = self.prepare_rl_data(train_df, data_type='train')  # Prepare RL-compatible data (e.g., windowed observations)
                valid_rl_data = self.prepare_rl_data(valid_df, data_type='valid')
                test_rl_data = self.prepare_rl_data(test_df, data_type='test')
                if not train_rl_data or not valid_rl_data or not test_rl_data:
                    raise ValueError(f"No RL data for mode {mode}")
                logging.info(f"FE Module - generate_experiment_data - Prepared RL data for mode {mode}: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")

                # Form split_dict with model_type
                split_dict = {
                    'train': train_rl_data,
                    'valid': valid_rl_data,
                    'test': test_rl_data,
                    'model_type': model_type
                }   # Create dictionary for this mode's splits
                if self.k_folds and self.k_folds > 1:
                    # If k-folds enabled, create cross-validation folds on train data
                    if self.cross_valid_mode == 'time_series':
                        split = TimeSeriesSplit(n_splits=self.k_folds)
                    elif self.cross_valid_mode == 'kfold':
                        split = KFold(n_splits=self.k_folds, shuffle=False)
                    train_folds = []
                    indices = np.arange(len(train_rl_data))
                    for train_idx, valid_idx in split.split(indices):
                        train_fold = [train_rl_data[i] for i in train_idx]
                        valid_fold = [train_rl_data[i] for i in valid_idx]
                        train_folds.append((train_fold, valid_fold))
                    split_dict['train_folds'] = train_folds

                exper_data_dict[mode] = split_dict  # Store split_dict in main dictionary
                self.fused_dfs[mode] = {'train': train_df, 'valid': valid_df, 'test': test_df}
                logging.info(f"FE Module - generate_experiment_data - Generated data for mode {mode}: train {len(split_dict['train'])}, valid {len(split_dict['valid'])}, test {len(split_dict['test'])}")

                self.news_engineer.text_cols = original_exper_news_cols # Restore original text columns after mode processing

            # Release FinBERT resources
            if hasattr(self.news_engineer, 'model') and self.news_engineer.finbert_model is not None:
                del self.news_engineer.finbert_model    # Delete model to free memory
                del self.news_engineer.finbert_tokenizer    # Delete tokenizer
                torch.cuda.empty_cache()    # Clear GPU cache if using CUDA
                logging.info("FE Module - generate_experiment_data - Released FinBERT resources")

            if single_mode and self.save_npz:
                self.save_exper_data_dict_npz(exper_data_dict)  # Save single mode data to NPZ
                logging.info(f"FE Module - generate_experiment_data - Save single mode data for {single_mode} successfully")
                logging.info(f"FE Module - generate_experiment_data - Return single mode train/valid/test data for {single_mode}")
                return exper_data_dict[single_mode]['train'], exper_data_dict[single_mode]['valid'], exper_data_dict[single_mode]['test']
            if self.save_npz:  # Default True
                try:
                    self.save_exper_data_dict_npz(exper_data_dict)
                except Exception as e:
                    logging.warning(f"FE Module - Failed to save NPZ files: {e}")
            logging.info(f"FE Module - generate_experiment_data - Save exper_data_dict successfully")
            return exper_data_dict  # Return the full experiment data dictionary
