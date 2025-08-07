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
import os
os.chdir('/Users/archy/Projects/finbert_trader/')

# %%
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import logging
import hashlib
import os
import json
import joblib

# %%
from finbert_trader.preprocessing.stock_features import StockFeatureEngineer
from finbert_trader.preprocessing.news_features import NewsFeatureEngineer

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
        self.window_size = self.config.window_size  # Window size for RL observation states
        self.prediction_days = self.config.prediction_days  # Number of future days for target prediction
        self.split_ratio = self.config.split_ratio  # Ratio for data splitting if split_mode='ratio'
        self.k_folds = self.config.k_folds  # Number of folds for cross-validation
        self.split_mode = self.config.split_mode  # Mode for data splitting ('date' or 'ratio')
        self.cross_valid_mode = self.config.cross_valid_mode  # Cross-validation type ('time_series' or 'kfold')
        self.exper_mode = self.config.exper_mode  # Dictionary of experiment modes/groups
        self.ind_mode = self.config.ind_mode  # Indicator mode for stock feature computation
        self.risk_mode = self.config.risk_mode  # Flag to enable risk score computation

        self.stock_engineer = StockFeatureEngineer(config)  # Instantiate stock feature engineer for technical indicators
        self.news_engineer = NewsFeatureEngineer(config)  # Instantiate news feature engineer for sentiment/risk
        self.fused_dfs = {}  # Initialize dict to cache fused DataFrames per mode for reuse

        # Config exper_data_dict save dir
        self.exper_data_path = self.config.EXPER_DATA_DIR  # Path for saving/loading experiment data (e.g., NPZ files)

        # Config train/valid/test date , reference from FinRL
        self.train_start_date = self.config.train_start_date  # Start date for training data
        self.train_end_date = self.config.train_end_date  # End date for training data
        self.valid_start_date = self.config.valid_start_date  # Start date for validation data
        self.valid_end_date = self.config.valid_end_date  # End date for validation data
        self.test_start_date = self.config.test_start_date  # Start date for test data
        self.test_end_date = self.config.test_end_date  # End date for test data

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
            logging.info(f"FE Module - Aggregated cleaned news: {len(aggregated_df)} rows")  # Log aggregated row count for data volume tracking
            return aggregated_df  # Return the full aggregated DataFrame
        logging.info("FE Module - No valid news chunks, returning empty DataFrame")  # Log if no data processed
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
        logging.info(f"FE Module - Filled {nulls_before - nulls_after} NaNs in {prefix} columns")  # Log filled count for data quality tracking
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
                    logging.info(f"FE Module - Fillna with value 3.0 in {col}")
                    df[col] = df[col].fillna(3.0)  # Neutral value for scores (e.g., 1-5 scale)
            elif any(keyword in col for keyword in ['macd', 'rsi', 'cci', 'sma', 'dx', 'boll', 'close']):
                # Handle technical indicator columns: Fill with 0 if NaNs exist
                if df[col].isna().any():
                    logging.info(f"FE Module - Fillna with value 0 in {col}")
                    df[col] = df[col].fillna(0)  # Zero for indicators to represent no signal
        return df  # Return the filled DataFrame

    def merge_features(self, stock_data_dict, sentiment_df, risk_df=None, ind_mode=None):
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
        ind_mode : str, optional
            Indicator mode for feature computation.

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
            processed_df = self.stock_engineer.compute_features(df, symbol, ind_mode)
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
        logging.info(f"FE Module - Filtered fused_df to positive Adj_Close: {fused_df.shape} rows")  # Log post-filter shape

        # Get columns without 'Date'
        cols = fused_df.columns.tolist()
        cols.remove('Date')  # Exclude Date for reordering
        # Reorder columns by field-type across symbols (group by symbol)
        symbols = self.config.symbols
        ordered_cols = ['Date']  # Start with Date
        for symbol in symbols:
            ordered_cols += [col for col in cols if col.endswith(f'_{symbol}')]  # Append columns per symbol

        fused_df = fused_df[ordered_cols]  # Reorder DF for symbol-grouped columns
        logging.info(f"FE Module - Fused features: {fused_df.shape} rows, with risk_mode={self.risk_mode}")  # Log final fused shape and mode

        fused_df = self._fill_nan_after_merge(fused_df)  # Final NaN fill after all operations

        return fused_df  # Return the fully fused and cleaned DataFrame

    def normalize_features(self, df, fit=False, means_stds=None, scaler_path=None):
        """
        Introduction
        ------------
        Normalize selected feature columns in the DataFrame using mean-std standardization.
        Supports fit mode (compute and save scaler) and transform mode (apply existing scaler).

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
        """
        to_normalize = self.stock_engineer.indicators + ['sentiment_score', 'risk_score']  # List of base columns to target for normalization
        present_cols = [col for col in df.columns 
                        if any(ind in col for ind in to_normalize) 
                        and not any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume']) 
                        and pd.api.types.is_numeric_dtype(df[col])]  # Filter columns: match targets, exclude raw prices/volumes, ensure numeric

        if 'Date' in df.columns:
            df = df.set_index('Date')  # Set Date as index to exclude from features if present

        if not present_cols:
            # Early exit if no valid columns found; avoid unnecessary processing
            logging.warning("FE Module - No valid columns to normalize.")
            return df, {}  # Always return tuple for consistency in fit mode

        if fit:
            # Fit mode: Compute or load means_stds
            if scaler_path and os.path.exists(scaler_path):
                means_stds = joblib.load(scaler_path)  # Load existing scaler from cache to reuse stats
                logging.info(f"FE Module - Loaded scaler from {scaler_path}")
            else:
                means_stds = {}  # Initialize dict for column-wise means and stds
                for col in present_cols:
                    mean = df[col].mean()  # Compute mean
                    std = df[col].std()  # Compute std
                    std = max(std, 1e-6)  # Enforce min std to prevent division by zero
                    means_stds[col] = (mean, std)  # Store tuple
                    df[col] = (df[col] - mean) / std  # Apply normalization
                if scaler_path:
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  # Ensure directory exists for saving
                    joblib.dump(means_stds, scaler_path)  # Save scaler for future use
                    logging.info(f"FE Module - Saved new scaler to {scaler_path}")
        else:
            # Transform mode: Apply provided or loaded means_stds
            if scaler_path and os.path.exists(scaler_path):
                means_stds = joblib.load(scaler_path)  # Load scaler if path provided
                logging.info(f"FE Module - Loaded scaler from {scaler_path} for transform")
            if not means_stds:
                # Fallback: If no means_stds, warn and use identity transform (mean=0, std=1)
                logging.warning("FE Module - No scaler found; defaulting to mean=0, std=1")
            for col in present_cols:
                mean, std = means_stds.get(col, (0, 1))  # Get stats or default to no-op
                std = max(std, 1e-6)  # Enforce min std
                df[col] = (df[col] - mean) / std  # Apply normalization
            return df  # Return only df in transform mode

        return df, means_stds  # Return normalized df and computed means_stds in fit mode

    def prepare_rl_data(self, fused_df, symbols=None):
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
        symbols = symbols or self.config.symbols  # Default to config symbols if not provided
        logging.info(f"FE Module - Begin preparing RL data with window={self.window_size}, prediction_days={self.prediction_days}, symbols={symbols}")  # Log preparation params
        assert isinstance(fused_df, pd.DataFrame), f"FE Module - Expected DataFrame, got {type(fused_df)}"  # Assert input type to prevent invalid data processing

        if fused_df.isna().sum().sum() > 0:
            # Check for NaNs; if found, fill with 0 to maintain data integrity for RL
            logging.warning(f"FE Module - NaN values found in fused_df: {fused_df.isna().sum().sum()} before RL window generation")
            fused_df = fused_df.fillna(0)  # Fill NaN with 0 to avoid NaN propagation in windows
        # Ensure excluded Date column
        if 'Date' in fused_df.columns:
            fused_df = fused_df.set_index('Date')  # Set Date as index to exclude from features
        logging.info(f"FE Module - Feature Columns: {fused_df.columns.tolist()}")  # Log feature columns for debugging

        self._update_features_categories(fused_df)       # Update features_* attributes to self.config for inheriting by ConfigTrading

        rl_data = []  # Initialize list to store RL data dicts
        dates = fused_df.index  # Extract dates for start_date assignment
        
        for i in range(len(fused_df) - self.window_size - self.prediction_days + 1):
            # Sliding window loop: Generate windows leaving room for prediction_days
            window_parts = []  # List to collect per-symbol window arrays for concat
            for symbol in symbols:
                symbol_cols = self.config.features_all[symbol]  # Get all features for this symbol from config
                symbol_window = fused_df.iloc[i:i+self.window_size][symbol_cols].values  # Extract (window_size, features_dim_per_symbol) array
                window_parts.append(symbol_window)  # Append for later concat

            # Concatenate all symbol windows along axis=1 to form 2D states (window_size, n_symbols * features_dim_per_symbol)
            states = np.concatenate(window_parts, axis=1)  # Unified 2D array for RL observation

            # Target: future Adj_Close of each stock
            target_cols = [f"Adj_Close_{symbol}" for symbol in symbols]  # Columns for targets
            target = fused_df.iloc[i+self.window_size:i+self.window_size+self.prediction_days][target_cols].values  # Extract targets: future Adj_Close for all symbols; shape (prediction_days, n_symbols)
            rl_data.append({'start_date': dates[i],
                            'states': states,   # 2D ndarray (window_size, n_symbols * features_dim_per_symbol)
                            'targets': target})     # 2D ndarray (prediction_days, n_stocks)

        logging.info(f"FE Module - Prepared {len(rl_data)} RL data")  # Log total prepared items
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
        assert isinstance(rl_data[0]['start_date'], pd.Timestamp), "FE Module - 'start_date' must be pandas.Timestamp"

        rl_data = sorted(rl_data, key=lambda x: x['start_date'])  # Sort RL data by start_date to maintain chronological order
        if self.split_mode == 'date':
            # Date-based split: Filter data within predefined date ranges for train/valid/test
            train_rl_data = [data for data in rl_data if pd.to_datetime(self.train_start_date) <= data['start_date'] <= pd.to_datetime(self.train_end_date)]
            valid_rl_data = [data for data in rl_data if pd.to_datetime(self.valid_start_date) <= data['start_date'] <= pd.to_datetime(self.valid_end_date)]
            test_rl_data = [data for data in rl_data if pd.to_datetime(self.test_start_date) <= data['start_date'] <= pd.to_datetime(self.test_end_date)]
            logging.info(f"FE Module - Split RL data: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")  # Log split sizes for monitoring
        elif self.split_mode == 'ratio':
            # Ratio-based split: Divide data sequentially using split_ratio for train, then 0.1 for valid, rest for test
            n = len(rl_data)
            train_end_idx = int(n * self.split_ratio)  # Calculate train end index
            valid_end_idx = int(n * (self.split_ratio + 0.1))  # Valid end: additional 10% after train
            train_rl_data = rl_data[:train_end_idx]
            valid_rl_data = rl_data[train_end_idx:valid_end_idx]
            test_rl_data = rl_data[valid_end_idx:]
            logging.info(f"FE Module - Split RL data: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")  # Log split sizes for monitoring
        else:
            raise ValueError(f"FE Module - Invalid split mode: {self.split_mode}")  # Error for unsupported split mode

        if len(train_rl_data) == 0 or len(valid_rl_data) == 0 or len(test_rl_data) == 0:
            # Fallback: If any split is empty, use all data as train to prevent training failure
            logging.warning("FE Module - Empty data split. Falling back to all data as train")
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
            logging.warning(f"FE Module - No {col} for mode {mode}; skipping check")
            return score_df
        
        score_var = score_df[col].var()  # Calculate variance of the score column
        score_mean = score_df[col].mean()  # Calculate mean for logging stats
        logging.info(f"FE Module - {col} stats for mode {mode}: var={score_var:.4f}, mean={score_mean:.4f}")  # Log initial statistics for debugging
        
        if score_var < 0.1:    # Check if variance is below threshold (increased from 0.05 to 0.1 for leniency)
            # Low variance detected; proceed to add noise to enhance data variability
            logging.warning(f"FE Module - Low var ({score_var:.4f}) for {col} in mode {mode}; adding mode-specific noise")
            seed = int(hashlib.sha256(mode.encode()).hexdigest(), 16) % (2**32) # Generate reproducible seed using hash of mode for consistency across runs
            np.random.seed(seed)  # Set seed for numpy random to ensure reproducibility
            # Add Gaussian noise and clip to maintain score range [1.0, 5.0] to prevent invalid values
            score_df[col] = np.clip(score_df[col] + np.random.normal(0, 0.3, len(score_df)), 1.0, 5.0)
            new_var = score_df[col].var()  # Recalculate variance after adjustment
            logging.info(f"FE Module - Adjusted var for {col} in mode {mode}: {new_var:.4f}")  # Log new variance to verify adjustment
        
        return score_df  # Return the potentially adjusted DataFrame
    
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
        logging.info("FE Module - Initial exper_data_dict save path")
        try:
            for mode, data_dict in exper_data_dict.items():
                mode_path = os.path.join(self.exper_data_path, f"{mode}_exper_data.npz")
                logging.info(f"FE Module - Initial exper_data_dict save path {mode_path} for mode {mode}")
                npz_data = {}

                for target in ['train', 'valid', 'test']:
                    records = data_dict.get(target, [])
                    # Convert list of dicts to structured arrays
                    if records:
                        start_dates = np.array([str(data['start_date']) for data in records])
                        states = np.array([data['states'] for data in records])
                        targets = np.array([data['targets'] for data in records])
                        npz_data[f"{target}_dates"] = start_dates
                        npz_data[f"{target}_states"] = states
                        npz_data[f"{target}_targets"] = targets
                np.savez_compressed(mode_path, **npz_data)
                logging.info(f"FE Module - Saved {mode} data to {mode_path}")
            logging.info(f"FE Module - Save exper_data_dict successfully")
        except Exception as e:
            logging.warning(f"FE Module - Save exper_data_dict failed: {e}")
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
            for filename in os.listdir(self.exper_data_path):
                logging.info(f"FE Module - Loading exper_data_dict from {filename}")
                if filename.endswith('.npz'):
                    mode = filename.replace('_exper_data.npz', '')
                    file_path = os.path.join(self.exper_data_path, filename)
                    data = np.load(file_path, allow_pickle=True)

                    mode_dict = {}
                    for target in ['train', 'valid', 'test']:
                        logging.info(f"FE Module - Loading exper_data_dict from {target} for mode {mode}")
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
                    logging.info(f"FE Module - Loaded {mode} data from {filename} completely")
            logging.info(f"FE Module - Load exper_data_dict successfully")
        except Exception as e:
            logging.warning(f"FE Module - Load exper_data_dict failed: {e}")
        return exper_data_dict

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
        assert isinstance(fused_df, pd.DataFrame), f"FE Module - Unexpected data type : {type(fused_df)}"  # Assert input is DataFrame to prevent invalid processing
        
        logging.info(f"FE Module - Start to update feature categories to ConfigSetup")  # Log start of update process
        logging.info(f"FE Module - Symbols to process: {self.config.symbols}")  # Log symbols for context
        try:
            for symbol in self.config.symbols:
                # Extract all columns for this symbol
                symbol_cols = [col for col in fused_df.columns if col.endswith(f"_{symbol}")]  # Filter cols ending with _sym, e.g., 'macd_AAPL'

                # Categorize: Price (Adj_Close), sentiment, risk
                price_cols = [col for col in symbol_cols if "Adj_Close" in col]  # Price-related columns
                senti_cols = [col for col in symbol_cols if "sentiment_score" in col]  # Sentiment columns
                risk_cols = [col for col in symbol_cols if "risk_score" in col]  # Risk columns

                # Derive indicators by exclusion
                exclude_cols = set(price_cols + senti_cols + risk_cols)  # Set of excluded categories
                ind_cols = [col for col in symbol_cols if col not in exclude_cols]  # Remaining as indicators

                # Debug log category counts
                logging.debug(f"{symbol} - price:{len(price_cols)}, ind:{len(ind_cols)}, senti:{len(senti_cols)}, risk:{len(risk_cols)}")
                
                # Update config with categorized lists per symbol
                self.config.features_price[symbol] = price_cols  # Store price cols
                self.config.features_ind[symbol] = ind_cols  # Store indicator cols
                self.config.features_senti[symbol] = senti_cols  # Store sentiment cols
                self.config.features_risk[symbol] = risk_cols  # Store risk cols
                self.config.features_all[symbol] = price_cols + ind_cols + senti_cols + risk_cols  # Store all combined
                logging.info(f"FE Module - Update feature categories for symbol: {symbol}")  # Log per-symbol update

            # Compute and update dimension per symbol (use first for symmetry assumption)
            self.features_dim_per_symbol = len(self.features_all.values[0]) # Update feautres dim per symbol to self.config for inheriting by ConfigTrading
            logging.info(f"FE Module - Update Features dim per symbol : {self.features_dim_per_symbol}")  # Log computed dim
            logging.info(f"FE Module - Successfully update all feature categories to ConfigSetup")  # Log overall success
        except Exception as e:
            logging.warning(f"FE Module - Failed to Update feature categories : {e} ")  # Log any exceptions without crashing

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
        if os.listdir(self.exper_data_path):
            # Check if experiment data directory is not empty; if so, load existing data to avoid regeneration
            logging.info("=========== Start to load experiment data dict ===========")
            logging.info(f"FE Module - Loading exper_data_dict from {self.exper_data_path}")
            return self.load_exper_data_dict_npz()  # Load pre-generated data from NPZ files for efficiency
        else:
            # Directory is empty; proceed to generate new experiment data
            logging.info("=========== Start to generate experiment data dict ===========")
            exper_data_dict = {}    # Initialize dictionary to store data for each mode
            exper_news_cols = {
                'benchmark': [],
                'title_only': ['Article_title'],
                'title_textrank': ['Article_title', 'Textrank_summary'],
                'title_fulltext': ['Article_title', 'Full_Text']
            }   # Define news column configurations for different modes
            ind_mode = self.ind_mode    # Retrieve indicator mode from instance
            if single_mode:
                # Single mode: Run only one specific mode for testing
                logging.info(f"FE Module - Running Single mode: {single_mode}")
                exper_modes = [single_mode]
                group = next((g for g, modes in self.exper_mode.items() if single_mode in modes), None) # Find group containing the single mode
                if not group:
                    raise ValueError(f"Unknown single_mode: {single_mode}") # Error if mode not found in any group
            elif exper_mode:
                # Experiment mode: Run all modes in a specified group
                logging.info(f"FE Module - Running Experiment mode: {exper_mode}")
                exper_modes = self.exper_mode.get(exper_mode, [])
                group = exper_mode
                if not exper_modes:
                    raise ValueError(f"Unknown exper_mode group: {exper_mode}") # Error if group not defined
            else:
                # No mode specified: Run all modes across all groups
                logging.info("FE Module - Running All experiment modes")
                exper_modes = sum(self.exper_mode.values(), [])
                group = None

            logging.info(f"FE Module - Experiment modes: {exper_modes} from group: {group}")    # Log the modes to be processed

            cleaned_news = self.process_news_chunks(news_chunks_gen)    # Process news generator into cleaned DataFrame
            logging.info(f"FE Module - Loaded and cleaned news: {len(cleaned_news)} rows")  # Log news data size after cleaning

            for mode in exper_modes:
                original_exper_news_cols = self.news_engineer.text_cols # Backup original text columns to restore later
                # Determine model_type and news_cols based on group
                group = next((group for group, modes in self.exper_mode.items() if mode in modes), None)    # Re-fetch group if needed
                if group == 'rl_algorithm':
                    self.news_engineer.text_cols = exper_news_cols['title_textrank']  # Fix to title_textrank, reference from FinRL_DeepSeek (4.2: stock recommendation prompt)
                    model_type = mode  # PPO, CPPO, etc.
                else:
                    self.news_engineer.text_cols = exper_news_cols.get(mode, [])    # Use mode-specific news columns
                    model_type = 'PPO'  # Default for indicator/news group
                logging.info(f"FE Module - Mode {mode} in group {group}, model_type={model_type}, news_cols={self.news_engineer.text_cols}")    # Log configuration for this mode

                # Compute sentiment
                if group == 'rl_algorithm' or mode != 'benchmark':
                    sentiment_score_df = self.news_engineer.compute_sentiment_risk_score(cleaned_news.copy(), senti_mode='sentiment')    # Compute sentiment scores using FinBERT or similar
                    sentiment_score_df = self._check_and_adjust_sentiment(sentiment_score_df, mode, col='sentiment_score')  # Adjust sentiment if needed (e.g., FinBERT adjustment)
                else:
                    sentiment_score_df = pd.DataFrame(columns=['Date', 'Symbol', 'sentiment_score'])    # Empty DF for benchmark mode
                    logging.info("FE Module - Benchmark mode: no sentiment")    # Log skipping sentiment for benchmarkv

                # Compute risk if enabled
                risk_score_df = None
                if self.risk_mode and (group == 'rl_algorithm' or mode != 'benchmark'):
                    risk_score_df = self.news_engineer.compute_sentiment_risk_score(cleaned_news.copy(), senti_mode='risk')  # Compute risk scores if risk mode is active
                    risk_score_df = self._check_and_adjust_sentiment(risk_score_df, mode, col='risk_score') # Adjust risk scores similarly

                # Merge features and split train/valid/test data by date
                fused_df = self.merge_features(stock_data_dict, sentiment_score_df, risk_score_df, ind_mode)    # Fuse stock data with sentiment/risk features
                if fused_df.empty:
                    raise ValueError(f"Fused DataFrame empty for mode {mode}")  # Error if fusion results in empty DF
                train_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.train_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.train_end_date))]
                valid_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.valid_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.valid_end_date))]
                test_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.test_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.test_end_date))]
                if train_df.empty or valid_df.empty or test_df.empty:
                    raise ValueError(f"Empty split for mode {mode}")    
                logging.info(f"FE Module - Split for mode {mode}: train {len(train_df)}, valid {len(valid_df)}, test {len(test_df)}")

                # Normalize indicators + sentiment + risk columns for RL training
                scaler_path = f"scaler_cache/scaler_train_{group}_{mode}.pkl"   # Dynamic per-group/mode
                train_df, means_stds = self.normalize_features(train_df, fit=True, scaler_path=scaler_path) # Load cache scaler if existed
                valid_df = self.normalize_features(valid_df, fit=False, means_stds=means_stds, scaler_path=scaler_path) # Load cache scaler if existed
                test_df = self.normalize_features(test_df, fit=False, means_stds=means_stds, scaler_path=scaler_path)   # Load cache scaler if existed

                # Prepare window for RL observation
                train_rl_data = self.prepare_rl_data(train_df)  # Prepare RL-compatible data (e.g., windowed observations)
                valid_rl_data = self.prepare_rl_data(valid_df)
                test_rl_data = self.prepare_rl_data(test_df)
                if not train_rl_data or not valid_rl_data or not test_rl_data:
                    raise ValueError(f"No RL data for mode {mode}")
                logging.info(f"FE Module - Prepared RL data for mode {mode}: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")

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
                logging.info(f"FE Module - Generated data for mode {mode}: train {len(split_dict['train'])}, valid {len(split_dict['valid'])}, test {len(split_dict['test'])}")

                self.news_engineer.text_cols = original_exper_news_cols # Restore original text columns after mode processing

            # Release FinBERT resources
            if hasattr(self.news_engineer, 'model') and self.news_engineer.model is not None:
                del self.news_engineer.model    # Delete model to free memory
                del self.news_engineer.tokenizer    # Delete tokenizer
                torch.cuda.empty_cache()    # Clear GPU cache if using CUDA
                logging.info("Released FinBERT resources")

            if single_mode:
                self.save_exper_data_dict_npz(exper_data_dict)  # Save single mode data to NPZ
                logging.info(f"FE Module - Save single mode data for {single_mode} successfully")
                logging.info(f"FE Module - Return single mode train/valid/test data for {single_mode}")
                return exper_data_dict[single_mode]['train'], exper_data_dict[single_mode]['valid'], exper_data_dict[single_mode]['test']
            self.save_exper_data_dict_npz(exper_data_dict)  # Save full dict to NPZ
            logging.info(f"FE Module - Save exper_data_dict successfully")
            return exper_data_dict  # Return the full experiment data dictionary

# %%
from finbert_trader.data.data_resource import DataResource
from finbert_trader.config_setup import ConfigSetup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
custom_setup = {
    'symbols': ['GOOGL'],  # Multi-stock for portfolio test
    # Optional: ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'AMD', 'TSLA', 'META']
    'start': '2015-01-01',
    'end': '2023-12-31',
    'train_start_date': '2015-01-01',
    'train_end_date': '2021-12-31',
    'valid_start_date': '2022-01-01',
    'valid_end_date': '2022-12-31',
    'test_start_date': '2023-01-01',
    'test_end_date': '2023-12-31',
    'exper_mode': {
        'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Includes CPPO, aligned with FinRL_DeepSeek
    }
}
setup_config = ConfigSetup(custom_setup)
logging.info(f"Main - ConfigSetup initialized with symbols: {setup_config.symbols}")

# %%
dr = DataResource(setup_config)
stock_data_dict = dr.fetch_stock_data()
if not stock_data_dict:
    raise ValueError("No stock data fetched")
logging.info(f"Main - Prepared stock data for next step")
stock_data_dict

# %%
cache_path, filtered_cache_path = dr.cache_path_config()
cache_path, filtered_cache_path

# %%
import pandas as pd

# %%
import os
print(os.getcwd())

# %% [markdown]
# origin_dir = '/Users/archy/Projects/finbert_trader/'
# cache_path = origin_dir + cache_path
# filtered_cache_path = origin_dir + filtered_cache_path
# cache_path, filtered_cache_path

# %%
filtered_df = pd.read_csv(filtered_cache_path)
filtered_df

# %%
news_chunks_gen = dr.load_news_data(cache_path, filtered_cache_path)

# %%
fe = FeatureEngineer(setup_config)

# %%
exper_data_dict = fe.generate_experiment_data(stock_data_dict, news_chunks_gen, exper_mode='rl_algorithm')
logging.info(f"Main - Generated experiment data for modes: {list(exper_data_dict.keys())}")

# %%
for mode, data_dict in exper_data_dict.items():
    print(f"Mode: {mode}")
    print(f"Data dict type: {type(data_dict)}")
    for target, data_list in data_dict.items():
        if target != 'model_type':
            print(f"target data: {target}")
            print(f"total data length: {len(data_list)}")
            print(f"data list keys: {data_list[0].keys()}")
            print(f"data shape: {data_list[0]['states'].shape, data_list[0]['targets'].shape}")

# %%
