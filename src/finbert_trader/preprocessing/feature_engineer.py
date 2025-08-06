# feature_engineer.py
# Module: FeatureEngineer
# Purpose: Orchestrator for feature engineering; delegates to StockFeatureEngineer and NewsFeatureEngineer.
# Design: Manages merge, normalize, prepare, split; supports experiment modes.
# Linkage: Inputs from DataResource; outputs split RL data for Environment/Agent.
# Extensibility: Supports 'rl_algorithm' group in exper_mode; fixed news processing to 'title_fulltext'.
# Robustness: Checks sentiment variance; adds mode-specific noise; validates splits.
# Updates: Added 'rl_algorithm' group handling; fixed news to 'title_fulltext'; added 'model_type' in exper_data_dict.
# Updates: Added risk_score computation if config.risk_mode; merged with sentiment in fused_df; adjusted _check_and_adjust_sentiment for both scores (var<0.1 add noise); extended feature_cols to include 'risk_score'; prepare_rl_data includes risk in states, reference from FinRL_DeepSeek (4.3: aggregate R_f for returns adjustment).

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import logging
import hashlib
import os
import joblib

from finbert_trader.preprocessing.stock_features import StockFeatureEngineer
from finbert_trader.preprocessing.news_features import NewsFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, config):
        """
        Initialize with config; instantiates sub-engineers.
        Updates: Added self.risk_mode from config for conditional risk computation.
        """
        self.config = config
        self.decay_lambda = self.config.decay_lambda
        self.window_size = self.config.window_size
        self.prediction_days = self.config.prediction_days
        self.split_ratio = self.config.split_ratio
        self.k_folds = self.config.k_folds
        self.split_mode = self.config.split_mode
        self.cross_valid_mode = self.config.cross_valid_mode
        self.exper_mode = self.config.exper_mode
        self.ind_mode = self.config.ind_mode
        self.risk_mode = self.config.risk_mode
        self.stock_engineer = StockFeatureEngineer(config)
        self.news_engineer = NewsFeatureEngineer(config)
        self.fused_dfs = {}

        # Config train/valid/test date , reference from FinRL
        self.train_start_date = self.config.train_start_date
        self.train_end_date = self.config.train_end_date
        self.valid_start_date = self.config.valid_start_date
        self.valid_end_date = self.config.valid_end_date
        self.test_start_date = self.config.test_start_date
        self.test_end_date = self.config.test_end_date

    def process_news_chunks(self, news_chunks_gen):
        """
        Process news chunks using NewsFeatureEngineer, cleaning only.
        """
        processed_chunks = []
        for chunk in news_chunks_gen:
            if chunk.empty:
                continue
            cleaned_chunk = self.news_engineer.clean_news_data(chunk)   # Drop useless columns and clean text for each column
            # filtered_chunk = self.news_engineer.filter_random_news(cleaned_chunk)   # Filter one random news for each symbol per day
            if not cleaned_chunk.empty:
                processed_chunks.append(cleaned_chunk)
        if processed_chunks:
            aggregated_df = pd.concat(processed_chunks, ignore_index=True)
            logging.info(f"FE Module - Aggregated cleaned news: {len(aggregated_df)} rows")
            return aggregated_df
        logging.info("FE Module - No valid news chunks, returning empty DataFrame")
        return pd.DataFrame(columns=['Date', 'Symbol', 'Article_title', 'Full_Text', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']) # Match original columns in FNSPID dataset

    def _fill_score_columns(df, prefix, fill_value=3.0):
        """
        Handle NaN values in columns by filling 3.0 as default
        """
        score_cols = [col for col in df.columns if col.startswith(prefix)]
        # Sum NaN values before filling
        nulls_before = df[score_cols].isna().sum().sum()
        df[score_cols] = df[score_cols].fillna(fill_value)
        # Sum Nan values after filling
        nulls_after = df[score_cols].isna().sum().sum()
        logging.info(f"FE Module - Filled {nulls_before - nulls_after} NaNs in {prefix} columns")
        return df
    
    def _fill_nan_after_merge(df):
        """
        Handle NaN values before return DataFrame in merge step
        """
        for col in df.columns:
            if 'sentiment_score_' in col or 'risk_score_' in col:
                if df[col].isna().any():
                    logging.info(f"FE Module - Fillna with value 3.0 in {col}")
                    df[col] = df[col].fillna(3.0)
            elif any(keyword in col for keyword in ['macd', 'rsi', 'cci', 'sma', 'dx', 'boll', 'close']):
                if df[col].isna().any():
                    logging.info(f"FE Module - Fillna with value 0 in {col}")
                    df[col] = df[col].fillna(0)
        return df

    def merge_features(self, stock_data_dict, sentiment_df, risk_df=None, ind_mode=None):
        """
        Merge stock and news features with decay fill.
        Updates: Added risk_df param; merge sentiment and risk; apply decay_fill to both; default mid 3.0, reference from FinRL_DeepSeek (4.3: R_f aggregation, but here per stock/day).
        """
        processed_stocks = []
        for symbol, df in stock_data_dict.items():
            processed_df = self.stock_engineer.compute_features(df, symbol, ind_mode)
            processed_stocks.append(processed_df.set_index('Date'))
        all_stock_df = pd.concat(processed_stocks, axis=1, join='outer').reset_index()  # Concat wide table

        for symbol in self.config.symbols:
            if f'Volume_{symbol}' not in all_stock_df.columns and 'Volume' in all_stock_df.columns:
                all_stock_df.rename(columns={'Volume': f'Volume_{symbol}'}, inplace=True)  # Fallback suffix if missed

        if not sentiment_df.empty:
            sentiment_df = sentiment_df.pivot(index='Date',
                                              columns='Symbol',
                                              values='sentiment_score').add_prefix('sentiment_score_')
            all_stock_df = pd.merge(all_stock_df, sentiment_df, left_on='Date', right_index=True, how='left')
            # Fill NaN value after merge
            all_stock_df = self._fill_score_columns(all_stock_df, 'sentiment_score_')

        if self.risk_mode and risk_df is not None and not risk_df.empty:
            risk_df = risk_df.pivot(index='Date',
                                    columns='Symbol',
                                    values='risk_score').add_prefix('risk_score_')
            all_stock_df = pd.merge(all_stock_df, risk_df, left_on='Date', right_index=True, how='left')
            # Fill NaN value after merge
            all_stock_df = self._fill_score_columns(all_stock_df, 'risk_score_')

        fused_df = all_stock_df.sort_values('Date').reset_index(drop=True)  # Ensure order, drop extra index

        # Global filter for positive Adj_Close per-symbol after merge (reference FinRL processor_yahoofinance.py)
        # Ensures prices positive without altering core merge/normalize
        for symbol in self.config.symbols:
            fused_df = fused_df[fused_df[f'Adj_Close_{symbol}'] > 0]
        logging.info(f"FE Module - Filtered fused_df to positive Adj_Close: {fused_df.shape} rows")

        # Get columns without 'Date'
        cols = fused_df.columns.tolist()
        cols.remove('Date')
        # Reorder columns by field-type across symbols (group by symbol)
        symbols = self.config.symbols
        ordered_cols = ['Date']
        for symbol in symbols:
            ordered_cols += [col for col in cols if col.endswith(f'_{symbol}')]

        fused_df = fused_df[ordered_cols]
        logging.info(f"FE Module - Fused features: {fused_df.shape} rows, with risk_mode={self.risk_mode}")

        fused_df = self._fill_nan_after_merge(fused_df)

        return fused_df

    def normalize_features(self, df, fit=False, means_stds=None, scaler_path=None):
        """
        Normalize indicators and sentiment/risk scores.
        - Only applies to numeric columns matching self.indicators + ['sentiment_score', 'risk_score']
        - Price/Volume columns excluded
        """
        to_normalize = self.stock_engineer.indicators + ['sentiment_score', 'risk_score']
        present_cols = [col for col in df.columns 
                        if any(ind in col for ind in to_normalize) 
                        and not any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume']) 
                        and pd.api.types.is_numeric_dtype(df[col])]

        if 'Date' in df.columns:
            df = df.set_index('Date')

        if not present_cols:
            logging.warning("FE Module - No valid columns to normalize.")
            return df, {}  # Always return tuple

        if fit:
            if scaler_path and os.path.exists(scaler_path):
                means_stds = joblib.load(scaler_path)
                logging.info(f"FE Module - Loaded scaler from {scaler_path}")
            else:
                means_stds = {}
                for col in present_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    std = max(std, 1e-6)
                    means_stds[col] = (mean, std)
                    df[col] = (df[col] - mean) / std
                if scaler_path:
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                    joblib.dump(means_stds, scaler_path)
                    logging.info(f"FE Module - Saved new scaler to {scaler_path}")
        else:
            if scaler_path and os.path.exists(scaler_path):
                means_stds = joblib.load(scaler_path)
                logging.info(f"FE Module - Loaded scaler from {scaler_path} for transform")
            if not means_stds:
                logging.warning("FE Module - No scaler found; defaulting to mean=0, std=1")
            for col in present_cols:
                mean, std = means_stds.get(col, (0, 1))
                std = max(std, 1e-6)
                df[col] = (df[col] - mean) / std

        return df, means_stds

    def prepare_rl_data(self, fused_df):
        """
        Prepare rolling windows for RL.
        Updates: Extended feature_cols to include 'risk_score'.
        """
        if fused_df.isna().sum().sum() > 0:
            logging.warning(f"FE Module - NaN values found in fused_df: {fused_df.isna().sum().sum()} before RL window generation")
            fused_df = fused_df.fillna(0)  # Fill NaN with 0

        rl_data = []
        dates = fused_df.index
        for i in range(len(fused_df) - self.window_size - self.prediction_days + 1):
            full_features_per_time = len(fused_df.columns) - 1  # Exclude 'Date'
            # Test window shape 
            expected_dim = self.window_size * (len(fused_df.columns) - 1)
            assert window.shape[0] == expected_dim, f"FE Module - Unexpected window shape: got {window.shape[0]}, expected {expected_dim}"
            window = fused_df.iloc[i:i+self.window_size].values.flatten()  # Flatten numeric to 1D array for RL observation
            if window.shape[0] < self.window_size:
                pad_rows = self.window_size - window.shape[0]
                pad_array = np.zeros((pad_rows, full_features_per_time))
                window = np.vstack((pad_array, window))  # Pad 0 rows at start for short window
            window = window.flatten()  # To 1D full length array

            target = fused_df.iloc[i+self.window_size:i+self.window_size+self.prediction_days][[f'Adj_Close_{symbol}' for symbol in self.config.symbols]].values.flatten()
            rl_data.append({'start_date': dates[i], 'states': window, 'targets': target})
        logging.info(f"FE Module - Prepared {len(rl_data)} RL windows")
        return rl_data

    def split_rl_data(self, rl_data):
        """
        Split rl_data by config.split_mode.
        """
        rl_data = sorted(rl_data, key=lambda x: x['start_date'])
        if self.split_mode == 'date':
            train_rl_data = [data for data in rl_data if pd.to_datetime(self.train_start_date) <= data['start_date'] <= pd.to_datetime(self.train_end_date)]
            valid_rl_data = [data for data in rl_data if pd.to_datetime(self.valid_start_date) <= data['start_date'] <= pd.to_datetime(self.valid_end_date)]
            test_rl_data = [data for data in rl_data if pd.to_datetime(self.test_start_date) <= data['start_date'] <= pd.to_datetime(self.test_end_date)]
            logging.info(f"FE Module - Split RL data: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")
        elif self.split_mode == 'ratio':
            n = len(rl_data)
            train_end_idx = int(n * self.split_ratio)
            valid_end_idx = int(n * (self.split_ratio + 0.1))
            train_rl_data = rl_data[:train_end_idx]
            valid_rl_data = rl_data[train_end_idx:valid_end_idx]
            test_rl_data = rl_data[valid_end_idx:]
            logging.info(f"FE Module - Split RL data: train {len(train_rl_data)}, valid {len(valid_rl_data)}, test {len(test_rl_data)}")
        else:
            raise ValueError(f"FE Module - Invalid split mode: {self.split_mode}")

        if len(train_rl_data) == 0 or len(valid_rl_data) == 0 or len(test_rl_data) == 0:
            logging.warning("FE Module - Empty data split. Falling back to all data as train")
            return {'train': rl_data, 'valid': [], 'test': []}

        if self.k_folds and self.k_folds > 1:
            if self.cross_valid_mode == 'time_series':
                split = TimeSeriesSplit(n_splits=self.k_folds)
            elif self.cross_valid_mode == 'kfold':
                split = KFold(n_splits=self.k_folds, shuffle=False)
            else:
                raise ValueError(f"Invalid cross validation mode: {self.cross_valid_mode}")
            train_folds = []
            indices = np.arange(len(train_rl_data))
            for train_idx, valid_idx in split.split(indices):
                train_fold = [train_rl_data[i] for i in train_idx]
                valid_fold = [train_rl_data[i] for i in valid_idx]
                train_folds.append((train_fold, valid_fold))
            return {'train_folds': train_folds, 'valid': valid_rl_data, 'test': test_rl_data}
        
        return {'train': train_rl_data, 'valid': valid_rl_data, 'test': test_rl_data}

    def _check_and_adjust_sentiment(self, score_df, mode, col='sentiment_score'):
        """
        Check score variance and adjust with mode-specific noise if low.
        Updates: Generalized for both sentiment/risk; increased noise scale if var<0.1.
        """
        if score_df.empty or col not in score_df.columns:
            logging.warning(f"FE Module - No {col} for mode {mode}; skipping check")
            return score_df
        
        score_var = score_df[col].var()
        score_mean = score_df[col].mean()
        logging.info(f"FE Module - {col} stats for mode {mode}: var={score_var:.4f}, mean={score_mean:.4f}")
        
        if score_var < 0.1:    # Raise 0.05 -> 0.1
            logging.warning(f"FE Module - Low var ({score_var:.4f}) for {col} in mode {mode}; adding mode-specific noise")
            seed = int(hashlib.sha256(mode.encode()).hexdigest(), 16) % (2**32) # Generate reproducible seed for each mode
            np.random.seed(seed)
            # Keep the score range [1.0, 5.0]
            score_df[col] = np.clip(score_df[col] + np.random.normal(0, 0.3, len(score_df)), 1.0, 5.0)
            new_var = score_df[col].var()
            logging.info(f"FE Module - Adjusted var for {col} in mode {mode}: {new_var:.4f}")
        
        return score_df

    def generate_experiment_data(self, stock_data_dict, news_chunks_gen, exper_mode=None, single_mode=None):
        """
        Generate rl_data for each exper_mode or single mode.
        Input: stock_data_dict, news_chunks_gen, exper_mode (str, default 'rl_algorithm'), single_mode (str, optional)
        Output: dict {'mode': {'train':, 'valid':, 'test':, 'model_type':}} or (train_rl_data, valid_rl_data, test_rl_data) if single_mode
        Logic: For 'rl_algorithm' group, fix news to 'title_fulltext'; for 'indicator/news', use defined cols.
        Robustness: Adds 'model_type' for TradingAgent; checks sentiment variance; releases FinBERT resources.
        Updates: Added risk computation if self.risk_mode; passed to merge_features; adjusted in _check_and_adjust_sentiment for risk_score; fixed news_cols for rl_algorithm to 'title_textrank' as in literature.
        """
        logging.info("=========== Start to generate experiment data ===========")
        exper_data_dict = {}
        exper_news_cols = {
            'benchmark': [],
            'title_only': ['Article_title'],
            'title_textrank': ['Article_title', 'Textrank_summary'],
            'title_fulltext': ['Article_title', 'Full_Text']
        }
        ind_mode = self.ind_mode
        if single_mode:
            logging.info(f"FE Module - Running Single mode: {single_mode}")
            exper_modes = [single_mode]
            group = next((g for g, modes in self.exper_mode.items() if single_mode in modes), None)
            if not group:
                raise ValueError(f"Unknown single_mode: {single_mode}")
        elif exper_mode:
            logging.info(f"FE Module - Running Experiment mode: {exper_mode}")
            exper_modes = self.exper_mode.get(exper_mode, [])
            group = exper_mode
            if not exper_modes:
                raise ValueError(f"Unknown exper_mode group: {exper_mode}")
        else:
            logging.info("FE Module - Running All experiment modes")
            exper_modes = sum(self.exper_mode.values(), [])
            group = None

        logging.info(f"FE Module - Experiment modes: {exper_modes} from group: {group}")

        cleaned_news = self.process_news_chunks(news_chunks_gen)
        logging.info(f"FE Module - Loaded and cleaned news: {len(cleaned_news)} rows")

        for mode in exper_modes:
            original_exper_news_cols = self.news_engineer.text_cols
            # Determine model_type and news_cols based on group
            group = next((group for group, modes in self.exper_mode.items() if mode in modes), None)
            if group == 'rl_algorithm':
                self.news_engineer.text_cols = exper_news_cols['title_textrank']  # Fix to title_textrank, reference from FinRL_DeepSeek (4.2: stock recommendation prompt)
                model_type = mode  # PPO, CPPO, etc.
            else:
                self.news_engineer.text_cols = exper_news_cols.get(mode, [])
                model_type = 'PPO'  # Default for indicator/news group
            logging.info(f"FE Module - Mode {mode} in group {group}, model_type={model_type}, news_cols={self.news_engineer.text_cols}")

            # Compute sentiment
            if group == 'rl_algorithm' or mode != 'benchmark':
                sentiment_score_df = self.news_engineer.compute_sentiment_score(cleaned_news.copy())
                sentiment_score_df = self._check_and_adjust_sentiment(sentiment_score_df, mode, col='sentiment_score')
            else:
                sentiment_score_df = pd.DataFrame(columns=['Date', 'Symbol', 'sentiment_score'])
                logging.info("FE Module - Benchmark mode: no sentiment")

            # Compute risk if enabled
            risk_score_df = None
            if self.risk_mode and (group == 'rl_algorithm' or mode != 'benchmark'):
                risk_score_df = self.news_engineer.compute_risk_score(cleaned_news.copy())
                risk_score_df = self._check_and_adjust_sentiment(risk_score_df, mode, col='risk_score')

            # Merge features and split train/valid/test data by date
            fused_df = self.merge_features(stock_data_dict, sentiment_score_df, risk_score_df, ind_mode)
            if fused_df.empty:
                raise ValueError(f"Fused DataFrame empty for mode {mode}")
            train_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.train_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.train_end_date))]
            valid_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.valid_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.valid_end_date))]
            test_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.test_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.test_end_date))]
            if train_df.empty or valid_df.empty or test_df.empty:
                raise ValueError(f"Empty split for mode {mode}")
            logging.info(f"FE Module - Split for mode {mode}: train {len(train_df)}, valid {len(valid_df)}, test {len(test_df)}")

            # Normalize RL data
            scaler_path = f"scaler_cache/scaler_train_{group}_{mode}.pkl"   # Dynamic per-group/mode
            train_df, means_stds = self.normalize_features(train_df, fit=True, scaler_path=scaler_path) # Load cache scaler if existed
            valid_df = self.normalize_features(valid_df, fit=False, means_stds=means_stds, scaler_path=scaler_path) # Load cache scaler if existed
            test_df = self.normalize_features(test_df, fit=False, means_stds=means_stds, scaler_path=scaler_path)   # Load cache scaler if existed

            # Prepare RL data
            train_rl_data = self.prepare_rl_data(train_df)
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
            }
            if self.k_folds and self.k_folds > 1:
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

            exper_data_dict[mode] = split_dict
            self.fused_dfs[mode] = {'train': train_df, 'valid': valid_df, 'test': test_df}
            logging.info(f"FE Module - Generated data for mode {mode}: train {len(split_dict['train'])}, valid {len(split_dict['valid'])}, test {len(split_dict['test'])}")

            self.news_engineer.text_cols = original_exper_news_cols

        # Release FinBERT resources
        if hasattr(self.news_engineer, 'model') and self.news_engineer.model is not None:
            del self.news_engineer.model
            del self.news_engineer.tokenizer
            torch.cuda.empty_cache()
            logging.info("Released FinBERT resources")

        if single_mode:
            return exper_data_dict[single_mode]['train'], exper_data_dict[single_mode]['valid'], exper_data_dict[single_mode]['test']
        return exper_data_dict