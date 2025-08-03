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
            cleaned_chunk = self.news_engineer.clean_news_data(chunk)
            if not cleaned_chunk.empty:
                processed_chunks.append(cleaned_chunk)
        if processed_chunks:
            aggregated_df = pd.concat(processed_chunks, ignore_index=True)
            logging.info(f"FE Module - Aggregated cleaned news: {len(aggregated_df)} rows")
            return aggregated_df
        logging.info("FE Module - No valid news chunks, returning empty DataFrame")
        return pd.DataFrame(columns=['Date', 'Symbol', 'Full_Text', 'Article_title', 'Textrank_summary'])

    def merge_features(self, stock_data_dict, sentiment_df, risk_df=None):
        """
        Merge stock and news features with decay fill.
        Updates: Added risk_df param; merge sentiment and risk; apply decay_fill to both; default mid 3.0, reference from FinRL_DeepSeek (4.3: R_f aggregation, but here per stock/day).
        """
        processed_stocks = []
        for symbol, df in stock_data_dict.items():
            processed_df = self.stock_engineer.compute_features(df, symbol)
            processed_stocks.append(processed_df.set_index('Date'))
        all_stock_df = pd.concat(processed_stocks, axis=1, join='outer').reset_index()  # Concat wide table

        for symbol in self.config.symbols:
            if f'Volume_{symbol}' not in all_stock_df.columns and 'Volume' in all_stock_df.columns:
                all_stock_df.rename(columns={'Volume': f'Volume_{symbol}'}, inplace=True)  # Fallback suffix if missed

        if not sentiment_df.empty:
            sentiment_df = sentiment_df.pivot(index='Date',
                                              columns='Symbol',
                                              values='sentiment_score').add_prefix('sentiment_score_').fillna(3.0)
            all_stock_df = pd.merge(all_stock_df, sentiment_df, left_on='Date', right_index=True, how='left')

        if self.risk_mode and risk_df is not None and not risk_df.empty:
            risk_df = risk_df.pivot(index='Date',
                                    columns='Symbol',
                                    values='risk_score').add_prefix('risk_score_').fillna(3.0)
            all_stock_df = pd.merge(all_stock_df, risk_df, left_on='Date', right_index=True, how='left')

        fused_df = all_stock_df.sort_values('Date').reset_index(drop=True)  # Ensure order, drop extra index

        # reorder columns by field-type across symbols (group by prefix)
        # Remove the resort-columns lines to keep original order per symbol for merge
        # all_cols = fused_df.columns.tolist()
        # sorted_cols = ['Date'] + sorted([c for c in all_cols if c != 'Date'], key=lambda x: (x.split('_')[0], x))  # Sort by field prefix then full (e.g., Adj_Close_AAPL before Adj_Close_MSFT)
        # fused_df = fused_df[sorted_cols]
        logging.info(f"FE Module - Fused features: {fused_df.shape} rows, with risk_mode={self.risk_mode}")

        return fused_df

    def normalize_features(self, df, fit=False, means_stds=None):
        """
        Normalize indicators and sentiment_score.
        Updates: Added 'risk_score' to to_normalize.
        """
        to_normalize = self.stock_engineer.indicators + ['sentiment_score', 'risk_score']
        present_cols = [col for col in df.columns if any(ind in col for ind in to_normalize)]  # e.g., 'macd_AAPL' contains 'macd'
        df = df.set_index('Date') if 'Date' in df.columns else df  # Set Date as index if present

        # Enhanced NaN handling: replace all NaN with 0 globally before normalization
        df = df.fillna(0)
        df = pd.DataFrame(np.nan_to_num(df.values, nan=0), columns=df.columns, index=df.index)  # Force num, handle inf/NaN

        df = df.select_dtypes(include=[np.number])  # Filter numeric, index preserved
        logging.info("FE Module - Set Date as index before numeric filter")
        if not present_cols:
            logging.warning("FE Module - No columns to normalize")
            return df, {} if fit else df
        
        df = df.fillna(0)  # Global fill 0 for all numeric (neutral/safe for indicators/sentiment/risk); or df.fillna(df.mean()) if prefer mean, but 0 simpler for NaN-heavy
        logging.info("FE Module - Global NaN filled with 0 before normalization")

        if fit:
            means_stds = {}
            for col in present_cols:
                mean = df[col].mean()
                std = df[col].std()
                std = max(std, 1e-6)  # Clip std > 1e-6 to prevent NaN
                means_stds[col] = (mean, std)
                df[col] = (df[col] - mean) / std if std != 0 else 0
            logging.info(f"FE Module - Fitted and normalized {len(present_cols)} columns")
            return df, means_stds
        else:
            for col in present_cols:
                mean, std = means_stds.get(col, (0, 1))
                std = max(std, 1e-6)  # Same clip
                df[col] = (df[col] - mean) / std if std != 0 else 0
            logging.info(f"FE Module - Transformed {len(present_cols)} columns")
            return df

    def prepare_rl_data(self, fused_df):
        """
        Prepare rolling windows for RL.
        Updates: Extended feature_cols to include 'risk_score'.
        """
        rl_data = []
        dates = fused_df.index
        for i in range(len(fused_df) - self.window_size - self.prediction_days + 1):
            full_features_per_time = len(fused_df.columns) - 1  # Exclude 'Date'
            window = fused_df.iloc[i:i+self.window_size].values.flatten()  # Flatten numeric (index=Date auto excluded)
            if window.shape[0] < self.window_size:
                pad_rows = self.window_size - window.shape[0]
                pad_array = np.zeros((pad_rows, full_features_per_time))
                window = np.vstack((pad_array, window))  # Pad 0 rows at start for short window
            window = window.flatten()  # To 1D full len

            # target_col_idx = fused_df.columns.get_loc(f'Adj_Close_{self.config.symbols[0]}') if self.config.symbols else 3  # Dynamic index for lead Adj_Close
            target = fused_df.iloc[i+self.window_size:i+self.window_size+self.prediction_days][[f'Adj_Close_{sym}' for sym in self.config.symbols]].values.flatten()
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
        
        if score_var < 0.05:
            logging.warning(f"FE Module - Low var ({score_var:.4f}) for {col} in mode {mode}; adding mode-specific noise")
            seed = int(hashlib.sha256(mode.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            noise = np.random.normal(0, 0.1, len(score_df))
            score_df[col] += noise
            new_var = score_df[col].var()
            logging.info(f"FE Module - Adjusted var for {col} in mode {mode}: {new_var:.4f}")
        
        return score_df

    def generate_experiment_data(self, stock_data_dict, news_chunks_gen, single_mode=None):
        """
        Generate rl_data for each test_mode or single mode.
        Input: stock_data_dict, news_chunks_gen, single_mode (str, optional)
        Output: dict {'mode': {'train':, 'valid':, 'test':, 'model_type':}} or (train_rl_data, valid_rl_data, test_rl_data) if single_mode
        Logic: For 'rl_algorithm' group, fix news to 'title_fulltext'; for 'indicator/news', use defined cols.
        Robustness: Adds 'model_type' for TradingAgent; checks sentiment variance; releases FinBERT resources.
        Updates: Added risk computation if self.risk_mode; passed to merge_features; adjusted in _check_and_adjust_sentiment for risk_score; fixed news_cols for rl_algorithm to 'title_textrank' as in literature.
        """
        logging.info("=========== Start to generate experiment data ===========")
        exper_data_dict = {}
        exper_cols = {
            'benchmark': [],
            'title_only': ['Article_title'],
            'title_textrank': ['Article_title', 'Textrank_summary'],
            'title_fulltext': ['Article_title', 'Full_Text']
        }
        available_modes = set(sum(self.exper_mode.values(), []))
        exper_modes = [single_mode] if single_mode else sum(self.exper_mode.values(), [])
        logging.info(f"FE Module - Experiment modes: {exper_modes}")

        # Load and clean news once
        cleaned_news = self.process_news_chunks(news_chunks_gen)
        logging.info(f"FE Module - Loaded and cleaned news: {len(cleaned_news)} rows")

        for mode in exper_modes:
            if mode not in available_modes:
                logging.warning(f"FE Module - Skipping unknown mode: {mode}")
                continue
            if single_mode and single_mode not in available_modes:
                raise ValueError(f"FE Module - Unknown single mode: {single_mode}, expected one of {available_modes}")

            original_exper_cols = self.news_engineer.text_cols
            # Determine model_type and news_cols based on group
            group = next((g for g, modes in self.exper_mode.items() if mode in modes), None)
            if group == 'rl_algorithm':
                self.news_engineer.text_cols = exper_cols['title_textrank']  # Fix to title_textrank, reference from FinRL_DeepSeek (4.2: stock recommendation prompt)
                model_type = mode  # PPO, CPPO, etc.
            else:
                self.news_engineer.text_cols = exper_cols.get(mode, [])
                model_type = 'PPO'  # Default for indicator/news group
            logging.info(f"FE Module - Mode {mode} in group {group}, model_type={model_type}, news_cols={self.news_engineer.text_cols}")

            # Compute sentiment
            if group == 'rl_algorithm' or mode != 'benchmark':
                sentiment_news = self.news_engineer.compute_sentiment(cleaned_news.copy())
                sentiment_news = self._check_and_adjust_sentiment(sentiment_news, mode, col='sentiment_score')
            else:
                sentiment_news = pd.DataFrame(columns=['Date', 'Symbol', 'sentiment_score'])
                logging.info("FE Module - Benchmark mode: no sentiment")

            # Compute risk if enabled
            risk_news = None
            if self.risk_mode and (group == 'rl_algorithm' or mode != 'benchmark'):
                risk_news = self.news_engineer.compute_risk(cleaned_news.copy())
                risk_news = self._check_and_adjust_sentiment(risk_news, mode, col='risk_score')

            # Merge and split
            fused_df = self.merge_features(stock_data_dict, sentiment_news, risk_news)
            if fused_df.empty:
                raise ValueError(f"Fused DataFrame empty for mode {mode}")
            train_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.train_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.train_end_date))]
            valid_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.valid_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.valid_end_date))]
            test_df = fused_df[(fused_df['Date'] >= pd.to_datetime(self.test_start_date)) & (fused_df['Date'] <= pd.to_datetime(self.test_end_date))]
            if train_df.empty or valid_df.empty or test_df.empty:
                raise ValueError(f"Empty split for mode {mode}")
            logging.info(f"FE Module - Split for mode {mode}: train {len(train_df)}, valid {len(valid_df)}, test {len(test_df)}")

            # Normalize
            train_df, means_stds = self.normalize_features(train_df, fit=True)
            valid_df = self.normalize_features(valid_df, fit=False, means_stds=means_stds)
            test_df = self.normalize_features(test_df, fit=False, means_stds=means_stds)

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

            self.news_engineer.text_cols = original_exper_cols

        # Release FinBERT resources
        if hasattr(self.news_engineer, 'model') and self.news_engineer.model is not None:
            del self.news_engineer.model
            del self.news_engineer.tokenizer
            torch.cuda.empty_cache()
            logging.info("Released FinBERT resources")

        if single_mode:
            return exper_data_dict[single_mode]['train'], exper_data_dict[single_mode]['valid'], exper_data_dict[single_mode]['test']
        return exper_data_dict