# preprocessing.py (Updated Version with Sub-Engineers)
# Module: Preprocessing
# Purpose: Orchestrator for feature engineering; calls StockFeatureEngineer and NewsFeatureEngineer.
# Design: Retains merge, normalize, prepare, split; delegates stock/news to sub-classes for modularity.
# Linkage: Inputs from DataResource; outputs split RL data for Environment/Agent.
# Extensibility: Config-driven split_ratio/k_folds for validation.
# Robustness: Added split_rl_data for train/val/cross-val.

import pandas as pd
import numpy as np
from finbert_trader.config_setup import ConfigSetup
from sklearn.model_selection import train_test_split, KFold  # For splitting
import logging

# Import sub-engineers (assume in same dir or imported)
from finbert_trader.preprocessing.stock_features import StockFeatureEngineer
from finbert_trader.preprocessing.news_features import NewsFeatureEngineer

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, config):
        """
        Initialize with config; instantiates sub-engineers.
        Added: 'split_ratio':0.8, 'k_folds':None (if int>1, use cross-val).
        """
        self.config = config
        self.decay_lambda = self.config.decay_lambda
        self.window_size = self.config.window_size
        self.prediction_days = self.config.prediction_days
        self.split_ratio = self.config.split_ratio
        self.k_folds = self.config.k_folds
        self.stock_engineer = StockFeatureEngineer(config)
        self.news_engineer = NewsFeatureEngineer(config)

    def process_news_chunks(self, news_chunks_gen):
        """
        Process news chunks using NewsFeatureEngineer.
        Input: news_chunks_gen
        Output: Aggregated sentiment_news_df
        Logic: Delegate clean/compute to sub-engineer; concat chunks.
        """
        processed_chunks = []
        for chunk in news_chunks_gen:
            if chunk.empty:
                continue
            cleaned_chunk = self.news_engineer.clean_news_data(chunk)
            sentiment_chunk = self.news_engineer.compute_sentiment(cleaned_chunk)
            processed_chunks.append(sentiment_chunk)
        if processed_chunks:
            aggregated_df = pd.concat(processed_chunks, ignore_index=True)
            logging.info(f"Aggregated {len(aggregated_df)} rows from chunks")
            return aggregated_df
        return pd.DataFrame()

    def merge_features(self, stock_data_dict, news_df):
        """
        Merge stock and news features.
        Input: stock_data_dict {symbol: df}, news_df
        Output: fused pd.DataFrame
        Logic: Delegate stock compute to StockFeatureEngineer; merge and decay fill.
        """
        processed_stocks = []
        for df in stock_data_dict.values():
            processed_df = self.stock_engineer.compute_features(df)
            processed_stocks.append(processed_df)
        all_stock_df = pd.concat(processed_stocks, ignore_index=True)
        
        fused_df = pd.merge(all_stock_df, news_df, on=['Date', 'Symbol'], how='left')
        fused_df.sort_values(['Symbol', 'Date'], inplace=True)
        
        def decay_fill(group):
            last_val = None
            last_date = None
            for idx, row in group.iterrows():
                if pd.notnull(row['sentiment_score']):
                    last_val = row['sentiment_score']
                    last_date = row['Date']
                elif last_val is not None:
                    t = (row['Date'] - last_date).days
                    group.at[idx, 'sentiment_score'] = last_val * np.exp(-self.decay_lambda * t)
            return group
        
        fused_df = fused_df.groupby('Symbol').apply(decay_fill).reset_index(drop=True)

        # Fill remaining NaN (leading) with neutral score 3
        fused_df['sentiment_score'].fillna(3, inplace=True)
        
        # Check for any remaining NaN to enhance robustness
        if fused_df['sentiment_score'].isna().any():
            logging.warning("Remaining NaN in sentiment_score after filling")

        logging.info(f"Fused features: {fused_df.shape} rows")
        return fused_df

    def normalize_features(self, fused_df):
        """
        Normalize numerical features.
        Input: fused_df
        Output: normalized fused_df
        Logic: Z-score for num cols.
        """
        num_cols = [col for col in fused_df.columns if col not in ['Date', 'Symbol'] and fused_df[col].dtype in ['float64', 'int64']]
        for col in num_cols:
            mean = fused_df[col].mean()
            std = fused_df[col].std()
            fused_df[col] = (fused_df[col] - mean) / std if std != 0 else 0
        logging.info("Normalized features")
        return fused_df

    def prepare_rl_data(self, fused_df):
        """
        Prepare rolling windows for RL.
        Input: fused_df
        Output: List of dicts {'states': array, 'targets': array, 'symbol', 'start_date'}
        Logic: Sliding windows per symbol.
        """
        feature_cols = [col for col in fused_df.columns if col not in ['Date', 'Symbol']]
        rl_data = []
        for symbol, group in fused_df.groupby('Symbol'):
            group = group.sort_values('Date')
            for i in range(len(group) - self.window_size - self.prediction_days + 1):
                window = group.iloc[i:i+self.window_size][feature_cols].values
                target = group.iloc[i+self.window_size:i+self.window_size+self.prediction_days]['Adj_Close'].values
                rl_data.append({'symbol': symbol, 'start_date': group.iloc[i]['Date'], 'states': window, 'targets': target})
        logging.info(f"Prepared {len(rl_data)} RL windows")
        return rl_data

    def split_rl_data(self, rl_data):
        """
        New: Split RL data into train/val sets, with optional cross-validation.
        Input: rl_data (list of dicts)
        Output: If k_folds=None: (train_list, val_list); else: list of (train, val) folds.
        Logic: Sort by start_date to avoid leak; ratio split or KFold (time-aware by index).
        Robustness: Shuffle=False for time series; log splits.
        Extensibility: Config-driven; supports future time-series CV.
        """
        # Sort rl_data by start_date for temporal order
        rl_data = sorted(rl_data, key=lambda x: x['start_date'])
        
        if self.k_folds and self.k_folds > 1:
            kf = KFold(n_splits=self.k_folds, shuffle=False)
            folds = []
            indices = np.arange(len(rl_data))
            for train_idx, val_idx in kf.split(indices):
                train = [rl_data[i] for i in train_idx]
                val = [rl_data[i] for i in val_idx]
                folds.append((train, val))
            logging.info(f"Created {self.k_folds} cross-validation folds")
            return folds
        else:
            train, val = train_test_split(rl_data, test_size=1 - self.split_ratio, shuffle=False)
            logging.info(f"Split RL data: train {len(train)}, val {len(val)}")
            return train, val

# Example usage
if __name__ == "__main__":
    config = ConfigSetup()
    # dr = DataResource(config)
    # stock_data = dr.fetch_stock_data()
    # news_chunks = dr.load_news_data('data_cache/nasdaq_exteral_data.csv')
    pp = FeatureEngineer(config)
    # sentiment_news = pp.process_news_chunks(news_chunks)
    # fused_df = pp.merge_features(stock_data, sentiment_news)
    # normalized_df = pp.normalize_features(fused_df)
    # rl_data = pp.prepare_rl_data(normalized_df)
    # splits = pp.split_rl_data(rl_data)  # (train, val) or folds