# stock_feature_engineer.py
# Module: StockFeatureEngineer
# Purpose: Dedicated class for computing stock technical features using TA-Lib.
# Design: Single responsibility for stock features; reusable independently.
# Linkage: Called by Preprocessing.merge_features to process stock data.
# Robustness: Enhanced NaN handling with mean/zero fill for stability.

import pandas as pd
import talib
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockFeatureEngineer:
    def __init__(self, config):
        """
        Initialize with config (e.g., 'indicators': list).
        """
        self.indicators = config.indicators # Access from config

    def compute_features(self, stock_df):
        """
        Compute TA-Lib features for a single stock DataFrame.
        Input: stock_df (pd.DataFrame) with 'Adj_Close', 'Adj_High', 'Adj_Low', etc.
        Output: stock_df with added indicator columns.
        Logic: Map indicators to talib functions; handle NaNs with ffill then mean/zero fill.
        Robustness: Use global mean for persistent NaNs (e.g., low volume stocks); log filled count.
        """
        if 'Adj_Close' not in stock_df.columns:
            raise ValueError("Stock DataFrame missing 'Adj_Close' column")
        
        indicator_funcs = {
            'macd': lambda df: talib.MACD(df['Adj_Close'])[0],
            'boll_ub': lambda df: talib.BBANDS(df['Adj_Close'])[0],
            'boll_lb': lambda df: talib.BBANDS(df['Adj_Close'])[2],
            'rsi_30': lambda df: talib.RSI(df['Adj_Close'], timeperiod=30),
            'cci_30': lambda df: talib.CCI(df['Adj_High'], df['Adj_Low'], df['Adj_Close'], timeperiod=30),
            'dx_30': lambda df: talib.DX(df['Adj_High'], df['Adj_Low'], df['Adj_Close'], timeperiod=30),
            'close_30_sma': lambda df: talib.SMA(df['Adj_Close'], timeperiod=30),
            'close_60_sma': lambda df: talib.SMA(df['Adj_Close'], timeperiod=60),
        }
        
        for ind in self.indicators:
            try:
                stock_df[ind] = indicator_funcs.get(ind, lambda df: np.nan)(stock_df)
            except KeyError:
                logging.warning(f"Indicator {ind} not supported")
        
        # NaN handling: ffill first, then fill remaining with mean (or 0 for neutral indicators like MACD)
        nan_count_before = stock_df[self.indicators].isna().sum().sum()
        stock_df[self.indicators] = stock_df[self.indicators].fillna(method='ffill')
        for ind in self.indicators:
            if ind in ['macd', 'rsi_30', 'cci_30', 'dx_30']:  # Neutral 0 for momentum
                stock_df[ind].fillna(0, inplace=True)
            else:  # Mean for others (e.g., SMA)
                mean_val = stock_df[ind].mean()
                stock_df[ind].fillna(mean_val, inplace=True)
        
        nan_count_after = stock_df[self.indicators].isna().sum().sum()
        logging.info(f"Computed stock features; filled {nan_count_before - nan_count_after} NaNs")
        return stock_df