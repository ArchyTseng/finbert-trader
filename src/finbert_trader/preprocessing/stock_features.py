# stock_features.py
# Module: StockFeatureEngineer
# Purpose: Dedicated class for computing stock technical features using TA-Lib.
# Design: Single responsibility for stock features; reusable independently.
# Linkage: Called by Preprocessing.merge_features to process stock data.
# Robustness: Enhanced NaN handling with mean/zero fill for stability.
# Updates: added logging for NaN count per indicator for debugging instability.

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
        self.indicators_mode = config.indicators_mode
        self.timeperiods = config.timeperiods
        self.ind_mode = config.ind_mode
        self.indicators = config.indicators # Access from config

    def compute_features(self, stock_df, symbol, ind_mode=None):
        """
        Compute TA-Lib features for a single stock DataFrame.
        Input: stock_df (pd.DataFrame) with 'Adj_Close', 'Adj_High', 'Adj_Low', etc.
        Output: stock_df with added indicator columns.
        Logic: Map indicators to talib functions; handle NaNs with ffill then mean/zero fill.
        Robustness: Use global mean for persistent NaNs (e.g., low volume stocks); log filled count per indicator.
        """
        if f'Adj_Close_{symbol}' not in stock_df.columns:
            raise ValueError(f"Stock DataFrame missing 'Adj_Close_{symbol}' column")
        
        # Map mode to time periods
        if ind_mode :
            tp = self.indicators_mode.get(ind_mode, 10)  # Default to short if unknown 
        else:
            tp = self.timeperiods       
        
        indicator_funcs = {
            'macd': lambda df: talib.MACD(df[f'Adj_Close_{symbol}'])[0],
            'boll_ub': lambda df: talib.BBANDS(df[f'Adj_Close_{symbol}'])[0],
            'boll_lb': lambda df: talib.BBANDS(df[f'Adj_Close_{symbol}'])[2],
            f'rsi_{tp}': lambda df: talib.RSI(df[f'Adj_Close_{symbol}'], timeperiod={tp}),
            f'cci_{tp}': lambda df: talib.CCI(df[f'Adj_High_{symbol}'], df[f'Adj_Low_{symbol}'], df[f'Adj_Close_{symbol}'], timeperiod={tp}),
            f'dx_{tp}': lambda df: talib.DX(df[f'Adj_High_{symbol}'], df[f'Adj_Low_{symbol}'], df[f'Adj_Close_{symbol}'], timeperiod={tp}),
            f'close_{tp}_sma': lambda df: talib.SMA(df[f'Adj_Close_{symbol}'], timeperiod={tp}),
            f'close_{tp * 2}_sma': lambda df: talib.SMA(df[f'Adj_Close_{symbol}'], timeperiod={tp * 2}),
        }
        
        for ind in self.indicators:
            try:
                func = indicator_funcs.get(ind, lambda df: np.nan)
                stock_df[f'{ind}_{symbol}'] = func(stock_df)  # Add _{symbol} suffix to indicators
            except KeyError:
                logging.warning(f"SF Module - Indicator {ind} not supported")
        
        # NaN handling: ffill first, then fill remaining with mean (or 0 for neutral indicators like MACD)
        for ind in self.indicators:
            col = f'{ind}_{symbol}'
            nan_count_before = stock_df[col].isna().sum()

            stock_df[col] = stock_df[col].ffill()
            if ind in ['macd', f'rsi_{tp}', f'cci_{tp}', f'dx_{tp}']:  # Neutral 0 for momentum
                stock_df[col].fillna(0, inplace=True)
            else:  # Mean for others (e.g., SMA)
                mean_val = stock_df[col].mean()
                stock_df[col].fillna(mean_val, inplace=True)

            nan_count_after = stock_df[col].isna().sum()
            if nan_count_before > nan_count_after:
                logging.info(f"SF Module - Filled {nan_count_before - nan_count_after} NaNs in {ind}_{symbol}")
        
        return stock_df