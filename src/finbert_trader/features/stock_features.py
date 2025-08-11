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
        Introduction
        ------------
        Initialize the StockFeatureEngineer with configuration parameters.
        Sets up indicators mode, time periods, and list of indicators for stock feature computation.

        Parameters
        ----------
        config : object
            Configuration object containing indicators_mode, timeperiods, ind_mode, indicators.

        Notes
        -----
        - Extracts parameters from config for modular and configurable technical indicator calculations.
        - indicators_mode: Defines computation modes for indicators.
        - timeperiods: List of time windows for indicators like SMA, RSI.
        - ind_mode: Current active indicator mode.
        - indicators: List of technical indicators to compute (e.g., MACD, RSI).
        """
        self.indicators_mode = config.indicators_mode  # Mode for computing technical indicators
        self.timeperiods = config.timeperiods  # List of time periods for window-based indicators
        self.ind_mode = config.ind_mode  # Current indicator mode for selective computation
        self.indicators = config.indicators # Access from config  # List of indicators to generate features for

    def compute_features(self, stock_df, symbol, ind_mode=None):
        """
        Introduction
        ------------
        Compute technical indicator features for a stock DataFrame using TA-Lib.
        Adds symbol-suffixed indicator columns based on configured indicators and time periods.
        Handles NaN filling with forward fill followed by type-specific values (0 for momentum, mean for others).

        Parameters
        ----------
        stock_df : pd.DataFrame
            Stock DataFrame with price columns like 'Adj_Close_{symbol}', 'Adj_High_{symbol}', etc.
        symbol : str
            Stock symbol for column suffixing.
        ind_mode : str, optional
            Indicator mode to select timeperiod; defaults to config.timeperiods if None.

        Returns
        -------
        pd.DataFrame
            Updated stock_df with added indicator columns.

        Notes
        -----
        - Requires 'Adj_Close_{symbol}' column; raises ValueError if missing.
        - Uses indicators_mode dict for timeperiod mapping; defaults to 10 if unknown.
        - Supported indicators: macd, boll_ub/lb, rsi, cci, dx, close_sma (with tp and tp*2).
        - NaN handling: ffill first, then 0 for macd/rsi/cci/dx, mean for others.
        - Logs filled NaNs per indicator for data quality tracking.
        - Warns on unsupported indicators.
        """
        logging.info(f"StockFeatureEngineer - compute_features - Start to Compute features for {symbol}")
        if f'Adj_Close_{symbol}' not in stock_df.columns:
            logging.error(f"StockFeatureEngineer - compute_features - Missing 'Adj_Close_{symbol}' column in stock_df")
            raise ValueError(f"Stock DataFrame missing 'Adj_Close_{symbol}' column")  # Ensure required close column exists
        
        # Map mode to time periods
        if ind_mode :
            logging.info(f"SF Module - compute_features - Indicator mode: {ind_mode}")
            tp = self.indicators_mode.get(ind_mode, 10)  # Default to short if unknown; retrieve timeperiod from mode
        else:
            tp = self.timeperiods  # Use full timeperiods list if no mode specified       
        
        indicator_funcs = {
            'macd': lambda df: talib.MACD(df[f'Adj_Close_{symbol}'])[0],  # MACD line
            'boll_ub': lambda df: talib.BBANDS(df[f'Adj_Close_{symbol}'])[0],  # Upper Bollinger Band
            'boll_lb': lambda df: talib.BBANDS(df[f'Adj_Close_{symbol}'])[2],  # Lower Bollinger Band
            f'rsi_{tp}': lambda df: talib.RSI(df[f'Adj_Close_{symbol}'], timeperiod=tp),  # RSI with timeperiod
            f'cci_{tp}': lambda df: talib.CCI(df[f'Adj_High_{symbol}'], df[f'Adj_Low_{symbol}'], df[f'Adj_Close_{symbol}'], timeperiod=tp),  # CCI with high/low/close
            f'dx_{tp}': lambda df: talib.DX(df[f'Adj_High_{symbol}'], df[f'Adj_Low_{symbol}'], df[f'Adj_Close_{symbol}'], timeperiod=tp),  # DX (Directional Movement Index)
            f'close_{tp}_sma': lambda df: talib.SMA(df[f'Adj_Close_{symbol}'], timeperiod=tp),  # SMA with tp
            f'close_{tp * 2}_sma': lambda df: talib.SMA(df[f'Adj_Close_{symbol}'], timeperiod=tp * 2),  # SMA with double tp
        }  # Dictionary of TA-Lib functions for each indicator
        
        for ind in self.indicators:
            # Loop to compute and add each indicator column
            try:
                func = indicator_funcs.get(ind, lambda df: np.nan)  # Get function or default to NaN if not found
                stock_df[f'{ind}_{symbol}'] = func(stock_df)  # Add _{symbol} suffix to indicators; compute and assign
            except KeyError:
                logging.warning(f"SF Module - compute_features - Indicator {ind} not supported")  # Warn if indicator key missing in dict
        
        # NaN handling: ffill first, then fill remaining with mean (or 0 for neutral indicators like MACD)
        for ind in self.indicators:
            col = f'{ind}_{symbol}'  # Construct column name
            nan_count_before = stock_df[col].isna().sum()  # Count NaNs before filling

            stock_df[col] = stock_df[col].ffill()  # Forward fill to propagate last known values
            if ind in ['macd', f'rsi_{tp}', f'cci_{tp}', f'dx_{tp}']:  # Neutral 0 for momentum indicators
                stock_df[col].fillna(0, inplace=True)  # Fill remaining with 0
            else:  # Mean for others (e.g., SMA, Bollinger)
                mean_val = stock_df[col].mean()  # Compute mean of non-NaN values
                stock_df[col].fillna(mean_val, inplace=True)  # Fill with mean

            nan_count_after = stock_df[col].isna().sum()  # Count after filling
            if nan_count_before > nan_count_after:
                logging.info(f"SF Module - compute_features - Filled {nan_count_before - nan_count_after} NaNs in {ind}_{symbol}")  # Log filled count if any
        logging.info(f"SF Module - compute_features - Successfully Compute features for {symbol}")
        logging.info(f"SF Module - compute_features - {symbol} stock_df shape: {stock_df.shape}, columns: {stock_df.columns.tolist()}")
        return stock_df  # Return updated DF with features