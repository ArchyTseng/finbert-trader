# config_setup.py
# Module: ConfigSetup
# Purpose: Centralized configuration for preprocessing modules.
# Design: Defaults + overrides + validation; supports train/val/test dates and experiment modes.
# Linkage: Passed to DataResource/Preprocessing; inherited by ConfigTrading.
# Robustness: Validate dates/order/format; ensure non-empty symbols list.
# Reusability: get_defaults for export.
# Updates: Modified exper_mode to include 'rl_algorithm' for RL comparison (PPO, CPPO, A2C, DDPG); retained indicator/news for extensibility; added 'risk_prompt' for news risk assessment, reference from FinRL_DeepSeek (3: Risk Assessment Prompt).
import os
import numpy as np
import json
from datetime import datetime
import logging

# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigSetup:
    # Referenc from FinRL_DeepSeek 
    nasdaq_100_tickers_july_17_2023 = [
        "ADBE", "ADP", "ABNB", "ALGN", "GOOGL", "GOOG", "AMZN", "AMD", "AEP", "AMGN",
        "ADI", "ANSS", "AAPL", "AMAT", "ASML", "AZN", "TEAM", "ADSK", "BKR", "BIIB",
        "BKNG", "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CTSH", "CMCSA", "CEG", "CPRT",
        "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DLTR", "EBAY", "EA",
        "ENPH", "EXC", "FAST", "FTNT", "GEHC", "GILD", "GFS", "HON", "IDXX", "ILMN",
        "INTC", "INTU", "ISRG", "JD", "KDP", "KLAC", "KHC", "LRCX", "LCID", "LULU",
        "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT", "MRNA", "MDLZ", "MNST",
        "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PANW", "PAYX", "PYPL",
        "PDD", "PEP", "QCOM", "REGN", "ROST", "SGEN", "SIRI", "SBUX", "SNPS", "TMUS",
        "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS"
    ]

    # Config directories, from FinRL reference
    RAW_DATA_DIR = 'raw_data_cache'
    PROCESSED_NEWS_DIR = 'processed_news_cache'
    FUSED_DATA_DIR = 'fused_data_cache'
    EXPER_DATA_DIR = 'exper_data_cache'
    PLOT_CACHE_DIR = 'plot_cache'
    PLOT_NEWS_DIR = 'plot_news_cache'
    RESULTS_CACHE_DIR = 'results_cache'
    EXPERIMENT_CACHE_DIR = 'exper_cache'
    SCALER_CACHE_DIR = 'scaler_cache'
    LOG_SAVE_DIR = 'logs'

    def __init__(self, custom_config=None):
        """
        Introduction
        ------------
        Initialize ConfigSetup with default parameters and optional overrides from custom_config.
        Sets up symbols, dates, indicators, experiment modes, and other RL pipeline configs.
        Validates dates, symbols, and parameters; computes derived values like features_dim_per_stock.

        Parameters
        ----------
        custom_config : dict, optional
            Dictionary to override default attributes (e.g., {'symbols': ['GOOG']}).

        Notes
        -----
        - Defaults centralized for easy maintenance; overrides logged for traceability.
        - Derived: timeperiods from ind_mode, indicators with dynamic tp, features_dim_per_stock.
        - exper_mode groups for multi-experiments; added 'rl_algorithm' with 'CPPO' for risk-sensitive RL (ref: FinRL_DeepSeek 4.3).
        - For split_mode='ratio', computes fallback dates based on global start/end.
        - Validates date sequence, chunksize, split_ratio, symbols; raises ValueError on issues.
        - risk_mode enables risk_prompt (ref: FinRL_DeepSeek 3).
        """
        # Default values centralized here
        self.symbols = ['AAPL']  # Example S&P500 symbols; extensible list
        self.start = '2000-01-01'   # Global fallback start for overall data range
        self.end = '2023-12-31'    # Global fallback end for overall data range
        self.chunksize = 100000  # For large data loading to manage memory in chunk processing
        self.indicators_mode = {'short': 10,
                                'middle': 20,
                                'long': 30,
                                'custom': None}  # Mapping of indicator modes to timeperiods
        self.ind_mode = 'short'  # Default indicator mode
        self.timeperiods = self.indicators_mode.get(self.ind_mode, 10)  # Default as short mode; retrieve timeperiod
        self.indicators = [
            "macd",
            "boll_ub",
            "boll_lb",
            f"rsi_{self.timeperiods}",
            f"cci_{self.timeperiods}",
            f"dx_{self.timeperiods}",
            f"close_sma_short_{self.timeperiods}",
            f"close_sma_long_{self.timeperiods * 2}",
        ]  # Reference from FinRL; dynamic indicators with timeperiods
        self.sentiment_keys = ['sentiment_score', 'risk_score']  # Keys for sentiment/risk features
        self.decay_lambda = 0.03  # From FNSPID paper; for potential decay in scoring

        self.window_size = 50  # For RL windows; observation history length
        self.window_factor = 2  # For scale RL window size in FeatureEngineer
        self.window_extend = 10 # For extend RL window size in FeatureEngineer

        self.smooth_window_size = 5 # Timeperiod for apply smoothing to features using a rolling mean
        self.prediction_days = 3  # For Short-term trading strategy; future days to predict
        self.batch_size = 32  # For FinBERT inference; balance memory and speed
        self.text_cols = ['Article_title', 'Textrank_summary']  # Default scheme from FNSPID; text fields for sentiment
        self.split_ratio = 0.8  # For train/val split if split_mode='ratio'
        self.k_folds = None  # If >1, enable cross-val on train data
        self.split_mode = 'date'  # 'date' (default) or 'ratio' for fallback splitting
        self.cross_valid_mode = 'time_series'  # 'time_series' (default for TimeSeriesSplit) or 'kfold'
        self.risk_mode = True  # Enable risk assessment prompt, reference from FinRL_DeepSeek (3: Risk Prompt)

        # Date params reference from FinRL
        self.train_start_date = '2010-01-01'  # Train period start
        self.train_end_date = '2021-12-31'  # Train period end
        self.valid_start_date = '2022-01-01'  # Validation period start
        self.valid_end_date = '2022-12-31'  # Validation period end
        self.test_start_date = '2023-01-01'  # Test period start
        self.test_end_date = '2023-12-31'  # Test period end

        # Config exper_mode for multi-mode experiments
        self.exper_mode = {
            'indicator/news': ['benchmark',
                               'title_only',
                               'title_textrank',
                               'title_fulltext'],  # Groups for indicator/news experiments
            'rl_algorithm': ['PPO',
                             'CPPO',
                             'A2C']  # Added 'CPPO' for comparison, reference from FinRL_DeepSeek; RL algorithm groups
        }

        # Override with custom_config

        # Initialize features attributes, updating in FeatureEngineer "prepare_rl_data()" function, inherited by ConfigTrading
        self.features_dim_per_symbol = None   # Expected: Adj_Close + indicators + sentiment + risk; compute total features per stock in FeatureEngineer Module
        self.features_price = {}
        self.features_ind = {}
        self.features_senti = {}
        self.features_risk = {}
        self.features_all = {}

        # Initialize thresholds for RL trading, updated in FeatureEngineer "prepare_rl_data()" function, inherited by ConfigTrading
        self.senti_threshold = {'train': {},
                                'valid': {},
                                'test': {}}
        self.risk_threshold = {'train': {},
                                'valid': {},
                                'test': {}}
        self.threshold_factor = 0.5

        self.cache_dir = 'config_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

        self.save_npz = True    # Control saveing .npz file in FeatureEngineer
        self.load_npz = False   # Control loading .npz file in FeatureEngineer

        self.force_normalize = True # Control normalize_features function
        self.filter_ind = []     # Control normalize_features function, fitered target indicators
        
        self.news_cache_path = None # Update news cache path dynamiclly

        self._features_initialized = self.load_or_init_features()   # Update Trur / False dynamiclly

        self.use_senti_factor = False   # Control sentiment score column
        self.use_risk_factor = False    # Control risk score column

        self.use_senti_features = False # Control sentiment features for window size, in StockTradingEnv
        self.use_risk_features = False  # Control risk features for window size, in StockTradingEnv

        self.use_senti_threshold = False    # Control sentiment threshold for S_f mechanism
        self.use_risk_threshold = False     # Control risk threshold for S_f mechanism

        self.use_dynamic_infusion = False   # Control dynamic infusion_strength mechanism

        self.use_symbol_name = True     # Control saved filename rule

        if custom_config:
            # Apply overrides from dict for flexibility
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)  # Set attribute if exists
                    logging.info(f"CS Module - __init__ - Overrode config: {key} = {value}")  # Log override for auditing
                else:
                    logging.warning(f"CS Module - __init__ - Ignored unknown config key: {key}")  # Warn on unknown keys

        # Fallback for train/valid/test dates if split_mode is 'ratio'
        if self.split_mode == 'ratio':
            # Compute dates based on ratio for non-date splitting
            start_date = datetime.strptime(self.start, '%Y-%m-%d')  # Parse global start
            end_date = datetime.strptime(self.end, '%Y-%m-%d')  # Parse global end
            total_days = (end_date - start_date).days  # Calculate total days
            if total_days <= 0:
                raise ValueError("End date must be after start date")  # Validate range
            train_days = max(int(total_days * self.split_ratio), 1)  # Ensure at least 1 day for train
            valid_days = max(int(total_days * 0.1), 1)  # 10% for valid, min 1
            test_days = total_days - train_days - valid_days  # Remaining for test
            if test_days < 0:
                # Adjust if test negative
                logging.warning("CS Module - __init__ - Test days <= 0, adjusting valid_days")
                valid_days = total_days - train_days - 1
                test_days = 1  # Min 1 for test
            self.train_end_date = (start_date + datetime.timedelta(days=train_days)).strftime('%Y-%m-%d')  # Set train end
            self.valid_start_date = self.train_end_date  # Valid starts after train
            self.valid_end_date = (start_date + datetime.timedelta(days=train_days + valid_days)).strftime('%Y-%m-%d')  # Set valid end
            self.test_start_date = self.valid_end_date  # Test starts after valid
            self.test_end_date = self.end  # Test ends at global end
            logging.info(f"CS Module - __init__ - Fallback dates set: train {self.train_start_date} to {self.train_end_date}, valid {self.valid_start_date} to {self.valid_end_date}, test {self.test_start_date} to {self.test_end_date}")  # Log computed dates

        # Basic validation
        if not isinstance(self.chunksize, int) or self.chunksize <= 0:
            raise ValueError("chunksize must be positive integer")  # Validate chunksize
        
        if self.split_ratio <= 0 or self.split_ratio >= 1:
            raise ValueError("split_ratio must be between 0 and 1")  # Validate split_ratio
        
        # Validate date sequence
        try:
            dates = [self.start,
                     self.end,
                     self.train_start_date,
                     self.train_end_date,
                     self.valid_start_date,
                     self.valid_end_date,
                     self.test_start_date,
                     self.test_end_date]  # Collect all dates for sequence check
            parsed = [datetime.strptime(date, '%Y-%m-%d') for date in dates]  # Parse to datetime
            if not (parsed[0] <= parsed[2] <= parsed[3] <= parsed[4] <= parsed[5] <= parsed[6] <= parsed[7] <= parsed[1]):
                raise ValueError("Date sequence invalid: global start <= train <= valid <= test <= global end")  # Ensure logical order
        except Exception as e:
            logging.error(f"CS Module - __init__ - Invalid date format: {e}")  # Log parsing errors
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")  # Raise on format issues

        # Validate symbols
        if not isinstance(self.symbols, list) or not self.symbols:
            raise ValueError("symbols must be a non-empty list")  # Ensure symbols valid

    def save_config_cache(self, filename="config_cache.json"):
        """Save dynamic attributes to cache file"""
        data = {
            "features_dim_per_symbol": self.features_dim_per_symbol,
            "features_price": self.features_price,
            "features_ind": self.features_ind,
            "features_senti": self.features_senti,
            "features_risk": self.features_risk,
            "features_all": self.features_all,
            "senti_threshold": self.senti_threshold,
            "risk_threshold": self.risk_threshold,
            "threshold_factor": self.threshold_factor
        }
        path = os.path.join(self.cache_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"CS Module - save_config_cache - [ConfigSetup] Saved config cache to {path}")

    def load_config_cache(self, filename="config_cache.json"):
        """Load dynamic attributes from cache file"""
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            logging.info(f"CS Module - load_config_cache - [ConfigSetup] No config cache found at {path}")
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        # Recover cache attributes values
        self.features_dim_per_symbol = data.get("features_dim_per_symbol")
        self.features_price = data.get("features_price", {})
        self.features_ind = data.get("features_ind", {})
        self.features_senti = data.get("features_senti", {})
        self.features_risk = data.get("features_risk", {})
        self.features_all = data.get("features_all", {})
        self.senti_threshold = data.get("senti_threshold", {})
        self.risk_threshold = data.get("risk_threshold", {})
        self.threshold_factor = data.get("threshold_factor", 0.5)
        logging.info(f"CS Module - load_config_cache - [ConfigSetup] Loaded config cache from {path}")
        return True
    
    def load_or_init_features(self):
        """First Running Pipeline initials config features, otherwise load from cache"""
        if self.load_config_cache() and self.load_npz:
            print("[ConfigSetup] Features already cached, will skip recalculation.")
            return True
        return False

    @staticmethod
    def get_defaults():
        """
        Static method to return default config dict for reference or export.
        """
        return {
            'symbols': ['AAPL'],
            'start': '2010-01-01',
            'end': '2023-12-31',
            'chunksize': 100000,
            'indicators_mode': {'short': 10,
                                'middle': 20,
                                'long': 30},
            'ind_mode': 'short',
            'timeperiods': 10,
            'indicators': ["macd",
                           "boll_ub",
                           "boll_lb",
                           "rsi_10",
                           "cci_10",
                           "dx_10",
                           "close_10_sma",
                           "close_20_sma"],
            'decay_lambda': 0.03,
            'window_size': 50,
            'window_factor': 2,
            'window_extend': 10,
            'prediction_days': 3,
            'batch_size': 32,
            'text_cols': ['Article_title', 'Textrank_summary'],
            'split_ratio': 0.8,
            'k_folds': None,
            'split_mode': 'date',
            'cross_valid_mode': 'time_series',
            'train_start_date': '2010-01-01',
            'train_end_date': '2021-12-31',
            'valid_start_date': '2022-01-01',
            'valid_end_date': '2022-12-31',
            'test_start_date': '2023-01-01',
            'test_end_date': '2023-12-31',
            'risk_mode': True,
            'save_npz': True,
            'load_npz': False,
            'load_fused_csv': False,
            'use_senti_factor': False,
            'use_risk_factor': False,
            'use_senti_features': False,
            'use_risk_features': False,
            'use_senti_threshold': False,
            'use_risk_threshold': False,
            'use_dynamic_infusion': False,
            'exper_mode': {
                'indicator/news': ['benchmark',
                                   'title_only',
                                   'title_textrank',
                                   'title_fulltext'],
                'rl_algorithm': ['PPO',
                                 'CPPO',
                                 'A2C']
            },
            
        }

if __name__ == "__main__":
    custom = {'symbols': ['AAPL'], 'batch_size': 32, 'exper_mode': {'rl_algorithm': ['PPO', 'CPPO', 'A2C']}}
    config = ConfigSetup(custom)
    print(config.symbols)  # ['AAPL']
    print(config.exper_mode)  # Updated exper_mode