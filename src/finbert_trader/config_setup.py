# config_setup.py
# Module: ConfigSetup
# Purpose: Centralized configuration for preprocessing modules.
# Design: Defaults + overrides + validation; supports train/val/test dates and experiment modes.
# Linkage: Passed to DataResource/Preprocessing; inherited by ConfigTrading.
# Robustness: Validate dates/order/format; ensure non-empty symbols list.
# Reusability: get_defaults for export.
# Updates: Modified exper_mode to include 'rl_algorithm' for RL comparison (PPO, CPPO, A2C, DDPG); retained indicator/news for extensibility; added 'risk_prompt' for news risk assessment, reference from FinRL_DeepSeek (3: Risk Assessment Prompt).

from datetime import datetime
import logging

# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigSetup:
    # Config directories, from FinRL reference
    DATA_SAVE_DIR = 'data_cache'
    EXPER_DATA_DIR = 'exper_cache'
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
                                'long': 30}  # Mapping of indicator modes to timeperiods
        self.ind_mode = 'short'  # Default indicator mode
        self.timeperiods = self.indicators_mode.get(self.ind_mode, 10)  # Default as short mode; retrieve timeperiod
        self.indicators = [
            "macd",
            "boll_ub",
            "boll_lb",
            f"rsi_{self.timeperiods}",
            f"cci_{self.timeperiods}",
            f"dx_{self.timeperiods}",
            f"close_{self.timeperiods}_sma",
            f"close_{self.timeperiods * 2}_sma"
        ]  # Reference from FinRL; dynamic indicators with timeperiods
        self.sentiment_keys = ['sentiment_score', 'risk_score']  # Keys for sentiment/risk features
        self.decay_lambda = 0.03  # From FNSPID paper; for potential decay in scoring
        self.window_size = 50  # For RL windows; observation history length
        self.prediction_days = 1  # For Short-term trading strategy; future days to predict
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

        if custom_config:
            # Apply overrides from dict for flexibility
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)  # Set attribute if exists
                    logging.info(f"CS Module - Overrode config: {key} = {value}")  # Log override for auditing
                else:
                    logging.warning(f"CS Module - Ignored unknown config key: {key}")  # Warn on unknown keys

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
                logging.warning("CS Module - Test days <= 0, adjusting valid_days")
                valid_days = total_days - train_days - 1
                test_days = 1  # Min 1 for test
            self.train_end_date = (start_date + datetime.timedelta(days=train_days)).strftime('%Y-%m-%d')  # Set train end
            self.valid_start_date = self.train_end_date  # Valid starts after train
            self.valid_end_date = (start_date + datetime.timedelta(days=train_days + valid_days)).strftime('%Y-%m-%d')  # Set valid end
            self.test_start_date = self.valid_end_date  # Test starts after valid
            self.test_end_date = self.end  # Test ends at global end
            logging.info(f"CS Module - Fallback dates set: train {self.train_start_date} to {self.train_end_date}, valid {self.valid_start_date} to {self.valid_end_date}, test {self.test_start_date} to {self.test_end_date}")  # Log computed dates

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
            logging.error(f"CS Module - Invalid date format: {e}")  # Log parsing errors
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")  # Raise on format issues

        # Validate symbols
        if not isinstance(self.symbols, list) or not self.symbols:
            raise ValueError("symbols must be a non-empty list")  # Ensure symbols valid

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
            'exper_mode': {
                'indicator/news': ['benchmark',
                                   'title_only',
                                   'title_textrank',
                                   'title_fulltext'],
                'rl_algorithm': ['PPO',
                                 'CPPO',
                                 'A2C']
            }
        }

if __name__ == "__main__":
    custom = {'symbols': ['AAPL'], 'batch_size': 32, 'exper_mode': {'rl_algorithm': ['PPO', 'CPPO', 'A2C']}}
    config = ConfigSetup(custom)
    print(config.symbols)  # ['AAPL']
    print(config.exper_mode)  # Updated exper_mode