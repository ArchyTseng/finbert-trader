# config_setup.py
# Module: ConfigSetup
# Purpose: Centralized configuration class for the entire pipeline.
# Design: Holds all default parameters; allows overrides via init dict; provides attribute access for ease.
# Linkage: Instantiated once, passed to all modules; enhances maintainability and extensibility.
# Robustness: Basic validation in __init__ for key params; can extend with full type checking.
# Reusability: Can load from file (e.g., JSON/YAML) in future; staticmethod for defaults.

from datetime import datetime
import logging

# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigSetup:
    # config directory , from FinRL reference
    DATA_SAVE_DIR = 'data_cache'
    LOG_SAVE_DIR = 'logs'

    def __init__(self, custom_config=None):
        """
        Initialize with optional custom_config dict to override defaults.
        Input: custom_config (dict, optional)
        Output: Self with attributes set.
        Logic: Set defaults; update from custom; basic validation.
        Extensibility: Add more params as pipeline grows; log overrides.
        """
        # Default values centralized here
        self.symbols = ['AAPL']  # Example S&P500 symbols
        self.start = '2010-01-01'
        self.end = datetime.today().strftime('%Y-%m-%d')    # Set Default End Date
        self.chunksize = 100000  # For large data loading
        self.indicators = [
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma"
        ]  # From FinRL reference
        self.decay_lambda = 0.03  # From FNSPID paper
        self.window_size = 50  # For RL windows
        self.prediction_days = 3  # For RL targets
        self.batch_size = 32  # For FinBERT inference
        self.text_cols = ['Article_title', 'Textrank_summary']  # Default scheme2. From FNSPID Datasets
        self.split_ratio = 0.8  # For train/val split
        self.k_folds = None  # If >1, enable cross-val

        # Override with custom_config
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logging.info(f"Overrode config: {key} = {value}")
                else:
                    logging.warning(f"Ignored unknown config key: {key}")

        # Basic validation (expand as needed)
        if not isinstance(self.chunksize, int) or self.chunksize <= 0:
            raise ValueError("chunksize must be positive integer")
        if self.split_ratio <= 0 or self.split_ratio >= 1:
            raise ValueError("split_ratio must be between 0 and 1")
        
        # Validate start/end format
        try:
            datetime.strptime(self.start, '%Y-%m-%d')
            datetime.strptime(self.end, '%Y-%m-%d')
        except Exception as e:
            logging.error(f"Invalid date format: {e}")
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")

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
            'indicators': ["macd",
                           "boll_ub",
                           "boll_lb",
                           "rsi_30",
                           "cci_30",
                           "dx_30",
                           "close_30_sma",
                           "close_60_sma"],
            'decay_lambda': 0.03,
            'window_size': 50,
            'prediction_days': 3,
            'batch_size': 32,
            'text_cols': ['Article_title', 'Textrank_summary'],
            'split_ratio': 0.8,
            'k_folds': None
        }

# Example instantiation
if __name__ == "__main__":
    custom = {'symbols': ['AAPL'], 'batch_size': 32}
    config = ConfigSetup(custom)
    print(config.symbols)  # ['AAPL']