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
    LOG_SAVE_DIR = 'logs'

    def __init__(self, custom_config=None):
        """
        Initialize with optional custom_config dict to override defaults.
        Input: custom_config (dict, optional)
        Output: Self with attributes set.
        Logic: Set defaults; update from custom; validate dates and symbols.
        Extensibility: Added 'rl_algorithm' in exper_mode for RL comparison; can extend with new groups.
        Updates: Added 'CPPO' to rl_algorithm for risk-sensitive comparison, reference from FinRL_DeepSeek (4.3: LLM-infused CPPO); added risk_prompt string for news_features, reference from FinRL_DeepSeek (3: Risk Prompt).
        """
        # Default values centralized here
        self.symbols = ['AAPL']  # Example S&P500 symbols; extensible list
        self.start = '2000-01-01'   # Global fallback start
        self.end = '2023-12-31'    # Global fallback end
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
        ]  # Reference from FinRL
        self.decay_lambda = 0.03  # From FNSPID paper
        self.window_size = 50  # For RL windows
        self.prediction_days = 3  # For RL targets
        self.batch_size = 32  # For FinBERT inference
        self.text_cols = ['Article_title', 'Textrank_summary']  # Default scheme from FNSPID
        self.split_ratio = 0.8  # For train/val split
        self.k_folds = None  # If >1, enable cross-val
        self.split_mode = 'date'  # 'date' (default) or 'ratio' for fallback
        self.cross_valid_mode = 'time_series'  # 'time_series' (default for TimeSeriesSplit) or 'kfold'
        self.risk_prompt = "You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk."  # Reference from FinRL_DeepSeek (3: Risk Assessment Prompt)
        self.risk_mode = True  # Enable risk assessment prompt, reference from FinRL_DeepSeek (3: Risk Prompt)

        # Date params reference from FinRL
        self.train_start_date = '2022-06-01'
        self.train_end_date = '2023-06-01'
        self.valid_start_date = '2023-06-02'
        self.valid_end_date = '2023-09-01'
        self.test_start_date = '2023-09-02'
        self.test_end_date = '2023-12-31'

        # Config exper_mode for multi-mode experiments
        self.exper_mode = {
            'indicator/news': ['benchmark', 'title_only', 'title_textrank', 'title_fulltext'],
            'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Added 'CPPO' for comparison, reference from FinRL_DeepSeek
        }

        # Override with custom_config
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logging.info(f"CS Module - Overrode config: {key} = {value}")
                else:
                    logging.warning(f"CS Module - Ignored unknown config key: {key}")

        # Fallback for train/valid/test dates if split_mode is 'ratio'
        if self.split_mode == 'ratio':
            start_date = datetime.strptime(self.start, '%Y-%m-%d')
            end_date = datetime.strptime(self.end, '%Y-%m-%d')
            total_days = (end_date - start_date).days
            if total_days <= 0:
                raise ValueError("End date must be after start date")
            train_days = max(int(total_days * self.split_ratio), 1)
            valid_days = max(int(total_days * 0.1), 1)
            test_days = total_days - train_days - valid_days
            if test_days < 0:
                logging.warning("CS Module - Test days <= 0, adjusting valid_days")
                valid_days = total_days - train_days - 1
                test_days = 1
            self.train_end_date = (start_date + datetime.timedelta(days=train_days)).strftime('%Y-%m-%d')
            self.valid_start_date = self.train_end_date
            self.valid_end_date = (start_date + datetime.timedelta(days=train_days + valid_days)).strftime('%Y-%m-%d')
            self.test_start_date = self.valid_end_date
            self.test_end_date = self.end
            logging.info(f"CS Module - Fallback dates set: train {self.train_start_date} to {self.train_end_date}, valid {self.valid_start_date} to {self.valid_end_date}, test {self.test_start_date} to {self.test_end_date}")

        # Basic validation
        if not isinstance(self.chunksize, int) or self.chunksize <= 0:
            raise ValueError("chunksize must be positive integer")
        if self.split_ratio <= 0 or self.split_ratio >= 1:
            raise ValueError("split_ratio must be between 0 and 1")
        
        # Validate date sequence
        try:
            dates = [self.start, self.end, self.train_start_date, self.train_end_date,
                     self.valid_start_date, self.valid_end_date, self.test_start_date, self.test_end_date]
            parsed = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
            if not (parsed[0] <= parsed[2] <= parsed[3] <= parsed[4] <= parsed[5] <= parsed[6] <= parsed[7] <= parsed[1]):
                raise ValueError("Date sequence invalid: global start <= train <= valid <= test <= global end")
        except Exception as e:
            logging.error(f"CS Module - Invalid date format: {e}")
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")

        # Validate symbols
        if not isinstance(self.symbols, list) or not self.symbols:
            raise ValueError("symbols must be a non-empty list")

    @staticmethod
    def get_defaults():
        """
        Static method to return default config dict for reference or export.
        """
        return {
            'symbols': ['AAPL'],
            'start': '2000-01-01',
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
            'k_folds': None,
            'split_mode': 'date',
            'cross_valid_mode': 'time_series',
            'train_start_date': '2010-01-01',
            'train_end_date': '2021-12-31',
            'valid_start_date': '2022-01-01',
            'valid_end_date': '2022-12-31',
            'test_start_date': '2023-01-01',
            'test_end_date': '2023-12-31',
            'risk_prompt': "You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk.",
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