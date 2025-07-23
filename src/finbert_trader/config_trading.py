# config_trade.py
# Module: ConfigTrading
# Purpose: Centralized configuration for trading modules, with decoupled model params for flexibility.
# Design: Supports model selection (e.g., 'PPO'); dynamically loads params dict; inherits from upstream ConfigSetup.
# Linkage: Receives upstream_config for shared params; model_params dict for Agent/Backtest.
# Extensibility: Add new models to SUPPORTED_MODELS and _model_defaults; enables easy switching for experiments.
# Robustness: Validate model; fallback to default 'PPO'; log selections.
# Reusability: get_model_params() returns dict for direct use in RL libs.

import logging

# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigTrading:
    # Supported models for switching
    SUPPORTED_MODELS = ['PPO', 'A2C', 'DDPG']  # Can add 'RLlib' etc. in future

    # Default params per model (reference: FinRL for PPO, Stable-Baselines3 for others)
    _model_defaults = {
        'PPO': {  # From FinRL rl_model.py
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 64,
        },
        'A2C': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
        },
        'DDPG': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.001,
            "batch_size": 128,
            "buffer_size": 1000000,
        },
    }

    # config directory
    SCALER_SAVE_DIR = 'scaler_cache'
    MODEL_SAVE_DIR = 'model_cache'
    RESULTS_SAVE_DIR = 'results_cache'
    LOG_SAVE_DIR = 'logs'

    def __init__(self, custom_config=None, upstream_config=None, model='PPO'):
        """
        Initialize with optional custom_config, upstream_config (ConfigSetup), and model str.
        Input: custom_config (dict), upstream_config (ConfigSetup), model (str, default 'PPO')
        Output: Self with attributes; self.model_params as selected dict.
        Logic: Set defaults; inherit upstream; set model and load params; override; validate.
        Extensibility: model_params can be overridden in custom_config; log model selection.
        Robustness: Check model in SUPPORTED_MODELS; fallback defaults.
        """
        # Default trading params (non-model specific)
        self.initial_cash = 10000
        self.transaction_cost = 0.001
        self.reward_scale = 1.0
        self.commission_rate = 0.0005
        self.slippage = 0.001
        self.total_timesteps = 100000  # Shared for training
        
        # Set model and load params
        self.model = model
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model}; supported: {self.SUPPORTED_MODELS}")
        self.model_params = self._model_defaults.get(self.model, {})
        logging.info(f"Selected model: {self.model} with params {self.model_params}")

        # Inherit from upstream_config if provided (linkage)
        if upstream_config:
            shared_params = ['window_size', 'prediction_days', 'split_ratio', 'k_folds', 'batch_size']
            for param in shared_params:
                if hasattr(upstream_config, param):
                    setattr(self, param, getattr(upstream_config, param))
                    logging.info(f"Inherited from upstream: {param} = {getattr(self, param)}")
                else:
                    logging.warning(f"Upstream missing {param}; using default if available")

        # Override with custom_config (can override model_params as dict)
        if custom_config:
            for key, value in custom_config.items():
                if key == 'model_params':
                    self.model_params.update(value)  # Merge override for params
                    logging.info(f"Overrode model_params: {value}")
                elif hasattr(self, key):
                    setattr(self, key, value)
                    logging.info(f"Overrode config: {key} = {value}")
                else:
                    logging.warning(f"Ignored unknown config key: {key}")

        # Basic validation
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.transaction_cost < 0 or self.transaction_cost > 1:
            raise ValueError("transaction_cost must be between 0 and 1")
        if hasattr(self, 'window_size') and self.window_size <= 0:
            raise ValueError("window_size must be positive (inherited or set)")

    def get_model_params(self):
        """
        Return the selected model's params dict (for Agent init).
        """
        return self.model_params

    @staticmethod
    def get_defaults(model='PPO'):
        """
        Static method to return default config dict for a model.
        """
        defaults = {
            'initial_cash': 10000,
            'transaction_cost': 0.001,
            'reward_scale': 1.0,
            'commission_rate': 0.0005,
            'slippage': 0.001,
            'total_timesteps': 100000,
            'model_params': ConfigTrading._model_defaults.get(model, {})
        }
        return defaults

# Example instantiation
if __name__ == "__main__":
    from finbert_trader.config_setup import ConfigSetup  # upstream
    upstream = ConfigSetup()
    custom = {'model_params': {'learning_rate': 0.0001}}  # Override PPO lr
    config_trade = ConfigTrading(custom, upstream, model='PPO')
    print(config_trade.model)  # 'PPO'
    print(config_trade.get_model_params())  # Updated dict

