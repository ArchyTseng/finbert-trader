# config_trading.py
# Module: ConfigTrading
# Purpose: Centralized configuration for trading modules, with decoupled model params for flexibility.
# Design: Supports model selection (e.g., 'PPO', 'CPPO'); dynamically loads params dict; inherits from upstream ConfigSetup.
# Linkage: Receives upstream_config for shared params; model_params dict for Agent/Backtest.
# Extensibility: Add new models to SUPPORTED_MODELS and _model_defaults; enables easy switching for experiments.
# Robustness: Validate model; fallback to default 'PPO'; log selections.
# Reusability: get_model_params() returns dict for direct use in RL libs.
# Updates: Added 'CPPO' support with CVaR params (alpha, lambda, beta), reference from FinRL_DeepSeek (4.1.2 CVaR-PPO formula); added infusion_strength for LLM/FinBERT injection tuning (0.001-0.1, reference from FinRL_DeepSeek 5.3); added risk_mode flag for risk assessment prompt.

import logging
import numpy as np

# Logging setup (shared)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigTrading:
    # Supported models for switching
    SUPPORTED_MODELS = ['PPO', 'A2C', 'DDPG', 'TD3', 'SAC', 'CPPO']  # Added 'CPPO' for risk-sensitive, reference from FinRL_DeepSeek (4.1.2)

    # Default params per model (reference: FinRL for PPO, Stable-Baselines3 for others; extended for CPPO)
    _model_defaults = {
        'PPO': {  # From FinRL rl_model.py
            "n_steps": 2048,
            "ent_coef": 0.2,   # Increased to encourage action entropy, from 0.01 -> 0.02 -> 0.05 -> 0.2, prevent zero-action policies
            "learning_rate": 0.00025,
            "batch_size": 64,
        },
        'A2C': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "ent_coef": 0.2,    # A2C has vf_coef/gae_lambda, but add ent_coef if applicable (Stable-Baselines3 A2C supports ent_coef=0.0 default, set to 0.2)
        },
        'DDPG': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.001,
            "batch_size": 128,
            "buffer_size": 1000000,
        },
        'TD3': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.001,
            "batch_size": 100,
            "buffer_size": 1000000,
        },
        'SAC': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.0003,
            "batch_size": 128,
            "buffer_size": 1000000,
        },
        'CPPO': {  # Extended from PPO with CVaR params, reference from FinRL_DeepSeek (4.1.2: alpha=0.05, lambda=1.0, beta=0.01)
            "n_steps": 2048,
            "ent_coef": 0.2,
            "learning_rate": 0.00025,
            "batch_size": 64,
            "alpha": 0.05,  # CVaR confidence level (worst 5%)
            "lambda_": 0.5,  # Reduced from 1.0 to 0.5 for lighter CVaR penalty, prevent excessive negative rewards, reference from FinRL_DeepSeek (4.1.2: tune lambda for balance)
            "beta": 0.005,   # Reduced from 0.01 for same reason
        },
    }

    # Valid parameters per model to ensure compatibility with stable_baselines3
    _valid_params = {
        'PPO': {'n_steps', 'ent_coef', 'learning_rate', 'batch_size', 'gamma', 'gae_lambda', 'vf_coef'},
        'A2C': {'learning_rate', 'n_steps', 'gamma', 'gae_lambda', 'vf_coef', 'ent_coef'},
        'DDPG': {'learning_rate', 'batch_size', 'buffer_size', 'gamma', 'tau', 'action_noise'},
        'TD3': {'learning_rate', 'batch_size', 'buffer_size', 'gamma', 'tau', 'action_noise', 'policy_delay'},
        'SAC': {'learning_rate', 'buffer_size', 'batch_size', 'ent_coef', 'action_noise', 'gamma'},
        'CPPO': {'n_steps', 'ent_coef', 'learning_rate', 'batch_size', 'gamma', 'gae_lambda', 'vf_coef', 'alpha', 'lambda_', 'beta'},  # Extended for CPPO
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
        Updates: Added infusion_strength (default 0.001 for 0.1% perturbation, reference from FinRL_DeepSeek 5.3.2: 0.1% improves CPPO); risk_mode (bool, for enabling risk prompt in news_features).
        """
        # Default trading params (non-model specific)
        self.initial_cash = 10000
        self.transaction_cost = 0.001
        self.reward_scale = 100
        self.commission_rate = 0.0005
        self.slippage = 0.001
        self.total_timesteps = 2e6  # Increased to 2M for stability, reference from FinRL_DeepSeek (5.2: 2M steps for convergence)
        self.infusion_strength = 0.001  # Default 0.1% for subtle injection, tunable 0.001-0.1, reference from FinRL_DeepSeek (5.3: 0.1% vs 10%)
        self.risk_mode = True  # Enable risk assessment prompt, reference from FinRL_DeepSeek (3: Risk Prompt)
        
        # Set model and load params
        self.model = model
        if self.model not in self.SUPPORTED_MODELS:
            logging.warning(f"CT Module - Unsupported model: {self.model}; falling back to 'PPO'")
            self.model = 'PPO'
        self.model_params = self._model_defaults.get(self.model, {})
        logging.info(f"CT Module - Selected model: {self.model} with params {self.model_params}")

        # Inherit from upstream_config if provided (linkage)
        if upstream_config:
            shared_params = ['symbols', 'window_size', 'prediction_days', 'split_ratio', 'k_folds', 'batch_size', 'exper_mode']
            for param in shared_params:
                if hasattr(upstream_config, param):
                    setattr(self, param, getattr(upstream_config, param))
                    logging.info(f"CT Module - Inherited from upstream: {param} = {getattr(self, param)}")
                else:
                    logging.warning(f"CT Module - Upstream missing {param}; using default if available")

        # Override with custom_config (can override model_params as dict)
        if custom_config:
            for key, value in custom_config.items():
                if key == 'model_params':
                    self.model_params.update(value)  # Merge override for params
                    logging.info(f"CT Module - Overrode model_params: {value}")
                elif hasattr(self, key):
                    setattr(self, key, value)
                    logging.info(f"CT Module - Overrode config: {key} = {value}")
                else:
                    logging.warning(f"CT Module - Ignored unknown config key: {key}")

        # Basic validation
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.transaction_cost < 0 or self.transaction_cost > 1:
            raise ValueError("transaction_cost must be between 0 and 1")
        if hasattr(self, 'window_size') and self.window_size <= 0:
            raise ValueError("window_size must be positive (inherited or set)")
        if self.infusion_strength < 0 or self.infusion_strength > 0.1:
            logging.warning(f"CT Module - infusion_strength {self.infusion_strength} out of [0, 0.1]; clipping")
            self.infusion_strength = np.clip(self.infusion_strength, 0, 0.1)

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
            'total_timesteps': 2000000,
            'infusion_strength': 0.001,
            'risk_mode': True,
            'model_params': ConfigTrading._model_defaults.get(model, {})
        }
        return defaults

# Example instantiation
if __name__ == "__main__":
    from finbert_trader.config_setup import ConfigSetup  # upstream
    upstream = ConfigSetup()
    custom = {'model_params': {'learning_rate': 0.0001}}  # Override PPO lr
    config_trade = ConfigTrading(custom, upstream, model='CPPO')
    print(config_trade.model)  # 'CPPO'
    print(config_trade.get_model_params())  # Updated dict