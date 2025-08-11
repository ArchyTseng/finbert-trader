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
    _model_params = {
        'PPO': {  # From FinRL rl_model.py
            "n_steps": 2048,
            "ent_coef": 0.1,   # Increased to encourage action entropy, from 0.01 -> 0.02 -> 0.05 -> 0.2, prevent zero-action policies
            "learning_rate": 0.00025,
            "batch_size": 64,
        },
        'A2C': {  # Default from Stable-Baselines3 docs
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "ent_coef": 0.1,    # A2C has vf_coef/gae_lambda, but add ent_coef if applicable (Stable-Baselines3 A2C supports ent_coef=0.0 default, set to 0.2)
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
            "ent_coef": 0.1,
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

    # config default directory
    SCALER_CACHE_DIR = 'scaler_cache'
    MODEL_CACHE_DIR = 'model_cache'
    TENSORBOARD_LOG_DIR = 'tensorboard_cache'
    PLOT_CACHE_DIR = 'plot_cache'
    RESULTS_CACHE_DIR = 'results_cache'
    EXPERIMENT_CACHE_DIR = 'exper_cache'
    LOG_SAVE_DIR = 'logs'

    def __init__(self, custom_config=None, upstream_config=None, model='PPO'):
        """
        Initialize the ConfigTrading instance for global configuration in a multi-stock trading backtesting system.

        This constructor sets default trading parameters, selects and configures the RL model,
        inherits shared parameters from an upstream configuration (e.g., ConfigSetup),
        overrides with custom settings, performs basic validation, and computes state/action dimensions.

        Parameters
        ----------
        custom_config : dict, optional
            Custom configuration dictionary to override default or inherited settings.
            Can include keys like 'initial_cash', 'model_params' (as dict for merging), etc.
            Default is None (no overrides).
        upstream_config : object, optional
            Upstream configuration instance (e.g., ConfigSetup) from which to inherit shared parameters
            such as 'symbols', 'indicators', 'window_size', etc. Default is None (no inheritance).
        model : str, optional
            The RL model to use (e.g., 'PPO', 'CPPO'). If unsupported, falls back to 'PPO'.
            Default is 'PPO'.
        Ablation Experiment:
            Introduce senti/risk_factor , senti/risk_features to switch different Ablation Experiment mode.

        Returns
        -------
        None
            Initializes the instance in place and does not return anything.

        Notes
        -----
        - Supported models are defined in self.SUPPORTED_MODELS (class attribute).
        - Inherited parameters include 'symbols', 'indicators', etc., for pipeline consistency.
        - References: FinRL_DeepSeek for parameters like total_timesteps (2M for convergence),
          infusion_strength (0.001-0.1 for subtle risk injection), and risk_mode.
        - State dimension is calculated as (window_size * num_flattened_features) + 3 (e.g., for cash, positions, etc.).
        - Action dimension equals the number of symbols (multi-stock setup).
        """
        # Set default trading parameters for backtesting realism
        self.initial_cash = 1e6  # Initial capital; large value for stability in simulations
        self.buy_cost_pct = 1e-3  # Buy transaction cost as percentage (0.1%)
        self.sell_cost_pct = 1e-3  # Sell transaction cost as percentage (0.1%)
        self.slippage_rate = 0.001  # Slippage factor to simulate market impact
        self.commission_rate = 0.0005  # Commission rate per trade
        self.reward_scaling = 1e-4  # Scale rewards to stabilize RL training gradients
        self.action_clip_range = (-1.0, 1.0)  # Clip actions within this range for bounded decisions
        self.cash_penalty_proportion = 0.01  # Penalty proportion for low cash to encourage balanced portfolios

        # Set risk and training hyperparameters with references for reproducibility
        self.risk_aversion = 0.0  # Default no aversion; can be tuned for conservative strategies
        self.total_timesteps = 2e6  # Increased to 2M for stability, reference from FinRL_DeepSeek (5.2: 2M steps for convergence)
        self.cvar_factor = 0.05 # Weight for CVaR downside risk adjustment, reference from FinRL_DeepSeek (4.1.2: CVaR shaping)
        self.cvar_alpha = 0.05  # CVaR confidence level, reference from FinRL_DeepSeek (4.1.2: alpha=0.05)
        self.cvar_min_history = 30 # CVaR minimum history, reference from FinRL_DeepSeek
        self.risk_mode = True  # Enable risk assessment prompt, reference from FinRL_DeepSeek (3: Risk Prompt)
        self.infusion_strength = 0.001  # Default 0.1% for subtle injection, tunable 0.001-0.1, reference from FinRL_DeepSeek (5.3: 0.1% vs 10%)
        
        # Ablation Experiment Controller
        self.use_senti_factor = True
        self.ust_risk_factor = True

        self.use_senti_features = True
        self.use_risk_features = True

        self.use_senti_threshold = True
        self.use_risk_threshold = True

        self.use_dynamic_infusion = False

        # Set model and load params with fallback for unsupported models
        self.model = model
        if self.model not in self.SUPPORTED_MODELS:  # Check against class-supported models for compatibility
            logging.warning(f"CT Module - Unsupported model: {self.model}; falling back to 'PPO'")
            self.model = 'PPO'  # Fallback to default stable model
        self.model_params = self._model_params.get(self.model, {})  # Load model-specific params from class dict
        logging.info(f"CT Module - Selected model: {self.model} with params {self.model_params}")

        # Inherit from upstream_config if provided (linkage to upstream pipeline)
        if upstream_config:
            shared_params = ['symbols',  # List of stock symbols
                             'indicators',  # Technical indicators
                             'sentiment_keys',  # Sentiment analysis keys
                             'window_size',  # Historical window for state
                             'prediction_days',  # Days ahead for predictions
                             'split_ratio',  # Train/valid/test split
                             'k_folds',  # Cross-validation folds
                             'batch_size',  # Training batch size
                             'exper_mode',  # Experiment mode
                             'features_dim_per_symbol',  # Feature dimensions per symbol
                             'features_price',  # Price-based features
                             'features_ind',  # Indicator features
                             'features_senti',  # Sentiment features
                             'features_risk',  # Risk features
                             'features_all',
                             'senti_threshold',
                             'risk_threshold',
                             'threshold_factor']  # All combined features
            for param in shared_params:
                if hasattr(upstream_config, param):  # Check if param exists in upstream
                    setattr(self, param, getattr(upstream_config, param))  # Inherit to maintain consistency
                    logging.info(f"CT Module - Inherited from upstream: {param} = {getattr(self, param)}")
                else:
                    logging.warning(f"CT Module - Upstream missing {param}; using default if available")  # Warn for missing to debug pipeline

        # Override with custom_config (can override model_params as dict)
        if custom_config:
            for key, value in custom_config.items():
                if key == 'model_params':
                    self.model_params.update(value)  # Merge override for params to allow partial updates
                    logging.info(f"CT Module - Overrode model_params: {value}")
                elif hasattr(self, key):  # Check if key is a valid attribute
                    setattr(self, key, value)  # Override existing attribute
                    logging.info(f"CT Module - Overrode config: {key} = {value}")
                else:
                    logging.warning(f"CT Module - Ignored unknown config key: {key}")  # Ignore unknowns to prevent silent errors

        # Basic validation to ensure config integrity
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")  # Prevent invalid simulations
        if self.buy_cost_pct < 0 or self.buy_cost_pct > 1:
            raise ValueError("buy_cost_pct must be between 0 and 1")  # Ensure percentage range
        if self.sell_cost_pct < 0 or self.sell_cost_pct > 1:
            raise ValueError("sell_cost_pct must be between 0 and 1")  # Ensure percentage range
        if hasattr(self, 'window_size') and self.window_size <= 0:
            raise ValueError("window_size must be positive (inherited or set)")  # Critical for state shaping
        if self.infusion_strength < 0 or self.infusion_strength > 0.1:
            logging.warning(f"CT Module - infusion_strength {self.infusion_strength} out of [0, 0.1]; clipping")
            self.infusion_strength = np.clip(self.infusion_strength, 0, 0.1)  # Clip to safe range for stability

        # Flatten feature categories for unified state representation
        self._flatten_features_categories()  # Internal method to combine features_price, ind, senti, risk into features_all_flatten

        # Compute dimensions for RL environment
        self.state_dim = self.window_size * len(self.features_all_flatten) + 3  # State: historical features + cash/positions/turbulence
        self.action_dim = len(self.symbols)  # Actions: one per symbol in multi-stock setup

    def _flatten_features_categories(self):
        """
        Flatten feature categories across all symbols into unified lists and compute their indices.

        This internal method processes inherited feature dictionaries (features_price, features_ind,
        features_senti, features_risk) by flattening them per symbol, combining into a single
        features_all_flatten list, validating the total length, and calculating indices for each category.

        Parameters
        ----------
        None
            This is an internal method with no parameters; relies on self.attributes from inheritance.

        Returns
        -------
        None
            Modifies instance attributes in place (e.g., features_all_flatten, category indices)
            and does not return anything.

        Notes
        -----
        - Assumes features_* are dictionaries with symbols as keys and lists of features as values.
        - Flattening ensures a consistent state representation for multi-stock RL environments.
        - Indices enable category-specific access or updates in downstream processing.
        - Length assertion checks consistency: total features = num_symbols * features_dim_per_symbol.
        """
        # Log start of flattening process for traceability
        logging.info("CT Module - Flattening features categories")
        try:
            # Initialize empty lists for each category to collect flattened features
            self.features_price_flatten = []  # Flattened price features across all symbols
            self.features_ind_flatten = []    # Flattened indicator features
            self.features_senti_flatten = []  # Flattened sentiment features
            self.features_risk_flatten = []   # Flattened risk features
            for symbol in self.symbols:
                # Extend lists with features for each symbol to create a sequential flatten
                self.features_price_flatten.extend(self.features_price[symbol])    # Price: e.g., open, close
                self.features_ind_flatten.extend(self.features_ind[symbol])        # Indicators: e.g., RSI, MACD
                self.features_senti_flatten.extend(self.features_senti[symbol])    # Sentiment: e.g., scores
                self.features_risk_flatten.extend(self.features_risk[symbol])      # Risk: e.g., volatility metrics

            # Combine all category lists into a single flattened features list for unified state
            self.features_all_flatten = (
                self.features_price_flatten +
                self.features_ind_flatten +
                self.features_senti_flatten +
                self.features_risk_flatten
            )

            # Assert total length matches expected dimension to catch data inconsistencies early
            assert len(self.features_all_flatten) == len(self.symbols) * self.features_dim_per_symbol, f"Error length of flattened all features: {len(self.features_all_flatten)}, Expected: {len(self.symbols) * self.features_dim_per_symbol}"
            # Log flattened size
            logging.info(f"CT Module - Flattened all features for symbol: {self.symbols}, size: {len(self.features_all_flatten)} ")

            # Compute indices for each category in the flattened list for quick access
            self.price_feature_index = [self.features_all_flatten.index(feature) for feature in self.features_price_flatten if feature in self.features_all_flatten]  # Indices for price features
            self.ind_feature_index = [self.features_all_flatten.index(feature) for feature in self.features_ind_flatten if feature in self.features_all_flatten]      # Indices for indicators
            self.senti_feature_index = [self.features_all_flatten.index(feature) for feature in self.features_senti_flatten if feature in self.features_all_flatten]  # Indices for sentiment
            self.risk_feature_index = [self.features_all_flatten.index(feature) for feature in self.features_risk_flatten if feature in self.features_all_flatten]    # Indices for risk
            # Log indices for debugging and verification of category separation
            logging.info(f"CT Module - Set indices for each feature categories: "
                         f"Price indices: {self.price_feature_index}, "
                         f"Indicator indices: {self.ind_feature_index}, "
                         f"Sentiment indices: {self.senti_feature_index}, "
                         f"Risk indices: {self.risk_feature_index}")
        except Exception as e:
            # Log error and raise specific ValueError for better error handling in callers
            logging.error(f"CT Module - Failed to flatten features categories: {e}")
            raise ValueError("Error in flattening features categories")
    
    def get_env_params(self):
        """
        Return the environment's params dict (for Agent init).
        """
        return {"initial_cash": self.initial_cash,
                "buy_cost_pct": self.buy_cost_pct,
                "sell_cost_pct": self.sell_cost_pct,
                "slippage_rate": getattr(self, 'slippage_rate', 0.001),
                "commission_rate": getattr(self, 'commission_rate', 0.0005),
                "reward_scaling": self.reward_scaling,
                "symbols": self.symbols,
                "window_size": self.window_size,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "features_all_flatten": self.features_all_flatten,
                "cvar_factor": self.cvar_factor,
                "risk_mode": getattr(self, 'risk_mode', True),
                "infusion_strength": getattr(self, 'infusion_strength', 0.001),}

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
            'initial_cash': 1e6,
            'buy_cost_pct': 1e-3,
            'sell_cost_pct': 1e-3,
            'slippage_rate': 0.001,
            'commission_rate': 0.0005,
            'reward_scale': 1e-4,
            'action_clip_range': (-1.0, 1.0),
            'cash_penalty_proportion': 0.1,
            'risk_aversion': 0.0,
            'total_timesteps': 2000000,
            'infusion_strength': 0.001,
            "cvar_factor": 0.05,
            'risk_mode': True,
            'model_params': ConfigTrading._model_params.get(model, {})
        }
        return defaults

# Example instantiation
if __name__ == "__main__":
    import os
    os.chdir('/Users/archy/Projects/finbert_trader')
    from finbert_trader.config_setup import ConfigSetup  # upstream
    setup_config = {'features_dim_per_symbol': 11}
    upstream = ConfigSetup(setup_config)
    custom = {'model_params': {'learning_rate': 0.0001}}  # Override PPO lr
    config_trade = ConfigTrading(custom, upstream, model='CPPO')
    print(config_trade.model)  # 'CPPO'
    print(config_trade.get_model_params())  # Updated dict
    print(config_trade.features_dim_per_symbol)