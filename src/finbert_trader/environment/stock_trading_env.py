# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# stock_trading_env.py
# Module: StockTradingEnv
# Purpose: Custom Gymnasium-compatible multi-stock trading environment with modular reward injection.
# Design: 
# - Multi-episode training architecture
# - Sliding window for state representation
# - Modular reward design (S_f: sentiment factor, R_f: risk factor, CVaR-aware)
# Linkage: Uses config_trading; receives rl_data from FeatureEngineer.
# Robustness: Handles invalid prices; logs state/action/reward.

# %%
import os
os.chdir('/Users/archy/Projects/finbert_trader/')

# %%
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import logging

# %%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
class StockTradingEnv(gym.Env):
    def __init__(self, config_trading, rl_data, env_type='train'):
        """
        Initialize trading environment.
        Parameters:
        - config_trading: Trading hyperparameters (commission, slippage, gamma, etc)
        - rl_data: dict_list of features, targets, and start_dates
        - env_type: 'train', 'valid', or 'test'
        """
        self.config = config_trading
        self.env_type = env_type
        self.data = rl_data.copy()
        # Core dimensions
        self.symbols = self.config.symbols
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.window_size = self.config.window_size
        self.features_all_flatten = self.config.features_all_flatten
        self.features_price_flatten = self.config.features_price_flatten
        self.features_ind_flatten = self.config.features_ind_flatten
        self.features_senti_flatten = self.config.features_senti_flatten
        self.features_risk_flatten = self.config.features_risk_flatten

        self.price_feature_index = self.config.price_feature_index
        self.ind_feature_index = self.config.ind_feature_index
        self.senti_feature_index = self.config.senti_feature_index
        self.risk_feature_index = self.config.risk_feature_index

        # Initialize environment state placeholders
        self.episode_idx = None
        self.trading_df = None
        self.targets = None
        self.terminal_step = None
        self.current_step = None

        self.position = None
        self.cash = None
        self.cost = None
        self.total_asset = None
        self.asset_memory = None
        self.returns_history = []

        # Experiment mode switches for ablation experiment (default True if not set)
        self.use_senti_factor = getattr(self.config, 'use_senti_factor', True)
        self.use_risk_factor = getattr(self.config, 'use_risk_factor', True)

        self.use_senti_features = getattr(self.config, 'use_senti_features', True)
        self.use_risk_features = getattr(self.config, 'use_risk_features', True)
        
        logging.info(f"STE Module - Env Init - Config symbols: {self.symbols}, window_size: {self.window_size}, features_all_flatten len: {len(self.features_all_flatten)}, state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        logging.info(f"STE Module - Env Init - rl_data len: {len(self.data)}, first episode states shape: {self.data[0]['states'].shape if self.data else 'Empty'}")
        logging.debug(f"STE Module - Env Init - Features flatten: price {len(self.features_price_flatten)}, ind {len(self.features_ind_flatten)}, senti {len(self.features_senti_flatten)}, risk {len(self.features_risk_flatten)}")
        logging.debug(f"STE Module - Env Init - Indices: price {self.price_feature_index}, ind {self.ind_feature_index}, senti {self.senti_feature_index}, risk {self.risk_feature_index}")
        # Gym Space Definitions
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.state_dim,),
                                     dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        logging.info(f"STE Modul - Env initialized: environment type: {self.env_type}, model={self.config.model}, state_dim={self.state_dim}")

        # Internal State
        self.last_prices = None  # Initial for slippage
        self.reset()

    def reset(self, seed=None, options=None):
        """Initialize multi-episode by randomly selecting a start point."""
        super().reset(seed=seed)
        try:
            self.episode_idx = np.random.randint(len(self.data))
            episode_data = self.data[self.episode_idx]
            self.trading_df = episode_data['states'] # [T, D]
            self.targets = episode_data['targets'] # [T, N]
            self.terminal_step = len(self.trading_df) - 1
            self.current_step = self.window_size
            # Agent state
            self.cash = 1.0
            self.position = np.zeros(self.action_dim, dtype=np.float32)
            self.cost = 0.0
            self.total_asset = 1.0
            self.asset_memory = [self.total_asset]
            self.returns_history = []
            # Initial last prices as the origin price when window set
            self.last_prices = self._get_current_prices()

            info = {'Environment Type': self.env_type,
                    'Episode Index': self.episode_idx,
                    'Episode Length': self.terminal_step + 1,
                    'Targets': self.targets[:5],
                    'Cash': self.cash,
                    'Position': self.position,
                    'Total Asset': self.total_asset,
                    'Last Prices': self.last_prices}

            logging.info(f"STE Module - Env Reset - Episode idx: {self.episode_idx}, trading_df shape: {self.trading_df.shape}, targets shape: {self.targets.shape}, terminal_step: {self.terminal_step}")
            logging.info(f"STE Module - Env Reset - Reset information: {info}")
            return self._get_states(), info
        except Exception as e:
            logging.error(f"STE Module - Env reset error: {e}")
            raise ValueError("Error in environment reset")

    def step(self, actions):
        """Execute trading step and compute modular reward."""
        try:
            try:
                # Clip actions to [-1, 1]
                actions = np.clip(actions, -1, 1).astype(np.float32)
                logging.info(f"STE Module - Env Step - Input actions shape: {actions.shape}, values: {actions}")
            except Exception as e:
                logging.error(f"STE Module - Error in actions clip: {e}")
                raise ValueError("Error in actions clip step")
            # Calculate S_f and R_f dynamically for multi-stock
            # Extract current sentiment and risk from window end (last row)
            current_row = self.trading_df[self.current_step - 1]  # Current after trade
            sentiment_per_stock = current_row[self.senti_feature_index] if self.use_senti_factor else np.zeros(self.action_dim, dtype=np.float32)  # (num_symbols,)
            risk_per_stock = current_row[self.risk_feature_index] if self.use_risk_factor else np.zeros(self.action_dim, dtype=np.float32)  # (num_symbols,)

            Senti_factor = np.ones(self.action_dim, dtype=np.float32)  # Initialize array
            if self.use_senti_factor:
                logging.info(f"STE Module - Env Step - Using sentiment factor, sentiment_per_stock: {sentiment_per_stock}")
                for i in range(self.action_dim):
                    if (sentiment_per_stock[i] > 0 and actions[i] > 0) or (sentiment_per_stock[i] < 0 and actions[i] < 0):
                        Senti_factor[i] = 1 + self.config.infusion_strength
                    elif (sentiment_per_stock[i] > 0 and actions[i] < 0) or (sentiment_per_stock[i] < 0 and actions[i] > 0):
                        Senti_factor[i] = 1 - self.config.infusion_strength

            mod_actions = np.clip(Senti_factor * actions, -1, 1)  # Optional re-clip after infusion

            # Execute trades
            self._execute_trades(mod_actions)
            logging.info(f"STE Module - Env Step - Executed trades, new cash: {self.cash}, position: {self.position}")
            logging.debug(f"STE Module - Env Step - After trades: cash: {self.cash}, position: {self.position}, total_asset: {self.total_asset}, cost: {self.cost}")

            Risk_factor = np.ones(self.action_dim)  # Initialize
            if self.use_risk_factor:
                logging.info(f"STE Module - Env Step - Using risk factor, risk_per_stock: {risk_per_stock}")
                for i in range(self.action_dim):
                    if risk_per_stock[i] > 0:
                        Risk_factor[i] = 1 + getattr(self.config, 'infusion_strength', 0.001) * (risk_per_stock[i] / 2)
                    elif risk_per_stock[i] < 0:
                        Risk_factor[i] = 1 - getattr(self.config, 'infusion_strength', 0.001) * (abs(risk_per_stock[i]) / 2)
                    else:
                        Risk_factor[i] = 1.0

            # Aggregate R_f as weighted mean (e.g., by position) for return adjustment
            weights = np.abs(self.position) / (np.sum(np.abs(self.position)) + 1e-8)  # Portfolio weights

            # Adjust raw_return with R_f (reference FinRL_DeepSeek 4.3)
            raw_return = self._calculate_return() * np.dot(weights, Risk_factor)
            if self.use_senti_factor and self.use_risk_factor:
                logging.info(f"STE Module - Env Step - Raw return: {raw_return}, R_f: {Risk_factor}, S_f: {Senti_factor}, sentiment_per_stock: {sentiment_per_stock}, risk_per_stock: {risk_per_stock}")
            
            pennalty = getattr(self.config, 'cash_penalty_proportion', 0.01)
            reward = np.float32(raw_return - self.cash * pennalty)
            self.returns_history.append(raw_return)
            # CVaR shaping
            if len(self.returns_history) >= getattr(self.config, 'cvar_min_history', 10) and getattr(self.config, 'cvar_factor', 0.05) > 0:
                cvar_alpha = getattr(self.config, 'cvar_alpha', 0.05) 
                returns_array = np.array(self.returns_history, dtype=np.float32)
                var = np.percentile(returns_array, 100 * cvar_alpha)  # VaR at alpha level
                cvar = returns_array[returns_array <= var].mean()    # CVaR: average returns which are lower than VaR
                # Smaller CVaR, Bigger Risk
                reward += np.float32(self.config.cvar_factor * cvar)

            self.asset_memory.append(self.total_asset)
            # Terminal condition
            self.current_step += 1
            done = (self.current_step >= self.terminal_step)
            truncated = False

            info = {'Total Asset': self.total_asset,
                    'Cash': self.cash,
                    'Position': self.position.copy(),
                    'Reward': reward,
                    'Cost': self.cost,
                    'Current Step': self.current_step,
                    'Sentiment Factor': Senti_factor,
                    'Risk Factor': Risk_factor,
                    'Done': done,
                    'Truncated': truncated}
            
            logging.info(f"STE Module - Env Step - Step information: {info}")
            return self._get_states(), reward, done, truncated, info
        except Exception as e:
            logging.error(f"STE Module - Env step error: {e}")
            raise ValueError("Error in environment step")
    
    def _get_states(self):
        try:
            window = self.trading_df[self.current_step - self.window_size : self.current_step]  # (window_size, D)
            price_features = window[:, self.price_feature_index].flatten()  # (window_size, len(price))
            ind_features = window[:, self.ind_feature_index].flatten()  # (window_size, len(ind))
            senti_features = window[:, self.senti_feature_index].flatten() if self.use_senti_features else np.zeros(self.window_size * len(self.senti_feature_index), dtype=np.float32)  # (window_size, len(senti))
            risk_features = window[:, self.risk_feature_index].flatten() if self.use_risk_features else np.zeros(self.window_size * len(self.risk_feature_index), dtype=np.float32)  # (window_size, len(risk))
            logging.debug(f"STE Module - Env _get_states - \
                          Window shape: {window.shape}, \
                          price_feats shape: {price_features.shape}, \
                          ind_feats: {ind_features.shape}, \
                          senti_feats: {senti_features.shape}, \
                          risk_feats: {risk_features.shape}")
            cash_state = np.array([self.cash], dtype=np.float32)
            position_state = np.array([np.sum(self.position)], dtype=np.float32)
            return_state = np.array([self.total_asset / self.asset_memory[0] - 1.0], dtype=np.float32)
            logging.debug(f"STE Module - Env _get_states - Cash: {cash_state}, position_state: {position_state}, return_state: {return_state}")
            # Set temp state for ablation experiment
            state_temp = [price_features, ind_features]
            # Switch sentiment features, default True
            if self.use_senti_features:
                logging.info(f"STE Module - Env _get_states - Introduce Sentiment features")
                state_temp.append(senti_features)
            else:
                logging.info(f"STE Module - Env _get_states - No Sentiment features mode")
                state_temp.append(np.zeros_like(senti_features))
            # Switch risk features, default True
            if self.use_risk_features:
                logging.info(f"STE Module - Env _get_states - Introduce Risk features")
                state_temp.append(risk_features)
            else:
                logging.info(f"STE Module - Env _get_states - No Risk features mode")
                state_temp.append(np.zeros_like(risk_features))

            state_temp.extend([cash_state, position_state, return_state])

            state = np.concatenate(state_temp).astype(np.float32)
            logging.info(f"STE Module - Env _get_states - Final state shape: {state.shape}, expected: {self.state_dim}")
            return state
        except Exception as e:
            logging.error(f"STE Module - Error in state retrieval: {e}")
            raise ValueError("Error in state retrieval")

    def _execute_trades(self, actions):
        """Update portfolio given actions."""
        try:
            # Get price info
            current_prices = self._get_current_prices()
            logging.debug(f"STE Module - Env _execute_trades - Current prices: {current_prices}, actions: {actions}")
            
            # Compute current allocation
            current_allocation = self.position * current_prices
            
            # Calculate target allocation(by weights) 
            weights = actions / (np.sum(np.abs(actions)) + 1e-8)
            target_allocation = self.total_asset * weights
            # Calculate trade diff
            trade_volume = np.abs(target_allocation - current_allocation)
            # Calculate cost
            commission_cost = np.sum(trade_volume) * getattr(self.config, 'commission_rate', 0.005)
            # Calculate slippage cost
            price_diff = np.abs(current_prices - self.last_prices)
            slippage_cost = np.sum(price_diff * np.abs(self.position)) * getattr(self.config, 'slippage_rate', 0.0)

            total_cost = commission_cost + slippage_cost
            self.cost += total_cost

            logging.debug(f"STE Module - Env _execute_trades - Weights: {weights}, desired_allocation: {target_allocation}, total cost: {total_cost}")

            # Update cash and position
            self.position = target_allocation / (current_prices + 1e-8)
            self.cash = self.total_asset - np.sum(self.position * current_prices) - total_cost
            self.total_asset = self.cash + np.sum(self.position * current_prices)
            # Update last prices
            self.last_prices = current_prices.copy()
        except Exception as e:
            logging.error(f"STE Module - Error in trade execution: {e}")
            raise ValueError("Error in trade execution")

    def _calculate_return(self):
        if len(self.asset_memory) == 0:
            return 0.0
        return_value = (self.total_asset / self.asset_memory[-1]) - 1.0
        logging.debug(f"STE Module - Env _calculate_return - Previous asset: {self.asset_memory[-1]}, current: {self.total_asset}")
        return return_value
    
    def _get_current_prices(self):
        """Extract adjusted close prices from window end."""
        last_row = self.trading_df[self.current_step]
        prices = last_row[self.price_feature_index]
        return prices.astype(np.float32)

    def render(self):
        print(f"Step: {self.current_step}, Asset: {self.total_asset:.4f}, Cash: {self.cash:.4f}")

    def close(self):
        pass


# %%
from finbert_trader.config_trading import ConfigTrading


# %%
# %load_ext autoreload
# %autoreload 2

# %%
class MockConfig:
    symbols = ['GOOGL', 'AAPL']
    window_size = 10
    features_all_flatten = [f'features_{i}_{symbol}' for symbol in symbols for i in range(11)]
    features_price_flatten = [f'Adj_Close_{symbol}' for symbol in symbols]
    features_ind_flatten = [f'ind_{i}_{symbol}' for symbol in symbols for i in range(8)]
    features_senti_flatten = [f'sentiment_score_{symbol}' for symbol in symbols]
    features_risk_flatten = [f'risk_score_{symbol}' for symbol in symbols]
    price_feature_index = [0, 11]
    ind_feature_index = list(range(1,9)) + list(range(12,20))
    senti_feature_index = [9, 20]
    risk_feature_index = [10, 21]
    state_dim = 10 * 22 + 3
    action_dim = 2
    model = 'PPO'
    infusion_strength = 0.001
    cvar_factor = 0.05
    commission_rate = 0.005


# %%
mock_config = MockConfig()

# %%
mock_rl_data = [{'start_date': '2015-01-01', 'states': np.random.rand(50, 22), 'targets': np.random.rand(50, 2)} for _ in range(3)]
mock_rl_data

# %%
env = StockTradingEnv(mock_config, mock_rl_data, env_type='test')

# %%
obs, info = env.reset()

# %%
actions = np.random.rand(env.action_dim) - 0.5

# %%
next_obs, reward, done, truncated, info = env.step(actions)

# %%
env.render()

# %%
env.close()

# %%
