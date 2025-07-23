# environment.py (Updated with Optimized ConfigTrading)
# Module: Environment
# Purpose: Build custom Gymnasium environment for stock trading RL.
# Updates: Uses optimized ConfigTrading; accesses self.config_trading.model_params if needed (e.g., for future extensions); reward scaled.
# Linkage: config_trading from main; rl_data from Preprocessing.
# Robustness: Log model from config_trading (for experiment tracking).

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockTradingEnv(gym.Env):
    def __init__(self, config_trading, rl_data, mode='train'):
        """
        Initialize Env with ConfigTrading instance and rl_data.
        Input: config_trading (ConfigTrading), rl_data (train/val list), mode ('train'/'val'/'test'), scaler_path (str).
        Output: Self as Gym Env.
        Logic: Use config_trading.initial_cash etc.; infer dims; fit/load scaler.
        Robustness: Check rl_data empty; log config params and model.
        """
        super(StockTradingEnv, self).__init__()
        self.config_trading = config_trading
        self.rl_data = rl_data
        self.mode = mode
        self.scaler_dir = config_trading.SCALER_SAVE_DIR
        # initial scaler save path
        self.scaler_path = os.path.join(self.scaler_dir, f"scaler_{self.mode}_{self.config_trading.model}.pkl")
        logging.info(f"Initial Scaler path: {self.scaler_path}")

        logging.info(f"Env config: model={self.config_trading.model}, initial_cash={self.config_trading.initial_cash}, transaction_cost={self.config_trading.transaction_cost}")
        
        if not self.rl_data:
            raise ValueError("Empty rl_data; check Preprocessing output")
        
        # Infer dims (features + position + cash)
        state_sample = self.rl_data[0]['states'][0]
        self.feature_dim = len(state_sample)    # Store original feature dim for scaler
        self.state_dim = self.feature_dim + 2
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        # Scaler(fit only on features, not on position/cash)
        self.scaler = StandardScaler()
        if self.mode == 'train':
            all_states = np.vstack([window['states'].reshape(-1, self.feature_dim) for window in self.rl_data])
            self.scaler.fit(all_states)
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)
            logging.info(f"Fitted and dumped scaler to {self.scaler_path}")
        else:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logging.info(f"Loaded scaler from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}; train first")
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if not self.rl_data:
            logging.warning("No rl_data; returning dummy state")
            return np.zeros(self.state_dim), {}
        
        self.current_window_idx = np.random.randint(0, len(self.rl_data))
        self.current_window = self.rl_data[self.current_window_idx]
        self.current_step = 0
        self.current_position = 0.0
        self.current_cash = self.config_trading.initial_cash
        self.previous_price = self.current_window['states'][0][-1]  # Assume Close last
        
        state = self._get_state()
        logging.info(f"Reset Env to window {self.current_window_idx}")
        return state, {}

    def step(self, action):
        action = np.clip(action[0], -1, 1)
        current_price = self.current_window['states'][self.current_step][-1]
        
        # Trade
        if action > 0:  # Buy
            shares_to_buy = (self.current_cash * action) / current_price
            cost = shares_to_buy * current_price * (1 + self.config_trading.transaction_cost)
            if cost <= self.current_cash:
                self.current_position += shares_to_buy
                self.current_cash -= cost
            else:
                logging.warning("Insufficient cash; no buy")
        elif action < 0:  # Sell
            shares_to_sell = self.current_position * abs(action)
            revenue = shares_to_sell * current_price * (1 - self.config_trading.transaction_cost)
            self.current_position -= shares_to_sell
            self.current_cash += revenue
        
        # Advance
        self.current_step += 1
        terminated = self.current_step >= self.config_trading.window_size  # Inherited
        truncated = False
        
        # Reward (scaled)
        current_portfolio = self.current_cash + self.current_position * current_price
        previous_portfolio = self.current_cash + self.current_position * self.previous_price
        reward = (current_portfolio - previous_portfolio) * self.config_trading.reward_scale
        
        self.previous_price = current_price
        next_state = self._get_state()
        
        info = {'portfolio_value': current_portfolio}
        logging.info(f"Step {self.current_step}: Action {action:.2f}, Reward {reward:.2f}")
        return next_state, reward, terminated, truncated, info

    def _get_state(self):
        features = self.current_window['states'][self.current_step]
        # Scale only the features(match fit dim), then append dynamic position/cash without scaling
        scaled_features = self.scaler.transform(features.reshape(1, -1))[0]
        state = np.append(scaled_features, [self.current_position, self.current_cash])
        # Log for debug
        logging.debug(f"State shape: {state.shape}, features dim: {len(scaled_features)}")
        return state

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.current_position}, Cash: {self.current_cash}")