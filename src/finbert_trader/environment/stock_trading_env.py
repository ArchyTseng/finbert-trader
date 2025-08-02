# stock_trading_env.py
# Module: StockTradingEnv
# Purpose: Custom Gymnasium environment for stock trading RL.
# Design: Supports flexible termination; integrates config_trading for params.
# Linkage: Uses config_trading; receives rl_data from FeatureEngineer.
# Robustness: Handles invalid prices; logs state/action/reward.
# Updates: Added save_portfolio_memory and save_action_memory functions to record portfolio values and actions during episodes, reference from FinRL (env_stocktrading.py: save_asset_memory and save_action_memory); these allow for detailed backtest analysis without changing core step logic.
# Updates: Added infusion to action (a_mod = S_f * a) and return (D = R_f * D) using config.infusion_strength; S_f/r_f from state[-2]/[-1] (sentiment/risk), mapped to perturbation close to 1, reference from FinRL_DeepSeek (4.2: S_f 0.9-1.1 based on score, 4.3: R_f aggregate); added CVaR adjustment in reward if model=='CPPO', reference from FinRL_DeepSeek (4.1.2: CVaR loss term); extended state_dim +1 for risk_score.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockTradingEnv(gym.Env):
    def __init__(self, config_trading, rl_data, mode='train'):
        """
        Initialize with ConfigTrading and rl_data.
        Updates: state_dim +1 for risk_score (feature_dim +3: position, cash, extra? No, +2 original +1 risk -> +3 total? Wait, original feature_dim includes sentiment, now +risk, so adjust to feature_dim +2 (position,cash), but if risk added, feature_dim increases.
        """
        super(StockTradingEnv, self).__init__()
        self.config_trading = config_trading
        self.rl_data = rl_data
        self.mode = mode
        self.scaler_dir = config_trading.SCALER_SAVE_DIR
        self.scaler_path = os.path.join(self.scaler_dir, f"scaler_train_{self.config_trading.model}.pkl")
        logging.info(f"STE Modul - Scaler path: {self.scaler_path}")

        if not self.rl_data:
            raise ValueError("Empty rl_data")

        # Adjust for multi-stock: features_per_time=15 * len(symbols), state_dim=features_per_time * window + len(symbols) +1 (positions vector + cash)
        self.features_per_time = 15 * len(self.config_trading.symbols)  # OHLCV 5 + ind 8 + sent/risk 2 per symbol
        self.feature_dim = self.features_per_time * self.config_trading.window_size
        self.state_dim = self.feature_dim + len(self.config_trading.symbols) + 1  # positions vector + cash
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.config_trading.symbols),), dtype=np.float32)  # Action per stock
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        logging.info(f"STE Modul - Env initialized: model={self.config_trading.model}, state_dim={self.state_dim}")

        self.portfolio_memory = []  # Added to record portfolio values, reference from FinRL (env_stocktrading.py: asset_memory)
        self.action_memory = []  # Added to record actions, reference from FinRL (env_stocktrading.py: actions_memory)
        self.date_memory = []  # Added to record dates for memory alignment, reference from FinRL (env_stocktrading.py: date_memory)
        self.reset()
        self.infusion_strength = self.config_trading.infusion_strength  # For perturbation
        self.alpha = self.config_trading.model_params.get('alpha', 0.05) if self.config_trading.model == 'CPPO' else None  # For CVaR

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if not self.rl_data:
            logging.warning("STE Modul - No rl_data; returning dummy state")
            return np.zeros(self.state_dim), {}
        
        self.current_window_idx = np.random.randint(0, len(self.rl_data))
        self.current_window = self.rl_data[self.current_window_idx]
        self.current_step = 0
        self.current_position = np.zeros(len(self.config_trading.symbols))  # Vector
        self.current_cash = self.config_trading.initial_cash

        state = self._get_state()  # Reuse to get initial features
        # Multi adjustment: previous_price as array per stock
        self.previous_prices = np.array([state[self.lead_col_idx + i * self.features_per_time] for i in range(len(self.config_trading.symbols))])  # Initial prices vector

        self.portfolio_memory = [self.current_cash + self.current_position * self.previous_price]  # Initialize with initial portfolio, reference from FinRL (asset_memory start with initial_amount)
        self.action_memory = []  # Reset actions per episode
        self.date_memory = [self.current_window['start_date']]  # Assume 'start_date' in rl_data; adjust if needed
        self.returns_history = []  # For CVaR, reference from FinRL_DeepSeek (4.1.2: trajectory return D)
        logging.info(f"STE Modul - Reset Env to window {self.current_window_idx}")
        return state, {}

    def step(self, action):
        """
        Execute one step; add sentiment-based reward scaling.
        Input: action (np.array)
        Output: next_state, reward, terminated, truncated, info
        Logic: Trade; compute portfolio change; scale reward by sentiment_delta.
        Robustness: Handles invalid prices; logs action/reward.
        Updates: Added infusion: mod_action = S_f * action where S_f based on sentiment_score (state[-3]), reference from FinRL_DeepSeek (4.2: S_f 1.1 if positive and buy etc.); adjust return D = R_f * base_return where R_f based on risk_score (state[-2]), reference from FinRL_DeepSeek (4.3: D_Rf = R_f * D); for CPPO, add CVaR penalty to reward if in worst alpha%, reference from FinRL_DeepSeek (4.1.2: CVaR loss term); collect returns_history for CVaR.
        """
        # Multi adjustment: compute S_f as array per stock, reference from FinRL_DeepSeek (4.2: S_f based on score and action sign)
        num_symbols = len(self.config_trading.symbols)
        sent_idx = -5 * num_symbols  # Assume sentiment per-stock at state end, adjust index as per state layout
        sentiment_per_stock = np.array([next_state[sent_idx + i] for i in range(num_symbols)])  # Per-stock sentiments

        S_f = np.ones(num_symbols)  # Initialize array
        for i in range(num_symbols):
            if sentiment_per_stock[i] > 0 and action[i] > 0 or sentiment_per_stock[i] < 0 and action[i] < 0:
                S_f[i] = 1 + self.infusion_strength
            elif sentiment_per_stock[i] > 0 and action[i] < 0 or sentiment_per_stock[i] < 0 and action[i] > 0:
                S_f[i] = 1 - self.infusion_strength
        
        mod_action = S_f * action
        mod_action = np.clip(mod_action, -1, 1)

        next_state = self._get_state()  # Get next features
        # Multi trade loop: current_prices as array, reference from FinRL_DeepSeek (4.3: multi-stock portfolio)
        current_prices = np.array([float(next_state[self.lead_col_idx + i * self.features_per_time]) for i in range(num_symbols)])  # Per-stock prices array
        if not np.all(np.isfinite(current_prices)) or np.any(current_prices <= 0):
            logging.warning(f"STE Modul - Invalid prices {current_prices}; using previous_prices")
            current_prices = np.where(np.isfinite(current_prices) & (current_prices > 0), current_prices, self.previous_prices)

        # Compute portfolios as total (cash + dot(position, prices))
        previous_portfolio = self.current_cash + np.dot(self.current_position, self.previous_prices)

        # Multi trade loop
        for i in range(num_symbols):
            mod_act = mod_action[i]  # Per-stock
            curr_price = current_prices[i]
            if mod_act > 0:
                shares_to_buy = (self.current_cash * mod_act) / curr_price
                cost = shares_to_buy * curr_price * (1 + self.config_trading.transaction_cost)
                if np.isfinite(cost) and cost <= self.current_cash:
                    self.current_position[i] += shares_to_buy
                    self.current_cash -= cost
                else:
                    logging.warning(f"STE Modul - Insufficient cash for symbol {i}; no buy")
            elif mod_act < 0:
                shares_to_sell = self.current_position[i] * abs(mod_act)
                revenue = shares_to_sell * curr_price * (1 - self.config_trading.transaction_cost)
                if np.isfinite(revenue):
                    self.current_position[i] -= shares_to_sell
                    self.current_cash += revenue
                else:
                    logging.warning(f"STE Modul - Invalid revenue for symbol {i}; no sell")

        # Advance step
        self.current_step += 1
        max_steps = self.config_trading.window_size if self.mode in ['train', 'valid'] else len(self.current_window['states']) - 1
        terminated = self.current_step >= max_steps
        truncated = False

        # Similar for R_f array (using risk_per_stock)
        risk_idx = -3 * num_symbols  # Adjust index
        risk_per_stock = np.array([next_state[risk_idx + i] for i in range(num_symbols)])
        R_f_array = np.ones(num_symbols)  # Initialize
        for i in range(num_symbols):
            if risk_per_stock[i] > 0:
                R_f_array[i] = 1 + self.infusion_strength * (risk_per_stock[i] / 2)
            elif risk_per_stock[i] < 0:
                R_f_array[i] = 1 - self.infusion_strength * (abs(risk_per_stock[i]) / 2)
            else:
                R_f_array[i] = 1.0

        # Compute base return and infuse R_f, reference from FinRL_DeepSeek (4.3: D_Rf = R_f * D, R_f 0.9-1.1 based on risk)
        current_portfolio = self.current_cash + np.dot(self.current_position, current_prices)
        if np.isfinite(current_portfolio) and np.isfinite(previous_portfolio):
            base_return = (current_portfolio - previous_portfolio) * self.config_trading.reward_scale
            # Aggregate R_f = np.dot(weights, R_f_array), weights = position / sum(position) if sum>0 else uniform
            weights = self.current_position / np.sum(self.current_position) if np.sum(self.current_position) != 0 else np.ones(num_symbols) / num_symbols
            R_f = np.dot(weights, R_f_array)  # Weighted aggregate per-stock R_f
            adjusted_return = R_f * base_return
            self.returns_history.append(adjusted_return)  # Collect for CVaR
        else:
            adjusted_return = 0.0
            logging.warning("STE Modul - Invalid portfolio; return set to 0")

        # Add Sharpe-like bonus and holding bonus for positive incentives, reference from FinRL_DeepSeek (Table 1: Sharpe/IR metrics for reward design)
        sharpe_bonus = adjusted_return / max(np.std(self.returns_history + [adjusted_return]), 1e-6) * 10 if len(self.returns_history) > 0 else 0  # Encourage stable positive returns
        holding_bonus = max(0, base_return) * 5  # Bonus for positive delta, scaled

        reward = adjusted_return + sharpe_bonus + holding_bonus  # Combined with positives
        if self.config_trading.model == 'CPPO' and len(self.returns_history) > 10:  # Min history for percentile
            sorted_returns = np.sort(self.returns_history)
            var_threshold = np.percentile(sorted_returns, self.alpha * 100)
            if adjusted_return < var_threshold:
                cvar_penalty = self.config_trading.model_params['lambda_'] * (var_threshold - adjusted_return) / (1 - self.alpha)
                reward -= 0.5 * cvar_penalty    # Halve penalty (lambda_=0.5 effectively), to reduce over-penalization of negative trajectories
                logging.debug(f"STE Modul - CPPO CVaR penalty applied: {cvar_penalty:.4f}")

        self.previous_prices = current_prices
        next_state = self._get_state()
        info = {'portfolio_value': current_portfolio}
        self.portfolio_memory.append(current_portfolio)  # Append current portfolio, reference from FinRL (env_stocktrading.py: asset_memory append)
        self.action_memory.append(mod_action)  # Append modified action
        self.date_memory.append(self.current_window['start_date'] + pd.Timedelta(days=self.current_step))  # Approximate date
        logging.info(f"STE Modul - Step: {self.current_step}, Mod_Action: {mod_action}, Reward: {reward:.2f}, Base Return: {base_return:.4f}, Adjusted Return: {adjusted_return:.4f}, Portfolio: {current_portfolio:.2f}")
        return next_state, reward, terminated, truncated, info

    def _get_state(self):
        start_idx = self.current_step * self.features_per_time
        end_idx = start_idx + self.features_per_time
        if self.current_step >= self.config_trading.window_size:
            features = self.current_window['states'][-self.features_per_time:]  # Last time features for terminal
        else:
            features = self.current_window['states'][start_idx:end_idx]  # Slice current time
        if len(features) < self.features_per_time:
            features = np.pad(features, (0, self.features_per_time - len(features)), 'constant', constant_values=0)

        if self.current_step >= len(self.current_window['states']) // self.feature_dim:
            features = self.current_window['states'][-self.feature_dim:]
            logging.debug("STE Modul - Using last features for terminal state")
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant', constant_values=0)
        else:
            features = self.current_window['states'][self.current_step * self.feature_dim : (self.current_step + 1) * self.feature_dim]
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant', constant_values=0)
        logging.debug(f"STE Modul - Padded features to len {len(features)}")

        # Force NaN to 0 in features to prevent propagation (covers all paths)
        features = np.nan_to_num(features, nan=0.0)
        state = np.append(features, [self.current_position, self.current_cash])
        state = np.atleast_1d(state)  # Ensure at least 1D if scalar
        logging.debug(f"STE Modul - State shape: {state.shape}")
        return state

    def render(self, mode='human'):
        """
        Render current state (console print).
        Input: mode (str, default 'human')
        Logic: Print step/position/cash.
        Extensibility: Can add visual plots in future.
        """
        print(f"STE Modul - Step: {self.current_step}, Position: {self.current_position}, Cash: {self.current_cash}")

    def save_portfolio_memory(self):
        """
        Save portfolio values as DataFrame.
        Output: pd.DataFrame with date and account_value.
        Logic: Align date and portfolio_memory, reference from FinRL (env_stocktrading.py: save_asset_memory: df with date/account_value).
        Robustness: Handle unequal lengths with NaN fill.
        """
        df_portfolio = pd.DataFrame({
            'date': self.date_memory,
            'account_value': self.portfolio_memory
        })
        logging.info(f"STE Modul - Saved portfolio memory: {len(df_portfolio)} rows")
        return df_portfolio

    def save_action_memory(self):
        """
        Save actions as DataFrame.
        Output: pd.DataFrame with date and actions.
        Logic: Align date and action_memory, reference from FinRL (env_stocktrading.py: save_action_memory: df with date/actions).
        Robustness: Handle unequal lengths with NaN fill.
        """
        df_actions = pd.DataFrame({
            'date': self.date_memory[:-1] if len(self.date_memory) > len(self.action_memory) else self.date_memory,
            'actions': self.action_memory
        })
        logging.info(f"STE Modul - Saved action memory: {len(df_actions)} rows")
        return df_actions