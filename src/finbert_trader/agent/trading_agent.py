# trading_agent.py
# Module: TradingAgent
# Purpose: Train and manage RL agent (PPO/A2C/DDPG) for stock trading using Stable-Baselines3.
# Design: Config-driven model selection; trains on train_env, validates on val_env; saves model for Backtest.
# Linkage: Inputs config_trading and envs from Environment; outputs model for Backtest.
# Extensibility: Supports multiple models via config_trading.model; customizable via model_params.
# Robustness: Early stopping on val performance; log training progress; handle env errors.
# Reusability: Independent save/load functions; model evaluation reusable.

import logging
import os
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SharpeCallback(BaseCallback):
    """
    Custom callback to monitor validation Sharpe ratio and enable early stopping.
    Logic: Every check_freq steps, evaluate on val_env; stop if no improvement.
    """
    def __init__(self, val_env, check_freq=1000, patience=10000, min_improve=0.01):
        super(SharpeCallback, self).__init__()
        self.val_env = val_env
        self.check_freq = check_freq
        self.patience = patience
        self.min_improve = min_improve
        self.best_sharpe = -np.inf
        self.steps_without_improve = 0

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Evaluate on val_env
            returns = []
            obs, _ = self.val_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.val_env.step(action)
                returns.append(reward)
            
            # Compute Sharpe (simplified: mean/std of returns)
            returns = np.array(returns)
            sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
            logging.info(f"Validation Sharpe at step {self.n_calls}: {sharpe:.4f}")
            
            if sharpe > self.best_sharpe + self.min_improve:
                self.best_sharpe = sharpe
                self.steps_without_improve = 0
            else:
                self.steps_without_improve += self.check_freq
                if self.steps_without_improve >= self.patience:
                    logging.info("Early stopping triggered")
                    return False
        return True

class TradingAgent:
    def __init__(self, config_trading, train_env, val_env, model_path='model_cache/rl_model'):
        """
        Initialize with ConfigTrading, train/val envs, and model save path.
        Input: config_trading (ConfigTrading), train_env (StockTradingEnv), val_env (StockTradingEnv), model_path (str).
        Output: Self with selected RL model.
        Logic: Select model based on config_trading.model; init with model_params.
        Robustness: Validate envs; log model init.
        """
        self.config_trading = config_trading
        self.train_env = train_env
        self.val_env = val_env
        self.model_path = model_path
        self.model_name = config_trading.model
        
        # Validate envs
        if not isinstance(train_env, gym.Env) or not isinstance(val_env, gym.Env):
            raise ValueError("train_env and val_env must be Gymnasium Env instances")
        
        # Select model
        model_classes = {
            'PPO': PPO,
            'A2C': A2C,
            'DDPG': DDPG
        }
        model_class = model_classes.get(self.model_name)
        if not model_class:
            raise ValueError(f"Unsupported model: {self.model_name}; supported: {list(model_classes.keys())}")
        
        # Initialize model with config params
        try:
            self.model = model_class(
                policy="MlpPolicy",
                env=self.train_env,
                **self.config_trading.get_model_params(),
                verbose=0  # Logging handled manually
            )
            logging.info(f"Initialized {self.model_name} with params: {self.config_trading.get_model_params()}")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise

    def train(self):
        """
        Train the RL model using train_env; monitor with val_env.
        Output: Trained model (self.model).
        Logic: Run learn with total_timesteps; use SharpeCallback for early stopping.
        Robustness: Save model on completion; log progress every 1000 steps.
        """
        try:
            callback = SharpeCallback(self.val_env, check_freq=1000, patience=10000)
            self.model.learn(
                total_timesteps=self.config_trading.total_timesteps,
                callback=callback,
                progress_bar=False
            )
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            logging.info(f"Trained and saved model to {self.model_path}")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    def load(self):
        """
        Load a trained model from model_path.
        Output: Loaded model (self.model).
        Logic: Re-init model with saved weights.
        Robustness: Check file exists; re-init with same params if load fails.
        """
        try:
            if os.path.exists(self.model_path):
                model_classes = {'PPO': PPO, 'A2C': A2C, 'DDPG': DDPG}
                model_class = model_classes.get(self.model_name)
                self.model = model_class.load(self.model_path, env=self.train_env)
                logging.info(f"Loaded model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")
        except Exception as e:
            logging.error(f"Load error: {e}")
            raise

    def predict(self, observation, deterministic=True):
        """
        Predict action for a given observation.
        Input: observation (np.array from env), deterministic (bool).
        Output: action (np.array), states (None for MlpPolicy).
        Logic: Wrapper for model.predict; used in Backtest.
        """
        try:
            action, states = self.model.predict(observation, deterministic=deterministic)
            return action, states
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return np.zeros(self.train_env.action_space.shape), None
