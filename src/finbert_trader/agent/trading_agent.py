# trading_agent.py
# Module: TradingAgent
# Purpose: Train and manage RL agent (PPO/A2C/DDPG/TD3/SAC/CPPO) for stock trading using Stable-Baselines3.
# Design: Config-driven model selection; trains on train_env, validates on val_env; saves model for Backtest.
# Linkage: Inputs config_trade and envs from Environment; outputs model for Backtest.
# Extensibility: Supports multiple models via config_trade.model; customizable via model_params.
# Robustness: Early stopping on val performance; log training progress; handle env errors; added action diversity check and dynamic ent_coef adjustment.
# Reusability: Independent save/load functions; model evaluation reusable.
# Updates: Extended model_classes to include 'CPPO' as custom class (below); _initialize_model handles CPPO init with custom class; increased total_timesteps to 2e6 in train if from config; enhanced SharpeCallback with CVaR metric for CPPO, reference from FinRL_DeepSeek (5.2.1: Table 1 CVaR); added CPPO class definition, reference from FinRL_DeepSeek (train_cppo.py: extend PPO with CVaR objective).

import logging
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC  # Classes for dynamic init
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from finbert_trader.environment.stock_trading_env import StockTradingEnv
import numpy as np
import pandas as pd
import torch # For seed and CVaR
import torch.nn.functional as F
import random  # For seed

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CPPO(PPO):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 0.05)
        self.lambda_ = kwargs.pop('lambda_', 1.0)
        self.beta = kwargs.pop('beta', 0.01)
        self.eta = 0.0  # Initial CVaR threshold
        self.kl_coef = kwargs.pop('kl_coef', None)  # Default None like PPO, or set to 0.2 if needed
        super(CPPO, self).__init__(*args, **kwargs)

    def train(self):
        # Compatible override: Copy PPO.train() logic and insert CVaR term into total_loss
        from stable_baselines3.common.utils import safe_mean

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantages
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_loss = entropy_loss * self.ent_coef
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + entropy_loss + self.vf_coef * value_loss

                # Insert CVaR term into loss (compatible integration)
                returns = rollout_data.returns.flatten()
                if len(returns) > 0:
                    returns_tensor = torch.tensor(returns, requires_grad=True, device=loss.device)
                    sorted_returns = torch.sort(returns_tensor)[0]
                    var_idx = int(self.alpha * len(sorted_returns))
                    var = sorted_returns[var_idx] if var_idx < len(sorted_returns) else sorted_returns[-1]
                    cvar_loss = torch.mean((self.eta - sorted_returns[:var_idx]).clamp(min=0)) / (1 - self.alpha)
                    cvar_term = self.lambda_ * (cvar_loss - self.eta + self.beta)
                    loss += cvar_term  # Add to total_loss
                    # Update eta (dual ascent)
                    self.eta += 0.01 * cvar_loss.item()  # Empirical step size

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(torch.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs = safe_mean(approx_kl_divs)
            if self.kl_coef is not None:
                self.kl_coef = self.kl_coef * (1 + (all_kl_divs - self.target_kl) / self.target_kl)

            if self.target_kl is not None and all_kl_divs > 1.5 * self.target_kl:
                continue_training = False
                break

        # Logs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("TA Module - train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("TA Module - train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("TA Module - train/value_loss", np.mean(value_losses))
        self.logger.record("TA Module - train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("TA Module - train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("TA Module - train/loss", loss.item())
        self.logger.record("TA Module - train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("TA Module - train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("TA Module - train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("TA Module - train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("TA Module - train/clip_range_vf", clip_range_vf)

# End of modified code in trading_agent.py

class SharpeCallback(BaseCallback):
    """
    Custom callback to monitor validation Sharpe ratio and enable early stopping.
    Logic: Every check_freq steps, evaluate on val_env; stop if no improvement.
    Updates: Added CVaR computation in eval if model=='CPPO', reference from FinRL_DeepSeek (5.2.1: CVaR metric).
    """
    def __init__(self, valid_env, check_freq=1000, patience=10000, min_improve=0.01):
        super(SharpeCallback, self).__init__()
        self.valid_env = valid_env
        self.check_freq = check_freq
        self.patience = patience
        self.min_improve = min_improve
        self.best_sharpe = -np.inf
        self.steps_without_improve = 0
        self.alpha = 0.05  # For CVaR

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Evaluate on val_env
            returns = []
            obs, _ = self.valid_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.valid_env.step(action)
                returns.append(reward)
            
            # Compute Sharpe (simplified: mean/std of returns)
            returns = np.array(returns)
            sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
            
            # Add CVaR if CPPO
            cvar = 0
            if self.model.__class__.__name__ == 'CPPO' and len(returns) > 0:
                sorted_returns = np.sort(returns)
                var_idx = int(self.alpha * len(sorted_returns))
                cvar = np.mean(sorted_returns[:var_idx]) if var_idx > 0 else 0  # Mean of worst alpha%
            logging.info(f"TA Module - Validation Sharpe {sharpe:.4f}, CVaR {cvar:.4f} at step {self.n_calls}")
            
            if sharpe > self.best_sharpe + self.min_improve:
                self.best_sharpe = sharpe
                self.steps_without_improve = 0
            else:
                self.steps_without_improve += self.check_freq
                if self.steps_without_improve >= self.patience:
                    logging.info("TA Module - Early stopping triggered")
                    return False
        return True

class TradingAgent:
    def __init__(self, config_trading, train_env=None, valid_env=None, model_path='model_cache/rl_model', seed=42, symbol=''):
        """
        Initialize with ConfigTrading, train/val envs, model save path, seed, and symbol for uniqueness.
        Input: config_trading (ConfigTrading), train_env (StockTradingEnv), val_env (StockTradingEnv), model_path (str), seed (int), symbol (str).
        Output: Self with selected RL model.
        Logic: Select model based on config_trading.model; init with model_params; set seeds.
        Robustness: Validate envs; log model init; seed all random sources; append symbol to base_model_path.
        Updates: Added 'CPPO' to model_classes as custom CPPO class.
        """
        self.config_trading = config_trading
        self.train_env = train_env
        self.valid_env = valid_env
        self.base_model_path = model_path   # Unified path without symbol
        self.model_name = config_trading.model
        self.seed = seed
        self.symbol = symbol  # For uniqueness in model path
        self._set_seeds()  # Set seeds early
        
        # Validate envs if provided
        if train_env is not None and valid_env is not None:
            if not isinstance(train_env, gym.Env) or not isinstance(valid_env, gym.Env):
                raise ValueError("train_env and valid_env must be Gymnasium Env instances")
        
        # Model classes for dynamic initialization
        self.model_classes = {
            'PPO': PPO,
            'A2C': A2C,
            'DDPG': DDPG,
            'TD3': TD3,
            'SAC': SAC,
            'CPPO': CPPO,  # Custom class defined above
        }
        
        # Initialize model if train_env provided
        if train_env is not None:
            try:
                self.model = self._initialize_model(train_env)
                logging.info(f"TA Module - Initialized {self.model_name} with params: {self.config_trading.get_model_params()}")
            except Exception as e:
                logging.error(f"TA Module - Error initializing model : {e}")
                raise
        else:
            self.model = None
            logging.info(f"TA Module - Deferred model initialization for {self.model_name}")

    def _set_seeds(self):
        """
        Set seeds for all random sources to ensure reproducibility.
        Logic: Seed numpy, random, torch, and gym env if available.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        if self.train_env:
            self.train_env.action_space.seed(self.seed)
            self.train_env.observation_space.seed(self.seed)
        logging.info(f"TA Module - Set all seeds to {self.seed} for reproducibility ({self.symbol})")

    def _initialize_model(self, env, model_type=None):
        """
        Initialize RL model with given environment and optional model_type.
        Input: env (gym.Env), model_type (str, optional for override)
        Output: Initialized model instance
        Logic: Select model class; handle noise if string; filter params; for CPPO use custom kwargs.
        """
        model_type = model_type or self.model_name
        model_class = self.model_classes.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model_type: {model_type}; supported: {list(self.model_classes.keys())}")
        
        model_params = self.config_trading.get_model_params()
        # Handle noise if string key
        if 'action_noise' in model_params and isinstance(model_params['action_noise'], str):
            noise_type = model_params['action_noise']
            if noise_type in self.config_trading.NOISE:
                n_actions = env.action_space.shape[-1]
                model_params['action_noise'] = self.config_trading.NOISE[noise_type](
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )
                logging.info(f"Applied noise {noise_type} for {model_type}")
            else:
                logging.warning(f"Invalid noise {noise_type}; removing")
                del model_params['action_noise']
        
        # Filter invalid params
        valid_params = self.config_trading._valid_params.get(model_type, set())
        filtered_params = {k: v for k, v in model_params.items() if k in valid_params}
        if filtered_params != model_params:
            logging.warning(f"Filtered params for {model_type}: {set(model_params) - valid_params}")
        
        # For CPPO, extract CPPO-specific params and pass as kwargs
        if model_type == 'CPPO':
            alpha = filtered_params.pop('alpha', 0.05)
            lambda_ = filtered_params.pop('lambda_', 1.0)
            beta = filtered_params.pop('beta', 0.01)
            return model_class(
                policy="MlpPolicy",
                env=env,
                **filtered_params,
                verbose=0,
                seed=self.seed,
                alpha=alpha,
                lambda_=lambda_,
                beta=beta
            )
        else:
            return model_class(
                policy="MlpPolicy",
                env=env,
                **filtered_params,
                verbose=0,
                seed=self.seed
            )

    def train(self):
        """
        Train the RL model using train_env; monitor with val_env.
        Output: Trained model (self.model).
        Logic: Run learn with total_timesteps; use SharpeCallback; check action diversity post-train, retrain if low.
        Robustness: Save model; log every 1000 steps; auto-adjust ent_coef if low std.
        Updates: Use config.total_timesteps (2e6); extended callback with CVaR for CPPO.
        """
        if self.train_env is None or self.valid_env is None:
            raise ValueError("train_env and valid_env must be initialized")
        if not isinstance(self.train_env, gym.Env) or not isinstance(self.valid_env, gym.Env):
            raise ValueError("Envs must be Gymnasium instances")

        if self.model is None:
            self.model = self._initialize_model(self.train_env)
            logging.info(f"TA Module - Initialized {self.model_name} with params: {self.config_trading.get_model_params()} for {self.symbol}")

        try:
            callback = SharpeCallback(self.valid_env, check_freq=1000, patience=10000)
            self.model.learn(
                total_timesteps=self.config_trading.total_timesteps,
                callback=callback,
                progress_bar=False
            )
            # Post-train action diversity check on valid_env (full episode)
            actions = []
            obs, _ = self.valid_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                actions.append(action[0])
                obs, _, done, _, _ = self.valid_env.step(action)
            action_std = np.std(actions)
            action_mean = np.mean(actions)
            logging.info(f"TA Module - Post-train action stats for {self.symbol}: mean {action_mean:.4f}, std {action_std:.4f}")
            
            if action_std < 0.1 or abs(action_mean) < 0.01:  # Low diversity threshold (empirical)
                logging.warning(f"TA Module - Low action diversity detected for {self.symbol}; increasing ent_coef and retraining")
                self.config_trading.model_params['ent_coef'] = min(self.config_trading.model_params.get('ent_coef', 0.05) + 0.05, 0.2)  # Cap at 0.2
                self.model = self._initialize_model(self.train_env)  # Reinit with new params
                self.model.learn(total_timesteps=self.config_trading.total_timesteps // 2, callback=callback)  # Shorter retrain
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            logging.info(f"TA Module - Trained and saved model to {self.model_path} for {self.symbol}")
        except Exception as e:
            logging.error(f"TA Module - Training error for {self.symbol}: {e}")
            raise

    def train_for_experiment(self, exper_data_dict, force_train=False):
        """
        Train models for each experiment mode (with cache check via force_train).
        Input: exper_data (dict of mode: split_dict with 'train', 'valid', 'test')
        Output: Dict of model paths per mode
        Logic: Create envs per mode; train model; save to mode-specific path with symbol.
        Robustness: Validate data non-empty; log per mode; parallel-safe (no shared resources).
        """
        logging.info("=========== Start to train for experiment ===========")
        models_paths = {}
        for mode, split_dict in exper_data_dict.items():
            train_rl_data = split_dict['train']
            valid_rl_data = split_dict['valid']
            model_type = split_dict.get('model_type', 'PPO')  # Default to PPO
            if not train_rl_data or not valid_rl_data:
                logging.warning(f"TA Module - Skipping mode {mode} due to empty train/valid data for {self.symbol}")
                continue

            mode_model_path = f"{self.base_model_path}_{mode}"  # base already has symbol
            zip_model_path = f"{mode_model_path}.zip"

            if os.path.exists(zip_model_path) and not force_train:
                logging.info(f"TA Module - Skipping training for mode {mode} (cache exists) for {self.symbol}")
                models_paths[mode] = mode_model_path
                continue
            else:
                logging.info(f"TA Module - Training for mode {mode} (force_train: {force_train}) for {self.symbol}")

            train_env = StockTradingEnv(self.config_trading, train_rl_data, env_type='train')
            valid_env = StockTradingEnv(self.config_trading, valid_rl_data, env_type='valid')

            self.train_env = train_env
            self.valid_env = valid_env
            self.model_path = mode_model_path
            self.model = self._initialize_model(train_env, model_type=model_type)
            self.train()
            models_paths[mode] = self.model_path
            logging.info(f"TA Module - Trained {model_type} for mode {mode} and saved to {self.model_path} for {self.symbol}")
        return models_paths

    def load(self, model_path=None, model_type=None):
        """
        Load a trained model from model_path with specified model_type.
        Output: Loaded model (self.model).
        Logic: Select class from model_classes; load weights.
        Robustness: Check file exists; validate model_type.
        Updates: For CPPO, use custom class for load.
        """
        model_path = model_path or self.model_path
        zip_path = f"{model_path}.zip"
        model_type = model_type or self.model_name
        model_class = self.model_classes.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model_type: {model_type}")
        try:
            if os.path.exists(zip_path):
                self.model = model_class.load(model_path, env=self.train_env)
                logging.info(f"TA Module - Loaded {model_type} from {model_path} for {self.symbol}")
            else:
                raise FileNotFoundError(f"Model zip not found at {zip_path} for {self.symbol}")
        except Exception as e:
            logging.error(f"TA Module - Load error for {self.symbol}: {e}")
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
            logging.error(f"TA Module - Prediction error for {self.symbol}: {e}")
            return np.zeros(self.train_env.action_space.shape), None
        
    def eval_on_test(self, test_rl_data, model_path, model_type=None, mode='test'):
        """
        Evaluate model on test data with specified model_type.
        Input: test_rl_data (list), model_path (str), model_type (str, optional), mode (str)
        Output: pd.DataFrame with step/reward/portfolio/sharpe
        Logic: Load model; run episode; compute Sharpe.
        Updates: Added CVaR computation in logs_df if CPPO.
        """
        self.load(model_path=model_path, model_type=model_type)
        test_env = StockTradingEnv(self.config_trading, test_rl_data, env_type=mode)
        logs = []
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        returns = []  # For CVaR
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            returns.append(reward)
            logs.append({'step': test_env.current_step, 'reward': reward, 'portfolio': info['portfolio_value']})
        
        logs_df = pd.DataFrame(logs)
        sharpe = logs_df['reward'].mean() / logs_df['reward'].std() if logs_df['reward'].std() != 0 else 0
        cvar = 0
        if model_type == 'CPPO' and len(returns) > 0:
            sorted_returns = np.sort(returns)
            var_idx = int(0.05 * len(sorted_returns))  # alpha=0.05
            cvar = np.mean(sorted_returns[:var_idx]) if var_idx > 0 else 0
        logs_df['sharpe'] = sharpe
        logs_df['cvar'] = cvar
        logging.info(f"TA Module - Evaluation on test for {mode} ({self.symbol}): Sharpe {sharpe:.4f}, CVaR {cvar:.4f}, Reward {episode_reward:.2f}")
        return logs_df
    
if __name__ == "__main__":
    env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    model = CPPO('MlpPolicy', env, verbose=0, alpha=0.1)
    print("CPPO initialized, alpha:", model.alpha)  # Expected: 0.1, no error