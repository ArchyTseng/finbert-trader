# trading_agent.py
# Module: TradingAgent
# Purpose: Centralized RL agent management for multi-stock trading with cross-validation and distributed training support.
# Design:
# - Model-agnostic interface supporting PPO, CPPO, A2C
# - Integrated cross-validation and distributed training
# - Automatic model saving/loading with versioning
# Linkage: Uses config_trading; interfaces with StockTradingEnv
# Robustness: Graceful error handling, logging, config validation
# Extensibility: Easy to add new algorithms by extending _create_model method
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
from datetime import datetime
import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
# RL Libraries
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:
    logging.error(f"TA Module - Required RL library not found: {e}")
    raise
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class TradingAgent:
    """
    Centralized RL agent management for multi-stock trading with cross-validation and distributed training.

    This class provides a unified interface for training, evaluating, and using RL agents
    in multi-stock trading environments. It supports multiple algorithms, cross-validation,
    distributed training, and automatic model management.

    Attributes
    ----------
    config : ConfigTrading
        Configuration object containing trading and model parameters
    model : stable_baselines3 base class
        The RL model instance (PPO, CPPO, A2C, etc.)
    model_name : str
        Name of the selected RL algorithm
    model_path : str
        Path for model saving/loading
    """

    def __init__(self, config_trading):
        """
        Initialize the TradingAgent with configuration.

        Parameters
        ----------
        config_trading : ConfigTrading
            Configuration object containing trading parameters and model settings

        Returns
        -------
        None
            Initializes the instance in place.
        """
        # Inherit configuration for pipeline consistency
        self.config = config_trading
        # Set model name from config
        self.model_name = self.config.model
        # Initialize model as None (created later in training)
        self.model = None
        # Set model save path, default to 'model_cache'
        self.model_path = getattr(self.config, 'MODEL_SAVE_DIR', 'model_cache')

        self.tensorboard_path = getattr(self.config, 'TENSORBOARD_LOG_DIR', 'tensorboard_cache')

        # Create model directory if not exists for robust file handling
        os.makedirs(self.model_path, exist_ok=True)

        # Validate model selection against supported list
        if self.model_name not in self.config.SUPPORTED_MODELS:
            raise ValueError(f"TA Module - Unsupported model: {self.model_name}")

        # Log initialization for traceability
        logging.info(f"TA Module - Initialized TradingAgent with {self.model_name}")
        logging.info(f"TA Module - Model parameters: {self.config.get_model_params()}")  # Assumes config has get_model_params()

    def _create_model(self, env):
        """
        Create and initialize the RL model based on configuration.

        Parameters
        ----------
        env : gym.Env or VecEnv
            The trading environment

        Returns
        -------
        stable_baselines3 base class
            Initialized RL model

        Notes
        -----
        - Uses MlpPolicy by default for multi-layer perceptron networks.
        - Tensorboard_cache logging enabled for training visualization.
        - Extensible: Add new model types by branching on model_name.
        """
        try:
            # Fetch model parameters from config
            model_params = self.config.get_model_params()
            model_name = self.model_name

            # Log creation details for debugging
            logging.info(f"TA Module - Creating {model_name} model with params: {model_params}")

            if model_name == 'PPO':
                # Create PPO model with standard params and TB log
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log= self.tensorboard_path,
                    **model_params
                )
            elif model_name == 'CPPO':
                # Delegate to custom CPPO creation for CVaR support
                model = self._create_cppo_model(env, model_params)
            elif model_name == 'A2C':
                # Create A2C model similarly
                model = A2C(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log= self.tensorboard_path,
                    **model_params
                )
            else:
                raise ValueError(f"TA Module - Model creation not implemented for: {model_name}")

            # Log successful creation
            logging.info(f"TA Module - Successfully created {model_name} model")
            return model

        except Exception as e:
            # Log and re-raise for upstream handling
            logging.error(f"TA Module - Error creating model: {e}")
            raise

    def _create_cppo_model(self, env, model_params):
        """
        Create Custom PPO (CPPO) model with CVaR support based on FinRL_DeepSeek.

        Parameters
        ----------
        env : gym.Env or VecEnv
            The trading environment
        model_params : dict
            Model parameters including CVaR-specific ones

        Returns
        -------
        PPO
            Custom PPO model with CVaR extensions

        Notes
        -----
        - Extracts CVaR params (alpha, lambda_, beta) and stores in model.cvar_params.
        - Base is PPO; custom training logic (e.g., CVaR in loss) would require subclassing or callbacks.
        - Reference: FinRL_DeepSeek for risk-aware extensions.
        """
        try:
            # Copy params to avoid mutation
            cppo_params = model_params.copy()

            # Extract and remove CVaR-specific params from base PPO kwargs
            cvar_params = {}
            for param in ['alpha', 'lambda_', 'beta']:
                if param in cppo_params:
                    cvar_params[param] = cppo_params.pop(param)

            # Create base PPO model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log= self.tensorboard_path,
                **cppo_params
            )

            # Attach CVaR params for use in custom training/callbacks
            model.cvar_params = cvar_params
            # Log CPPO specifics
            logging.info(f"TA Module - Created CPPO model with CVaR params: {cvar_params}")

            return model

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error creating CPPO model: {e}")
            raise

    def train(self, train_env, valid_env=None, total_timesteps=None,
              eval_freq=10000, n_eval_episodes=10, eval_callback=True):
        """
        Train the RL agent with optional validation and callbacks.

        Parameters
        ----------
        train_env : gym.Env or VecEnv
            Training environment
        valid_env : gym.Env or VecEnv, optional
            Validation environment for evaluation callbacks
        total_timesteps : int, optional
            Total training timesteps (defaults to config.total_timesteps)
        eval_freq : int, optional
            Evaluation frequency for callbacks
        n_eval_episodes : int, optional
            Number of episodes for evaluation
        eval_callback : bool, optional
            Whether to use evaluation callback

        Returns
        -------
        stable_baselines3 base class
            Trained model

        Notes
        -----
        - Uses EvalCallback for periodic validation and best model saving.
        - TB log name includes timestamp for unique runs.
        """
        try:
            # Fallback to config timesteps if not provided
            if total_timesteps is None:
                total_timesteps = getattr(self.config, 'total_timesteps', 1000000)

            # Log training start
            logging.info(f"TA Module - Starting training for {total_timesteps} timesteps")

            # Create model if not already initialized
            if self.model is None:
                self.model = self._create_model(train_env)

            # Setup evaluation callback if enabled and valid_env provided
            callback = None
            if eval_callback and valid_env is not None:
                callback = EvalCallback(
                    valid_env,
                    best_model_save_path=self.model_path,
                    log_path=self.model_path,
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False
                )
                logging.info("TA Module - Evaluation callback enabled")

            # Execute training with callback and unique TB log
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Log completion
            logging.info("TA Module - Training completed successfully")
            return self.model

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error during training: {e}")
            raise

    def cross_validate(self, cv_data_list, n_folds=None, total_timesteps_per_fold=None):
        """
        Perform cross-validation training across multiple data folds.

        Parameters
        ----------
        cv_data_list : list
            List of (train_env, valid_env) tuples for each fold
        n_folds : int, optional
            Number of folds (defaults to length of cv_data_list)
        total_timesteps_per_fold : int, optional
            Training timesteps per fold

        Returns
        -------
        dict
            Cross-validation results including metrics for each fold

        Notes
        -----
        - Trains fresh model per fold for independent evaluation.
        - Selects best fold based on mean_reward and sets as self.model.
        - Computes aggregate stats (mean/std/min/max reward).
        """
        try:
            # Determine folds from list length if not specified
            if n_folds is None:
                n_folds = len(cv_data_list)

            # Log CV start
            logging.info(f"TA Module - Starting cross-validation with {n_folds} folds")

            # Initialize results dict
            cv_results = {
                'fold_results': [],
                'average_performance': {},
                'best_fold': None,
                'best_performance': float('-inf')
            }

            # Track best model and fold
            best_model = None
            best_fold_idx = -1

            # Loop over folds
            for fold_idx, (train_env, valid_env) in enumerate(cv_data_list[:n_folds]):
                # Log fold start
                logging.info(f"TA Module - Training fold {fold_idx + 1}/{n_folds}")

                # Create new model for each fold to avoid leakage
                fold_model = self._create_model(train_env)

                # Train with reduced timesteps per fold
                fold_model.learn(
                    total_timesteps=total_timesteps_per_fold or getattr(self.config, 'total_timesteps', 1000000) // n_folds,
                    tb_log_name=f"{self.model_name}_fold_{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                # Evaluate on valid_env
                mean_reward, std_reward = evaluate_policy(
                    fold_model, valid_env, n_eval_episodes=10, deterministic=True
                )

                # Store fold results
                fold_result = {
                    'fold': fold_idx,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'model': fold_model
                }

                cv_results['fold_results'].append(fold_result)
                # Log fold performance
                logging.info(f"TA Module - Fold {fold_idx + 1} - Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

                # Update best if better
                if mean_reward > cv_results['best_performance']:
                    cv_results['best_performance'] = mean_reward
                    cv_results['best_fold'] = fold_idx
                    best_model = fold_model
                    best_fold_idx = fold_idx

            # Compute average stats across folds
            rewards = [result['mean_reward'] for result in cv_results['fold_results']]
            cv_results['average_performance'] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards)
            }

            # Set best model to self.model
            if best_model is not None:
                self.model = best_model
                logging.info(f"TA Module - Best fold: {best_fold_idx + 1} with reward: {cv_results['best_performance']:.4f}")

            # Log CV summary
            logging.info(f"TA Module - Cross-validation completed. Average reward: {cv_results['average_performance']['mean_reward']:.4f}")
            return cv_results

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error during cross-validation: {e}")
            raise

    def evaluate(self, test_env, n_eval_episodes=10, deterministic=True):
        """
        Evaluate the trained agent on test environment.

        Parameters
        ----------
        test_env : gym.Env or VecEnv
            Test environment
        n_eval_episodes : int, optional
            Number of evaluation episodes
        deterministic : bool, optional
            Whether to use deterministic actions

        Returns
        -------
        dict
            Evaluation results including rewards and other metrics

        Notes
        -----
        - Uses SB3's evaluate_policy for mean/std reward over episodes.
        - Extensible: Add custom metrics (e.g., Sharpe ratio) here.
        """
        try:
            # Check for trained model
            if self.model is None:
                raise ValueError("TA Module - No trained model available for evaluation")

            # Log evaluation start
            logging.info(f"TA Module - Evaluating model on {n_eval_episodes} episodes")

            # Run policy evaluation
            mean_reward, std_reward = evaluate_policy(
                self.model, test_env, n_eval_episodes=n_eval_episodes, deterministic=deterministic
            )

            # Compile results dict
            eval_results = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'n_episodes': n_eval_episodes,
                'deterministic': deterministic
            }

            # Log results
            logging.info(f"TA Module - Evaluation completed - Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")
            return eval_results

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error during evaluation: {e}")
            raise

    def predict(self, observation, deterministic=True):
        """
        Generate action prediction for given observation.

        Parameters
        ----------
        observation : np.ndarray
            Current observation/state
        deterministic : bool, optional
            Whether to use deterministic actions

        Returns
        -------
        tuple
            (action, state) - predicted action and hidden state (if applicable)

        Notes
        -----
        - For inference in production; returns action in env's space.
        """
        try:
            # Check for trained model
            if self.model is None:
                raise ValueError("TA Module - No trained model available for prediction")

            # Call model's predict method
            action, state = self.model.predict(observation, deterministic=deterministic)
            return action, state

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error during prediction: {e}")
            raise

    def save_model(self, filename=None, include_config=True):
        """
        Save the trained model to disk.

        Parameters
        ----------
        filename : str, optional
            Custom filename (defaults to auto-generated)
        include_config : bool, optional
            Whether to save configuration along with model

        Returns
        -------
        str
            Path to saved model file

        Notes
        -----
        - Filename auto-generated with timestamp for versioning.
        - Config saved as separate .pkl for easy reloading.
        """
        try:
            # Check for trained model
            if self.model is None:
                raise ValueError("TA Module - No trained model to save")

            # Generate timestamped filename if none provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.model_name}_model_{timestamp}.zip"

            # Construct full save path
            save_path = os.path.join(self.model_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save SB3 model (as .zip)
            self.model.save(save_path)
            logging.info(f"TA Module - Model saved to: {save_path}")

            # Optionally save config as pickle
            if include_config:
                config_filename = save_path.replace('.zip', '_config.pkl')
                with open(config_filename, 'wb') as f:
                    pickle.dump(self.config, f)
                logging.info(f"TA Module - Configuration saved to: {config_filename}")

            return save_path

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error saving model: {e}")
            raise

    def load_model(self, filepath, load_config=False):
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model file
        load_config : bool, optional
            Whether to load configuration from file

        Returns
        -------
        stable_baselines3 base class
            Loaded model

        Notes
        -----
        - Config loaded from adjacent _config.pkl if exists.
        - Model class determined by model_name.
        """
        try:
            # Load config if requested and file exists
            if load_config:
                config_filepath = filepath.replace('.zip', '_config.pkl')
                if os.path.exists(config_filepath):
                    with open(config_filepath, 'rb') as f:
                        self.config = pickle.load(f)
                    logging.info(f"TA Module - Configuration loaded from: {config_filepath}")
                else:
                    logging.warning(f"TA Module - Configuration file not found: {config_filepath}")

            # Load model based on type
            if self.model_name == 'PPO' or self.model_name == 'CPPO':
                self.model = PPO.load(filepath)
            elif self.model_name == 'A2C':
                self.model = A2C.load(filepath)
            else:
                raise ValueError(f"TA Module - Model loading not implemented for: {self.model_name}")

            # Log success
            logging.info(f"TA Module - Model loaded from: {filepath}")
            return self.model

        except Exception as e:
            # Log and re-raise
            logging.error(f"TA Module - Error loading model: {e}")
            raise

    def get_model_info(self):
        """
        Get information about the current model.

        Returns
        -------
        dict
            Model information including name, parameters, and status

        Notes
        -----
        - Useful for debugging or logging model status.
        """
        # Compile info dict
        info = {
            'model_name': self.model_name,
            'model_params': self.config.get_model_params(),
            'is_trained': self.model is not None,
            'model_path': self.model_path
        }

        # Add model type if initialized
        if self.model is not None:
            info['model_type'] = type(self.model).__name__

        return info
# Utility functions for distributed training
def make_env(env_class, rank, seed=0, **kwargs):
    """
    Utility function for creating environments for vectorized training.

    Parameters
    ----------
    env_class : class
        Environment class to instantiate
    rank : int
        Process rank for seed differentiation
    seed : int, optional
        Base random seed
    **kwargs : dict
        Additional arguments for environment constructor

    Returns
    -------
    function
        Environment creation function

    Notes
    -----
    - Sets unique seed per rank for reproducibility in parallel envs.
    - Compatible with SB3's VecEnv.
    """
    # Define inner init function
    def _init():
        # Instantiate env with kwargs
        env = env_class(**kwargs)
        # Set seed offset by rank
        env.seed(seed + rank)
        return env
    # Set global seed
    set_random_seed(seed)
    return _init
def create_vectorized_env(env_class, n_envs=4, **kwargs):
    """
    Create vectorized environments for parallel training.

    Parameters
    ----------
    env_class : class
        Environment class to instantiate
    n_envs : int, optional
        Number of parallel environments
    **kwargs : dict
        Arguments for environment constructor

    Returns
    -------
    VecEnv
        Vectorized environment

    Notes
    -----
    - Uses DummyVecEnv for single (sequential) or SubprocVecEnv for multi-process.
    - Accelerates training by parallel episode rollouts.
    """
    # Use Dummy for n=1 (sequential)
    if n_envs == 1:
        return DummyVecEnv([make_env(env_class, i, **kwargs) for i in range(n_envs)])
    else:
        # Use Subproc for parallel (multi-core)
        return SubprocVecEnv([make_env(env_class, i, **kwargs) for i in range(n_envs)])