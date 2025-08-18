# -*- coding: utf-8 -*-
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
# import os
# os.chdir('/Users/archy/Projects/finbert_trader/')

# %%
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# %%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
class StockTradingEnv(gym.Env):
    """
    Custom multi-stock trading environment for reinforcement learning with dynamic signal analysis.

    This environment simulates trading across multiple stocks, using a sliding window of historical
    features (prices, indicators, sentiment, risk) as the state. It supports modular reward shaping
    with sentiment/risk factors and CVaR-aware risk management. Key features include intelligent
    action interpretation based on technical signals and dynamic thresholding per symbol/episode.

    Attributes
    ----------
    config : ConfigTrading
        Configuration object containing trading parameters and feature definitions.
    data : list
        List of episode data dictionaries (states, targets) for training/validation/testing.
    env_type : str
        Type of environment ('train', 'valid', 'test') for data selection and logging.
    filter_ind : list
        List of indicator names to use for signal generation.
    I_s_thr : dict
        Static indicator thresholds loaded from config for fallback and percentile reference.
    use_dynamic_threshold : bool
        Flag to enable/disable dynamic threshold calculation (from config).
    symbols : list
        List of stock symbols being traded.
    window_size : int
        Historical window size for state representation.
    state_dim : int
        Dimension of the observation space.
    action_dim : int
        Dimension of the action space (number of stocks).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config_trading, rl_data, env_type='train'):
        """
        Initialize the StockTradingEnv, a Gym-compatible environment for multi-stock RL trading backtesting.

        This constructor sets up trading hyperparameters, loads episode data, initializes state placeholders,
        configures ablation experiment switches, defines Gym spaces, and resets the environment.

        Parameters
        ----------
        config_trading : object
            Configuration object containing trading parameters (e.g., initial_cash, symbols, window_size,
            state_dim, action_dim, features_all_flatten, etc.) inherited from ConfigTrading.
        rl_data : list of dict
            List of dictionaries, each containing 'states' (feature array), 'targets' (price targets),
            and 'start_date' for episodes. Copied internally to avoid modifying the original.
        env_type : str, optional
            Environment type: 'train', 'valid', or 'test'. Default is 'train'.

        Returns
        -------
        None
            Initializes the instance in place and does not return anything.

        Notes
        -----
        - Ablation switches (e.g., use_senti_factor) default to True for full feature usage; set to False for experiments.
        - Observation space is unbounded floats for flexibility in features; action space is continuous [-1, 1] per symbol
          (e.g., -1 for full sell, 1 for full buy).
        - Calls self.reset() at the end to prepare the initial state.
        - Logs key dimensions and data shapes for debugging and verification.
        """
        # Inherit trading configuration for pipeline consistency
        self.config = config_trading    
        self.env_type = env_type    # Set environment type for data selection (train/valid/test)
        self.raw_data = rl_data.copy()  # Copy data to prevent side-effects on original rl_data
        # Ensure test dates are datetime objects
        self.test_start_date = pd.to_datetime(getattr(self.config, 'test_start_date', None))
        self.test_end_date = pd.to_datetime(getattr(self.config, 'test_end_date', None))
        self.if_test = (self.env_type == 'test')  # Switch for training mode (test sequential, train/valid randomly)
        self.filter_ind = getattr(self.config, 'filter_ind', [])    # Get filterd indicators for identifying signals
        self.indicators = getattr(self.config, 'indicators', [])    # Get indicators for identifying signals

        # Core dimensions from config for state and action spaces
        self.symbols = getattr(self.config, 'symbols', [])  # List of stock symbols
        self.state_dim = getattr(self.config, 'state_dim', (50, ))  # self.config.state_dim  # Total state dimension
        self.action_dim = getattr(self.config, 'action_dim', (1,))  # self.config.action_dim  # Action dimension (one per symbol)
        self.window_size = getattr(self.config, 'window_size', 50)  # self.config.window_size  # Historical window size for states
        self.features_all_flatten = getattr(self.config, 'features_all_flatten', [])  # self.config.features_all_flatten  # Flattened all features

        # Core factor for sentiment / risk
        self.senti_threshold = getattr(self.config, 'senti_threshold', {})  # Sentiment thresholds for environment
        self.risk_threshold = getattr(self.config, 'risk_threshold', {})  # Risk thresholds for environment
        self.threshold_factor = getattr(self.config, 'threshold_factor', 0.5)   # Control Senti/Risk strength

        # Feature category indices for selective access/updates
        self.price_feature_index = getattr(self.config, 'price_feature_index', [])  # Get price_feature_index from Config
        self.ind_feature_index = getattr(self.config, 'ind_feature_index', [])  # Get ind_feature_index from Config
        self.senti_feature_index = getattr(self.config, 'senti_feature_index', [])  # Get senti_feature_index from Config
        self.risk_feature_index = getattr(self.config, 'risk_feature_index', [])  # Get risk_feature_index from Config

        self.bypass_interpretation = getattr(self.config, 'bypass_interpretation', True)    # Swith action strategy
        self.use_signal_consistency_bonus = getattr(self.config, 'use_signal_consistency_bonus', True)  # Use signal consistency bonus
        self.use_dynamic_ind_threshold = getattr(self.config, 'use_dynamic_ind_threshold', True)

        # Experiment mode switches for ablation (default True to enable full features)
        self.use_senti_factor = getattr(self.config, 'use_senti_factor', True)  # Use sentiment factor in rewards/actions
        self.use_risk_factor = getattr(self.config, 'use_risk_factor', True)    # Use risk factor in rewards/actions

        self.use_senti_features = getattr(self.config, 'use_senti_features', True)  # Include sentiment features in state
        self.use_risk_features = getattr(self.config, 'use_risk_features', True)    # Include risk features in state

        # self.use_senti_threshold = getattr(self.config, 'use_senti_threshold', True)  # Apply sentiment thresholds
        # self.use_risk_threshold = getattr(self.config, 'use_risk_threshold', True)    # Apply risk thresholds

        # self.use_dynamic_infusion = getattr(self.config, 'use_dynamic_infusion', False)  # Use dynamic vs static thresholds

        # Daynamic trading configuration
        self.signal_threshold = getattr(self.config, 'signal_threshold', 0.3)  # Singal threshold for trading actions
        self.min_trade_amount = getattr(self.config, 'min_trade_amount', 0.01)  # Mininum trading amount
        self.hold_threshold = getattr(self.config, 'hold_threshold', 0.1)  # Hold threshold for holding actions

        logging.info(f"STE Module - Env Init - Config symbols: {self.symbols}, window_size: {self.window_size}, features_all_flatten len: {len(self.features_all_flatten)}, state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        logging.info(f"STE Module - Env Init - rl_data len: {len(self.raw_data)}, first episode states shape: {self.raw_data[0]['states'].shape if self.raw_data else 'Empty'}")

        # Gym Space Definitions for RL compatibility
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.state_dim,),
                                     dtype=np.float32)  # Unbounded observation space for features
        # Action Space Semantics:
        # Each element `a[i]` in the action vector `a` represents the desired proportion of the total asset to be allocated to the i-th stock at the current time step.
        # For example, with 3 stocks:
        #   a = [0.5, 0.3, 0.2] means allocating 50% of assets to stock 0, 30% to stock 1, and 20% to stock 2.
        #   a = [0.5, -0.2, 0.1] if shorting is supported, means allocating 50% of assets to stock 0, shorting 20% of stock 1, and allocating 10% to stock 2.
        #   Note: This requires the `_execute_trades` logic to correctly handle negative positions (shorts).
        #   If shorting is not supported, negative values need to be handled (e.g., clipped or ignored) within `_execute_trades`.
        #   sum(abs(a)) represents the total proportion of assets involved in trading. If sum(abs(a)) > 1, it implies leverage or excessive short selling (beyond available cash).
        #   Cash holding proportion = 1 - sum(a) (if `a` can be positive or negative and represents net allocation) or 1 - sum(abs(a)) (if `a` represents absolute allocation比例).
        #   The agent needs to learn to output an appropriate `a` to optimize its long-term return.
        # Log initialization summary
        self.action_space = Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)  # Continuous actions [-1,1] per symbol
        logging.info(f"STE Modul - Env initialized: environment type: {self.env_type}, model={self.config.model}, state_dim={self.state_dim}")

        # Initialize environment state placeholders for episode management
        self.episode_idx = None  # Current episode index
        self.trading_df = None   # DataFrame for trading records
        self.targets = None      # Price targets for current episode
        self.terminal_step = None  # End step of episode
        self.current_step = None   # Current timestep in episode

        # Trading state variables for asset tracking
        self.position = None     # Current positions (shares held per symbol)
        self.cash = None         # Available cash
        self.cost = None         # Transaction costs accumulated
        self.total_asset = None  # Total portfolio value
        self.asset_memory = None  # History of total assets
        self.returns_history = []  # List of returns for performance metrics

        self.I_s_thr = getattr(self.config, 'ind_signal_threshold', {}) # Get default indicator signal threshold form Config
        self._validate_signal_thresholds()  # Validate the signal thresholds for indicators to ensure they are in defined indicator list.

        # Internal State for slippage calculation
        self.last_prices = None  # Initial for slippage
        self.last_actions = None    # Penalty for trading frequency

        # Pre-compute relative indices for indicator value lookup (after indices are set)
        # This should be called after self.ind_feature_index and related attributes are initialized
        # but before they are needed (e.g., before reset if reset uses them, or just once here)
        self.symbol_to_ind_feature_relative_index = self._generate_ind_relative_index()

        # Call reset to initialize the environment state
        self.reset()

    def _generate_ind_relative_index(self):
        """
        Pre-compute relative indices for faster indicator value lookup in _analyze_technical_signals.
        This mapping is {symbol: {indicator_name: relative_index_in_ind_features}}.
        It is computed based on self.ind_feature_index and self.features_all_flatten.
        """
        logging.info("STE Module - Pre-computing symbol-to-indicator relative indices.")
        try:
            symbol_to_ind_feature_relative_index = {}
            target_indicator_base_names = self.filter_ind if self.filter_ind else self.indicators
            if not target_indicator_base_names:
                logging.info("STE Module - No target indicators configured. Skipping relative index generation.")
                return symbol_to_ind_feature_relative_index

            # Create a mapping from global feature index to its base indicator name
            # This avoids repeatedly searching self.features_all_flatten
            global_idx_to_base_ind_name = {}
            for global_idx in self.ind_feature_index:
                if 0 <= global_idx < len(self.features_all_flatten):
                    full_name = self.features_all_flatten[global_idx]
                    # Core improvement：Fetch indicator which has same format in self.indicators
                    # Assume indicatorName_params_symbol (e.g., rsi_30_AMD, macd_AMD, close_sma_short_30_AMD)
                    # Target: 'indicatorName_params' 部分作为 base_indicator_name
                    # Get symbol suffix
                    found_symbol_suffix = None
                    for s in self.symbols:
                        if full_name.endswith(f"_{s}"):
                            found_symbol_suffix = f"_{s}"
                            break
                    # Remove suffix
                    if found_symbol_suffix:
                        name_without_symbol = full_name[:-len(found_symbol_suffix)]
                        # name_without_symbol 现在是 'indicatorName_params' (e.g., 'rsi_30', 'macd', 'close_sma_short_30')
                        global_idx_to_base_ind_name[global_idx] = name_without_symbol.lower()

            # For each symbol, find its indicators and their relative positions
            for symbol_idx, symbol in enumerate(self.symbols):
                symbol_to_ind_feature_relative_index[symbol] = {}
                
                # Get the list of global indices that belong to this symbol's indicators
                # We iterate through self.ind_feature_index and check the feature name
                symbol_global_indices = []
                for global_idx in self.ind_feature_index:
                    if 0 <= global_idx < len(self.features_all_flatten):
                        full_name = self.features_all_flatten[global_idx]
                        if full_name.endswith(f"_{symbol}"):
                            symbol_global_indices.append(global_idx)
                
                # Now, map each target indicator to its relative index within this symbol's slice
                for base_ind_name in target_indicator_base_names:
                    # Find the global index for this indicator and symbol
                    found_global_idx = None
                    for g_idx in symbol_global_indices:
                        if global_idx_to_base_ind_name.get(g_idx) == base_ind_name:
                            found_global_idx = g_idx
                            break
                    
                    if found_global_idx is not None:
                        # Calculate the relative index within the ind_features array
                        # ind_features = current_row[self.ind_feature_index]
                        # relative_index = index of found_global_idx within self.ind_feature_index
                        try:
                            relative_index = self.ind_feature_index.index(found_global_idx)
                            symbol_to_ind_feature_relative_index[symbol][base_ind_name] = relative_index
                        except ValueError:
                            # Should not happen if logic is consistent, but good to catch
                            logging.error(f"STE Module - Relative Index - Global index {found_global_idx} not found in self.ind_feature_index for symbol {symbol}, indicator {base_ind_name}")
                    # else: Indicator not found for this symbol, entry remains missing in dict

            logging.debug(f"STE Module - Pre-computed symbol_to_ind_feature_relative_index: {symbol_to_ind_feature_relative_index}")
            return symbol_to_ind_feature_relative_index

        except Exception as e:
            logging.error(f"STE Module - Error generating symbol-to-indicator relative indices: {e}")
            return {}

    def _validate_signal_thresholds(self):
        """
        Validate the signal thresholds for indicators to ensure they are in defined indicator list.

        This method checks if the signal thresholds are defined for target indicators in the configuration.
        If any signal thereshold is missing, it'll be set a default value by self._set_default_threshold().

        Parameters
        ----------
        None
            This method does not take any parameters; relies on self.config.

        Returns
        -------
        None
            This method does not return anything; it either passes the validation or raises a warning.

        Notes
        -----
        - This method is called during initialization to ensure target indicators have a defined signal threshold.
        - It logs a message for each validated indicator and raises a warning if any indicator is missing a threshold.
        """
        required_inds = self.config.filter_ind
        for ind in required_inds:
            if ind not in self.I_s_thr:
                logging.warning(f"STE Moule - _validate_signal_threshold - Missing threshold config for indicator: {ind}")
                self._set_default_threshold(ind)

    def _set_default_threshold(self, ind):
        """
        Set a default threshold for an indicator if it's missing from the configuration.

        This method sets a default threshold for an indicator that is missing from the configuration.
        It updates the configuration with the default threshold and logs the action.

        Parameters
        ----------
        ind : str
            The indicator for which the default threshold is being set.

        Returns
        -------
        None
            This method does not return anything; it updates the configuration in place.

        Notes
        -----
        - This method is called when a missing indicator is encountered during validation.
        - It sets a default threshold and logs the action for debugging and future reference.
        """
        defaults = {
            "rsi": {"oversold": 30, "overbought": 70},
            "boll_ub": {"ub": 1.02, "dev": 0.03},
            "boll_lb": {"lb": 0.98, "dev": 0.03},
            "close_sma": {"below": 0.97, "above": 1.03, "dev": 0.05},
            "macd": {"positive": 0.25, "negative": -0.25, "max_range": 0.5},
            "cci": {"oversold": -100, "overbought": 100, "neutral_range": 50},
            "dx": {"trend_threshold": 25, "strong_trend": 40}
        }
        if ind in defaults:
            self.I_s_thr[ind] = defaults[ind]
            logging.info(f"STE Module - Set default threshold for {ind}: {defaults[ind]}")

    def _generate_trading_dates(self, start_date: pd.Timestamp, num_trading_days: int) -> Optional[List[pd.Timestamp]]:
        """
        Generates a list of precise trading dates based on a start date and number of trading days.
        Uses pandas_market_calendars for accuracy, with a fallback to pd.bdate_range.

        Parameters
        ----------
        start_date : pd.Timestamp
            The starting date for the trading period.
        num_trading_days : int
            The total number of trading days to generate.

        Returns
        -------
        list of pd.Timestamp, or None
            A list of trading dates, or None if generation fails or inputs are invalid.
        """
        if not isinstance(start_date, pd.Timestamp):
            logging.error(f"STE Module - _generate_trading_dates - Invalid start_date type: {type(start_date)}")
            return None
        if num_trading_days <= 0:
            logging.warning(f"STE Module - _generate_trading_dates - Invalid num_trading_days: {num_trading_days}")
            return []

        try:
            # Introduce pandas_market_calendars
            import pandas_market_calendars as mcal
            # Initialize exchange calendar (assuming US stocks, e.g., NASDAQ or NYSE)
            exchange_calendar = mcal.get_calendar('NASDAQ') 
            # Estimate an end date to get enough schedule entries
            # Prepare at least num_trading_days business days
            estimated_end_date = pd.bdate_range(start=start_date, periods=2 * num_trading_days)[-1]
            # Get the trading schedule
            schedule = exchange_calendar.schedule(start_date=start_date, end_date=estimated_end_date)
            # Take the first num_trading_days valid trading days
            actual_trading_days_index = schedule.index[:num_trading_days]
            # Convert to list of Pandas datetime objects
            trading_dates = actual_trading_days_index.tolist()
            
            logging.debug(f"STE Module - _generate_trading_dates - Generated {len(trading_dates)} EXACT trading dates "
                          f"from {start_date} using calendar.")
            if trading_dates:
                logging.debug(f"STE Module - _generate_trading_dates - Date range: {trading_dates[0]} to {trading_dates[-1]}")
            return trading_dates

        except ImportError:
            logging.warning("STE Module - _generate_trading_dates - 'pandas-market-calendars' not installed. "
                            "Falling back to pd.bdate_range for trading dates.")
            # Fallback to simple business day range
            trading_dates = pd.bdate_range(start=start_date, periods=num_trading_days).tolist()
            return trading_dates
            
        except Exception as e:
            logging.error(f"STE Module - _generate_trading_dates - Error generating trading dates: {e}", exc_info=True)
            return None # Gracefully handle date generation errors

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        In training/validation mode, this method selects a random episode from the data.
        In testing mode, it prepares the environment to iterate through the entire
        predefined test period sequentially, which must be covered by the episode data.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator (for reproducibility). Default is None.
        options : dict, optional
            Additional options for reset (Gym compatibility, often unused). Default is None.

        Returns
        -------
        tuple
            - state : ndarray
                Initial observation state (flattened historical features + agent states).
            - info : dict
                Dictionary with reset information (e.g., episode index, initial cash/position).

        Notes
        -----
        - Training/Validation mode uses random episode selection for diverse training.
        - Testing mode uses the full test data range for sequential evaluation.
          It requires the episode data to cover the entire test_start_date to test_end_date.
        - Current step starts at window_size to include historical context in the state.
        - Agent states are initialized with absolute values (using initial_cash) for clarity.
        - Info dict aids in monitoring and debugging, compatible with Stable Baselines3.
        """
        # Call superclass reset for Gym compatibility and seed handling
        super().reset(seed=seed)
        try:
            if not self.if_test:    # Cover 'train' and 'valid'
                # *** Training/Validation Mode Logic ***
                # Randomly select an episode with sufficient length
                valid_episodes = []
                for idx, episode_data in enumerate(self.raw_data):
                    if len(episode_data['states']) > self.window_size:
                        valid_episodes.append(idx)
                
                if not valid_episodes:
                    raise ValueError("No episodes with sufficient length found")
                
                # Randomly select an episode indexx from valid episodes
                self.episode_idx = np.random.choice(valid_episodes)
                episode_data = self.raw_data[self.episode_idx]
                # Extract trading data (states) and targets (next-day prices) for the episode
                # Ensure data is C-contiguous for efficient access
                self.trading_df = np.ascontiguousarray(episode_data['states'], dtype=np.float32)
                self.targets = np.ascontiguousarray(episode_data['targets'], dtype=np.float32) if 'targets' in episode_data else None
                # Log shapes for verification
                logging.info(f"STE Module - Env Reset - Selected episode {self.episode_idx} with shapes - "
                        f"states: {self.trading_df.shape}, targets: {self.targets.shape}")
                
                # Count available trading decisions
                available_trading_decisions = len(self.trading_df) - self.window_size
                prediction_days = len(self.targets) if self.targets is not None else 0
                logging.info(f"STE Module - Env Reset - Available trading decisions: {available_trading_decisions}")
                logging.info(f"STE Module - Env Reset - Prediction days: {prediction_days}")

                # Varify minimum required decisions
                min_required_decisions = 10  # At least 10 desicions
                if available_trading_decisions < min_required_decisions:
                    raise ValueError(f"Episode too short for trading: {available_trading_decisions} < {min_required_decisions}")

                # Set terminal step and current step
                self.terminal_step = self.window_size + available_trading_decisions - 1
                self.current_step = self.window_size
            
                logging.info(f"STE Module - Env Reset - Trading steps available: {available_trading_decisions}")
                logging.info(f"STE Module - Env Reset - Terminal step: {self.terminal_step}")

                # Generate EXACT trading dates for current episode
                self.trading_dates = None # Initialize
                try:
                    # Fetch start_date from the selected episode data
                    start_date_raw = episode_data.get('start_date', None)
                    if start_date_raw is not None:
                        start_date = pd.to_datetime(start_date_raw)
                        logging.debug(f"STE Module - Env Reset (Train/Valid) - Episode {self.episode_idx} start date: {start_date}")

                        # Calculate number of trading days in the episode
                        num_trading_days = len(self.trading_df)
                        logging.debug(f"STE Module - Env Reset (Train/Valid) - Episode length (trading_df rows): {num_trading_days}")

                        if num_trading_days > 0:
                            self.trading_dates = self._generate_trading_dates(start_date, num_trading_days)
                            if self.trading_dates is None:
                                logging.warning("STE Module - Env Reset (Train/Valid) - Failed to generate trading dates using helper function.")
                        else:
                            logging.warning(f"STE Module - Env Reset (Train/Valid) - trading_df length is zero, cannot generate dates.")
                    else:
                        logging.warning("STE Module - Env Reset (Train/Valid) - 'start_date' not found in episode data.")
                except Exception as e:
                    logging.error(f"STE Module - Env Reset (Train/Valid) - Error generating EXACT trading dates: {e}", exc_info=True)
                    self.trading_dates = None # Gracefully handle date generation errors
            else:
                # *** Testing Mode Logic ***
                logging.info(f"STE Module - Env Reset (Test Mode) - Preparing for full test period: "
                            f"{self.config.test_start_date} to {self.config.test_end_date}")
                
                # Check date configuration
                if self.test_start_date is None or self.test_end_date is None:
                    logging.error("STE Module - Env Reset (Test Mode) - test_start_date or test_end_date not found in config.")
                    raise ValueError("Test start and end dates must be provided in config for test mode.")
                
                # Verify test episode
                test_episode_data = None
                test_episode_idx = -1
                found_episode_start_date = None
                found_episode_end_date = None

                # Traverse episode to get target period
                logging.info(f"STE Module - Env Reset (Test Mode) - Attempting to find episode with start_date exactly matching test_start_date: {self.test_start_date}")
                for idx, episode_data in enumerate(self.raw_data):
                    episode_start_raw = episode_data.get('start_date')
                    if episode_start_raw is not None:
                        episode_start_date = pd.to_datetime(episode_start_raw)
                        # Match start_date
                        if episode_start_date == self.test_start_date:
                            test_episode_data = episode_data
                            test_episode_idx = idx
                            found_episode_start_date = episode_start_date
                            logging.info(f"STE Module - Env Reset (Test Mode) - Found matching test episode at index {idx} with start date {episode_start_date}")
                            # Verify episode coverage
                            episode_length = len(test_episode_data['states'])
                            if episode_length > self.window_size:
                                generated_dates_for_length_check = self._generate_trading_dates(found_episode_start_date, episode_length)
                                if generated_dates_for_length_check:
                                    found_episode_end_date = generated_dates_for_length_check[-1]
                            logging.info(f"STE Module - Env Reset (Test Mode) - Found EXACT MATCH test episode at index {idx} with start date {episode_start_date}. "
                                        f"Estimated end date: {found_episode_end_date if found_episode_end_date else 'N/A'}")
                            break
                         
                if test_episode_data is None:
                    logging.info(f"STE Module - Env Reset (Test Mode) - No exact match found. Searching for episode that covers test period ({self.test_start_date} to {self.test_end_date}).")
                for idx, episode_data in enumerate(self.raw_data):
                    episode_start_raw = episode_data.get('start_date')
                    if episode_start_raw is not None:
                        episode_start_date = pd.to_datetime(episode_start_raw)
                        # Check episode start date
                        if episode_start_date <= self.test_start_date:
                            episode_length = len(episode_data['states'])
                            if episode_length > self.window_size:
                                # Generate episode end date
                                generated_dates_for_check = self._generate_trading_dates(episode_start_date, episode_length)
                                if generated_dates_for_check:
                                    episode_end_date = generated_dates_for_check[-1]
                                    # Safe padding several days 
                                    if episode_end_date >= (self.test_end_date - pd.Timedelta(days=5)): # Default 5 days
                                        test_episode_data = episode_data
                                        test_episode_idx = idx
                                        found_episode_start_date = episode_start_date
                                        found_episode_end_date = episode_end_date
                                        logging.info(f"STE Module - Env Reset (Test Mode) - Found CONTAINING test episode at index {idx}. "
                                                    f"Start: {episode_start_date}, Estimated End: {episode_end_date}. "
                                                    f"This episode covers the configured test period.")
                                        break 
                # Log for debugging
                if test_episode_data is None:
                    error_msg = (f"STE Module - Env Reset (Test Mode) - "
                                f"CRITICAL: No suitable episode found in self.data for the test period "
                                f"{self.test_start_date} to {self.test_end_date}. "
                                f"An episode is required that either: "
                                f"1) Starts exactly on {self.test_start_date}, or "
                                f"2) Starts on or before {self.test_start_date} and has sufficient data to cover the period "
                                f"(allowing for a small buffer). "
                                f"Please check your data preparation pipeline (e.g., feature_engineer.py) "
                                f"to ensure it creates such an episode for the 'test' split.")
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                    
                # Set test data
                self.episode_idx = test_episode_idx
                self.trading_df = np.ascontiguousarray(test_episode_data['states'], dtype=np.float32)
                self.targets = np.ascontiguousarray(test_episode_data.get('targets'), dtype=np.float32) if test_episode_data.get('targets') is not None else None

                # Use _generate_trading_dates to generate EXACT episode end_date
                self.trading_dates = None
                try:
                    num_trading_days = len(self.trading_df)
                    if found_episode_start_date is not None and num_trading_days > 0:
                        self.trading_dates = self._generate_trading_dates(found_episode_start_date, num_trading_days)
                        if self.trading_dates is None:
                            logging.warning("STE Module - Env Reset (Test Mode) - Failed to generate trading dates for test episode.")
                    else:
                        logging.warning("STE Module - Env Reset (Test Mode) - Could not generate trading dates: missing start date or zero length.")
                except Exception as e:
                    logging.error(f"STE Module - Env Reset (Test Mode) - Error generating trading dates for test episode: {e}", exc_info=True)
                    self.trading_dates = None

                # Set terminal_step and current_step
                available_test_decisions = len(self.trading_df) - self.window_size

                if available_test_decisions < 1:
                    error_msg = (f"STE Module - Env Reset (Test Mode) - "
                                 f"Test episode data too short for window size {self.window_size}. "
                                 f"Available decisions: {available_test_decisions}")
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                # Terminal step is the index of the last data point that can be used for a decision
                self.terminal_step = len(self.trading_df) - 1 # 0-based index
                # Current step starts after the initial window to have history
                self.current_step = self.window_size

                logging.info(f"STE Module - Env Reset (Test Mode) - Configured for sequential processing of test episode {self.episode_idx}: "
                             f"window_size={self.window_size}, terminal_step={self.terminal_step}, "
                             f"available_decisions={available_test_decisions}")

            # Common Initialization for both Train/Valid and Test
            # Initial cash, dtype as float
            self.initial_cash = float(getattr(self.config, 'initial_cash', 1e6))
            # Reset agent state to initial values using absolute cash value
            self.cash = self.initial_cash  # Initial cash in absolute terms
            self.position = np.zeros(self.action_dim, dtype=np.float32)  # Zero positions (no holdings)
            self.cost = 0.0  # Reset accumulated transaction costs
            self.total_asset = self.initial_cash  # Initial total asset value (cash only)
            self.asset_memory = [self.total_asset]  # Start asset history tracking
            self.returns_history = []  # Clear returns history for new episode
            # Set initial last prices from current step for slippage calculations
            self.last_prices = self._get_current_prices()  # Fetches prices at current_step
            self.last_actions = None

            available_trading_decisions = max(0, len(self.trading_df) - self.window_size)
            prediction_days = len(self.targets) if self.targets is not None else 0

            # Calculate each indicator statistic information per symbol
            # to generate dynamic threshold from current episode if enabled
            # This step analyzes the *entire selected episode's data* to set thresholds,
            # which are then used in _analyze_technical_signals for the episode.
            if self.use_dynamic_ind_threshold:
                logging.info("STE Module - Env Reset - Dynamic indicator threshold calculation is enabled.")
                self._calculate_dynamic_ind_threshold()
            else:
                logging.info("STE Module - Env Reset - Dynamic indicator threshold calculation is disabled.")
                # Ensure dynamic_thresholds is cleared or None if not used
                if hasattr(self, 'dynamic_thresholds'):
                    del self.dynamic_thresholds

            # Build info dict for reset details and monitoring
            info = {'Environment Type': self.env_type,  # Train/valid/test
                    'Episode Index': self.episode_idx,  # Selected episode
                    'Episode Length': self.terminal_step + 1,  # Full length including window
                    'Trading Steps': available_trading_decisions,
                    'Prediction Days': prediction_days,
                    'Initial Cash': self.initial_cash, # Log initial cash
                    'Cash': self.cash,
                    'Position': self.position,
                    'Total Asset': self.total_asset,
                    'Last Prices': self.last_prices,
                    'Use Dynamic Ind Threshold': self.use_dynamic_ind_threshold # Log switch state
                    }
            # Log reset details for debugging and verification
            if not self.if_test:
                logging.info(f"STE Module - Env Reset (Train/Valid) - Episode idx: {self.episode_idx}, trading_df shape: {self.trading_df.shape}, "
                             f"targets shape: {self.targets.shape if self.targets is not None else 'None'}, terminal_step: {self.terminal_step}")
            else:
                 logging.info(f"STE Module - Env Reset (Test Mode) - Using test episode {self.episode_idx}, trading_df shape: {self.trading_df.shape}, "
                              f"terminal_step: {self.terminal_step}")

            logging.info(f"STE Module - Env Reset - Reset information: {info}")

            # Return initial state and info
            try:
                initial_state = self._get_states()
                logging.info(f"STE Module - Env Reset - Initial state shape: {initial_state.shape}")
                return initial_state, info
            except Exception as e:
                logging.error(f"STE Module - Env Reset - Error getting initial state: {e}")
                # Return a dummy state on error
                dummy_state = np.zeros(self.state_dim, dtype=np.float32)
                return dummy_state, info
        except Exception as e:
            logging.error(f"STE Module - Env Reset - Unexpected error during reset: {e}", exc_info=True)
            # Return a dummy state and empty info on critical error
            dummy_state = np.zeros(self.state_dim, dtype=np.float32)
            dummy_info = {'Error': str(e)}
            return dummy_state, dummy_info
    
    def _identify_trading_signals(self, current_row):
        """
        Identify trading signals for all symbols based on current indicator features.

        This method extracts indicator features for the current timestep, analyzes them
        per symbol using _analyze_technical_signals, and compiles a dictionary of signals.

        Parameters
        ----------
        current_row : ndarray
            Flattened feature array for the current timestep (shape: (D,)).

        Returns
        -------
        dict
            Dictionary mapping symbol names to their signal analysis results.
        """
        try:
            # Extract indicator features for the current row using precomputed indices
            ind_features = current_row[self.ind_feature_index]
            signals = {}  # Initialize dict to store signals per symbol
            # Loop over each symbol to analyze its specific indicators
            for i, symbol in enumerate(self.symbols):
                # Analyze technical signals for the current symbol using its features
                signal_analysis = self._analyze_technical_signals(ind_features, symbol, i)
                signals[symbol] = signal_analysis  # Store analysis under symbol key
            # Log number of signals identified for monitoring
            logging.debug(f"STE Module - _identify_trading_signals - Identified signals for {len(signals)} symbols")
            return signals  # Return compiled signal dictionary
        except Exception as e:
            # Log error and raise specific ValueError for upstream handling
            logging.error(f"STE Module - Error in signal identification: {e}")
            raise ValueError("Error in signal identification")
    
    def _calculate_dynamic_ind_threshold(self):
        """
        Calculates dynamic thresholds for technical indicators per symbol based on historical episode data.
        This method computes indicator percentiles (e.g., 10th, 50th, 90th) for each symbol in the current episode,
        using the static threshold configuration as a reference for which percentiles to calculate.
        The results are stored in self.dynamic_thresholds for use in signal analysis.

        Example:
            If config defines rsi.oversold=30, this method calculates the 30rd percentile of RSI for each symbol.

        Notes:
        - This version correctly uses self.ind_feature_index and self.features_all_flatten
        which are assumed to be ordered symbol-wise (e.g., [S1_Features, S2_Features, ...]).
        - It no longer relies on self.features_ind_flatten.
        """
        logging.info("STE Module - Dynamic Threshold - Starting calculation for current episode.")
        try:
            # Initialize the main dynamic thresholds dictionary
            # Structure: {symbol_name: {indicator_name: {percentile_value: calculated_value}}}
            self.dynamic_thresholds = {}

            # Get the indicator data for the entire current episode (shape: [Time, Num_Ind_Features])
            # This is the correct data slice based on the verified indices.
            if not hasattr(self, 'ind_feature_index') or not self.ind_feature_index:
                logging.warning("STE Module - Dynamic Threshold - ind_feature_index is empty or not set. Skipping calculation.")
                return
            episode_ind_data = self.trading_df[:, self.ind_feature_index]
            logging.debug(f"STE Module - Dynamic Threshold - episode_ind_data shape: {episode_ind_data.shape}")

            # Initialize dict, expected structure: {indicator_full_base_name: [(local_index_in_episode_ind_data, symbol_name), ...]}
            indicator_to_local_indices_and_symbols = {}
            
            # Iterate through each indicator global index found in self.ind_feature_index
            for local_index_in_episode_ind_data, global_feature_index in enumerate(self.ind_feature_index):
                if 0 <= global_feature_index < len(self.features_all_flatten):
                    full_feature_name = self.features_all_flatten[global_feature_index]
                    logging.debug(f"STE Module - Dynamic Threshold - Processing feature: {full_feature_name} (Global Index: {global_feature_index}, Local Index: {local_index_in_episode_ind_data})")
                    
                    # Exclude other features
                    if any(keyword in full_feature_name.lower() for keyword in ['adj_close', 'sentiment', 'risk']):
                        logging.debug(f"STE Module - Dynamic Threshold - Skipping non-indicator feature: {full_feature_name}")
                        continue 

                    # Match symbol
                    found_symbol = None
                    for s in self.symbols:
                        if full_feature_name.endswith(f"_{s}"):
                            found_symbol = s
                            break
                    
                    if found_symbol:
                        # Expected feature name format: indicatorName_params_symbol (e.g., rsi_30_AMD, close_sma_long_60_AMD)
                        # Remove symbol suffix
                        if full_feature_name.endswith(f"_{found_symbol}"):
                            name_without_symbol = full_feature_name[:-(len(found_symbol) + 1)] # +1 for the '_'
                            # Expected ind name format: 'indicatorName_params' (e.g., 'rsi_30', 'close_sma_long_60')
                            base_indicator_name = name_without_symbol.lower()
                            
                            if base_indicator_name not in indicator_to_local_indices_and_symbols:
                                indicator_to_local_indices_and_symbols[base_indicator_name] = []
                            # Save to dict
                            indicator_to_local_indices_and_symbols[base_indicator_name].append((local_index_in_episode_ind_data, found_symbol))
                            logging.debug(f"STE Module - Dynamic Threshold - Mapped {full_feature_name} to indicator '{base_indicator_name}' for symbol '{found_symbol}' at local index {local_index_in_episode_ind_data}")
                    else:
                        logging.warning(f"STE Module - Dynamic Threshold - Could not map feature '{full_feature_name}' to any symbol in {self.symbols}.")
                else:
                    logging.error(f"STE Module - Dynamic Threshold - Global feature index {global_feature_index} is out of bounds for features_all_flatten (length {len(self.features_all_flatten)}).")

            # Iterate through each configured indicator type found in the data
            # Get target_inds from config, including raw indicator name ['rsi_30', 'boll_ub', ...]
            target_inds = [ind.lower() for ind in (self.filter_ind if self.filter_ind else self.indicators)]
            logging.info(f"STE Module - Dynamic Threshold - Target indicators to process: {target_inds}")
            for ind in target_inds: # e.g., ind = 'rsi_30'
                if ind not in indicator_to_local_indices_and_symbols:
                    logging.warning(f"STE Module - Dynamic Threshold - Indicator '{ind}' not found in data features for any symbol. Skipping.")
                    continue
                    
                logging.info(f"STE Module - _calculate_dynamic_ind_threshold - Calculating dynamic threshold for {ind}")
                
                # Get the list of (local_index_in_episode_ind_data, symbol_name) for this specific indicator type
                local_index_symbol_pairs = indicator_to_local_indices_and_symbols[ind] # e.g., [(0, 'AMD'), (1, 'SBUX'), ...]
                if not local_index_symbol_pairs:
                    logging.warning(f"STE Module - Dynamic Threshold - No symbol data found for indicator '{ind}'. Skipping.")
                    continue

                # Extract the column indices for this indicator from the local index list
                local_indices_for_this_ind = [pair[0] for pair in local_index_symbol_pairs]
                symbol_names_for_this_ind = [pair[1] for pair in local_index_symbol_pairs]
                
                # Extract data for this specific indicator type for the whole episode
                # episode_ind_data shape: [Time, Num_Total_Ind_Features_From_Index]
                # ind_data_for_type shape: [Time, Num_Symbols_Having_This_Ind]
                ind_data_for_type = episode_ind_data[:, local_indices_for_this_ind]
                logging.debug(f"STE Module - Dynamic Threshold - Data shape for indicator '{ind}': {ind_data_for_type.shape}")
                
                # Get the static threshold configuration for this indicator
                static_thr = self.I_s_thr.get(ind, {})
                logging.debug(f"STE Module - Dynamic Threshold - Static thresholds for '{ind}': {static_thr}")
                
                # Determine which percentiles to calculate based on static config keys
                percentiles_to_calculate = set()
                # Map common static threshold keys to percentile values
                percentile_mapping = {
                    'oversold': 30, 'overbought': 70,   # rsi, cci
                    'below': 10, 'above': 90, 'dev': 50,    # close_sma
                    'ub': 90, 'lb': 10, # boll_ub, boll_lb
                    'positive': 75, 'negative': 25, 'max_range': 50,    # macd
                    'trend_threshold': 50, 'strong_trend': 75   # dx
                }
                for key, default_val in static_thr.items():
                    percentile_val = percentile_mapping.get(key, default_val)
                    if isinstance(percentile_val, (int, float)) and 0 <= percentile_val <= 100:
                        percentiles_to_calculate.add(percentile_val)
                    else:
                        logging.warning(f"STE Module - Dynamic Threshold - Invalid percentile value '{percentile_val}' for key '{key}' in indicator '{ind}'. Skipping.")
                
                if not percentiles_to_calculate:
                    logging.info(f"STE Module - Dynamic Threshold - No valid percentiles found for '{ind}' based on config. Using defaults 25, 50, 75.")
                    percentiles_to_calculate = {25, 50, 75}
                
                # Calculate all required percentiles for this indicator type once, for all symbols
                ind_percentile_results = {}
                for p in percentiles_to_calculate:
                    # Calculate percentile 'p' for all symbols of this indicator type
                    # axis=0 calculates percentile along the time dimension for each symbol column
                    # Add small epsilon to handle potential NaNs in ind_data_for_type
                    try:
                        ind_percentile_results[p] = np.nanpercentile(ind_data_for_type, p, axis=0)
                    except Exception as e:
                        logging.error(f"STE Module - Dynamic Threshold - Error calculating percentile {p} for {ind}: {e}")
                        # Assign a default value (e.g., 0.0) or skip this percentile
                        ind_percentile_results[p] = np.full(ind_data_for_type.shape[1], 0.0) 
                
                # Iterate through the symbol names corresponding to the columns in ind_data_for_type
                for symbol_idx_in_result, symbol_name in enumerate(symbol_names_for_this_ind):
                    # Initialize dynamic threshold dict for this specific symbol if not exists
                    if symbol_name not in self.dynamic_thresholds:
                        self.dynamic_thresholds[symbol_name] = {}
                    
                    # Create a dictionary for this specific indicator and symbol
                    symbol_dynamic_thr = {}
                    for p in percentiles_to_calculate:
                        symbol_dynamic_thr[p] = ind_percentile_results[p][symbol_idx_in_result] # Use symbol_idx_in_result
                    
                    # Store the calculated thresholds for this symbol and indicator
                    self.dynamic_thresholds[symbol_name][ind] = symbol_dynamic_thr
                    
                    # Log for the first symbol of each indicator
                    if symbol_idx_in_result == 0:
                        log_values = {p: f"{v:.4f}" for p, v in symbol_dynamic_thr.items()}
                        logging.debug(f"STE Module - Dynamic Threshold - {ind} percentiles for {symbol_name}: {log_values}")

            logging.info("STE Module - Dynamic Threshold - Calculation completed for current episode.")
            logging.debug(f"STE Module - Dynamic Threshold - Final dynamic_thresholds keys: {list(self.dynamic_thresholds.keys()) if self.dynamic_thresholds else 'None'}")

        except Exception as e:
            logging.error(f"STE Module - Dynamic Threshold - Error during calculation: {e}", exc_info=True)

    def _analyze_technical_signals(self, ind_features, symbol, symbol_index):
        """
        Analyze technical signals for a given symbol based on indicator features.
        
        This method processes multiple technical indicators to identify buy/sell signals,
        calculates signal strength, and computes confidence based on indicator consensus.
        The method supports configurable thresholds and handles multiple indicator types.
        
        Parameters:
        -----------
        ind_features : numpy.ndarray
            Array of indicator feature values for current timestep
        symbol : str
            Stock symbol being analyzed
        symbol_index : int
            Index of symbol in feature array
        
        Returns:
        --------
        dict
            Dictionary containing signal type, strength, confidence, and indicator details
        """
        # Initialize signal variables
        signal_type = "hold"        # Default signal type: hold, buy, sell
        signal_strength = 0.0       # Signal strength (0.0-1.0)
        confidence = 0.0            # Signal confidence based on indicator consensus
        indicators = {}             # Dictionary to store indicator values
        signal_votes = {"buy": 0, "sell": 0, "hold": 0}  # Vote counting for signal fusion
        total_strength = 0.0        # Accumulated signal strength from all indicators
        signal_count = 0            # Count of indicators providing signals

        # Get filtered indicators from configuration
        target_inds = self.filter_ind if self.filter_ind else self.indicators

        # Process each filtered indicator
        for ind in target_inds:
            ind_value = None    # Initialize
            # Use the pre-computed relative index
            # ind_features is the slice of features for ALL symbols relevant to indicators
            # We need the relative index for THIS symbol's indicator within THIS symbol's slice
            relative_index_dict_for_symbol = self.symbol_to_ind_feature_relative_index.get(symbol, {})
            relative_index = relative_index_dict_for_symbol.get(ind, None)
            
            if relative_index is not None and 0 <= relative_index < len(ind_features):
                ind_value = ind_features[relative_index]
            else:
                # Log warning if indicator not found for symbol or index invalid
                logging.warning(f"STE Module - _analyze_technical_signals - Indicator '{ind}' not found or invalid index for symbol '{symbol}'. Index: {relative_index}, ind_features length: {len(ind_features)}")
                # ind_value remains None

            # Process indicator only if value is valid
            if ind_value is not None:
                indicators[ind] = ind_value
                # Determine thresholds to use: dynamic or static
                use_dynamic_ind = self.use_dynamic_ind_threshold # Use the instance flag
                
                # Check if dynamic thresholds are available for this specific symbol and indicator
                dynamic_thr = None
                if use_dynamic_ind and hasattr(self, 'dynamic_thresholds') and self.dynamic_thresholds is not None:
                    # Safely navigate the nested dictionary structure
                    symbol_dyn_thrs = self.dynamic_thresholds.get(symbol, {})
                    if symbol_dyn_thrs: # Check if symbol entry exists
                         dynamic_thr = symbol_dyn_thrs.get(ind, None)
                         # At this point, dynamic_thr is a dict like {30: val, 70: val, ...} 
                         # or None if not found

                static_thr = self.I_s_thr.get(ind, {}) # Get static thresholds

                # RSI (Relative Strength Index) - Momentum oscillator
                if ind == "rsi":
                    # Determine which percentiles were used for RSI based on config
                    rsi_oversold_percentile = 30 # Default or get from a mapping like before
                    rsi_overbought_percentile = 70 # Default or get from a mapping like before
                    
                    if dynamic_thr is not None and rsi_oversold_percentile in dynamic_thr and rsi_overbought_percentile in dynamic_thr:
                        # Use dynamic thresholds
                        oversold = dynamic_thr[rsi_oversold_percentile]
                        overbought = dynamic_thr[rsi_overbought_percentile]
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: oversold={oversold:.4f}, overbought={overbought:.4f}")
                    else:
                        # Fallback to static thresholds
                        oversold = static_thr.get("oversold", 30)
                        overbought = static_thr.get("overbought", 70)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: oversold={oversold}, overbought={overbought}")

                    # Buy/Sell logic remains the same, but now uses potentially dynamic values
                    if ind_value < oversold:
                        signal_votes["buy"] += 1
                        range_for_strength = max(overbought - oversold, 1e-6) # Avoid division by zero
                        strength = min((oversold - ind_value) / range_for_strength, 1.0)
                        total_strength += strength
                        signal_count += 1
                    elif ind_value > overbought:
                        signal_votes["sell"] += 1
                        range_for_strength = max(overbought - oversold, 1e-6)
                        strength = min((ind_value - overbought) / range_for_strength, 1.0)
                        total_strength += strength
                        signal_count += 1

                elif ind == "macd":
                     macd_pos_percentile = 75 # Example mapping from static config key 'positive'
                     macd_neg_percentile = 25 # Example mapping from static config key 'negative'
                     # For max_range, we might use the 50th percentile as a proxy for 'typical' range
                     # or calculate it as (P_high - P_low) / 2 if P_high and P_low are available.
                     # Here, we simplify by using the 50th percentile value itself or a static fallback.
                     if dynamic_thr is not None and macd_pos_percentile in dynamic_thr and macd_neg_percentile in dynamic_thr:
                         positive_threshold = dynamic_thr[macd_pos_percentile]
                         negative_threshold = dynamic_thr[macd_neg_percentile]
                         # Simplified range calculation for strength
                         max_range = max(abs(positive_threshold), abs(negative_threshold), 1e-6)
                         logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: pos={positive_threshold:.4f}, neg={negative_threshold:.4f}")
                     else:
                         positive_threshold = static_thr.get("positive", 0.1)
                         negative_threshold = static_thr.get("negative", -0.1)
                         max_range = static_thr.get("max_range", 0.5)
                         logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: pos={positive_threshold}, neg={negative_threshold}")

                     if ind_value > positive_threshold:
                         signal_votes["buy"] += 1
                         strength = min((ind_value - positive_threshold) / max_range, 1.0)
                         total_strength += strength
                         signal_count += 1
                     elif ind_value < negative_threshold:
                         signal_votes["sell"] += 1
                         strength = min((negative_threshold - ind_value) / max_range, 1.0)
                         total_strength += strength
                         signal_count += 1

                elif ind == "boll_ub":
                    boll_ub_high_percentile = 90 # Mapping 'ub' -> 90th percentile
                    boll_ub_low_percentile = 10  # Mapping 'dev' (for lower bound of deviation) -> 10th percentile
                    if dynamic_thr is not None and boll_ub_high_percentile in dynamic_thr and boll_ub_low_percentile in dynamic_thr:
                        upper_bound = dynamic_thr[boll_ub_high_percentile]
                        # Deviation is calculated as distance from median (50th percentile) to upper/lower bound
                        median_val = dynamic_thr.get(50, upper_bound) # Fallback to upper if 50 not available
                        deviation = max(upper_bound - median_val, 1e-6)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: ub={upper_bound:.4f}, dev={deviation:.4f}")
                    else:
                        upper_bound = static_thr.get("ub", 1.02)
                        deviation = static_thr.get("dev", 0.03)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: ub={upper_bound}, dev={deviation}")

                    if ind_value > upper_bound:
                        signal_votes["sell"] += 1
                        strength = min((ind_value - upper_bound) / deviation, 1.0)
                        total_strength += strength
                        signal_count += 1

                elif ind == "boll_lb":
                    boll_lb_low_percentile = 10 # Mapping 'lb' -> 10th percentile
                    boll_lb_high_percentile = 90 # Mapping 'dev' (for upper bound of deviation) -> 90th percentile
                    if dynamic_thr is not None and boll_lb_low_percentile in dynamic_thr and boll_lb_high_percentile in dynamic_thr:
                        lower_bound = dynamic_thr[boll_lb_low_percentile]
                        median_val = dynamic_thr.get(50, lower_bound) # Fallback to lower if 50 not available
                        deviation = max(median_val - lower_bound, 1e-6)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: lb={lower_bound:.4f}, dev={deviation:.4f}")
                    else:
                        lower_bound = static_thr.get("lb", 0.98)
                        deviation = static_thr.get("dev", 0.03)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: lb={lower_bound}, dev={deviation}")

                    if ind_value < lower_bound:
                        signal_votes["buy"] += 1
                        strength = min((lower_bound - ind_value) / deviation, 1.0)
                        total_strength += strength
                        signal_count += 1

                elif ind == "close_sma":
                    sma_below_percentile = 10 # Mapping 'below' -> 10th percentile
                    sma_above_percentile = 90 # Mapping 'above' -> 90th percentile
                    sma_dev_percentile = 50   # Mapping 'dev' -> 50th percentile (median)
                    if dynamic_thr is not None and sma_below_percentile in dynamic_thr and sma_above_percentile in dynamic_thr:
                        below_threshold = dynamic_thr[sma_below_percentile]
                        above_threshold = dynamic_thr[sma_above_percentile]
                        # Deviation can be half the range between above and below thresholds
                        deviation = max((above_threshold - below_threshold) / 2.0, 1e-6)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: below={below_threshold:.4f}, above={above_threshold:.4f}")
                    else:
                        below_threshold = static_thr.get("below", 0.97)
                        above_threshold = static_thr.get("above", 1.03)
                        deviation = static_thr.get("dev", 0.05)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: below={below_threshold}, above={above_threshold}")

                    if ind_value < below_threshold:
                        signal_votes["buy"] += 1
                        strength = min((below_threshold - ind_value) / deviation, 1.0)
                        total_strength += strength
                        signal_count += 1
                    elif ind_value > above_threshold:
                        signal_votes["sell"] += 1
                        strength = min((ind_value - above_threshold) / deviation, 1.0)
                        total_strength += strength
                        signal_count += 1

                elif ind == "cci":
                    cci_oversold_percentile = 30 # Mapping 'oversold' -> 30th percentile (example)
                    cci_overbought_percentile = 70 # Mapping 'overbought' -> 70th percentile (example)
                    cci_neutral_percentile = 50   # Mapping 'neutral_range' proxy -> 50th percentile
                    if dynamic_thr is not None and cci_oversold_percentile in dynamic_thr and cci_overbought_percentile in dynamic_thr:
                        oversold = dynamic_thr[cci_oversold_percentile]
                        overbought = dynamic_thr[cci_overbought_percentile]
                        neutral_median = dynamic_thr[cci_neutral_percentile]
                        # Neutral range could be defined as distance from median to oversold/overbought
                        neutral_range = max(min(neutral_median - oversold, overbought - neutral_median), 1e-6)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: oversold={oversold:.4f}, overbought={overbought:.4f}")
                    else:
                        oversold = static_thr.get("oversold", -100)
                        overbought = static_thr.get("overbought", 100)
                        neutral_range = static_thr.get("neutral_range", 50)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: oversold={oversold}, overbought={overbought}")

                    if ind_value < oversold:
                        signal_votes["buy"] += 1
                        strength = min((oversold - ind_value) / (abs(oversold) + neutral_range), 1.0)
                        total_strength += strength
                        signal_count += 1
                    elif ind_value > overbought:
                        signal_votes["sell"] += 1
                        strength = min((ind_value - overbought) / (overbought + neutral_range), 1.0)
                        total_strength += strength
                        signal_count += 1

                elif ind == "dx":
                    dx_trend_percentile = 50 # Mapping 'trend_threshold' -> 50th percentile (example)
                    dx_strong_percentile = 75 # Mapping 'strong_trend' -> 75th percentile (example)
                    if dynamic_thr is not None and dx_trend_percentile in dynamic_thr and dx_strong_percentile in dynamic_thr:
                        trend_threshold = dynamic_thr[dx_trend_percentile]
                        strong_trend = dynamic_thr[dx_strong_percentile]
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using dynamic thresholds: trend={trend_threshold:.4f}, strong={strong_trend:.4f}")
                    else:
                        trend_threshold = static_thr.get("trend_threshold", 25)
                        strong_trend = static_thr.get("strong_trend", 40)
                        logging.debug(f"STE Module - _analyze_technical_signals - {symbol} {ind} using static thresholds: trend={trend_threshold}, strong={strong_trend}")

                    # DX confirms trend strength, doesn't generate primary buy/sell signals directly here
                    # It can be used to weight other signals or adjust confidence/strength
                    # For simplicity, we can increment a "trend_confirm" vote if threshold is met
                    # This vote doesn't directly influence final signal type but affects confidence/weighting
                    # if ind_value > trend_threshold:
                    #     signal_votes["trend_confirm"] += 1 # Hypothetical use
                    pass # DX logic for confirmation can be added here if needed

        # Signal fusion: Combine multiple indicator signals
        if signal_count > 0:
            # Calculate average signal strength across all indicators
            avg_strength = total_strength / signal_count
            
            # Determine final signal based on voting consensus
            if signal_votes["buy"] > signal_votes["sell"] and signal_votes["buy"] > 0:
                # Buy signal wins by majority vote
                signal_type = "buy"
                signal_strength = avg_strength
                # Confidence based on vote consensus (higher consensus = higher confidence)
                confidence = signal_votes["buy"] / signal_count
                
            elif signal_votes["sell"] > signal_votes["buy"] and signal_votes["sell"] > 0:
                # Sell signal wins by majority vote
                signal_type = "sell"
                signal_strength = avg_strength
                confidence = signal_votes["sell"] / signal_count
                
            else:
                # Equal votes or no clear signal, maintain hold position
                signal_type = "hold"
                signal_strength = 0.0
                confidence = 0.0

        # Log signal analysis results for debugging and monitoring
        logging.info(f"STE Module - _analyze_technical_signals - Symbol: {symbol}, "
                    f"Signal: {signal_type}, Strength: {signal_strength:.3f}, "
                    f"Confidence: {confidence:.3f}")
        
        # Return comprehensive signal analysis results
        return {
            "type": signal_type,           # Final signal type (buy/sell/hold)
            "strength": signal_strength,   # Signal strength (0.0-1.0)
            "confidence": confidence,      # Signal confidence based on consensus
            "indicators": indicators,      # Raw indicator values for debugging
            "signal_votes": signal_votes   # Vote distribution for transparency
        }
        
    def _interpret_actions_strategy(self, raw_actions, signals):
        """
        Intelligently interpret actions: decide whether to trade based on signal strength and action alignment.
        
        This method analyzes trading signals and determines appropriate final actions. It aims to allow the 
        agent's actions to have a meaningful impact while still incorporating technical signals as guidance.
        The original complex modulation logic that hindered learning has been simplified.

        Parameters:
        -----------
        raw_actions : numpy.ndarray
            Raw actions from RL agent (clipped to [-1, 1], shape: (action_dim,)).
        signals : dict
            Trading signals for each symbol, generated by _identify_trading_signals.

        Returns:
        --------
        numpy.ndarray
            Final actions array (shape: (action_dim,)) to be passed to _execute_trades.
            
        Notes:
        ------
        - Simplified logic to prevent overwriting agent's intent.
        - Strong signals that align with agent's action direction are more likely to be executed.
        - Weak signals or misaligned actions are dampened towards zero (hold).
        - A bypass option is included for debugging the base RL pipeline.
        """
        try:
            # Set switch for Using Signal-Driven actions strategy
            # Help verify comunication between Env and Agent
            bypass_interpretation = self.bypass_interpretation if self.bypass_interpretation else True # Default True

            if bypass_interpretation:
                logging.debug("STE Module - _interpret_actions_strategy - Bypassing interpretation, using raw actions.")
                # Ensure dtype
                return np.clip(raw_actions, -1.0, 1.0).astype(np.float32)
            
            # Initialize final actions array with zeros (default to hold)
            final_actions = np.zeros(self.action_dim, dtype=np.float32)
            # Get holding threshold from config or default
            hold_threshold = getattr(self.config, 'action_threshold', 0.1)
            # Get signal strength threshold from config or default
            signal_strength_threshold = getattr(self.config, 'signal_strength_threshold', 0.2)

            # Loop over each symbol and its corresponding action/signal
            for i, symbol in enumerate(self.symbols):
                if symbol in signals:
                    signal = signals[symbol]
                    raw_action = raw_actions[i] if i < len(raw_actions) else 0.0

                    sig_type = signal['type']
                    sig_strength = signal['strength']
                    sig_confidence = signal.get('confidence', 0.5) # Default confidence if not present

                    # Only act if signal is strong enough
                    if sig_strength > signal_strength_threshold:
                        # If agent action same with signal direction
                        if (sig_type == 'buy' and raw_action > hold_threshold) or \
                        (sig_type == 'sell' and raw_action < -hold_threshold):
                            # final_actions[i] = np.clip(raw_action * (1.0 + 0.1 * sig_strength), -1.0, 1.0) # This line encourage action slightly
                            pass # Default set raw_action remain
                            
                        # If agent action diff from signal direction
                        elif (sig_type == 'buy' and raw_action < -hold_threshold) or \
                            (sig_type == 'sell' and raw_action > hold_threshold):
                            # Strong signal with strong opposing action, suppress to near 0
                            final_actions[i] = raw_action * (1.0 - sig_confidence) # Suppress based on confidence
                            logging.debug(f"STE Module - _interpret_actions_strategy - Symbol {symbol}: Strong {sig_type} signal "
                                        f"conflicts with action {raw_action:.3f}. Dampened to {final_actions[i]:.3f}.")
                        
                        # If agent's action is weak (near hold), then gently push based on signal confidence
                        else: # abs(raw_action) <= hold_threshold
                            push_factor = sig_confidence * sig_strength # Combine strength and confidence
                            if sig_type == 'buy':
                                final_actions[i] = np.clip(final_actions[i] + push_factor, -1.0, 1.0)
                            elif sig_type == 'sell':
                                final_actions[i] = np.clip(final_actions[i] - push_factor, -1.0, 1.0)
                            # If it's a 'hold' signal, then don't push, maintain original action (near 0)
                    else:
                        # Signal is weak, only act if raw action is very strong
                        if abs(raw_action) > hold_threshold:
                             # Slightly dampen strong actions without strong signals
                            dampen_factor = 0.5 # Configurable suppression factor
                            final_actions[i] = raw_action * dampen_factor
                            logging.debug(f"STE Module - _interpret_actions_strategy - Symbol {symbol}: Weak signal, "
                                        f"dampening strong action {raw_action:.3f} to {final_actions[i]:.3f}.")
                        else:
                            # Weak signal, weak action -> hold
                            final_actions[i] = 0.0
                else:
                    # No signal for this symbol, use raw action if it's significant
                    raw_action = raw_actions[i] if i < len(raw_actions) else 0.0
                    if abs(raw_action) > hold_threshold:
                        final_actions[i] = raw_action * 0.8 # Mild suppression
                        logging.debug(f"STE Module - _interpret_actions_strategy - Symbol {symbol}: No signal, "
                                    f"dampening action {raw_action:.3f} to {final_actions[i]:.3f}.")
                    else:
                        final_actions[i] = 0.0

            # Debug log raw and final actions for auditing
            logging.debug(f"STE Module - _interpret_actions_strategy - Raw actions: {raw_actions}")
            logging.debug(f"STE Module - _interpret_actions_strategy - Final actions: {final_actions}")   
            return final_actions
        except Exception as e:
            # Log error and return zero actions as fallback
            logging.error(f"STE Module - Error in _interpret_actions_strategy: {e}")
            return np.zeros(self.action_dim, dtype=np.float32)


    def _calculate_strategy_reward(self, raw_return, raw_actions, final_actions, signals, sentiment_per_stock=None, risk_per_stock=None):
        """
        Calculate a robust reward function to guide the agent towards profitable and stable strategies,
        incorporating sentiment and risk factors passed from the step function.

        This reward function balances multiple objectives:
        1. Portfolio returns (primary objective)
        2. Cash penalty (encourage investment)
        3. Trade frequency penalty (reduce unnecessary churn)
        4. (Optional/Weak) Signal consistency bonus (align actions with technical analysis)
        5. Sentiment-based reward/penalty (encourage/disourage trades based on sentiment)
        6. Risk-based penalty (penalize holding high-risk assets)

        Parameters
        ----------
        raw_return : float
            Raw portfolio return for the current step (e.g., (V_t / V_{t-1}) - 1).
        raw_actions : ndarray
            Raw actions from the agent before any interpretation (-1 to 1).
        final_actions : ndarray
            Final actions taken after interpretation/modulation.
        signals : dict
            Trading signals used for action interpretation (for potential bonus).
        sentiment_per_stock : ndarray, optional
            Array of sentiment scores for each stock (shape: (action_dim,)). Default is None.
        risk_per_stock : ndarray, optional
            Array of risk scores for each stock (shape: (action_dim,)). Default is None.

        Returns
        -------
        float
            Computed reward value.

        Notes
        -----
        - Raw return is the core component, scaled by config.reward_scaling.
        - Cash penalty discourages holding excessive cash (promotes investment).
        - Trade penalty discourages frequent, small trades (reduces transaction costs).
        - Signal bonus (if used) rewards actions that align with technical indicator signals,
        but is weakened to prevent overriding the primary return objective.
        - Sentiment/Risk factors are used to shape the reward based on the environment's
        current state and the agent's actions, aligning with the project's core goal.
        - CVaR shaping is handled externally in step() for risk-awareness.
        """
        try:
            # Initialize reward components
            reward = 0.0
            scaled_return_component = 0.0
            cash_penalty_component = 0.0
            trade_penalty_component = 0.0
            signal_consistency_bonus_component = 0.0
            sentiment_reward_component = 0.0
            risk_penalty_component = 0.0

            # Core improvement: Raw return robustness
            if raw_return is None:
                raw_return = 0.0
            reward_scaling = float(getattr(self.config, 'reward_scaling', 1e-4))

            # Core reward: Scaled raw portfolio return
            # This is the primary driver for learning profitable strategies.
            scaled_return_component = raw_return * reward_scaling
            reward += scaled_return_component

            # Cash penalty: Penalize holding too much cash (encourages investment)
            # Using total_asset as denominator is more stable than cash alone.
            # Squaring the penalty makes it stronger at high cash levels.
            cash_penalty_proportion = float(getattr(self.config, 'cash_penalty_proportion', 0.1))
            if self.total_asset > 0: # Prevent division by zero or negative assets
                cash_ratio = self.cash / self.total_asset
                # Core improvement: Use squared term to strengthen cash penalty
                cash_penalty_component = -cash_penalty_proportion * (cash_ratio ** 2)
            else:
                cash_penalty_component = 0.0 # No penalty if no asset
            reward += cash_penalty_component
            
            # Trade frequency penalty: Penalize excessive trading (reduces costs)
            # This helps stabilize actions and prevent jitter.
            if self.last_actions is not None:
                # Calculate action difference to measure trade magnitude
                action_diff = np.abs(final_actions - self.last_actions)
                # Penalty proportional to sum of absolute differences
                # Core improvement: Rescale trading penalty for consistency with reward terms
                trade_penalty_strength = 0.0001 # Can be customed conditionally
                trade_penalty_component = -trade_penalty_strength * np.sum(action_diff)
                reward += trade_penalty_component
            
            # Initial switch for signal consistency bonus
            use_signal_consistency_bonus = self.use_signal_consistency_bonus if self.use_signal_consistency_bonus else False
            # Signal consistency bonus: Reward actions aligned with technical signals
            # Core improvement: Weakened to prevent overriding the primary return objective.
            if use_signal_consistency_bonus:
                signal_bonus_strength = 0.00001 # Set 0.001 -> 0.00001
                for i, symbol in enumerate(self.symbols):
                    if symbol in signals:
                        signal = signals[symbol]
                        action = final_actions[i] if i < len(final_actions) else 0.0
                        # Get holding threshold for consistency check
                        hold_threshold = float(getattr(self.config, 'hold_threshold', 0.1)) # 或 'action_threshold'
                        # Reward when actions align with strong signals (but with very weak weight)
                        if signal['type'] == 'buy' and action > hold_threshold:
                            # Core improvement: Combine strength and confidence
                            signal_strength = signal.get('strength', 0.5)
                            signal_confidence = signal.get('confidence', 0.5)
                            signal_consistency_bonus_component += signal_strength * signal_confidence * signal_bonus_strength
                        elif signal['type'] == 'sell' and action < -hold_threshold:
                            signal_strength = signal.get('strength', 0.5)
                            signal_confidence = signal.get('confidence', 0.5)
                            signal_consistency_bonus_component += signal_strength * signal_confidence * signal_bonus_strength
                reward += signal_consistency_bonus_component

            # Core improvement: Implement Sentiment/Risk factor mechanism

            # Sentiment Reward
            # Reward: buy/hold with high positive sentiment; sell with high negative. 
            # Penalty: sell with high positive; buy with high negative
            if sentiment_per_stock is not None and self.use_senti_factor:
                # Get sentiment threshold
                pos_senti_thr = self.senti_threshold.get(self.env_type, {}).get('pos_threshold', 0.0)
                neg_senti_thr = self.senti_threshold.get(self.env_type, {}).get('neg_threshold', 0.0)
                
                sentiment_reward_strength = float(getattr(self.config, 'sentiment_reward_strength', 0.0001)) # 可配置的奖励系数

                for i in range(self.action_dim):
                    action = final_actions[i]
                    senti_score = sentiment_per_stock[i] if i < len(sentiment_per_stock) else 0.0

                    # Encourage buying/holding under strong positive sentiment
                    if senti_score > pos_senti_thr and action > 0:
                        sentiment_reward_component += senti_score * action * sentiment_reward_strength
                    # Encourage selling under strong negative sentiment
                    elif senti_score < neg_senti_thr and action < 0:
                        sentiment_reward_component += abs(senti_score) * abs(action) * sentiment_reward_strength
                    # Penalize selling under strong positive sentiment
                    elif senti_score > pos_senti_thr and action < 0:
                        sentiment_reward_component -= senti_score * abs(action) * sentiment_reward_strength
                    # Penalize buying/holding under strong negative sentiment
                    elif senti_score < neg_senti_thr and action > 0:
                        sentiment_reward_component -= abs(senti_score) * action * sentiment_reward_strength
                        
            reward += sentiment_reward_component

            # Risk Penalty
            # Penalize holding large positions under high risk
            if risk_per_stock is not None and self.use_risk_factor:
                # Get risk threshold
                pos_risk_thr = self.risk_threshold.get(self.env_type, {}).get('pos_threshold', 0.0)
                # neg_risk_thr = self.risk_threshold.get(self.env_type, {}).get('neg_threshold', 0.0) # Generally monitor high risk threshold
                
                risk_penalty_strength = float(getattr(self.config, 'risk_penalty_strength', 0.0001)) # Configurable penalty coefficient
                
                for i in range(self.action_dim):
                    action = final_actions[i]
                    risk_score = risk_per_stock[i] if i < len(risk_per_stock) else 0.0

                    # Penalize holding large positions under high risk (whether long or short, high risk is undesirable)
                    if risk_score > pos_risk_thr:
                        # Use abs(action) to measure position size
                        risk_penalty_component -= risk_score * abs(action) * risk_penalty_strength
                        
            reward += risk_penalty_component
                
            # Ensure final reward is a standard Python float
            final_reward = float(reward)

            # Log reward components for debugging and monitoring
            logging.debug(
                f"STE Module - _calculate_strategy_reward - Components: "
                f"Scaled Return: {scaled_return_component:.6f}, "
                f"Cash Penalty: {cash_penalty_component:.6f}, "
                f"Trade Penalty: {trade_penalty_component:.6f}, "
                f"Signal Bonus: {signal_consistency_bonus_component:.6f}, "
                f"Sentiment Reward: {sentiment_reward_component:.6f}, "
                f"Risk Penalty: {risk_penalty_component:.6f}, "
                f"Final Reward: {final_reward:.6f}"
            )
            return final_reward
        except Exception as e:
            logging.error(f"STE Module - Error in _calculate_strategy_reward: {e}", exc_info=True)
            # Return a neutral reward as a robust fallback
            return 0.0
        
    def _get_senti_risk_features(self, current_row: np.ndarray, feature_type: str) -> np.ndarray:
        """
        Extracts sentiment or risk features for the current step from the current data row.

        This method retrieves the specific feature values needed for reward shaping or
        action interpretation based on the configured indices, regardless of whether
        these features are included in the state observation.

        Parameters
        ----------
        current_row : np.ndarray
            The flattened feature array for the current timestep (shape: (D,)).
        feature_type : str
            Type of feature to extract. Must be either 'sentiment' or 'risk'.

        Returns
        -------
        np.ndarray
            Array of feature values for each symbol (shape: (action_dim,)).
            Returns an array of zeros if the feature type is disabled or indices are missing.
        """
        if feature_type == 'sentiment' and self.use_senti_factor:
            if hasattr(self, 'senti_feature_index') and self.senti_feature_index:
                # Safely extract features, handling potential index errors or shape mismatches
                try:
                    # Ensure current_row is long enough
                    if len(current_row) > max(self.senti_feature_index):
                        senti_features = current_row[self.senti_feature_index]
                        # Ensure output shape is (action_dim,)
                        if len(senti_features) == self.action_dim:
                            return senti_features.astype(np.float32)
                        else:
                            logging.warning(f"STE Module - _get_senti_risk_features (senti) - "
                                            f"Mismatch in extracted feature length. Expected {self.action_dim}, got {len(senti_features)}. "
                                            f"Returning zeros.")
                    else:
                        logging.warning(f"STE Module - _get_senti_risk_features (senti) - "
                                        f"current_row length ({len(current_row)}) is not sufficient for senti_feature_index. "
                                        f"Returning zeros.")
                except (IndexError, TypeError) as e:
                    logging.error(f"STE Module - _get_senti_risk_features (senti) - Error extracting features: {e}")
            else:
                logging.debug("STE Module - _get_senti_risk_features (senti) - senti_feature_index is empty or not set.")
                
        elif feature_type == 'risk' and self.use_risk_factor:
            if hasattr(self, 'risk_feature_index') and self.risk_feature_index:
                try:
                    if len(current_row) > max(self.risk_feature_index):
                        risk_features = current_row[self.risk_feature_index]
                        if len(risk_features) == self.action_dim:
                            return risk_features.astype(np.float32)
                        else:
                            logging.warning(f"STE Module - _get_senti_risk_features (risk) - "
                                            f"Mismatch in extracted feature length. Expected {self.action_dim}, got {len(risk_features)}. "
                                            f"Returning zeros.")
                    else:
                        logging.warning(f"STE Module - _get_senti_risk_features (risk) - "
                                        f"current_row length ({len(current_row)}) is not sufficient for risk_feature_index. "
                                        f"Returning zeros.")
                except (IndexError, TypeError) as e:
                    logging.error(f"STE Module - _get_senti_risk_features (risk) - Error extracting features: {e}")
            else:
                logging.debug("STE Module - _get_senti_risk_features (risk) - risk_feature_index is empty or not set.")
        else:
            # If feature_type is invalid, or the corresponding factor switch is off
            logging.debug(f"STE Module - _get_senti_risk_features - Request for '{feature_type}' features skipped "
                        f"(factor switch: senti={self.use_senti_factor}, risk={self.use_risk_factor}).")

        # Return zeros if any condition above is not met
        return np.zeros(self.action_dim, dtype=np.float32)

    def step(self, actions):
        """
        Execute a trading step: process actions, perform trades, compute reward, and advance the environment.

        This method processes clipped actions, identifies trading signals, interprets actions,
        executes trades, calculates rewards (with senti/risk factors passed for internal reward shaping),
        updates states, and checks for termination. It integrates CVaR shaping for risk awareness.

        Parameters
        ----------
        actions : ndarray
            Action vector of shape (action_dim,) in [-1, 1], representing target asset allocations.

        Returns
        -------
        tuple
            - state : ndarray
                Next observation state (flattened historical features + agent states).
            - reward : float
                Reward for the action taken, shaped by portfolio return, risk (CVaR), etc.
            - done : bool
                Whether the episode has terminated (reached terminal_step).
            - truncated : bool
                Whether the episode was truncated (always False in this env).
            - info : dict
                Dictionary with step information (e.g., cash, position, total asset, date).

        Notes
        -----
        - Actions are clipped to [-1, 1] for bounded decisions.
        - Trading signals are identified but action interpretation is simplified.
        - Trades are executed based on interpreted actions and current prices.
        - Reward is shaped by return, CVaR, cash/turnover penalties, and senti/risk factors.
        - State advances to the next timestep.
        - Done is determined by reaching self.terminal_step (valid for train/valid/test).
        - Info dict includes Date, which relies on self.trading_dates set correctly in reset().
        - Senti/Risk factors are passed to the reward function for appropriate shaping.
        """
        logging.debug(f"STE Module - Env Step - Input actions: {actions}")
        try:
            # Clip actions to valid range [-1, 1] for bounded decisions
            actions = np.clip(actions, -1, 1).astype(np.float32)
            # Log input actions for debugging
            logging.info(f"STE Module - Env Step - Input actions shape: {actions.shape}")
            
            # Check if reaching terminal step
            # This condition is correct for both train/valid (random episode) and test (full period) modes
            if self.current_step >= self.terminal_step:
                done = True
                truncated = False
                reward = 0.0
                info = {
                    'Done': done,
                    'Truncated': truncated,
                    'Reward': reward,
                    'Total Asset': self.total_asset
                }
                # Ensure a valid state is returned even at termination
                dummy_state = np.zeros(self.state_dim, dtype=np.float32)
                return dummy_state, reward, done, truncated, info

            # Get current data row for decision making
            current_row = self.trading_df[self.current_step - 1]  # Data used for decision at current_step

            # Extract sentiment/risk per stock if factors enabled, else zeros
            # Core Improvement: Retain feature extraction, but no longer use it to directly modulate actions or rewards
            sentiment_per_stock = current_row[self.senti_feature_index] if self.use_senti_factor and len(self.senti_feature_index) > 0 else np.zeros(self.action_dim, dtype=np.float32)
            risk_per_stock = current_row[self.risk_feature_index] if self.use_risk_factor and len(self.risk_feature_index) > 0 else np.zeros(self.action_dim, dtype=np.float32)

            # Identify trading signals based on current indicators
            signals = self._identify_trading_signals(current_row)

            # Interpret actions using signals (strategy layer)
            final_actions = self._interpret_actions_strategy(actions, signals)
            # Ensure dtype
            final_actions = np.clip(final_actions, -1.0, 1.0).astype(np.float32)

            # Execute trades with actions generated by _interpret_actions_strategy()
            self._execute_trades(final_actions)
            # Calculate raw portfolio return after trade execution
            raw_return = self._get_portfolio_return()
            logging.info(f"STE Module - Env Step - After trade execution - Cash: {self.cash}, Positions: {self.position}, Total Asset: {self.total_asset}")

            # Core improvement: Reward calculation (sent senti/risk factor to _calculate_strategy_reward() function)
            reward = self._calculate_strategy_reward(
                raw_return, actions, final_actions, signals, 
                sentiment_per_stock, risk_per_stock
            )

            # Core improvement: CVaR shaping for risk awareness 
            # Calculate CVaR (Conditional Value at Risk) adjustment based on recent returns
            # This helps penalize strategies with high tail risk, promoting risk-aware learning.
            try:
                # Fetch CVaR alpha threshold from config (e.g., 0.05 for 5%)
                cvar_alpha = float(getattr(self.config, 'cvar_alpha', 0.05))
                # Ensure alpha is within valid range (0, 0.5)
                cvar_alpha = np.clip(cvar_alpha, 0.001, 0.499)
                # Fetch recent returns history from asset memory
                returns_history = self.returns_history
                # Check if sufficient history is available for robust CVaR calculation
                min_cvar_history = 5 # Minimum history points required
                if len(returns_history) >= min_cvar_history:
                    # Convert list to NumPy array for efficient computation
                    returns_array = np.array(returns_history, dtype=np.float32)
                    # Calculate Value at Risk (VaR) at alpha level using percentile
                    var = np.percentile(returns_array, 100 * cvar_alpha)
                    # Calculate CVaR as the mean of returns below VaR (tail loss)
                    # Add epsilon to prevent empty array access
                    below_var_returns = returns_array[returns_array <= var]
                    if len(below_var_returns) > 0:
                        cvar = below_var_returns.mean()
                        # Apply CVaR adjustment to reward (subtract because cvar is negative for losses)
                        # Scale factor for cvar impact is fetched from config
                        cvar_adjustment = float(self.config.cvar_factor * cvar)
                        reward += cvar_adjustment
                        logging.debug(f"STE Module - Step - CVaR adjustment applied: {cvar_adjustment:.6f}")
                    else:
                        logging.debug("STE Module - Step - No returns below VaR, CVaR adjustment skipped.")
                else:
                    logging.debug(f"STE Module - Step - Insufficient return history for CVaR ({len(returns_history)} < {min_cvar_history}), skipped.")
            except Exception as e:
                logging.error(f"STE Module - Step - Error calculating CVaR adjustment: {e}", exc_info=True)

            # Track total asset history
            self.asset_memory.append(float(self.total_asset))   # Ensure dtype

            # Advance timestep
            self.current_step += 1
            # Record last actions for trade frequency penalty in next step
            self.last_actions = final_actions.copy()
            # Check if episode done (reached terminal step)
            done = (self.current_step >= self.terminal_step)
            truncated = False  # No truncation in this environment

            # Build info dict for step monitoring
            info = {
                'Environment Type': self.env_type,
                'Episode Index': int(self.episode_idx),
                'Current Step': int(self.current_step),
                'Total Asset': float(self.total_asset),
                'Cash': float(self.cash),
                'Position': self.position.copy(), # Already float32 from _execute_trades
                'Cost': float(self.cost),
                'Raw Return': float(raw_return),
                'Reward': float(reward),
                # Log signals and actions for analysis
                'Signals': signals, # For debugging
                'Raw Actions': actions.tolist(), # Convert to list for JSON serialization in info
                'Final Actions': final_actions.tolist(),
                # Core improvement: Add Senti/Risk factor for behaviour analysis after training
                'Sentiment Per Stock': sentiment_per_stock.tolist(),
                'Risk Per Stock': risk_per_stock.tolist(),
                # Log ablation switches
                'Use Senti Factor': bool(self.use_senti_factor),
                'Use Risk Factor': bool(self.use_risk_factor),
                'Use Senti Features': bool(self.use_senti_features),
                'Use Risk Features': bool(self.use_risk_features),
                'Use Dynamic Ind Threshold': bool(self.use_dynamic_ind_threshold),
            }
            # Save Date index for visualization
            # This relies on self.trading_dates being correctly populated by reset() for both train/valid/test.
            # In test mode, reset() now ensures self.trading_dates covers the full test period.
            info['Date'] = None # Initialize Date as None
            try:
                # The date corresponds to the decision made at the *previous* step (current_step - 1)
                # because that's when the action was taken based on the data.
                # asset_memory[0] -> initial asset (before any decision, no date)
                # asset_memory[1] -> asset after decision at current_step=window_size -> date is trading_dates[0]
                # ...
                # asset_memory[t] -> asset after decision at current_step=window_size+t-1 -> date is trading_dates[t-1]
                # So, for the decision made at current_step-1, the date index is (current_step - 1) - window_size
                # which is current_step - window_size - 1.
                # BUT self.trading_dates is generated to correspond directly to decision points.
                # In reset(): self.trading_dates = self.trading_df.index[window_size : terminal_step]
                # So, trading_dates[0] corresponds to the decision made at step window_size (index 0 in trading_dates)
                # So, for current_step, the decision point index in trading_dates is current_step - window_size.
                # The date for the decision made at current_step-1 is therefore trading_dates[(current_step-1) - window_size]
                # Simplified: date_index_for_previous_decision = (self.current_step - 1) - self.window_size
                date_index_for_current_decision = self.current_step - self.window_size

                if (hasattr(self, 'trading_dates') and self.trading_dates is not None and
                    0 <= date_index_for_current_decision < len(self.trading_dates)):
                    current_date = self.trading_dates[date_index_for_current_decision]
                    # Ensure it's a datetime object before formatting
                    if isinstance(current_date, (pd.Timestamp, datetime)):
                        info['Date'] = current_date.strftime('%Y-%m-%d')
                    else:
                        # If it's already a string or other format, try to parse and format
                        info['Date'] = pd.to_datetime(current_date).strftime('%Y-%m-%d')
                    logging.debug(f"STE Module - Env Step - Set Date info to: {info['Date']} for step {self.current_step}")
                else:
                    if not hasattr(self, 'trading_dates') or self.trading_dates is None:
                        logging.debug("STE Module - Env Step - 'trading_dates' is not available or is None. Date info will be None.")
                    elif not (0 <= date_index_for_current_decision < len(self.trading_dates)):
                        logging.debug(f"STE Module - Env Step - Date index {date_index_for_current_decision} is out of bounds "
                                    f"[0, {len(self.trading_dates)-1}] for trading_dates (len={len(self.trading_dates)}). Date info will be None.")

            except (IndexError, TypeError, ValueError, AttributeError) as e:
                logging.warning(f"STE Module - Env Step - Error formatting date at internal index {self.current_step - self.window_size}: {e}")
                info['Date'] = None

            logging.info(f"STE Module - Env Step - Info: Total Asset={info['Total Asset']:.2f}, Reward={info['Reward']:.6f}")

            # Return Gym-compatible tuple
            return self._get_states(), reward, done, truncated, info

        except Exception as e:
            logging.error(f"STE Module - Env step error: {e}", exc_info=True)  # Add exc_info for stack trace
            # Return a valid state, zero reward, done=True to terminate episode on error
            dummy_state = np.zeros(self.state_dim, dtype=np.float32)
            return dummy_state, 0.0, True, False, {'Error': str(e)}
    
    def _get_states(self):
        """
        Retrieve the current observation state by concatenating historical features and agent states.

        This method constructs the observation by flattening a window of historical features
        (price, indicators, sentiment, risk) and appending current agent states (cash, positions, returns).

        Returns
        -------
        ndarray
            Flattened observation state array (shape: (state_dim,)).

        Notes
        -----
        - Uses a rolling window (current_step - window_size to current_step) for historical context.
        - Handles potential mismatch in feature dimensions with zero-padding or truncation (not implemented here).
        - Sentiment/risk features are zeroed if switches are False, maintaining dimension for ablation experiments.
        - Additional states: cash (available funds), position sum (aggregate holdings), relative return (total_asset / initial - 1).
        - Final shape is logged and should match self.state_dim for consistency.
        """
        try:
            # Extract historical window from trading_df (shape: (window_size, D features))
            window = self.trading_df[self.current_step - self.window_size : self.current_step] # (window_size, D)
            # Flatten price features from window using indices
            price_features = window[:, self.price_feature_index].flatten() # (window_size * num_symbols,)
            # Flatten indicator features
            ind_features = window[:, self.ind_feature_index].flatten() # (window_size * num_ind_features,)
            # Initialize state components list
            state_temp = [price_features, ind_features]
            # Conditionally add sentiment features or zeros
            if self.use_senti_features:
                logging.info(f"STE Module - Env _get_states - Introduce Sentiment features")
                senti_features = window[:, self.senti_feature_index].flatten() # (window_size * num_senti_features,)
                state_temp.append(senti_features) # Include for full model
            else:
                logging.info(f"STE Module - Env _get_states - No Sentiment features mode")
                # Zero-fill to maintain consistent state dimension for ablation studies
                zero_senti_shape = (self.window_size * len(self.senti_feature_index),)
                state_temp.append(np.zeros(zero_senti_shape, dtype=np.float32))
            # Conditionally add risk features or zeros
            if self.use_risk_features:
                logging.info(f"STE Module - Env _get_states - Introduce Risk features")
                risk_features = window[:, self.risk_feature_index].flatten() # (window_size * num_risk_features,)
                state_temp.append(risk_features) # Include for full model
            else:
                logging.info(f"STE Module - Env _get_states - No Risk features mode")
                # Zero-fill to maintain consistent state dimension for ablation studies
                zero_risk_shape = (self.window_size * len(self.risk_feature_index),)
                state_temp.append(np.zeros(zero_risk_shape, dtype=np.float32))

            # --- Agent States ---
            # Normalized cash (relative to initial asset)
            cash_state = np.array([self.cash], dtype=np.float32)
            # Normalized sum of absolute positions (proxy for portfolio concentration)
            position_state = np.array([np.sum(np.abs(self.position))], dtype=np.float32)
            # Relative portfolio return since episode start
            if len(self.asset_memory) > 1:
                return_state = np.array([(self.total_asset / self.asset_memory[0]) - 1.0], dtype=np.float32)
            else:
                return_state = np.array([0.0], dtype=np.float32)
            # Extend with agent states
            state_temp.extend([cash_state, position_state, return_state])
            # Concatenate all parts into final state array
            state = np.concatenate(state_temp).astype(np.float32)
            # Log final shape and compare to expected state_dim for integrity check
            logging.info(f"STE Module - Env _get_states - Final state shape: {state.shape}, expected: {self.state_dim}")
            assert state.shape == (self.state_dim,), f"State shape mismatch: {state.shape} vs {self.state_dim}"
            return state
        except Exception as e:
            # Log error and raise specific ValueError for upstream handling
            logging.error(f"STE Module - Error in state retrieval: {e}")
            raise ValueError("Error in state retrieval")

    def _execute_trades(self, actions):
        """
        Execute trades by updating the portfolio based on target asset allocation proportions.

        This internal method computes target allocations directly from the action vector,
        calculates trade volumes and costs (commission and slippage), and updates
        cash, positions, and total assets accordingly.

        The action semantics are:
        - `actions[i]` represents the target proportion of the total portfolio value
        to allocate to stock `i`.
        - E.g., `actions = [0.5, -0.2, 0.1]` means:
            - Allocate 50% of total asset to stock 0 (long).
            - Allocate -20% of total asset to stock 1 (short, if supported).
            - Allocate 10% of total asset to stock 2 (long).
        - The sum of `actions` determines the net leverage. sum(abs(actions)) determines
        the total traded value relative to the portfolio.

        Parameters
        ----------
        actions : ndarray
            Action vector of shape (action_dim,) in [-1, 1], representing target
            asset allocation proportions per symbol.

        Returns
        -------
        None
            Updates instance attributes (position, cash, total_asset, cost, last_prices) in place.

        Notes
        -----
        - Actions are interpreted directly as target allocations, not normalized weights.
        - Commission cost = sum(trade_volume) * commission_rate.
        - Slippage cost = sum(|price_diff| * |position_change|) * slippage_rate.
        - Uses epsilon (1e-8) to prevent division by zero.
        - Negative target allocations imply shorting (requires data/environment support).
        - Small trades below `min_trade_amount` threshold are ignored.
        """
        try:
            # Fetch current prices from targets for accurate valuation
            current_prices = self._get_current_prices() # Shape: (action_dim,)
            if current_prices is None or np.any(current_prices <= 0):
                logging.warning("STE Module - _execute_trades - Invalid current prices. Skipping trade.")
                return
            
            # Compute current value allocation
            current_allocation = self.position * current_prices # Shape: (action_dim,)
                
            # Core improvement: Calculate target allocation directly based on total asset
            # action[i] : percentage of target allocation
            # target_allocation[i] = action[i] * total_asset
            target_allocation = actions * self.total_asset # Shape: (action_dim,)
            # e.g.
            # actions = [0.5, -0.2, 0.1], total_asset = 100000
            # target_allocation = [50000, -20000, 10000]
            
            # Core improvement: Calculate value for trading
            trade_value = target_allocation - current_allocation # Shape: (action_dim,)
            # e.g.
            # current_allocation = [30000, 0, 5000]
            # trade_value = [50000-30000, -20000-0, 10000-5000] = [20000, -20000, 5000]
            # Positive: buy, Negative: sell

            # Core improvement: Apply minimum trade amount
            # Obtain minimum trade amount threshold
            min_trade_amount = float(getattr(self.config, 'min_trade_amount', 0.01) * self.total_asset)
            # Create a mask to mark trades to be executed
            trade_mask = np.abs(trade_value) >= min_trade_amount
            # If all trades are too small, skip execution
            if not np.any(trade_mask):
                logging.debug("STE Module - _execute_trades - All trade values below minimum threshold. Skipping trade.")
                self.last_prices = current_prices.copy()
                return # Return early without performing any trades or cost calculations

            # Core improvement: Calculate necessary trading cost
            total_trade_volume = np.sum(np.abs(trade_value[trade_mask])) # Based on trade mask
            commission_cost = total_trade_volume * float(getattr(self.config, 'commission_rate', 0.005))
            
            # Compute price difference for slippage
            price_diff = np.abs(current_prices - self.last_prices) # Shape: (action_dim,)
            
            # Compute slippage cost based on price movement, only for real trading
            position_change = np.abs(self.position - (target_allocation / (current_prices + 1e-8))) # Shape: (action_dim,)
            slippage_cost = np.sum(price_diff * position_change * trade_mask) * float(getattr(self.config, 'slippage_rate', 0.0))
            
            # Sum costs and accumulate to total cost
            total_cost = commission_cost + slippage_cost
            self.cost += total_cost
            
            # Core improvement: Update positions (shares)
            # New position = target position value / current price
            # Note: if target_allocation[i] is negative and current_prices[i] > 0,
            # then self.position[i] will be negative, representing a short position
            # Important: ensure your backtesting logic supports short positions
            # If shorts are not supported, process target_allocation here, e.g.:
            # target_allocation = np.maximum(target_allocation, 0)  # enforce non-negative
            eplision = 1e-8
            self.position = target_allocation / (current_prices + eplision)
            
            # Core improvement: Update cash
            # Cash change = - (target position value - current position value) - trading costs
            # Cash = previous cash - change in position value - costs
            # Change in position value = sum(target_allocation - current_allocation) = sum(trade_value)
            # Only executed trades are considered in the position value change
            executed_trade_value = np.sum(trade_value * trade_mask)
            cash_change = -executed_trade_value - total_cost
            self.cash += cash_change
            
            # Core improvement: Recalculate total assets for consistency
            # Total assets = cash + position value (considering short positions)
            # Position value = sum(position * price). If position is negative, value is negative.
            recalculated_total_asset = self.cash + np.sum(self.position * current_prices)
            # Minor differences may occur due to floating-point precision; log or adjust if needed
            # Simply set total assets = self.cash + position value
            self.total_asset = recalculated_total_asset
            
            # Update last_prices for next step's slippage calculation
            self.last_prices = current_prices.copy()

            # Debug log for auditing
            logging.debug(
                f"STE Module - _execute_trades - "
                f"Actions: {actions}, "
                f"Target Alloc: {target_allocation}, "
                f"Current Alloc: {current_allocation}, "
                f"Trade Value: {trade_value}, "
                f"Executed Trade Value: {executed_trade_value}, "
                f"Cost: {total_cost:.2f}, "
                f"New Cash: {self.cash:.2f}, "
                f"New Total Asset: {self.total_asset:.2f}"
            )
        except Exception as e:
            # Log error and raise specific ValueError for upstream handling
            logging.error(f"STE Module - Error in trade execution: {e}")
    
    def _get_current_prices(self):
        """
        Retrieve current adjusted closing prices for all symbols from the trading dataframe.

        This helper method fetches the latest price data for all symbols at the current timestep,
        using precomputed indices (self.price_feature_index) for efficiency. It includes boundary checks and handles
        invalid prices.

        Returns
        -------
        ndarray
            Array of current prices for each symbol (shape: (num_symbols,)), guaranteed to be > 0.
            Returns a fallback (e.g., last valid prices or ones) if current prices are invalid.

        Notes
        -----
        - Assumes price_feature_index points to adjusted close prices or similar in the features within features_all_flatten.
        - Used in trade execution and return calculations for real-time valuation.
        - Includes checks for step bounds and non-positive prices.
        - Falls back to last_prices or ones if current prices are invalid.
        """
        try:
            # Core improvement: Ensure current_step range
            if self.current_step >= len(self.trading_df) or self.current_step < 0:
                logging.warning(
                    f"STE Module - _get_current_prices - Current step {self.current_step} "
                    f"is out of bounds [0, {len(self.trading_df) - 1}]."
                )
                # Use the last available step if step is too large, or first step if negative (robustness)
                safe_step = max(0, min(self.current_step, len(self.trading_df) - 1))
                last_row = self.trading_df[safe_step] # Shape: (D,) where D is total features
            else:
                last_row = self.trading_df[self.current_step] # Shape: (D,) where D is total features

            # Extract only price-related features using indices and convert to float32 for precision
            # Fetch target price from raw data 
            prices = last_row[self.price_feature_index].astype(np.float32) # Shape: (num_symbols,)
            
            logging.debug(f"STE Debug - Using price_feature_index: {self.price_feature_index}")
            logging.debug(f"STE Debug - Raw prices fetched at step {self.current_step}: {prices}")

            # Core improvement: Check and handle negative price
            if np.any(prices <= 0):
                logging.warning(
                    f"STE Module - _get_current_prices - Non-positive prices found at step {self.current_step}: {prices}. "
                    f"Attempting fallback."
                )
                # Use last valid price if available
                if hasattr(self, 'last_prices') and self.last_prices is not None and not np.any(self.last_prices <= 0):
                    logging.info("STE Module - _get_current_prices - Using last valid prices as fallback.")
                    return self.last_prices.copy()
                else:
                    # Otherwise, return 1.0 as placeholder to avoid errors
                    # Strong assumption; better to have proper default or cleaned data
                    logging.warning(
                        "STE Module - _get_current_prices - No valid last_prices or last_prices also invalid. "
                        "Returning array of ones as fallback."
                    )
                    # Returned array length must match action_dim
                    return np.ones(self.action_dim, dtype=np.float32) 
                    
            return prices
        except Exception as e:
            logging.error(
                f"STE Module - _get_current_prices - Unexpected error at step {self.current_step}: {e}", 
                exc_info=True # Include traceback for detailed debugging
            )
            # Fallback to previous prices or ones if any unexpected error occurs
            if hasattr(self, 'last_prices') and self.last_prices is not None:
                # Prefer last_prices even if possibly invalid, as it reflects the latest state
                logging.info("STE Module - _get_current_prices - Unexpected error, returning last_prices.")
                return self.last_prices.copy()
            else:
                logging.warning("STE Module - _get_current_prices - Unexpected error and no last_prices, returning ones.")
                return np.ones(self.action_dim, dtype=np.float32)

            
    def _get_portfolio_return(self):
        """
        Calculate the relative portfolio return for the current step based on asset memory.

        This helper method computes the percentage change in total asset value from the
        previous timestep (last element in asset_memory) to the current one (self.total_asset),
        serving as the raw return component for rewards.

        Returns
        -------
        float
            Relative portfolio return ((current_asset / previous_asset) - 1).
            Returns 0.0 if asset_memory is empty or previous_asset is non-positive.

        Notes
        -----
        - Assumes asset_memory is appended after each step; uses last element as previous value.
        - Useful for reward components in RL, promoting strategies that maximize asset growth.
        - Includes checks for empty memory and non-positive previous asset value.
        - Debug logging aids in verifying asset transitions during backtesting.
        """
        try:
            # Core improvement: Check asset_memory 
            # Check if asset_memory is empty to handle initial step safely
            if len(self.asset_memory) == 0:
                logging.debug("STE Module - _get_portfolio_return - Asset memory is empty. Returning 0.0 return.")
                return 0.0 # Default to zero return at episode start or if memory is unexpectedly empty
            
            # Core improvement: Retrieve total assets from the previous step and check for validity
            previous_asset = self.asset_memory[-1]

            # Check for non-positive previous asset value to prevent division by zero or invalid calculations
            if previous_asset <= 0:
                logging.warning(
                    f"STE Module - _get_portfolio_return - Previous asset value is non-positive: {previous_asset}. "
                    f"Cannot calculate return. Returning 0.0."
                )
                return 0.0 # Return zero return if previous asset value is invalid

            # Core improvement: Calculate returns
            # Compute relative return: (current / previous) - 1 for percentage change
            # Ensure self.total_asset is also valid (though _execute_trades should maintain it)
            if self.total_asset <= 0:
                logging.warning(
                    f"STE Module - _get_portfolio_return - Current total_asset is non-positive: {self.total_asset}. "
                    f"Cannot calculate return. Returning 0.0."
                )
                return 0.0
                
            return_value = (self.total_asset / previous_asset) - 1.0

            # Debug log previous and current assets for return validation
            logging.debug(
                f"STE Module - _get_portfolio_return - Previous asset: {previous_asset:.4f}, "
                f"Current asset: {self.total_asset:.4f}, Calculated Return: {return_value:.6f}"
            )
            return np.float32(return_value)
        except Exception as e:
            logging.error(f"STE Module - _get_portfolio_return - Unexpected error: {e}", exc_info=True)
            # Return a neutral return as a robust fallback in case of any calculation error
            return 0.0

    def render(self, mode='human'):
        """
        Render the environment's current state for monitoring (simplified console output).

        This method provides a human-readable summary of the environment's key states,
        including current step, total asset value, cash, and positions.

        Parameters
        ----------
        mode : str, optional
            Rendering mode (default: 'human' for console). Other modes ignored.

        Returns
        -------
        None
            Prints formatted state summary to console.

        Notes
        -----
        - Useful for debugging and real-time monitoring during episodes.
        - Shows step progress, asset value, cash, and position vector.
        """
        # Print formatted state summary for human-readable feedback
        print(f"STE Module - Render - Step {self.current_step} / {self.terminal_step}"
              f"| Asset: {self.total_asset:.6f}"
              f"| Cash: {self.cash:.6f}"
              f"| Positions: {self.position}")

    def close(self):
        """
        Clean up resources when environment is closed (placeholder).

        This method is called when the environment is no longer needed.
        Currently a placeholder for potential future cleanup logic.

        Returns
        -------
        None
        """
        pass


# %%
