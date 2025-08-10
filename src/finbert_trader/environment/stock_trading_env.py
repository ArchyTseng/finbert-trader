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
        # Set environment type for data selection (train/valid/test)
        self.env_type = env_type
        # Copy data to prevent side-effects on original rl_data
        self.data = rl_data.copy()
        # Core dimensions from config for state and action spaces
        self.symbols = self.config.symbols  # List of stock symbols
        self.state_dim = self.config.state_dim  # Total state dimension
        self.action_dim = self.config.action_dim  # Action dimension (one per symbol)
        self.window_size = self.config.window_size  # Historical window size for states
        self.features_all_flatten = self.config.features_all_flatten  # Flattened all features
        self.features_price_flatten = self.config.features_price_flatten  # Flattened price features
        self.features_ind_flatten = self.config.features_ind_flatten  # Flattened indicator features
        self.features_senti_flatten = self.config.features_senti_flatten  # Flattened sentiment features
        self.features_risk_flatten = self.config.features_risk_flatten  # Flattened risk features

        # Feature category indices for selective access/updates
        self.price_feature_index = self.config.price_feature_index
        self.ind_feature_index = self.config.ind_feature_index
        self.senti_feature_index = self.config.senti_feature_index
        self.risk_feature_index = self.config.risk_feature_index

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

        # Experiment mode switches for ablation (default True to enable full features)
        self.use_senti_factor = getattr(self.config, 'use_senti_factor', True)  # Use sentiment factor in rewards/actions
        self.use_risk_factor = getattr(self.config, 'use_risk_factor', True)    # Use risk factor in rewards/actions

        self.use_senti_features = getattr(self.config, 'use_senti_features', True)  # Include sentiment features in state
        self.use_risk_features = getattr(self.config, 'use_risk_features', True)    # Include risk features in state

        self.use_senti_threshold = getattr(self.config, 'use_senti_threshold', True)  # Apply sentiment thresholds
        self.use_risk_threshold = getattr(self.config, 'use_risk_threshold', True)    # Apply risk thresholds

        self.use_dynamic_threshold = getattr(self.config, 'use_dynamic_threshold', False)  # Use dynamic vs static thresholds

        # Log core config for debugging and verification
        logging.info(f"STE Module - Env Init - Config symbols: {self.symbols}, window_size: {self.window_size}, features_all_flatten len: {len(self.features_all_flatten)}, state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        # Log data length and first episode shape (handle empty data gracefully)
        logging.info(f"STE Module - Env Init - rl_data len: {len(self.data)}, first episode states shape: {self.data[0]['states'].shape if self.data else 'Empty'}")
        # Debug log for feature breakdowns
        logging.debug(f"STE Module - Env Init - Features flatten: price {len(self.features_price_flatten)}, ind {len(self.features_ind_flatten)}, senti {len(self.features_senti_flatten)}, risk {len(self.features_risk_flatten)}")
        # Debug log for indices
        logging.debug(f"STE Module - Env Init - Indices: price {self.price_feature_index}, ind {self.ind_feature_index}, senti {self.senti_feature_index}, risk {self.risk_feature_index}")
        # Gym Space Definitions for RL compatibility
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.state_dim,),
                                     dtype=np.float32)  # Unbounded observation space for features
        self.action_space = Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)  # Continuous actions [-1,1] per symbol
        # Log initialization summary
        logging.info(f"STE Modul - Env initialized: environment type: {self.env_type}, model={self.config.model}, state_dim={self.state_dim}")

        # Internal State for slippage calculation
        self.last_prices = None  # Initial for slippage
        # Call reset to initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state by randomly selecting an episode.

        This method selects a random episode from the data, initializes trading states,
        resets agent variables (cash, position, etc.), and prepares the initial observation.

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
        - Random episode selection promotes diverse training and prevents overfitting to sequences.
        - Current step starts at window_size to include historical context in the state.
        - Agent states are normalized (cash=1.0 relative) for scale-invariant RL training.
        - Info dict aids in monitoring and debugging, compatible with Stable Baselines3.
        """
        # Call superclass reset for Gym compatibility and seed handling
        super().reset(seed=seed)
        try:
            # Randomly select an episode index for multi-episode diversity
            self.episode_idx = np.random.randint(len(self.data))  # Ensures varied starting points across resets
            # Extract data for the selected episode
            episode_data = self.data[self.episode_idx]
            self.trading_df = episode_data['states']  # [T, D]: Time x Features array
            self.targets = episode_data['targets']    # [T, N]: Time x Symbols price targets
            # Set terminal step as the last index of the episode
            self.terminal_step = len(self.trading_df) - 1
            # Start current step after window to include history
            self.current_step = self.window_size
            # Reset agent state to initial values
            self.cash = 1.0  # Normalized initial cash (relative to portfolio value)
            self.position = np.zeros(self.action_dim, dtype=np.float32)  # Zero positions (no holdings)
            self.cost = 0.0  # Reset accumulated transaction costs
            self.total_asset = 1.0  # Initial total asset value (cash only)
            self.asset_memory = [self.total_asset]  # Start asset history tracking
            self.returns_history = []  # Clear returns history for new episode
            # Set initial last prices from current step for slippage calculations
            self.last_prices = self._get_current_prices()  # Fetches prices at current_step

            # Build info dict for reset details and monitoring
            info = {'Environment Type': self.env_type,  # Train/valid/test
                    'Episode Index': self.episode_idx,  # Selected episode
                    'Episode Length': self.terminal_step + 1,  # Full length including window
                    'Targets': self.targets[:5],  # Sample targets for quick check
                    'Cash': self.cash,
                    'Position': self.position,
                    'Total Asset': self.total_asset,
                    'Last Prices': self.last_prices}

            # Log reset details for debugging and verification
            logging.info(f"STE Module - Env Reset - Episode idx: {self.episode_idx}, trading_df shape: {self.trading_df.shape}, targets shape: {self.targets.shape}, terminal_step: {self.terminal_step}")
            logging.info(f"STE Module - Env Reset - Reset information: {info}")
            # Return initial state and info
            return self._get_states(), info
        except Exception as e:
            # Log error and raise specific ValueError for caller handling
            logging.error(f"STE Module - Env reset error: {e}")
            raise ValueError("Error in environment reset")

    def _get_threshold_and_strength(self, factor_array, threshold_cfg, use_threshold, use_dynamic):
        """
        Compute positive/negative thresholds and infusion strength based on configuration and dynamic options.

        This internal method determines thresholds for factor application (e.g., sentiment/risk)
        and computes infusion strength, either statically from config or dynamically from factor statistics.

        Parameters
        ----------
        factor_array : ndarray
            Array of factor values (e.g., sentiment scores or risk metrics) for the current step.
        threshold_cfg : dict
            Configuration dictionary with thresholds per env_type (e.g., {'train': {'pos_threshold': 0.5, 'neg_threshold': -0.5}}).
        use_threshold : bool
            Flag to enable/disable threshold application; if False, thresholds default to 0.0.
        use_dynamic : bool
            Flag to enable dynamic strength computation based on factor_array statistics.

        Returns
        -------
        tuple
            (pos_threshold: float, neg_threshold: float, infusion_strength: float)

        Notes
        -----
        - Thresholds are fetched from threshold_cfg for the current env_type; defaults to 0.0 if missing or disabled.
        - Dynamic strength = clip(std / max(|mean|, 1e-6), 0.001, 0.01) to adapt to data variability while bounding values.
        - Falls back to self.config.infusion_strength (default 0.001) if not dynamic or array is empty.
        """
        # Determine thresholds based on use_threshold flag
        if use_threshold:
            # Fetch env-specific thresholds from config, fallback to defaults if not found
            thresholds = threshold_cfg.get(self.env_type, {'pos_threshold': 0.0, 'neg_threshold': 0.0}) \
                        or {'pos_threshold': 0.0, 'neg_threshold': 0.0}
        else:
            # Disable thresholds by setting to zero if flag is False
            thresholds = {'pos_threshold': 0.0, 'neg_threshold': 0.0}

        # Compute infusion strength dynamically if enabled and array is non-empty
        if use_dynamic and factor_array.size > 0:
            # Calculate strength as std / |mean| (normalized variability), with safeguards against zero mean
            infusion_strength = np.clip(factor_array.std() / max(abs(factor_array.mean()), 1e-6), 0.001, 0.01)
        else:
            # Fallback to static config value for consistency when dynamic is off or data insufficient
            infusion_strength = getattr(self.config, 'infusion_strength', 0.001)

        logging.info(f"STE Module - _get_threshold_and_strength - Dynamic: {use_dynamic}, Infusion strength: {infusion_strength}, Positiv Thresholds: {thresholds['pos_threshold']}, Negative Thresholds: {thresholds['neg_threshold']}")
        # Return the computed values as a tuple
        return thresholds['pos_threshold'], thresholds['neg_threshold'], infusion_strength


    def step(self, actions):
        """
        Execute a trading step: modulate actions with factors, perform trades, compute modular reward, and advance the environment.

        This method processes clipped actions, infuses sentiment/risk factors (if enabled), executes trades,
        calculates risk-weighted returns with penalties and CVaR shaping, updates states, and checks termination.

        Parameters
        ----------
        actions : ndarray
            Action vector of shape (action_dim,) in [-1, 1], representing buy/sell proportions per symbol.

        Returns
        -------
        tuple
            - state : ndarray
                Next observation state.
            - reward : float
                Computed reward (raw return - cash penalty + CVaR adjustment).
            - done : bool
                True if episode terminated.
            - truncated : bool
                Always False (no truncation in this env).
            - info : dict
                Dictionary with step details (e.g., assets, factors, switches).

        Notes
        -----
        - Actions are modulated by sentiment factors (enhance/penalize based on thresholds and direction).
        - Returns are weighted by risk factors and positions for multi-stock balance.
        - CVaR shaping uses historical returns (min 10) to penalize tail risks, configurable via cvar_alpha/factor.
        - Vectorized operations ensure efficiency for high-dimensional states/actions.
        """
        try:
            # Clip actions to valid range [-1, 1] for bounded decisions
            actions = np.clip(actions, -1, 1).astype(np.float32)
            # Log input actions for debugging
            logging.info(f"STE Module - Env Step - Input actions shape: {actions.shape}")

            # Fetch current feature row (t-1, as step advances after trade)
            current_row = self.trading_df[self.current_step - 1]  # Current after trade
            # Extract sentiment/risk per stock if factors enabled, else zeros
            sentiment_per_stock = current_row[self.senti_feature_index] if self.use_senti_factor else np.zeros(self.action_dim, dtype=np.float32)
            risk_per_stock = current_row[self.risk_feature_index] if self.use_risk_factor else np.zeros(self.action_dim, dtype=np.float32)

            # Get thresholds and infusion strengths for sentiment
            pos_senti_thr, neg_senti_thr, senti_inf_strength = self._get_threshold_and_strength(
                sentiment_per_stock, self.config.senti_threshold, self.use_senti_threshold, self.use_dynamic_threshold)

            # Get thresholds and infusion strengths for risk
            pos_risk_thr, neg_risk_thr, risk_inf_strength = self._get_threshold_and_strength(
                risk_per_stock, self.config.risk_threshold, self.use_risk_threshold, self.use_dynamic_threshold)

            # Initialize factors as ones (no effect baseline)
            Senti_factor = np.ones(self.action_dim, dtype=np.float32)
            Risk_factor = np.ones(self.action_dim, dtype=np.float32)

            # Apply sentiment infusion factor (vectorized for efficiency)
            if self.use_senti_factor:
                # Masks for positive/negative sentiment exceeding thresholds
                mask_pos = sentiment_per_stock > pos_senti_thr
                mask_neg = sentiment_per_stock < neg_senti_thr

                # Enhance if aligned (pos senti & buy, neg & sell), penalize if opposed
                Senti_factor[np.where((mask_pos & (actions > 0)) | (mask_neg & (actions < 0)))] = 1 + senti_inf_strength
                Senti_factor[np.where((mask_pos & (actions < 0)) | (mask_neg & (actions > 0)))] = 1 - senti_inf_strength

                # Debug log sentiment factors
                logging.debug(f"STE Module - Env Step - Sentiment factor: {Senti_factor}")

            # Apply risk infusion factor (vectorized)
            if self.use_risk_factor:
                # Masks for positive/negative risk exceeding thresholds
                mask_pos = risk_per_stock > pos_risk_thr
                mask_neg = risk_per_stock < neg_risk_thr

                # Penalize if aligned (high risk & buy, low & sell), enhance if opposed
                Risk_factor[np.where((mask_pos & (actions > 0)) | (mask_neg & (actions < 0)))] = 1 - risk_inf_strength
                Risk_factor[np.where((mask_pos & (actions < 0)) | (mask_neg & (actions > 0)))] = 1 + risk_inf_strength

                # Debug log risk factors
                logging.debug(f"STE Module - Env Step - Risk factor: {Risk_factor}")

            # Modulate actions with sentiment factor, re-clip to [-1,1]
            mod_actions = np.clip(Senti_factor * actions, -1, 1)

            # Execute trades with modulated actions
            self._execute_trades(mod_actions)
            # Log post-trade state
            logging.info(f"STE Module - Env Step - Executed trades, cash: {self.cash}, position: {self.position}")

            # Compute position weights (normalized absolute positions, +epsilon for stability)
            weights = np.abs(self.position) / (np.sum(np.abs(self.position)) + 1e-8)
            # Calculate raw return and weight by risk factor (dot product for multi-stock)
            raw_return = self._calculate_return() * np.dot(weights, Risk_factor)

            # Debug log if both factors enabled
            if self.use_senti_factor and self.use_risk_factor:
                logging.debug(f"STE Module - Env Step - Raw return: {raw_return}, Risk_factor: {Risk_factor}, Sentiment_factor: {Senti_factor}")

            # Apply cash penalty to encourage investment
            penalty = getattr(self.config, 'cash_penalty_proportion', 0.01)
            reward = np.float32(raw_return - self.cash * penalty)

            # CVaR shaping for risk-aware rewards
            self.returns_history.append(raw_return)
            if len(self.returns_history) >= getattr(self.config, 'cvar_min_history', 10) and getattr(self.config, 'cvar_factor', 0.05) > 0:
                # Fetch CVaR params from config
                cvar_alpha = getattr(self.config, 'cvar_alpha', 0.05)
                # Convert history to array for percentile/mean
                returns_array = np.array(self.returns_history, dtype=np.float32)
                # Compute VaR at alpha percentile
                var = np.percentile(returns_array, 100 * cvar_alpha)
                # CVaR as mean of returns below VaR
                cvar = returns_array[returns_array <= var].mean()
                # Adjust reward by CVaR (penalize tail risks)
                reward += np.float32(self.config.cvar_factor * cvar)

            # Track total asset history
            self.asset_memory.append(self.total_asset)

            # Advance timestep
            self.current_step += 1
            # Check if episode done (reached terminal step)
            done = (self.current_step >= self.terminal_step)
            truncated = False  # No truncation in this environment

            # Build info dict for step monitoring
            info = {
                'Total Asset': self.total_asset,
                'Cash': self.cash,
                'Position': self.position.copy(),  # Copy to avoid reference issues
                'Reward': reward,
                'Cost': self.cost,
                'Current Step': self.current_step,
                'Sentiment Factor': Senti_factor,
                'Risk Factor': Risk_factor,
                'Done': done,
                'Truncated': truncated,
                'Use Senti Factor': self.use_senti_factor,
                'Use Risk Factor': self.use_risk_factor,
                'Use Senti Threshold': self.use_senti_threshold,
                'Use Risk Threshold': self.use_risk_threshold,
                'Use Dynamic Threshold': self.use_dynamic_threshold
            }

            # Log step info
            logging.info(f"STE Module - Env Step - Step info: {info}")

            # Return Gym-compatible tuple
            return self._get_states(), reward, done, truncated, info

        except Exception as e:
            # Log error and raise for upstream handling
            logging.error(f"STE Module - Env step error: {e}")
            raise ValueError("Error in environment step")
    
    def _get_states(self):
        """
        Retrieve the current observation state by concatenating historical features and agent states.

        This internal method extracts a window of features from trading_df, flattens them by category
        (price, indicators, sentiment, risk), optionally zeros out sentiment/risk for ablation,
        and appends cash, total position, and relative return states.

        Parameters
        ----------
        None
            Relies on instance attributes like trading_df, current_step, window_size, feature indices,
            and ablation switches (use_senti_features, use_risk_features).

        Returns
        -------
        ndarray
            Flattened state array of shape (state_dim,) and dtype float32, ready for RL input.

        Notes
        -----
        - Window covers [current_step - window_size : current_step] for historical context.
        - Sentiment/risk features are zeroed if switches are False, maintaining dimension for ablation experiments.
        - Additional states: cash (available funds), position sum (aggregate holdings), relative return (total_asset / initial - 1).
        - Final shape is logged and should match self.state_dim for consistency.
        """
        try:
            # Extract historical window from trading_df (shape: (window_size, D features))
            window = self.trading_df[self.current_step - self.window_size : self.current_step]  # (window_size, D)
            # Flatten price features from window using indices
            price_features = window[:, self.price_feature_index].flatten()  # (window_size * len(price))
            # Flatten indicator features
            ind_features = window[:, self.ind_feature_index].flatten()  # (window_size * len(ind))
            # Flatten sentiment features if enabled, else zero vector for dimension consistency
            senti_features = window[:, self.senti_feature_index].flatten() if self.use_senti_features else np.zeros(self.window_size * len(self.senti_feature_index), dtype=np.float32)  # (window_size * len(senti))
            # Flatten risk features if enabled, else zero vector
            risk_features = window[:, self.risk_feature_index].flatten() if self.use_risk_features else np.zeros(self.window_size * len(self.risk_feature_index), dtype=np.float32)  # (window_size * len(risk))
            # Debug log feature shapes for verification
            logging.debug(f"STE Module - Env _get_states - "
                          f"Window shape: {window.shape}, "
                          f"price_feats shape: {price_features.shape}, "
                          f"ind_feats: {ind_features.shape}, "
                          f"senti_feats: {senti_features.shape}, "
                          f"risk_feats: {risk_features.shape}")
            # Prepare scalar states as 1D arrays
            cash_state = np.array([self.cash], dtype=np.float32)  # Current cash level
            position_state = np.array([np.sum(self.position)], dtype=np.float32)  # Sum of positions across symbols for aggregate exposure
            return_state = np.array([self.total_asset / self.asset_memory[0] - 1.0], dtype=np.float32)  # Relative return since episode start (scale-invariant)
            # Debug log additional states
            logging.debug(f"STE Module - Env _get_states - Cash: {cash_state}, position_state: {position_state}, return_state: {return_state}")
            # Initialize temp list with base features (price + ind)
            state_temp = [price_features, ind_features]
            # Conditionally add sentiment features or zeros based on switch
            if self.use_senti_features:
                logging.info(f"STE Module - Env _get_states - Introduce Sentiment features")
                state_temp.append(senti_features)  # Include for full model
            else:
                logging.info(f"STE Module - Env _get_states - No Sentiment features mode")
                state_temp.append(np.zeros_like(senti_features))  # Zero-fill for ablation consistency
            # Conditionally add risk features or zeros
            if self.use_risk_features:
                logging.info(f"STE Module - Env _get_states - Introduce Risk features")
                state_temp.append(risk_features)  # Include for full model
            else:
                logging.info(f"STE Module - Env _get_states - No Risk features mode")
                state_temp.append(np.zeros_like(risk_features))  # Zero-fill for ablation

            # Extend with agent states
            state_temp.extend([cash_state, position_state, return_state])

            # Concatenate all parts into final state array
            state = np.concatenate(state_temp).astype(np.float32)
            # Log final shape and compare to expected state_dim for integrity check
            logging.info(f"STE Module - Env _get_states - Final state shape: {state.shape}, expected: {self.state_dim}")
            return state
        except Exception as e:
            # Log error and raise specific ValueError for upstream handling
            logging.error(f"STE Module - Error in state retrieval: {e}")
            raise ValueError("Error in state retrieval")

    def _execute_trades(self, actions):
        """
        Execute trades by updating the portfolio based on given actions.

        This internal method computes target allocations from actions, calculates trade volumes and costs
        (commission and slippage), and updates cash, positions, and total assets accordingly.

        Parameters
        ----------
        actions : ndarray
            Modulated action vector of shape (action_dim,) in [-1, 1], representing desired buy/sell proportions per symbol.

        Returns
        -------
        None
            Updates instance attributes (position, cash, total_asset, cost, last_prices) in place.

        Notes
        -----
        - Actions are normalized to weights summing to 1 (considering absolutes for long/short).
        - Commission cost = sum(trade_volume) * commission_rate.
        - Slippage cost = sum(|price_diff| * |position|) * slippage_rate, simulating market impact.
        - Uses epsilon (1e-8) to prevent division by zero in weights and positions.
        """
        try:
            # Fetch current prices from targets for accurate valuation
            current_prices = self._get_current_prices()
            # Debug log prices and actions for trade verification
            logging.debug(f"STE Module - Env _execute_trades - Current prices: {current_prices}, actions: {actions}")
            
            # Compute current value allocation (positions * prices)
            current_allocation = self.position * current_prices
            
            # Normalize actions to weights (sum abs(weights) = 1, handles long/short)
            weights = actions / (np.sum(np.abs(actions)) + 1e-8)
            # Compute target allocation as total_asset * weights
            target_allocation = self.total_asset * weights
            # Calculate absolute trade volume (difference in allocations)
            trade_volume = np.abs(target_allocation - current_allocation)
            # Compute commission cost based on trade volume
            commission_cost = np.sum(trade_volume) * getattr(self.config, 'commission_rate', 0.005)
            # Compute price difference for slippage
            price_diff = np.abs(current_prices - self.last_prices)
            # Compute slippage cost based on price movement and existing positions
            slippage_cost = np.sum(price_diff * np.abs(self.position)) * getattr(self.config, 'slippage_rate', 0.0)

            # Sum costs and accumulate to total cost
            total_cost = commission_cost + slippage_cost
            self.cost += total_cost

            # Debug log weights, targets, and costs for auditing
            logging.debug(f"STE Module - Env _execute_trades - Weights: {weights}, desired_allocation: {target_allocation}, total cost: {total_cost}")

            # Update positions as target_allocation / current_prices (shares/holdings)
            self.position = target_allocation / (current_prices + 1e-8)
            # Update cash: subtract new holdings value and costs from total_asset
            self.cash = self.total_asset - np.sum(self.position * current_prices) - total_cost
            # Recompute total_asset to ensure balance (cash + holdings value)
            self.total_asset = self.cash + np.sum(self.position * current_prices)
            # Update last_prices for next step's slippage calculation
            self.last_prices = current_prices.copy()
        except Exception as e:
            # Log error and raise specific ValueError for upstream handling
            logging.error(f"STE Module - Error in trade execution: {e}")
            raise ValueError("Error in trade execution")

    def _calculate_return(self):
        """
        Calculate the single-step return based on change in total asset value.

        This internal method computes the relative return as (current_total_asset / previous_total_asset) - 1.
        Returns 0.0 if no previous asset value is available.

        Parameters
        ----------
        None
            Relies on instance attributes: total_asset and asset_memory (list of historical total assets).

        Returns
        -------
        float
            The computed return value, as a percentage change (e.g., 0.01 for 1% gain).

        Notes
        -----
        - Assumes asset_memory is appended after each step; uses last element as previous value.
        - Useful for reward components in RL, promoting strategies that maximize asset growth.
        - Debug logging aids in verifying asset transitions during backtesting.
        """
        # Check if asset_memory is empty to handle initial step safely
        if len(self.asset_memory) == 0:
            return 0.0  # Default to zero return at episode start
        # Compute relative return: (current / previous) - 1 for percentage change
        return_value = (self.total_asset / self.asset_memory[-1]) - 1.0
        # Debug log previous and current assets for return validation
        logging.debug(f"STE Module - Env _calculate_return - Previous asset: {self.asset_memory[-1]}, current: {self.total_asset}")
        return return_value
    
    def _get_current_prices(self):
        """
        Extract the current adjusted close prices from the end of the feature window.

        This internal method retrieves the price features from the current step's row
        in trading_df using pre-defined price_feature_index.

        Parameters
        ----------
        None
            Relies on instance attributes: trading_df (feature array), current_step,
            and price_feature_index (list of indices for price features).

        Returns
        -------
        ndarray
            Array of current prices per symbol, dtype float32.

        Notes
        -----
        - Assumes price_feature_index points to adjusted close prices or similar in the features.
        - Used in trade execution and return calculations for real-time valuation.
        - No boundary checks; assumes current_step is valid (handled upstream).
        """
        # Fetch the entire feature row at current_step
        last_row = self.trading_df[self.current_step]  # Shape: (D,) where D is total features
        # Extract only price-related features using indices and convert to float32 for precision
        prices = last_row[self.price_feature_index].astype(np.float32)  # Shape: (num_symbols,)
        return prices

    def render(self, mode='human'):
        """
        Render the current environment state for visualization.

        This method prints the current step, total asset, cash, and positions
        in 'human' mode for console-based monitoring.

        Parameters
        ----------
        mode : str, optional
            Rendering mode: 'human' for console print. Default is 'human'.

        Returns
        -------
        None
            Prints to console in 'human' mode.

        Notes
        -----
        - Compatible with Gym API for easy integration with RL libraries.
        - Precision formatting (:.6f) ensures clean output for floats.
        """
        # Print formatted state summary for human-readable feedback
        print(f"Step {self.current_step} / {self.terminal_step} | Asset: {self.total_asset:.6f} | Cash: {self.cash:.6f} | Positions: {self.position}")

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
