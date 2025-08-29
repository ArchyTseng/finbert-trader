# trading_backtest.py
# Module: TradingBacktest
# Purpose: Comprehensive backtesting framework for multi-stock RL trading agents with detailed performance metrics.
# Design:
# - RL-optimized backtesting without Backtrader dependency
# - Rich performance metrics including Sharpe, Max Drawdown, CAGR, CVaR
# - Detailed trade logging and asset tracking
# - Batch backtesting support for multiple algorithms
# Linkage: Integrates with TradingAgent, StockTradingEnv, ConfigTrading
# Robustness: Comprehensive error handling, data validation, metric computation
# Extensibility: Easy to add new metrics or backtest modes
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
import yfinance as yf
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingBacktest:
    """
    Comprehensive backtesting framework for multi-stock RL trading agents.

    This class provides a complete backtesting solution for RL-based trading agents,
    including performance metrics computation, detailed trade logging, and batch testing
    capabilities. It's designed specifically for reinforcement learning environments
    and does not rely on traditional backtesting libraries like Backtrader.

    Attributes
    ----------
    config : ConfigTrading
        Configuration object containing trading parameters
    results_cache_dir : str
        Directory for caching backtest results
    metrics : dict
        Computed performance metrics
    trade_history : list
        Detailed history of all trades
    asset_history : list
        Historical asset values over time
    """

    def __init__(self, config_trading):
        """
        Initialize the TradingBacktest with configuration.

        Parameters
        ----------
        config_trading : ConfigTrading
            Configuration object containing trading parameters

        Returns
        -------
        None
            Initializes the instance in place.
        """
        # Inherit configuration for pipeline consistency
        self.config = config_trading
        # Set results cache directory, default to 'results_cache'
        self.results_cache_dir = getattr(self.config, 'RESULTS_CACHE_DIR', 'results_cache')
        # Create directory if not exists for robust file handling
        os.makedirs(self.results_cache_dir, exist_ok=True)
        # Set benchmark data cache directory, default to 'benchmark_cache'
        self.benchmark_cache_dir = getattr(self.config, 'BENCHMARK_CACHE_DIR', 'benchmark_cache')
        # Create directory if not exists for robust file handling
        os.makedirs(self.benchmark_cache_dir, exist_ok=True)

        self.proxy = "127.0.0.1:7897"

        # Initialize result containers as empty
        self.metrics = {}
        self.trade_history = []
        self.asset_history = []
        self.position_history = []
        self.action_history = []

        # Log initialization for traceability
        logging.info("TB Module - Initialized TradingBacktest")
        logging.info(f"TB Module - Results cache directory: {self.results_cache_dir}")

    def _get_nasdaq100_benchmark(self) -> pd.Series:
        """
        Fetch Nasdaq-100 ETF benchmark
        
        Parameters
        ----------
        start_date : str
            start format 'YYYY-MM-DD'
        end_date : str
            end format 'YYYY-MM-DD'
            
        Returns
        -------
        pd.Series
            Nasdaq-100 ETF price series
        """
        try:
            nasdaq_data = None
            # Default get QQQ benchmark
            ticker = "QQQ"
            start = getattr(self.config, 'test_start_date', self.config.start)
            end = getattr(self.config, 'test_end_date', self.config.end)

            start_format = str(start).split()[0]
            end_format = str(end).split()[0]
            # Fetch Nasdaq-100 ETF (QQQ) data from cache
            if not os.path.exists(self.benchmark_cache_dir):
                os.makedirs(self.benchmark_cache_dir)
                logging.info(f"TB Module - _get_nasdaq100_benchmark - Created cache directory: {self.benchmark_cache_dir}")
            benchmark_file_list = os.listdir(self.benchmark_cache_dir)
            if len(benchmark_file_list) > 0:
                logging.info(f"TB Module - _get_nasdaq100_benchmark - Existed benchmark data {benchmark_file_list}")
                for file in benchmark_file_list:
                    if file == f"{ticker}_{start_format}_{end_format}.csv":
                        nasdaq_data = pd.read_csv(f"{self.benchmark_cache_dir}/{file}", index_col=0, parse_dates=True)
                        logging.info(f"TB Module - _get_nasdaq100_benchmark - Fetched Nasdaq-100 data from cache: {nasdaq_data.head()}")
                        return nasdaq_data
                    
            # Fetch Nasdaq-100 ETF (QQQ) data by yfinance
            nasdaq_data = yf.download(ticker, start=start, end=end, proxy=self.proxy)
            
            if nasdaq_data.empty:
                logging.warning("TB Module - _get_nasdaq100_benchmark - Failed to fetch Nasdaq-100 data, using fallback")
                return None
            
            # Initial DataResource instance for cleaning raw data
            from .data_resource import DataResource
            dr = DataResource(self.config)

            # Clean raw data, convert columns name to ['Adj_Close_QQQ', ...]
            nasdaq_data = dr.clean_yf_ohlcv(nasdaq_data, ticker)
            nasdaq_data = nasdaq_data[f"Adj_Close_{ticker}"]
            logging.debug(f"TB Module - _get_nasdaq100_benchmark - Downloaded Nasdaq-100 benchmark data: {nasdaq_data.head()}")
            # Save and return Adj_Close price
            nasdaq_data.to_csv(f"{self.benchmark_cache_dir}/{ticker}_{start_format}_{end_format}.csv", index=True)
            return nasdaq_data
            
        except Exception as e:
            logging.error(f"TB Module -  _get_nasdaq100_benchmark - Error fetching Nasdaq-100 benchmark: {e}")
            return None

    def _calculate_benchmark_returns(self, benchmark_prices: pd.Series) -> np.ndarray:
        """
        Calculate Benchmark returns
        
        Parameters
        ----------
        benchmark_prices : pd.Series
            Benchmark price series
            
        Returns
        -------
        np.ndarray
            Benchmark returns
        """
        if benchmark_prices is None or len(benchmark_prices) < 2:
            return np.array([])
        
        # Calculate daily returns
        benchmark_returns = benchmark_prices.pct_change().dropna()
        return benchmark_returns.values

    def run_backtest(self, agent, test_env, render=False, record_trades=True, use_benchmark=True):
        """Run backtest on test environment with trained agent.

        Parameters
        ----------
        agent : TradingAgent
            Trained trading agent.
        test_env : StockTradingEnv
            Test environment for backtesting.
        render : bool, optional
            Whether to render environment steps. Default is False.
        record_trades : bool, optional
            Whether to record detailed trade history. Default is True.
        use_benchmark : bool, optional
            Whether to compute benchmark-relative metrics. Default is True.

        Returns
        -------
        dict
            Backtest results including metrics and history.
        """
        try:
            # Log backtest start
            logging.info("TB Module - Starting backtest")

            # Reset environment to initial state
            state, _ = test_env.reset()

            # Initialize loop variables
            done = False
            truncated = False
            step_count = 0

            # Clear histories if recording
            if record_trades:
                self.trade_history = []
                self.asset_history = []
                self.position_history = []
                self.action_history = []

            # Main backtest loop
            while not (done or truncated):
                # Get deterministic action from agent
                action, _ = agent.predict(state, deterministic=True)

                # Step environment with action
                next_state, reward, done, truncated, info = test_env.step(action)

                # Record detailed trade info if enabled
                if record_trades:
                    trade_record = {
                        'step': step_count,
                        'date': info.get('Date', None),  # Try to fetch Date from info
                        'action': info.get('Final Actions', []).copy(),  # Copy to avoid reference changes
                        'position': info.get('Position', []).copy(),
                        'cash': info.get('Cash', 0.0),
                        'total_asset': info.get('Total Asset', 0.0),
                        'reward': reward,
                    }
                    self.trade_history.append(trade_record)
                    # Note: Asset history recorded here is redundant with test_env.asset_memory[1:]
                    # We will use test_env.asset_memory for consistency and completeness
                    self.asset_history.append(info.get('Total Asset', 0.0))
                    self.position_history.append(info.get('Position', []).copy())
                    # Action history recording total_asset seems like a typo, but kept for compatibility
                    self.action_history.append(info.get('Total Asset', 0.0))

                # Update state and counter
                state = next_state
                step_count += 1

            # Data alignment and build results
            # Get entire data with asset and date information from test_env
            logging.info("TB Module - Backtest loop finished. Starting data alignment and result compilation.")

            # Get asset memory and trading dates
            full_asset_memory = getattr(test_env, 'asset_memory', [])
            full_trading_dates = getattr(test_env, 'trading_dates', [])
            
            logging.info(f"TB Module - Collected full asset memory length: {len(full_asset_memory)}")
            logging.info(f"TB Module - Collected full trading dates length: {len(full_trading_dates)}")

            # Calculate EXACT trading steps
            # Actually trading step = Environment executed steps = len(asset_memory) - 1 (asset_memory[0] records initial asset)
            actual_executed_steps = len(full_asset_memory) - 1 if len(full_asset_memory) > 0 else 0
            logging.info(f"TB Module - Determined actual executed trading steps: {actual_executed_steps}")

            # Align asset and dates based on actually executed steps
            # Get strategy asset (without initial asset)
            strategy_asset_values_raw = full_asset_memory[1:] if len(full_asset_memory) > 1 else []
            logging.debug(f"TB Module - Raw strategy asset values count (after initial): {len(strategy_asset_values_raw)}")

            # Get dates of executed steps
            # Get actual_executed_steps dates
            if full_trading_dates and len(full_trading_dates) >= actual_executed_steps:
                aligned_trading_dates_final = full_trading_dates[:actual_executed_steps]
                logging.debug(f"TB Module - Truncated trading dates to match executed steps. New count: {len(aligned_trading_dates_final)}")
            elif full_trading_dates:
                # If trading dates less than executed steps
                logging.warning(f"TB Module - Pre-generated dates ({len(full_trading_dates)}) < executed steps ({actual_executed_steps}). "
                                f"Using all available dates ({len(full_trading_dates)}). Adjusting asset count.")
                aligned_trading_dates_final = full_trading_dates
                # Adjust length of asset list to align with dates length
                strategy_asset_values_raw = strategy_asset_values_raw[:len(aligned_trading_dates_final)]
                actual_executed_steps = len(aligned_trading_dates_final) # Update steps
            else:
                # If no dates information
                aligned_trading_dates_final = []
                logging.warning("TB Module - No trading dates available from env.")

            # Final chekt
            final_asset_count = len(strategy_asset_values_raw)
            final_date_count = len(aligned_trading_dates_final)
            
            if final_asset_count != final_date_count:
                logging.warning(f"TB Module - Final alignment mismatch before Series creation: "
                                f"Assets ({final_asset_count}) vs Dates ({final_date_count}). "
                                f"Using minimum length to force strict alignment.")
                min_final_length = min(final_asset_count, final_date_count)
                # Force clipping to same length
                final_aligned_assets = strategy_asset_values_raw[:min_final_length]
                final_aligned_dates = aligned_trading_dates_final[:min_final_length]
            else:
                # If same length
                final_aligned_assets = strategy_asset_values_raw
                final_aligned_dates = aligned_trading_dates_final

            logging.info(f"TB Module - Final aligned data - Assets: {len(final_aligned_assets)}, Dates: {len(final_aligned_dates)}")

            # Create strategy asset Series with date index
            if (final_aligned_dates and final_aligned_assets and 
                len(final_aligned_dates) == len(final_aligned_assets)):
                try:
                    # Create Pandas Series with date index, ensuring length match
                    strategy_assets_series = pd.Series(
                        final_aligned_assets, 
                        index=final_aligned_dates, 
                        dtype=np.float32
                    )
                    # Ensure index is DatetimeIndex
                    # strategy_assets_series.index = pd.to_datetime(strategy_assets_series.index) 
                    strategy_assets_series.sort_index(inplace=True) # Ensure sorted
                    logging.info(f"TB Module - Successfully created strategy asset series with date index. "
                                f"Length: {len(strategy_assets_series)}, "
                                f"Date range: {strategy_assets_series.index.min()} to {strategy_assets_series.index.max()}")
                except Exception as e:
                    logging.error(f"TB Module - Failed to create strategy asset series: {e}")
                    strategy_assets_series = pd.Series(dtype=np.float32) # 返回空的 Series
            else:
                logging.warning("TB Module - Could not create strategy asset series due to missing or mismatched final data.")
                strategy_assets_series = pd.Series(dtype=np.float32) # 返回空的 Series

            # Calculate benchmark returns for metrics
            benchmark_returns_for_metrics = None  # Pass to _compute_metrics()
            benchmark_prices_with_date = None  # For visualization based on Date index

            # Get benchmark data
            if use_benchmark:
                try:
                    # Fetch Nasdaq-100 Benchmark data
                    benchmark_prices_with_date = self._get_nasdaq100_benchmark()
                    if benchmark_prices_with_date is not None and not benchmark_prices_with_date.empty:
                        # Ensure benchmark data with date index
                        benchmark_prices_with_date.index = pd.to_datetime(benchmark_prices_with_date.index)
                        benchmark_prices_with_date.sort_index(inplace=True)

                        # Align strategy asset dates range with benchmark
                        if not strategy_assets_series.empty:
                            # Get strategy asset dates range
                            strategy_start_date = strategy_assets_series.index.min()
                            strategy_end_date = strategy_assets_series.index.max()

                            # Use strategy asset dates range to clip benchmark
                            benchmark_prices_with_date = benchmark_prices_with_date[
                                (benchmark_prices_with_date.index >= strategy_start_date) &
                                (benchmark_prices_with_date.index <= strategy_end_date)
                            ]

                            if not benchmark_prices_with_date.empty:
                                # Calculate benchmark returns based on aligned prices
                                benchmark_returns_for_metrics = self._calculate_benchmark_returns(benchmark_prices_with_date)
                                logging.info(f"TB Module - Benchmark data aligned and fetched, "
                                            f"{len(benchmark_returns_for_metrics)} returns calculated. "
                                            f"Aligned date range: {benchmark_prices_with_date.index.min()} to {benchmark_prices_with_date.index.max()}")
                            else:
                                logging.warning("TB Module - Benchmark data is empty after alignment with strategy dates.")
                        else:
                            # Fallback if no strategy data
                            benchmark_returns_for_metrics = self._calculate_benchmark_returns(benchmark_prices_with_date)
                            logging.info(f"TB Module - Benchmark data fetched (no strategy alignment), "
                                        f"{len(benchmark_returns_for_metrics)} returns calculated.")
                    else:
                        logging.warning("TB Module - Failed to fetch or process benchmark data.")
                except Exception as e:
                    logging.warning(f"TB Module - Error processing benchmark for metrics: {e}", exc_info=True)

            # Compute and store metrics
            metrics = self._compute_metrics(benchmark_returns_for_metrics)
            logging.info("TB Module - Backtest completed")
            self.metrics = metrics

            # Update results dict
            logging.warning(f"TB Module - run_backtest - Before return results, print {benchmark_prices_with_date.head()}")
            results = {
                'metrics': metrics,
                'trade_history': self.trade_history if record_trades else [],
                'asset_history': self.asset_history if record_trades else [], # Keep original asset history
                'position_history': self.position_history if record_trades else [],
                'action_history': self.action_history if record_trades else [],
                'total_steps': step_count,
                # Update episode_length actually trading times
                # test_env.asset_memory includes initial asset and the asset after trading
                # Total trading times = len(asset_memory) - 1
                'episode_length': actual_executed_steps, 
                # Key update: Pandas Series aligned strategy assets with date index
                'strategy_assets_with_date': strategy_assets_series, 
                # Update benchmarkd 
                'benchmark_prices_with_date': benchmark_prices_with_date, 
                # Update benchmark returns
                'benchmark_returns': benchmark_returns_for_metrics 
            }
            
            # Debug
            logging.debug(f"TB Module - run_backtest - Final results keys: {list(results.keys())}")
            logging.debug(f"TB Module - run_backtest - strategy_assets_with_date exists: {'strategy_assets_with_date' in results}")
            if 'strategy_assets_with_date' in results:
                series_obj = results['strategy_assets_with_date']
                if isinstance(series_obj, pd.Series):
                    logging.debug(f"  - 'strategy_assets_with_date' is a pandas Series")
                    logging.debug(f"  - Series is empty: {series_obj.empty}")
                    if not series_obj.empty:
                        logging.debug(f"  - Series length: {len(series_obj)}")
                        logging.debug(f"  - Series index type: {type(series_obj.index)}")
                        logging.debug(f"  - Series index length: {len(series_obj.index)}")
                        logging.debug(f"  - Sample index values: {series_obj.index[:3] if len(series_obj.index) >= 3 else series_obj.index}")
                        logging.debug(f"  - Sample series values: {series_obj.iloc[:3].tolist() if len(series_obj) >= 3 else series_obj.tolist()}")
                    else:
                        logging.debug(f"  - Series is EMPTY")
                else:
                    logging.debug(f"  - 'strategy_assets_with_date' is NOT a pandas Series, type is: {type(series_obj)}")
            else:
                logging.debug("  - 'strategy_assets_with_date' key is MISSING from results dictionary")

            logging.debug(f"TB Module - run_backtest - All keys in results dictionary: {list(results.keys())}")

            return results

        except Exception as e:
            # Log and re-raise for upstream handling
            logging.error(f"TB Module - Error during backtest: {e}", exc_info=True)
            raise


    def _compute_metrics(self, benchmark_returns: np.ndarray = None):
        """
        Compute comprehensive performance metrics from backtest results.

        Parameters
        ----------
        benchmark_returns : np.ndarray, optional
            Benchmark returns for relative performance comparison

        Returns
        -------
        dict
            Dictionary of computed performance metrics
        """
        try:
            # Validate asset history
            if not self.asset_history:
                raise ValueError("TB Module - No asset history available for metrics computation")

            # Convert to numpy for vectorized ops
            assets = np.array(self.asset_history)
            # Compute daily returns
            returns = np.diff(assets) / assets[:-1] if len(assets) > 1 else np.array([0.0])

            # Handle no returns case
            if len(returns) == 0:
                returns = np.array([0.0])

            # Log computation start
            logging.info(f"TB Module - Computing metrics for {len(returns)} returns")

            # Total Return: overall growth
            total_return = (assets[-1] / assets[0] - 1) if len(assets) > 1 else 0.0

            # CAGR: annualized compounding
            if len(assets) > 1:
                total_days = len(assets)
                cagr = (assets[-1] / assets[0]) ** (252 / total_days) - 1  # 252 trading days
            else:
                cagr = 0.0

            # Volatility: std dev annualized
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

            # Sharpe Ratio: risk-adjusted, assume rf=0
            risk_free_rate = 0.0  # Simplification; can be from config
            sharpe_ratio = (np.mean(returns) - risk_free_rate) / (np.std(returns) + 1e-8) * np.sqrt(252) \
                          if len(returns) > 1 and np.std(returns) > 0 else 0.0

            # Maximum Drawdown: peak-to-trough
            if len(assets) > 1:
                rolling_max = np.maximum.accumulate(assets)
                drawdown = (assets - rolling_max) / (rolling_max + 1e-8)
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0.0

            # Calmar Ratio: CAGR / |MDD|
            calmar_ratio = cagr / (abs(max_drawdown) + 1e-8) if max_drawdown < 0 else 0.0

            # Profit Factor and Win Rate: trade efficiency
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            total_positive = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
            total_negative = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-8

            profit_factor = total_positive / total_negative if total_negative > 0 else 0.0
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

            # CVaR at 5%: tail risk
            if len(returns) > 10:  # Sufficient data threshold
                alpha = 0.05  # 5% CVaR
                var_index = int(alpha * len(returns))
                sorted_returns = np.sort(returns)
                cvar = np.mean(sorted_returns[:var_index]) if var_index > 0 else np.min(returns)
            else:
                cvar = np.min(returns) if len(returns) > 0 else 0.0

            # Additional: Annual Return
            annual_return = np.mean(returns) * 252 if len(returns) > 0 else 0.0

            # Sortino Ratio: downside-focused
            negative_returns_squared = np.square(returns[returns < 0])
            downside_deviation = np.sqrt(np.mean(negative_returns_squared)) * np.sqrt(252) \
                               if len(negative_returns_squared) > 0 else 1e-8
            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation \
                          if downside_deviation > 0 else 0.0

            # Max Consecutive Wins/Losses: streak analysis
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_win_streak = 0
            current_loss_streak = 0

            for ret in returns:
                if ret > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
                elif ret < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
                else:
                    current_win_streak = 0
                    current_loss_streak = 0

            # Volatility-Adjusted Return: return per vol unit
            volatility_adjusted_return = annual_return / (volatility + 1e-8) if volatility > 0 else 0.0

            # Initial benchmark metrics
            metrics = {
                'total_return': float(total_return),
                'cagr': float(cagr),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'calmar_ratio': float(calmar_ratio),
                'profit_factor': float(profit_factor),
                'win_rate': float(win_rate),
                'cvar_5_percent': float(cvar),
                'max_consecutive_wins': int(max_consecutive_wins),
                'max_consecutive_losses': int(max_consecutive_losses),
                'volatility_adjusted_return': float(volatility_adjusted_return),
                'total_steps': len(returns),
                'final_asset': float(assets[-1]) if len(assets) > 0 else 0.0,
                'initial_asset': float(assets[0]) if len(assets) > 0 else 0.0
            }

            # Calculate relative metrics based on benchmark data
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align lengths ，take the shorter one
                min_length = min(len(returns), len(benchmark_returns))
                aligned_strategy_returns = returns[:min_length]
                aligned_benchmark_returns = benchmark_returns[:min_length]

                # Alpha and Beta compute
                if len(aligned_strategy_returns) > 1 and np.std(aligned_benchmark_returns) > 0:
                    if isinstance(aligned_benchmark_returns, np.ndarray):
                        logging.warning(f"TB Module - _compute_metrics - Sample of aligned_benchmark_returns: {aligned_benchmark_returns[:5] if len(aligned_benchmark_returns) >= 5 else aligned_benchmark_returns}")
                    if isinstance(aligned_strategy_returns, np.ndarray):
                        logging.warning(f"TB Module - _compute_metrics - Sample of aligned_strategy_returns: {aligned_strategy_returns[:5] if len(aligned_strategy_returns) >= 5 else aligned_strategy_returns}")
                    
                    # Ensure shape is 1D array
                    if hasattr(aligned_benchmark_returns, 'ravel'):
                        aligned_benchmark_returns = aligned_benchmark_returns.ravel()
                    if hasattr(aligned_strategy_returns, 'ravel'):
                        aligned_strategy_returns = aligned_strategy_returns.ravel()
                    
                    logging.warning(f"TB Module - _compute_metrics - After ravel - Shape of aligned_benchmark_returns: {getattr(aligned_benchmark_returns, 'shape', 'No shape attr')}")
                    logging.warning(f"TB Module - _compute_metrics - After ravel - Shape of aligned_strategy_returns: {getattr(aligned_strategy_returns, 'shape', 'No shape attr')}")
                    # Calculate Alpha and Beta using linear regression
                    beta, alpha = np.polyfit(aligned_benchmark_returns, aligned_strategy_returns, 1)
                    alpha_annualized = alpha * 252  # Calculate annualized Alpha
                else:
                    beta = 0.0
                    alpha_annualized = 0.0

                # Information ratio
                tracking_error = np.std(aligned_strategy_returns - aligned_benchmark_returns) * np.sqrt(252)
                information_ratio = (np.mean(aligned_strategy_returns - aligned_benchmark_returns) * 252) / (tracking_error + 1e-8) \
                                  if tracking_error > 0 else 0.0

                # Excess returns
                excess_return = np.mean(aligned_strategy_returns - aligned_benchmark_returns) * 252

                # Benchmark related metrics
                benchmark_total_return = (1 + aligned_benchmark_returns).prod() - 1
                benchmark_volatility = np.std(aligned_benchmark_returns) * np.sqrt(252)
                benchmark_cagr = (1 + benchmark_total_return) ** (252 / len(aligned_benchmark_returns)) - 1

                # Relative win rate
                relative_win_rate = np.mean(aligned_strategy_returns > aligned_benchmark_returns)

                # Add benchmark-related metrics
                metrics.update({
                    'benchmark_total_return': float(benchmark_total_return),
                    'benchmark_cagr': float(benchmark_cagr),
                    'benchmark_volatility': float(benchmark_volatility),
                    'alpha': float(alpha_annualized),
                    'beta': float(beta),
                    'information_ratio': float(information_ratio),
                    'excess_return': float(excess_return),
                    'relative_win_rate': float(relative_win_rate),
                    'tracking_error': float(tracking_error)
                })

            # Log key metrics summary
            logging.info("TB Module - Metrics computation completed")
            logging.info(f"TB Module - Key metrics - CAGR: {cagr:.4f}, Sharpe: {sharpe_ratio:.4f}, Max Drawdown: {max_drawdown:.4f}")

            return metrics

        except Exception as e:
            # Log and re-raise
            logging.error(f"TB Module - Error computing metrics: {e}")
            raise

    def batch_backtest(self, agent_configs: List[Tuple], test_env):
        """
        Run batch backtest for multiple agent configurations.

        Parameters
        ----------
        agent_configs : list of tuples
            List of (agent, name) tuples for batch testing
        test_env : StockTradingEnv
            Test environment for backtesting

        Returns
        -------
        dict
            Batch backtest results with comparison metrics

        Notes
        -----
        - Runs independent backtests per agent, stores individual results.
        - Generates comparison report with rankings.
        """
        try:
            # Log batch start
            logging.info(f"TB Module - Starting batch backtest for {len(agent_configs)} agents")

            # Initialize results dict
            batch_results = {}

            # Loop over agents
            for i, (agent, name) in enumerate(agent_configs):
                # Log current agent
                logging.info(f"TB Module - Running backtest {i+1}/{len(agent_configs)}: {name}")

                # Run single backtest
                results = self.run_backtest(agent, test_env, record_trades=True)

                # Store in batch
                batch_results[name] = {
                    'metrics': results['metrics'],
                    'trade_history': results['trade_history'],
                    'asset_history': results['asset_history']
                }

                # Log agent summary
                logging.info(f"TB Module - Completed backtest for {name}")
                logging.info(f"TB Module - {name} - CAGR: {results['metrics']['cagr']:.4f}, Sharpe: {results['metrics']['sharpe_ratio']:.4f}")

            # Generate and add comparison
            comparison_report = self._generate_comparison_report(batch_results)

            final_results = {
                'individual_results': batch_results,
                'comparison_report': comparison_report
            }

            return final_results

        except Exception as e:
            # Log and re-raise
            logging.error(f"TB Module - Error during batch backtest: {e}")
            raise

    def _generate_comparison_report(self, batch_results):
        """
        Generate comparison report from batch backtest results.

        Parameters
        ----------
        batch_results : dict
            Results from batch backtest

        Returns
        -------
        dict
            Comparison report with rankings and statistics

        Notes
        -----
        - Uses pd.DataFrame for easy ranking (ascending/descending per metric).
        - Overall Rank as mean of key ranks; sorts df by it.
        - Extensible: Add metrics to key_metrics list.
        """
        try:
            # Collect data for df
            comparison_data = []
            for name, results in batch_results.items():
                metrics = results['metrics']
                comparison_data.append({
                    'Algorithm': name,
                    'CAGR': metrics['cagr'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Calmar Ratio': metrics['calmar_ratio'],
                    'Profit Factor': metrics['profit_factor'],
                    'Win Rate': metrics['win_rate'],
                    'CVaR (5%)': metrics['cvar_5_percent'],
                    'Volatility': metrics['volatility'],
                    'Total Return': metrics['total_return']
                })

            # Create df
            df = pd.DataFrame(comparison_data)

            # Define key metrics for ranking
            rankings = {}
            key_metrics = ['CAGR', 'Sharpe Ratio', 'Calmar Ratio', 'Profit Factor']

            for metric in key_metrics:
                # Rank higher-better metrics descending
                if metric in ['CAGR', 'Sharpe Ratio', 'Calmar Ratio', 'Profit Factor']:
                    df[f'{metric} Rank'] = df[metric].rank(ascending=False)
                # Rank lower-better ascending (though not in key, for completeness)
                elif metric in ['Max Drawdown', 'CVaR (5%)']:
                    df[f'{metric} Rank'] = df[metric].rank(ascending=True)

            # Compute overall rank as mean of rank columns
            rank_columns = [col for col in df.columns if 'Rank' in col]
            if rank_columns:
                df['Overall Rank'] = df[rank_columns].mean(axis=1)
                df = df.sort_values('Overall Rank')

            # Compile report
            comparison_report = {
                'summary_table': df.to_dict('records'),
                'best_performer': df.iloc[0]['Algorithm'] if len(df) > 0 else None,
                'rankings': df.set_index('Algorithm')[rank_columns + ['Overall Rank']].to_dict('index') if rank_columns else {}
            }

            # Log success
            logging.info("TB Module - Comparison report generated")
            return comparison_report

        except Exception as e:
            # Log and re-raise
            logging.error(f"TB Module - Error generating comparison report: {e}")
            raise

    def save_results(self, results, filename=None):
        """
        Save backtest results to file.

        Parameters
        ----------
        results : dict
            Backtest results to save
        filename : str, optional
            Custom filename (defaults to auto-generated)

        Returns
        -------
        str
            Path to saved results file

        Notes
        -----
        - Uses pickle for dict serialization.
        - Timestamped filename for versioning.
        """
        try:
            # Generate timestamped name if none
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"backtest_results_{timestamp}.pkl"

            # Construct path
            save_path = os.path.join(self.results_cache_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save with pickle
            with open(save_path, 'wb') as f:
                import pickle
                pickle.dump(results, f)

            # Log path
            logging.info(f"TB Module - Results saved to: {save_path}")
            return save_path

        except Exception as e:
            # Log and re-raise
            logging.error(f"TB Module - Error saving results: {e}")
            raise

    def load_results(self, filepath):
        """
        Load backtest results from file.

        Parameters
        ----------
        filepath : str
            Path to saved results file

        Returns
        -------
        dict
            Loaded backtest results
        """
        try:
            # Load with pickle
            with open(filepath, 'rb') as f:
                import pickle
                results = pickle.load(f)

            # Log success
            logging.info(f"TB Module - Results loaded from: {filepath}")
            return results

        except Exception as e:
            # Log and re-raise
            logging.error(f"TB Module - Error loading results: {e}")
            raise

    def generate_detailed_report(self, results, report_name=None):
        """
        Generate detailed backtest report with metrics and analysis.

        Parameters
        ----------
        results : dict
            Backtest results
        report_name : str, optional
            Custom report name

        Returns
        -------
        dict
            Detailed report with formatted metrics and analysis

        Notes
        -----
        - Formats metrics with %/$ for readability.
        - Includes qualitative analysis_summary and recommendation.
        """
        try:
            metrics = results.get('metrics', {})

            # Format metrics into readable dict
            formatted_metrics = {
                'Performance Metrics': {
                    'Total Return': f"{metrics.get('total_return', 0)*100:.2f}%",
                    'CAGR': f"{metrics.get('cagr', 0)*100:.2f}%",
                    'Annual Return': f"{metrics.get('annual_return', 0)*100:.2f}%",
                    'Final Asset Value': f"${metrics.get('final_asset', 0):,.2f}",
                    'Initial Asset Value': f"${metrics.get('initial_asset', 0):,.2f}"
                },
                'Risk Metrics': {
                    'Volatility (Annualized)': f"{metrics.get('volatility', 0)*100:.2f}%",
                    'Maximum Drawdown': f"{metrics.get('max_drawdown', 0)*100:.2f}%",
                    'CVaR (5%)': f"{metrics.get('cvar_5_percent', 0)*100:.2f}%",
                    'Max Consecutive Losses': f"{metrics.get('max_consecutive_losses', 0)}"
                },
                'Risk-Adjusted Metrics': {
                    'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.4f}",
                    'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.4f}",
                    'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.4f}",
                    'Volatility-Adjusted Return': f"{metrics.get('volatility_adjusted_return', 0):.4f}"
                },
                'Trade Metrics': {
                    'Profit Factor': f"{metrics.get('profit_factor', 0):.4f}",
                    'Win Rate': f"{metrics.get('win_rate', 0)*100:.2f}%",
                    'Max Consecutive Wins': f"{metrics.get('max_consecutive_wins', 0)}",
                    'Total Trading Steps': f"{metrics.get('total_steps', 0)}"
                }
            }

            # Generate summary
            analysis_summary = self._generate_analysis_summary(metrics)

            # Compile report
            detailed_report = {
                'report_name': report_name or f"Backtest Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'generated_at': datetime.now().isoformat(),
                'formatted_metrics': formatted_metrics,
                'raw_metrics': metrics,
                'analysis_summary': analysis_summary,
                'trade_history': results.get('trade_history', []),
                'asset_history': results.get('asset_history', [])
            }

            return detailed_report

        except Exception as e:
            # Log and re-raise
            logging.error(f"TB Module - Error generating detailed report: {e}")
            raise

    def _generate_analysis_summary(self, metrics):
        """
        Generate analysis summary based on computed metrics.

        Parameters
        ----------
        metrics : dict
            Computed performance metrics

        Returns
        -------
        dict
            Analysis summary with interpretations

        Notes
        -----
        - Threshold-based qualitative assessments (e.g., 'Excellent' for CAGR >20%).
        - Calls _generate_recommendation for final advice.
        """
        try:
            summary = {}

            # Performance: threshold on CAGR
            cagr = metrics.get('cagr', 0)
            if cagr > 0.2:
                performance = "Excellent"
            elif cagr > 0.1:
                performance = "Good"
            elif cagr > 0.05:
                performance = "Moderate"
            elif cagr > 0:
                performance = "Poor"
            else:
                performance = "Negative"

            # Risk: on abs(MDD)
            max_dd = abs(metrics.get('max_drawdown', 0))
            if max_dd < 0.1:
                risk_level = "Low"
            elif max_dd < 0.2:
                risk_level = "Moderate"
            elif max_dd < 0.3:
                risk_level = "High"
            else:
                risk_level = "Very High"

            # Risk-Return: on Sharpe
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 2:
                risk_return = "Excellent"
            elif sharpe > 1:
                risk_return = "Good"
            elif sharpe > 0.5:
                risk_return = "Moderate"
            else:
                risk_return = "Poor"

            # Consistency: on win_rate/profit_factor
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            if win_rate > 0.6 and profit_factor > 2:
                consistency = "High"
            elif win_rate > 0.5 and profit_factor > 1.5:
                consistency = "Moderate"
            else:
                consistency = "Low"

            summary = {
                'performance_assessment': f"{performance} - CAGR of {cagr*100:.2f}%",
                'risk_assessment': f"{risk_level} - Max Drawdown of {max_dd*100:.2f}%",
                'risk_return_assessment': f"{risk_return} - Sharpe Ratio of {sharpe:.2f}",
                'consistency_assessment': f"{consistency} - Win Rate: {win_rate*100:.1f}%, Profit Factor: {profit_factor:.2f}",
                'recommendation': self._generate_recommendation(performance, risk_level, risk_return, consistency)
            }

            return summary

        except Exception as e:
            # Log and return empty
            logging.error(f"TB Module - Error generating analysis summary: {e}")
            return {}

    def _generate_recommendation(self, performance, risk_level, risk_return, consistency):
        """
        Generate recommendation based on assessment.

        Parameters
        ----------
        performance : str
            Performance assessment
        risk_level : str
            Risk level assessment
        risk_return : str
            Risk-return assessment
        consistency : str
            Consistency assessment

        Returns
        -------
        str
            Recommendation

        Notes
        -----
        - Rule-based: combines assessments for 'Buy'/'Sell' advice.
        """
        try:
            # Rule-based logic
            if performance in ["Excellent", "Good"] and risk_level in ["Low", "Moderate"] and risk_return in ["Excellent", "Good"]:
                return "Strong Buy - Excellent risk-adjusted returns with acceptable risk levels"
            elif performance in ["Moderate"] and risk_level in ["Low", "Moderate"] and risk_return in ["Moderate", "Good"]:
                return "Buy - Good performance with reasonable risk"
            elif performance in ["Poor", "Negative"] or risk_level in ["High", "Very High"]:
                return "Sell/Hold - Poor performance or excessive risk"
            else:
                return "Hold - Mixed signals, monitor closely"

        except Exception as e:
            # Log and default
            logging.error(f"TB Module - Error generating recommendation: {e}")
            return "Hold - Unable to generate recommendation"

    def plot_performance_comparison(self, results, benchmark_prices=None):
        """
        Generate strategy vs benchmark performance comparison plot
        
        Parameters
        ----------
        results : dict
            Backtest results
        benchmark_prices : pd.Series, optional
            Benchmark price series
        """
        try:
            asset_history = results.get('asset_history', [])
            if not asset_history:
                logging.warning("TB Module - No asset history for plotting")
                return

            # Generate figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Normalize asset values to initial value of 1
            initial_asset = asset_history[0] if len(asset_history) > 0 else 1
            normalized_strategy = [asset / initial_asset for asset in asset_history]
            
            # Strategy cumulative returns curve
            ax1.plot(normalized_strategy, label='Trading Strategy', linewidth=2)
            
            # Plot benchmark curve 
            if benchmark_prices is not None and len(benchmark_prices) > 0:
                # Normalize benchmark prices
                normalized_benchmark = benchmark_prices / benchmark_prices.iloc[0]
                # Align lengths
                min_length = min(len(normalized_strategy), len(normalized_benchmark))
                ax1.plot(normalized_benchmark.iloc[:min_length].values, 
                        label='Nasdaq-100 Benchmark', linewidth=2)
            
            ax1.set_title('Performance Comparison: Strategy vs Benchmark')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Normalized Value (Initial = 1.0)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot maximum drawdown curve
            strategy_assets = np.array(asset_history)
            if len(strategy_assets) > 1:
                rolling_max = np.maximum.accumulate(strategy_assets)
                drawdown = (strategy_assets - rolling_max) / (rolling_max + 1e-8) * 100
                
                ax2.plot(drawdown, label='Strategy Drawdown', color='red', linewidth=2)
                ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
                
                ax2.set_title('Drawdown Analysis')
                ax2.set_xlabel('Trading Days')
                ax2.set_ylabel('Drawdown (%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"TB Module - Error plotting performance comparison: {e}")

# Utility functions for common backtest scenarios
def run_single_backtest(config_trading, agent, test_env, save_results=True):
    """
    Utility function to run a single backtest with standard configuration.

    Parameters
    ----------
    config_trading : ConfigTrading
        Trading configuration
    agent : TradingAgent
        Trained agent
    test_env : StockTradingEnv
        Test environment
    save_results : bool, optional
        Whether to save results

    Returns
    -------
    dict
        Backtest results

    Notes
    -----
    - Wrapper for TradingBacktest.run_backtest with defaults.
    """
    # Instantiate backtester
    backtester = TradingBacktest(config_trading)
    # Run and optionally save
    results = backtester.run_backtest(agent, test_env, record_trades=True)

    if save_results:
        backtester.save_results(results)

    return results

def run_batch_comparison(config_trading, agent_configs, test_env, save_results=True):
    """
    Utility function to run batch comparison of multiple agents.

    Parameters
    ----------
    config_trading : ConfigTrading
        Trading configuration
    agent_configs : list
        List of (agent, name) tuples
    test_env : StockTradingEnv
        Test environment
    save_results : bool, optional
        Whether to save results

    Returns
    -------
    dict
        Batch comparison results

    Notes
    -----
    - Wrapper for batch_backtest with timestamped save.
    """
    # Instantiate
    backtester = TradingBacktest(config_trading)
    # Run backtest
    results = backtester.batch_backtest(agent_configs, test_env)

    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_backtest_results_{timestamp}.pkl"
        backtester.save_results(results, filename)

    return results