# trading_backtest.py
# Module: Backtest
# Purpose: Backtest models for modes on test data.
# Design: Run bt.Strategy with model.predict; output logs df/figs like FinRL.
# Linkage: config_trade; modes_data['test']; models_paths.
# Robustness: Date filter in df; log errors.
# Outputs: df to_string for tables; plt save for pics.
# Updates: Added symbol to results_save_dir (results_cache/{symbol}) to avoid overwrite; enhanced logging with symbol; created subdir for per-symbol isolation.
# Updates: Integrated DRL_prediction-like logic to simulate prediction in env (not backtrader), collect account_value/actions, reference from FinRL (DRL_prediction: run episode, save asset/action memory); added DJI benchmark comparison in run, reference from FinRL backtest.ipynb (dji index plot); plot account_value like FinRL.
# Updates: Dynamic model loading based on exper_data[mode]['model_type'], using self.model_classes for flexibility; this allows loading different models without hardcoding, reference from FinRL (DRLAgent.DRL_prediction: load model dynamically).
# Updates: Replaced Pendulum-v1 dummy_env with StockTradingEnv using empty rl_data to match observation/action spaces, fixing space mismatch error during model loading (ValueError: Observation spaces do not match).
# Updates: Extended compute_metrics with Information Ratio (sharpe-like but vs benchmark), CVaR (worst alpha% mean), Rachev Ratio (CVaR_positive / CVaR_negative), reference from FinRL_DeepSeek (Table 1-3: IR, CVaR, Rachev); added benchmark (Nasdaq-100 or DJI) fetch in run; plot multiple curves like Figure 1-4; table with all metrics.

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import yfinance as yf  # For benchmark download
from collections import deque   # Import deque from collections for history buffer
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC  # Import classes for dynamic load
from finbert_trader.agent.trading_agent import CPPO     # reference from FinRL_DeepSeek
from finbert_trader.environment.stock_trading_env import StockTradingEnv  # For dummy env and simulation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RLStrategy(bt.Strategy):
    params = (('model', None),)

    def __init__(self):
        self.model = self.p.model
        self.actions = []  # Collect actions for entropy/win_rate
        self.window_size = 50  # From config, align with env
        self.features_per_time = 15  # OHLCV 5 + ind 8 + sent + risk
        self.feature_history = deque(maxlen=self.window_size)  # Buffer for window
        # Multi-stock: map datas by name=symbol
        self.symbol_datas = {date._name: date for date in self.datas}  # Dict of per-symbol data
        self.num_symbols = len(self.symbol_datas)  # Number of symbols
        self.current_positions = np.zeros(self.num_symbols)  # Positions array (updated in next())

    def next(self):
        # Multi-stock: collect features per symbol into multi array
        features_list = []
        for symbol_idx, (symbol, data) in enumerate(self.symbol_datas.items()):
            sym_features = np.array([
                data.open[0],
                data.high[0],
                data.low[0],
                data.close[0],
                data.volume[0],
                data.macd[0],
                data.boll_ub[0],
                data.boll_lb[0],
                data.rsi_30[0],
                data.cci_30[0],
                data.dx_30[0],
                data.close_30_sma[0],
                data.close_60_sma[0],
                data.sentiment_score[0],
                data.risk_score[0]  # Added risk_score
            ])
            sym_features = np.nan_to_num(sym_features, nan=0.0)  # Clean NaN
            features_list.append(sym_features)

        features = np.concatenate(features_list)  # Concat to multi-features (15*num_symbols)
        
        # Append to history (now multi-features per time)
        self.feature_history.append(features)
        
        # Build window with padding
        if len(self.feature_history) < self.window_size:
            pad = np.zeros((self.window_size - len(self.feature_history), self.features_per_time * self.num_symbols))
            window = np.vstack((pad, np.array(self.feature_history)))
        else:
            window = np.array(self.feature_history)
        
        # Force NaN to 0 in window
        window = np.nan_to_num(window, nan=0.0)

        # Flatten window and append positions array + cash
        cash = self.broker.getcash()
        # Update positions from broker (multi-stock: getposition per data)
        self.current_positions = np.array([self.getposition(data).size for data in self.symbol_datas.values()])
        state = np.append(window.flatten(), np.append(self.current_positions, cash))  # Multi-state: flatten + positions vec + cash
        
        # Predict action vector
        action, _ = self.model.predict(state, deterministic=True)
        
        # Clip small actions to 0 (hold) to prevent no-trades, reference from FinRL_DeepSeek (5.3: action threshold for stability)
        action[np.abs(action) < 0.1] = 0.0  # Vector clip

        # Multi trade loop: per symbol/action[i]
        for i, (symbol, data) in enumerate(self.symbol_datas.items()):
            act = action[i]
            price = data.close[0]
            if act > 0:
                shares_to_buy = (cash * act) / price
                # Fixes attribute error without altering trade logic.
                cost = shares_to_buy * price * (1 + self.broker.get_commissioninfo(data).p.commission)  # Use get_commissioninfo(data).p.commission
                if np.isfinite(cost) and cost <= cash:
                    self.buy(data=data, size=shares_to_buy)  # Buy on specific data
            elif act < 0:
                shares_to_sell = self.current_positions[i] * abs(act)
                revenue = shares_to_sell * price * (1 - self.broker.get_commissioninfo(data).p.commission)  # Use get_commissioninfo(data).p.commission
                if np.isfinite(revenue):
                    self.sell(data=data, size=shares_to_sell)  # Sell on specific data
        
        self.actions.append(action)  # Append vector action

class CustomPandasData(bt.feeds.PandasData):
    lines = ('macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'sentiment_score', 'risk_score')  # Added risk_score
    params = (
        ('macd', 'macd'),
        ('boll_ub', 'boll_ub'),
        ('boll_lb', 'boll_lb'),
        ('rsi_30', 'rsi_30'),
        ('cci_30', 'cci_30'),
        ('dx_30', 'dx_30'),
        ('close_30_sma', 'close_30_sma'),
        ('close_60_sma', 'close_60_sma'),
        ('sentiment_score', 'sentiment_score'),
        ('risk_score', 'risk_score'),  # Added
    )

class PortfolioAnalyzer(bt.Analyzer):
    def start(self):
        self.portfolio_values = []

    def next(self):
        self.portfolio_values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return self.portfolio_values

class Backtest:
    def __init__(self, config_trade, exper_data, models_paths, fused_dfs, stock_data_dict, symbol=''):
        """
        Initialize Backtest with config, experiment data, model paths, fused dataframes, stock data, and symbol.
        Updates: Use StockTradingEnv with dummy rl_data as dummy_env to match observation/action spaces.
        """
        self.config_trade = config_trade
        self.exper_data = exper_data
        self.models_paths = models_paths
        self.fused_dfs = fused_dfs
        self.stock_data_dict = stock_data_dict
        self.symbol = symbol
        self.results_cache = os.path.join(self.config_trade.RESULTS_SAVE_DIR, self.symbol) if self.symbol else self.config_trade.RESULTS_SAVE_DIR
        os.makedirs(self.results_cache, exist_ok=True)
        logging.info(f"TB Module - Results cache for {self.symbol}: {self.results_cache}")

        # Model classes dict for dynamic loading, reference from FinRL (DRLAgent.py: MODELS dict for dynamic class selection)
        self.model_classes = {
            'PPO': PPO,
            'A2C': A2C,
            'DDPG': DDPG,
            'TD3': TD3,
            'SAC': SAC,
            'CPPO': CPPO,  # Custom from trading_agent.py
        }

        # Dummy env for model loading (required for TD3/DDPG/SAC _setup_model)
        # Use StockTradingEnv with minimal rl_data to match spaces
        dummy_rl_data = [{'states': [[0.0] * 15], 'start_date': '2000-01-01'}]  # 15 features: OHLCV+8 ind+sentiment+risk
        self.dummy_env = StockTradingEnv(self.config_trade, dummy_rl_data, env_type='test')
        logging.info(f"TB Module - Initialized dummy_env with observation_space: {self.dummy_env.observation_space}, action_space: {self.dummy_env.action_space}")

        if not isinstance(self.exper_data, dict) or not isinstance(self.fused_dfs, dict):
            raise ValueError("Invalid input: exper_data and fused_dfs must be dictionaries")
    
    def compute_metrics(self, daily_rets, portfolio_series, actions, trade_rewards, benchmark_rets):
        """
        Compute comprehensive metrics from daily returns, portfolio, actions, and benchmark.
        Input: daily_rets (pd.Series), portfolio_series (pd.Series), actions (list), trade_rewards (list), benchmark_rets (pd.Series)
        Output: dict of metrics
        Logic: Calculate each metric with safeguards; annualized 252 days; added IR (mean excess / tracking error), CVaR (mean worst alpha%), Rachev (CVaR_pos / CVaR_neg), reference from FinRL_DeepSeek (Table 1: IR, CVaR, Rachev Ratio).
        Robustness: Handle empty/zero with defaults; align index for benchmark.
        """
        metrics = {}
        alpha = 0.05  # For CVaR

        # Use len() >0 for empty check to avoid ambiguous truth value, reference from FinRL_DeepSeek (5.2: safe metrics computation)
        if len(benchmark_rets) > 0 and len(daily_rets) > 0:
            # Align before subtraction
            aligned_daily, aligned_bench = daily_rets.align(benchmark_rets, join='inner', method='ffill')
            excess_rets = aligned_daily - aligned_bench.fillna(0)
            tracking_error = excess_rets.std()
            metrics['information_ratio'] = (excess_rets.mean() / tracking_error * np.sqrt(252)) if tracking_error != 0 else 0.0
        else:
            metrics['information_ratio'] = 0.0
        
        try: 
            if len(portfolio_series) <= 1:
                logging.warning("TB Module - No portfolio changes detected; setting metrics to 0")
                metrics = {k: 0.0 for k in ['sharpe', 'information_ratio', 'total_returns', 'total_rewards', 'annualized_return', 'max_drawdown', 'cvar', 'rachev_ratio', 'action_entropy', 'win_rate', 'num_trades']}
                return metrics  # Early return defaults without raise
            
            std_ret = daily_rets.std()
            metrics['sharpe'] = (daily_rets.mean() / std_ret * np.sqrt(252)) if std_ret != 0 else 0.0
            
            metrics['total_returns'] = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) if len(portfolio_series) > 1 else 0.0

            metrics['total_rewards'] = portfolio_series.diff().sum() if not portfolio_series.empty else 0.0
            
            if len(portfolio_series) > 1:
                years = (portfolio_series.index[-1] - portfolio_series.index[0]).days / 365.25
                metrics['annualized_return'] = (1 + metrics['total_returns']) ** (1 / years) - 1 if years > 0 else 0.0
            else:
                metrics['annualized_return'] = 0.0

            if not portfolio_series.empty:
                peak = portfolio_series.cummax()
                drawdown = (portfolio_series - peak) / peak
                metrics['max_drawdown'] = drawdown.min() if not drawdown.empty else 0.0
            else:
                metrics['max_drawdown'] = 0.0
        
            # CVaR: mean of worst alpha% returns
            if len(daily_rets) > 0:
                sorted_rets = np.sort(daily_rets)
                var_idx = int(alpha * len(sorted_rets))
                metrics['cvar'] = np.mean(sorted_rets[:var_idx]) if var_idx > 0 else 0.0
            else:
                metrics['cvar'] = 0.0
            
            # Rachev Ratio: CVaR_positive (best (1-alpha)%) / CVaR_negative (worst alpha%)
            if len(daily_rets) > 0:
                pos_idx = int((1 - alpha) * len(sorted_rets))
                cvar_pos = np.mean(sorted_rets[pos_idx:]) if pos_idx < len(sorted_rets) else 0.0
                cvar_neg = abs(metrics['cvar']) if metrics['cvar'] < 0 else 1e-6  # Avoid div0
                metrics['rachev_ratio'] = cvar_pos / cvar_neg if cvar_neg != 0 else 1.0
            else:
                metrics['rachev_ratio'] = 1.0
            
            if actions:
                bins = np.array([np.sum(np.array(actions) > 0.05), np.sum(np.abs(np.array(actions)) <= 0.05), np.sum(np.array(actions) < -0.05)])
                probs = bins / np.sum(bins) if np.sum(bins) > 0 else np.array([1/3, 1/3, 1/3])
                metrics['action_entropy'] = -np.sum(probs * np.log(probs + 1e-10))
                if metrics['action_entropy'] < 0.1:
                    logging.warning(f"TB Module - Low entropy detected; scaling rewards in future runs suggested")
            else:
                metrics['action_entropy'] = 0.0
            
            metrics['win_rate'] = sum(np.array(trade_rewards) > 0) / len(trade_rewards) if len(trade_rewards) > 0 else 0.0

            metrics['num_trades'] = sum(np.abs(np.array(actions)) > 0.1) if actions else 0
            
            logging.debug(f"TB Module - Action bins: buy={bins[0]}, hold={bins[1]}, sell={bins[2]}")

        except Exception as e:
            logging.error(f"TB Module - Error in computing metrics: {e}")
            metrics = {k: 0.0 for k in metrics}  # Reset to 0 on failure
        
        return metrics

    def simulate_prediction(self, mode):
        """
        Simulate prediction in env to collect account_value and actions.
        Output: df_account_value (pd.DataFrame), df_actions (pd.DataFrame)
        Logic: Run full episode in test_env, collect portfolio and actions, reference from FinRL (DRL_prediction.py: run step loop, save asset/action memory).
        Robustness: Use test_rl_data from exper_data; handle empty data.
        """
        test_rl_data = self.exper_data.get(mode, {}).get('test', [])
        model_path = self.models_paths.get(mode)
        zip_path = f"{model_path}.zip" if model_path else None
        logging.info(f"TB Module - Simulation for mode {mode} ({self.symbol}): test_rl_data len {len(test_rl_data)}, model_path {model_path}, zip exists {os.path.exists(zip_path) if zip_path else False}")

        if not test_rl_data or not model_path or not os.path.exists(zip_path):
            logging.warning(f"TB Module - Skipping simulation for mode {mode} ({self.symbol}): missing data or model")
            return None, None

        model_type = self.exper_data.get(mode, {}).get('model_type', 'PPO')
        model_class = self.model_classes.get(model_type)
        if not model_class:
            logging.warning(f"TB Module - Unsupported model_type {model_type} for simulation {mode}; skipping")
            return None, None

        model = model_class.load(model_path, env=self.dummy_env)  # Use dummy_env for load

        # Create test env from rl_data
        test_env = StockTradingEnv(self.config_trade, test_rl_data, env_type='test')  # Use 'backtest' mode for full run

        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)

        # Save memories from env, reference from FinRL (DRL_prediction.py: collect after loop)
        df_account_value = test_env.save_portfolio_memory()  # Get portfolio df
        df_actions = test_env.save_action_memory()  # Get actions df
        logging.info(f"TB Module - Simulation for mode {mode} ({self.symbol}): account_value shape {df_account_value.shape}, actions shape {df_actions.shape}")
        return df_account_value, df_actions

    def run_backtest_for_mode(self, mode):
        """
        Run backtest for a single mode on test data.
        Updates: Dynamic model loading; use dummy_env for load; added benchmark fetch (NDX for Nasdaq-100, reference from FinRL_DeepSeek 5); passed to compute_metrics; plot with benchmark curve.
        """
        fused_test_df = self.fused_dfs.get(mode, {}).get('test', pd.DataFrame())
        model_path = self.models_paths.get(mode)
        zip_path = f"{model_path}.zip" if model_path else None
        logging.info(f"TB Module - Backtest for mode {mode} ({self.symbol}): fused_test shape {fused_test_df.shape}, model_path {model_path}, zip exists {os.path.exists(zip_path) if zip_path else False}")

        if fused_test_df.empty or not model_path or not os.path.exists(zip_path):
            logging.warning(f"TB Module - Skipping backtest for mode {mode} ({self.symbol}): missing data or model")
            return None, None
        
        # Dynamic model class selection
        model_type = self.exper_data.get(mode, {}).get('model_type', 'PPO')
        model_class = self.model_classes.get(model_type)
        if not model_class:
            logging.warning(f"TB Module - Unsupported model_type {model_type} for mode {mode}; skipping")
            return None, None
        
        # Load model dynamically with dummy_env
        try:
            model = model_class.load(model_path, env=self.dummy_env)
            logging.info(f"TB Module - Loaded {model_type} from {model_path} for mode {mode} ({self.symbol})")
        except Exception as e:
            logging.error(f"TB Module - Model load failed for {model_type} in mode {mode} ({self.symbol}): {e}")
            return None, None
        
        # Upstream repair: Filter test period
        mode_df = fused_test_df.copy().sort_values('Date')  # Use test df directly
        mode_df = mode_df.reset_index() # Ensure 'Date' as columns
        
        cerebro = bt.Cerebro()
        
        # Split wide df to per-symbol sub-df with generic columns
        symbol_cols = {symbol: [col for col in mode_df.columns if col.endswith(f'_{symbol}')] for symbol in self.config_trade.symbols}
        global_ind_cols = [col for col in mode_df.columns if col != 'Date' and not any(col.endswith(f'_{symbol}') for symbol in self.config_trade.symbols)]
        for symbol in self.config_trade.symbols:
            if not symbol_cols[symbol]:
                logging.warning(f"TB Module - No columns for symbol {symbol} in mode {mode}; skipping")
                continue
            # Merge symbol-specific and global indicator columns
            all_symbol_cols = symbol_cols[symbol] + global_ind_cols
            # Rename symbol-specific columns to generic names (remove suffix)
            symbol_df = mode_df[['Date'] + all_symbol_cols].rename(columns={col: col.replace(f'_{symbol}', '') for col in all_symbol_cols if col.endswith(f'_{symbol}')})

            data = CustomPandasData(dataname=symbol_df.set_index('Date'))
            cerebro.adddata(data, name=symbol)  # Name per symbol for strategy access
        
        cerebro.addstrategy(RLStrategy, model=model)
        cerebro.broker.setcash(self.config_trade.initial_cash)
        cerebro.broker.setcommission(self.config_trade.commission_rate)
        cerebro.addanalyzer(PortfolioAnalyzer, _name='portfolio')
        
        try:
            thestrats = cerebro.run()
            portfolio_values = thestrats[0].analyzers.portfolio.get_analysis()
            actions = thestrats[0].actions
            trade_rewards = []
            for i in range(1, len(portfolio_values)):
                # Fixes ambiguous error without altering logic.
                if (np.abs(actions[i-1]) > 0.05).any(): # Use .any() for array condition, check if any action > threshold
                    delta = portfolio_values[i] - portfolio_values[i-1]
                    trade_rewards.append(delta)
            portfolio_series = pd.Series(portfolio_values, index=mode_df['Date'])
            daily_rets = portfolio_series.pct_change().dropna()
            
            # Fetch benchmark (Nasdaq-100 '^NDX', reference from FinRL_DeepSeek 5)
            benchmark_df = yf.download('^NDX', start=portfolio_series.index[0], end=portfolio_series.index[-1])
            benchmark_rets = benchmark_df['Close'].pct_change().dropna()
            benchmark_series = benchmark_df['Close'] / benchmark_df['Close'].iloc[0] * self.config_trade.initial_cash  # Normalize to initial_cash for plot/comparison
            
            metrics = self.compute_metrics(daily_rets, portfolio_series, actions, trade_rewards, benchmark_rets)
            logging.info(f"TB Module - Backtest for mode {mode} ({self.symbol}): {metrics}")
            
            fig, ax = plt.subplots()
            ax.plot(portfolio_series.index, portfolio_series, label=mode)
            ax.plot(benchmark_series.index, benchmark_series, label='Nasdaq-100', linestyle='--')
            ax.legend()
            fig_path = f"{self.results_cache}/{mode}_portfolio.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logging.info(f"TB Module - Saved fig to {fig_path} for {self.symbol}")
            
            return {'metrics': metrics, 'portfolio_series': portfolio_series}, fig_path
        except Exception as e:
            logging.error(f"TB Module - Backtest error for mode {mode} ({self.symbol}): {e}")
            return None, None

    def run(self):
        logging.info("=========== Start to run Backtest ===========")
        all_results = {}
        
        # Simplified plotting loop:
        for group, modes in self.config_trade.exper_mode.items():
            logging.info(f"TB Module - Running backtest for group {group} ({self.symbol})")
            for mode in modes:
                results_dict, fig_path = self.run_backtest_for_mode(mode)
                if results_dict:
                    all_results[mode] = (results_dict, fig_path)

        # Aggregate plot/table across symbols per mode
        all_series = {}  # {mode: {symbol: series}}
        all_metrics = []  # list of {'mode':, 'symbol':, **metrics}
        for mode in sum(self.config_trade.exper_mode.values(), []):
            mode_series = {}
            for symbol in self.config_trade.symbols:
                symbol_results = all_results.get(symbol, {}).get(mode, (None, None))[0]
                if symbol_results and 'portfolio_series' in symbol_results:
                    mode_series[symbol] = symbol_results['portfolio_series']
                    all_metrics.append({'mode': mode, 'symbol': symbol, **symbol_results['metrics']})
            all_series[mode] = mode_series

        # Plot multi-symbol per mode
        for mode, mode_series in all_series.items():
            if mode_series:
                fig_multi, ax_multi = plt.subplots()
                for sym, ser in mode_series.items():
                    ax_multi.plot(ser.index, ser, label=f'{mode}-{sym}')
                # Add benchmark (fetch once outside if needed)
                benchmark_df = yf.download('^NDX', start=min(s.index[0] for s in mode_series.values()), end=max(s.index[-1] for s in mode_series.values()))
                benchmark_series = benchmark_df['Close'] / benchmark_df['Close'].iloc[0] * self.config_trade.initial_cash
                ax_multi.plot(benchmark_series.index, benchmark_series, label='Nasdaq-100', linestyle='--')
                ax_multi.set_title(f'Portfolio Value Comparison - {mode} (Multi-Symbol)')
                ax_multi.legend()
                multi_fig_path = f"{self.results_cache}/{mode}_multi_portfolio.png"
                fig_multi.savefig(multi_fig_path)
                plt.close(fig_multi)
                logging.info(f"TB Module - Saved multi-symbol portfolio fig to {multi_fig_path} for {mode}")

        # Combined metrics table across all
        if all_metrics:
            results_df = pd.DataFrame(all_metrics).sort_values(by=['mode', 'sharpe'], ascending=[True, False])
            logging.info(f"TB Module - Multi-symbol results table:\n{results_df.to_string()}")
            table_path = f"{self.results_cache}/metrics_table_multi.csv"
            results_df.to_csv(table_path, index=False)
            logging.info(f"TB Module - Saved multi-symbol metrics table to {table_path}")
        
        return all_results