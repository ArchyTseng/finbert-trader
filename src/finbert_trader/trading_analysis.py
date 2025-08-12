# trading_analysis.py (or potentially extend trading_backtest.py)

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# Assuming logging is already configured in your pipeline
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_trade_history(trade_history: List[Dict[str, Any]], 
                          initial_asset_value: float = 1.0, # Assuming normalized env
                          symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyzes the detailed trade history from a backtest to extract key performance
    and behavioral metrics.

    This function provides deeper insights into the agent's trading behavior beyond
    standard backtest metrics, focusing on transaction patterns, holding dynamics,
    and risk distribution. The output dictionary is designed to be easily integrated
    into the main experiment log (e.g., experiment_log.json).

    Parameters
    ----------
    trade_history : List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents the state and
        actions at a specific timestep from the backtest. Expected keys include
        'step', 'action', 'reward', 'total_asset', 'cash', 'position', 'cost',
        'sentiment_factor', 'risk_factor'.
    initial_asset_value : float, optional
        The initial total asset value used for percentage calculations.
        Defaults to 1.0, assuming a normalized environment.
    symbols : List[str], optional
        A list of symbol names corresponding to the action/position dimensions.
        If provided, enables symbol-specific analysis. If None, generic indices
        (e.g., 'Symbol_0', 'Symbol_1') are used.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various calculated metrics and statistics
        describing the trading behavior. This structure is suitable for JSON
        serialization and inclusion in experiment logs.

        Example structure:
        {
            "summary": { ... },
            "transactions": { ... },
            "holdings": { ... },
            "returns": { ... },
            "risk_factors": { ... } # If factors were tracked
        }
    """
    if not trade_history:
        logging.warning("TA Module - analyze_trade_history - Trade history is empty.")
        return {}

    try:
        # --- 1. Convert to DataFrame for easier analysis ---
        df_history = pd.DataFrame(trade_history)
        
        # Ensure numeric types for calculations
        numeric_cols = ['reward', 'total_asset', 'cash', 'cost']
        for col in numeric_cols:
            if col in df_history.columns:
                 # pd.to_numeric with errors='coerce' handles potential string 'np.float32(...)' issues
                df_history[col] = pd.to_numeric(df_history[col], errors='coerce') 

        num_assets = len(df_history['position'].iloc[0]) if 'position' in df_history.columns and len(df_history['position'].iloc[0]) > 0 else 1
        if symbols is None:
            symbols = [f"Symbol_{i}" for i in range(num_assets)]
        elif len(symbols) != num_assets:
             logging.warning(f"TA Module - analyze_trade_history - Mismatch between provided symbols ({len(symbols)}) and position dimensions ({num_assets}). Using generic names.")
             symbols = [f"Symbol_{i}" for i in range(num_assets)]

        # Expand list-like columns (action, position, etc.) into separate columns
        if 'action' in df_history.columns:
            actions_df = pd.DataFrame(df_history['action'].tolist(), 
                                      columns=[f"Action_{s}" for s in symbols],
                                      index=df_history.index)
            df_history = pd.concat([df_history, actions_df], axis=1)
            
        if 'position' in df_history.columns:
            positions_df = pd.DataFrame(df_history['position'].tolist(), 
                                        columns=[f"Position_{s}" for s in symbols],
                                        index=df_history.index)
            df_history = pd.concat([df_history, positions_df], axis=1)
            
        if 'sentiment_factor' in df_history.columns:
            senti_factors_df = pd.DataFrame(df_history['sentiment_factor'].tolist(), 
                                            columns=[f"Senti_Factor_{s}" for s in symbols],
                                            index=df_history.index)
            df_history = pd.concat([df_history, senti_factors_df], axis=1)
            
        if 'risk_factor' in df_history.columns:
            risk_factors_df = pd.DataFrame(df_history['risk_factor'].tolist(), 
                                           columns=[f"Risk_Factor_{s}" for s in symbols],
                                           index=df_history.index)
            df_history = pd.concat([df_history, risk_factors_df], axis=1)

        # --- 2. Calculate derived series ---
        df_history['returns'] = df_history['total_asset'].pct_change().fillna(0)
        df_history['cumulative_returns'] = (1 + df_history['returns']).cumprod() - 1
        df_history['drawdown'] = (df_history['total_asset'] / df_history['total_asset'].cummax()) - 1
        
        # --- 3. Initialize results dictionary ---
        trading_analy_dict = {
            "summary": {},
            "transactions": {},
            "holdings": {},
            "returns": {},
            "risk_exposure": {} # Renamed from risk_factors for clarity if factors are used for exposure
        }

        # --- 4. Summary Statistics ---
        total_steps = len(df_history)
        total_return = (df_history['total_asset'].iloc[-1] / initial_asset_value) - 1 if total_steps > 0 else 0.0
        
        trading_analy_dict["summary"] = {
            "total_trading_steps": total_steps,
            "initial_asset_value": initial_asset_value,
            "final_asset_value": df_history['total_asset'].iloc[-1] if total_steps > 0 else initial_asset_value,
            "total_return": float(total_return),
            "total_transaction_cost": float(df_history['cost'].sum()) if 'cost' in df_history.columns else 0.0,
            "final_cash_level": float(df_history['cash'].iloc[-1]) if 'cash' in df_history.columns and total_steps > 0 else initial_asset_value,
            "max_drawdown": float(df_history['drawdown'].min()) if 'drawdown' in df_history.columns else 0.0
        }

        # --- 5. Transaction Analysis ---
        # Simple proxy for transactions: significant change in position
        # A more precise method would require tracking buys/sells explicitly in the env
        transaction_counts = {}
        turnover_rates = {} # Based on position changes relative to asset value
        avg_holding_periods = {} # Simplified estimate

        for i, symbol in enumerate(symbols):
            pos_col = f"Position_{symbol}"
            if pos_col in df_history.columns:
                position_series = df_history[pos_col]
                # Count significant changes in position (e.g., crossing zero, large relative changes)
                # This is a heuristic; actual trade count needs env logic
                abs_changes = position_series.diff().abs()
                # Assume a transaction if position change is > 1% of total asset (adjustable heuristic)
                significant_changes = abs_changes > (0.01 * df_history['total_asset'])
                transaction_counts[symbol] = int(significant_changes.sum())
                
                # Turnover: Sum of absolute position changes / Average total asset
                # (This approximates the volume of assets traded relative to portfolio size)
                total_abs_position_change = abs_changes.sum()
                avg_total_asset = df_history['total_asset'].mean()
                if avg_total_asset > 0:
                    turnover_rates[symbol] = float(total_abs_position_change / avg_total_asset)
                else:
                    turnover_rates[symbol] = 0.0
                
                # Average Holding Period (simplified)
                # Calculate average time positions are held non-zero
                non_zero_mask = position_series != 0
                if non_zero_mask.any():
                    # Find transitions to/from zero to estimate holding periods
                    # This is a rough approximation
                    is_non_zero = non_zero_mask.astype(int)
                    transitions = is_non_zero.diff().fillna(0)
                    enter_long_signals = (transitions == 1)
                    exit_long_signals = (transitions == -1)
                    
                    # Find pairs of enter/exit (simplified, might not be perfect)
                    enter_indices = enter_long_signals[enter_long_signals].index
                    exit_indices = exit_long_signals[exit_long_signals].index
                    
                    if len(enter_indices) > 0 and len(exit_indices) > 0:
                        # Pair exits with nearest previous enter
                        holding_periods = []
                        for exit_idx in exit_indices:
                            prior_enters = enter_indices[enter_indices < exit_idx]
                            if len(prior_enters) > 0:
                                enter_idx = prior_enters[-1]
                                holding_periods.append(exit_idx - enter_idx)
                        
                        if holding_periods:
                            avg_holding_periods[symbol] = float(np.mean(holding_periods))
                        else:
                            avg_holding_periods[symbol] = float('nan') # Not enough data
                    else:
                         avg_holding_periods[symbol] = float('nan') # Not enough data
                else:
                     avg_holding_periods[symbol] = float('nan') # Never held

        trading_analy_dict["transactions"] = {
            "estimated_trade_counts": transaction_counts,
            "turnover_rates": turnover_rates,
            "average_holding_periods_steps": avg_holding_periods # In steps, not time
        }

        # --- 6. Holdings Analysis ---
        time_in_market = {}
        max_leverage = 0.0
        avg_abs_exposure = 0.0
        
        if any(f"Position_{s}" in df_history.columns for s in symbols):
            total_abs_positions = df_history[[f"Position_{s}" for s in symbols]].abs().sum(axis=1)
            total_exposure = total_abs_positions / (df_history['total_asset'] + 1e-8) # Ratio of invested capital
            time_in_market = {s: float((df_history[f'Position_{s}'].abs() > 1e-6).mean()) for s in symbols if f'Position_{s}' in df_history.columns}
            max_leverage = float(total_exposure.max())
            avg_abs_exposure = float(total_exposure.mean())
            
        # Positional Concentration (Herfindahl-Hirschman Index - HHI)
        concentration_per_step = []
        for _, row in df_history.iterrows():
             abs_positions = [abs(row.get(f'Position_{s}', 0.0)) for s in symbols]
             total_abs_pos = sum(abs_positions)
             if total_abs_pos > 1e-8:
                 weights = [p / total_abs_pos for p in abs_positions]
                 hhi = sum(w**2 for w in weights)
                 concentration_per_step.append(hhi)
             else:
                 concentration_per_step.append(1.0) # No position, fully concentrated in cash (conceptually)
                 
        avg_concentration = float(np.mean(concentration_per_step)) if concentration_per_step else 1.0

        trading_analy_dict["holdings"] = {
            "time_spent_in_market_per_symbol": time_in_market,
            "average_absolute_exposure_ratio": avg_abs_exposure,
            "maximum_leverage_ratio": max_leverage,
            "average_position_concentration_hhi": avg_concentration # 1: fully concentrated, 1/n: equally distributed
        }

        # --- 7. Returns Analysis ---
        returns_series = df_history['returns']
        if len(returns_series) > 1:
            # Basic stats
            trading_analy_dict["returns"] = {
                "mean_return_per_step": float(returns_series.mean()),
                "std_return_per_step": float(returns_series.std()),
                "min_return": float(returns_series.min()),
                "max_return": float(returns_series.max()),
                "skewness": float(returns_series.skew()) if hasattr(returns_series, 'skew') else float('nan'),
                "kurtosis": float(returns_series.kurtosis()) if hasattr(returns_series, 'kurtosis') else float('nan'),
                # Downside risk
                "downside_deviation": float(np.sqrt(np.mean(np.minimum(returns_series, 0)**2))),
                # Drawdown stats
                "max_drawdown": float(df_history['drawdown'].min()),
                "avg_drawdown": float(df_history['drawdown'].mean())
            }
        else:
            trading_analy_dict["returns"] = {"error": "Insufficient data for return analysis"}

        # --- 8. Risk Factor Analysis (if available) ---
        if 'sentiment_factor' in df_history.columns or 'risk_factor' in df_history.columns:
            factor_analysis = {}
            for factor_type, prefix in [("sentiment", "Senti_Factor"), ("risk", "Risk_Factor")]:
                factor_cols = [f"{prefix}_{s}" for s in symbols]
                available_factor_cols = [col for col in factor_cols if col in df_history.columns]
                if available_factor_cols:
                    factor_df = df_history[available_factor_cols]
                    # Analyze how factors deviate from neutral (1.0)
                    deviations_from_neutral = (factor_df - 1.0).abs()
                    factor_analysis[factor_type] = {
                        "mean_abs_deviation_from_neutral": float(deviations_from_neutral.mean().mean()),
                        "max_abs_deviation_from_neutral": float(deviations_from_neutral.max().max()),
                        "percentage_of_time_factors_active": float((deviations_from_neutral > 0.01).mean().mean()) # E.g., >1% deviation
                    }
            trading_analy_dict["risk_exposure"] = factor_analysis
        else:
            trading_analy_dict["risk_exposure"] = {"note": "Sentiment/Risk factors not tracked in trade history"}

        logging.info("TA Module - analyze_trade_history - Completed trade history analysis.")
        return trading_analy_dict

    except Exception as e:
        logging.error(f"TA Module - analyze_trade_history - Error during analysis: {e}")
        # Return a minimal dict indicating failure
        return {"error": f"Analysis failed: {str(e)}"}
