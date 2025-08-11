# src/finbert_trader/__init__.py
"""
FinBERT-Driven Multi-Stock RL Trading System - Source Code
"""

__version__ = "1.0.0"
__author__ = "Zeng Shicheng"

# Package imports for easy access within the package

from .config_setup import ConfigSetup
from .data_resource import DataResource
from .features.feature_engineer import FeatureEngineer
from .config_trading import ConfigTrading
from .stock_trading_env import StockTradingEnv
from .trading_agent import TradingAgent
from .trading_backtest import TradingBacktest
from .exper_tracker import ExperimentTracker
from .exper_scheme import ExperimentScheme
from .visualize.visualize_backtest import VisualizeBacktest

__all__ = [
    'ConfigSetup',
    'DataResource', 
    'FeatureEngineer',
    'ConfigTrading',
    'StockTradingEnv',
    'TradingAgent',
    'TradingBacktest',
    'ExperimentTracker',
    'ExperimentScheme',
    'VisualizeBacktest'
]