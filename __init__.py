# Projects/finbert_trader/__init__.py
"""
FinBERT-Driven Multi-Stock RL Trading System
"""

__version__ = "1.0.0"
__author__ = "Zeng Shicheng"

# Package imports for easy access
from src.finbert_trader.config_setup import ConfigSetup
from src.finbert_trader.data_resource import DataResource
from src.finbert_trader.features.feature_engineer import FeatureEngineer
from src.finbert_trader.config_trading import ConfigTrading
from src.finbert_trader.stock_trading_env import StockTradingEnv
from src.finbert_trader.trading_agent import TradingAgent
from src.finbert_trader.trading_backtest import TradingBacktest
from src.finbert_trader.exper_tracker import ExperimentTracker
from src.finbert_trader.visualize.visualize_backtest import VisualizeBacktest

__all__ = [
    'ConfigSetup',
    'DataResource', 
    'FeatureEngineer',
    'ConfigTrading',
    'StockTradingEnv',
    'TradingAgent',
    'TradingBacktest',
    'ExperimentTracker',
    'VisualizeBacktest'
]