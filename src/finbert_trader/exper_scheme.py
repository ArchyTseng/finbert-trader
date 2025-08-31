# src/finbert_trader/exper_scheme.py
"""
Experiment Scheme Module for FinBERT-Driven Trading System
Purpose: Provide systematic experiment schemes for parameter tuning and validation
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import project modules
from .config_setup import ConfigSetup
from .config_trading import ConfigTrading
from .data_resource import DataResource
from .features.feature_engineer import FeatureEngineer
from .stock_trading_env import StockTradingEnv
from .trading_agent import TradingAgent
from .trading_backtest import TradingBacktest
from .exper_tracker import ExperimentTracker
from .trading_analysis import analyze_trade_history

class ExperimentScheme:
    """
    Class for managing experiment schemes and parameter tuning strategies.
    
    This class provides both quick experiments for rapid feedback and full experiments
    for comprehensive validation, supporting systematic parameter exploration.
    """
    
    def __init__(self, config: ConfigSetup):
        """
        Initialize ExperimentScheme with unified configuration.
        
        Parameters
        ----------
        config : ConfigSetup
            Configuration setup instance containing all cache directory paths
        """
        self.config = config
        self.symbols = self.config.symbols
        self.filter_ind = self.config.filter_ind
        self.use_experiment_sequence = getattr(self.config, 'use_experiment_sequence', False)
        
        # Use unified cache directories from ConfigSetup
        self.config_cache_dir = getattr(config, 'CONFIG_CACHE_DIR', 'config_cache')
        self.raw_data_dir = getattr(config, 'RAW_DATA_DIR', 'raw_data_cache')
        self.processed_news_dir = getattr(config, 'PROCESSED_NEWS_DIR', 'processed_news_cache')
        self.fused_data_dir = getattr(config, 'FUSED_DATA_DIR', 'fused_data_cache')
        self.exper_data_dir = getattr(config, 'EXPER_DATA_DIR', 'exper_data_cache')
        self.plot_features_dir = getattr(config, 'PLOT_FEATURES_DIR', 'plot_features_cache')
        self.plot_news_dir = getattr(config, 'PLOT_NEWS_DIR', 'plot_news_cache')
        self.plot_exper_dir = getattr(config, 'PLOT_EXPER_DIR', 'plot_exper_cache')
        self.results_cache_dir = getattr(config, 'RESULTS_CACHE_DIR', 'results_cache')
        self.experiment_cache_dir = getattr(config, 'EXPERIMENT_CACHE_DIR', 'exper_cache')
        self.scaler_cache_dir = getattr(config, 'SCALER_CACHE_DIR', 'scaler_cache')
        self.log_dir = getattr(config, 'LOG_SAVE_DIR', 'logs')
        
        # Ensure all directories exist
        for dir_path in [self.config_cache_dir, self.raw_data_dir, self.exper_data_dir, 
                         self.processed_news_dir, self.fused_data_dir,self.scaler_cache_dir,
                         self.plot_features_dir,self.plot_news_dir, self.plot_exper_dir,
                        self.results_cache_dir, self.experiment_cache_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize experiment tracker with unified config
        self.experiment_tracker = ExperimentTracker(self.config)

        
        logging.info("ES Module - Initialized ExperimentScheme with unified configuration")
        logging.info(f"ES Module - Raw data directory: {self.raw_data_dir}")
        logging.info(f"ES Module - Experiment data directory: {self.exper_data_dir}")
        logging.info(f"ES Module - Plot cache directory: {self.plot_features_dir}")
        logging.info(f"ES Module - Results cache directory: {self.results_cache_dir}")
        logging.info(f"ES Module - Experiment cache directory: {self.experiment_cache_dir}")
    
        from .visualize.visualize_experiment import VisualizeExperiment
        self.experiment_visualizer = VisualizeExperiment(config)
        logging.info("ES Module - Initialized ExperimentScheme with experiment visualization support")

    # ==================== QUICK EXPERIMENTS ====================
    def quick_exper_1(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Quick Experiment 1: Basic Parameter Validation
        Purpose: Validate basic pipeline with minimal parameters
        Focus: Asset calculation, reward scaling, basic trading mechanics
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols (default: ['GOOGL', 'AAPL'])
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Quick Experiment 1: Basic Parameter Validation")
        
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': self.config.exper_mode,
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': True,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': False,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': False,
            'use_risk_factor': False,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 150000,  
            'reward_scaling': 1e-3,
            'cash_penalty_proportion': 0.0001,
            'commission_rate': 0.0001
        }

        model_params = {
            'PPO': {
                "n_steps": 2048,
                "ent_coef": 0.02,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
        }
        
        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_1',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Quick validation of basic parameters and pipeline",
            notes="No indicator signal strategy, No sent/risk factor, Basic Benchmark"
        )
    
    def quick_exper_2(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Quick Experiment 2: Reward Function Optimization
        Purpose: Test enhanced reward signals and learning efficiency
        Focus: Reward scaling, cash penalty, infusion strength
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Quick Experiment 2: Reward Function Optimization")
        
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': self.config.exper_mode,
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': False,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': False,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': False,
            'use_risk_factor': False,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 150000,  
            'reward_scaling': 1e-3,
            'cash_penalty_proportion': 0.0001,
            'commission_rate': 0.0001
        }

        model_params = {
            'PPO': {
                "n_steps": 2048,
                "ent_coef": 0.02,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
        }
        
        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_2',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Quick optimization of reward function parameters",
            notes="Use indicator signal strategy. No senti/risk factor. Indicator Benchmark"
        )
    
    def quick_exper_3(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Quick Experiment 3: RL Hyperparameter Tuning
        Purpose: Test optimized RL algorithm parameters
        Focus: Learning rate, entropy coefficient, batch size
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Quick Experiment 3: RL Hyperparameter Tuning")
        
        # Minimal configuration for quick validation
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': self.config.exper_mode,
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': True,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': True,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 150000,  
            'reward_scaling': 1e-3,
            'cash_penalty_proportion': 0.0001,
            'commission_rate': 0.0001
        }
        
        model_params = {
            'PPO': {
                "n_steps": 2048,
                "ent_coef": 0.02,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
        }

        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_3',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Quick validation of basic parameters and pipeline",
            notes="No indicator signal strategy. Use senti/rik factor. Sentiment Benchmark"
        )
    
    def quick_exper_4(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Quick Experiment 4: Reward Function Optimization
        Purpose: Test enhanced reward signals and learning efficiency
        Focus: Reward scaling, cash penalty, infusion strength
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Quick Experiment 2: Reward Function Optimization")
        
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': self.config.exper_mode,
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': False,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': True,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 150000,  
            'reward_scaling': 1e-3,
            'cash_penalty_proportion': 0.0001,
            'commission_rate': 0.0001
        }
        
        model_params = {
            'PPO': {
                "n_steps": 2048,
                "ent_coef": 0.02,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
        }

        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_4',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Quick tuning of RL hyperparameters",
            notes="Use indicator signal. Use senti/risk factor. Full mode test."
        )
    
    # ==================== FULL EXPERIMENTS ====================
    def full_exper_1(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Full Experiment 1: Comprehensive Basic Validation
        Purpose: Full validation of basic pipeline with all algorithms
        Focus: Multi-algorithm comparison, extended time period, full feature set
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Full Experiment 1: Comprehensive Basic Validation")
        
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Focus on top performers
            },
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': False,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': True,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,
            'total_timesteps': 200000,  # Extended training
            'reward_scaling': 1e-3,
            'cash_penalty_proportion': 0.001,
            'commission_rate': 0.001,
            'infusion_strength': 0.01
        }
        
        return self._execute_experiment(
            experiment_id='full_exp_1',
            setup_config=setup_config,
            trading_config=trading_config,
            description="Full validation with all algorithms and extended training",
            notes="Comprehensive baseline experiment with all features enabled"
        )
    
    def full_exper_2(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Full Experiment 2: Advanced Reward Engineering
        Purpose: Comprehensive reward function optimization
        Focus: Enhanced reward signals, risk-adjusted returns, multi-factor integration
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Full Experiment 2: Advanced Reward Engineering")
        
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO']  # Focus on top performers
            },
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': False,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': True,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,
            'total_timesteps': 300000,  # Long training for complex reward function
            'reward_scaling': 1e-2,
            'cash_penalty_proportion': 0.0005,
            'infusion_strength': 0.05,
            'commission_rate': 0.0005,
            'cvar_factor': 0.1,  # Enhanced CVaR weighting
        }
        
        model_params = {
            'PPO': {
                "n_steps": 2048,
                "ent_coef": 0.01,
                "learning_rate": 0.00025,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
            'CPPO': {
                "n_steps": 2048,
                "ent_coef": 0.01,
                "learning_rate": 0.00025,
                "batch_size": 64,
                "alpha": 0.05,
                "lambda_": 0.5,
                "beta": 0.005
            }
        }
        
        return self._execute_experiment(
            experiment_id='full_exp_2',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Advanced reward engineering with multi-factor integration",
            notes="Enhanced reward function with comprehensive risk and sentiment integration"
        )
    
    def full_exper_3(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Full Experiment 3: Production-Ready Optimization
        Purpose: Final optimization for production deployment
        Focus: Best parameters, extensive validation, performance benchmarking
        
        Parameters
        ----------
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Experiment results
        """
        logging.info("ES Module - Running Full Experiment 3: Production-Ready Optimization")
        
        setup_config = {
            'symbols': self.symbols,
            # Train/valid/test Date split
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            # Config exper_mode, extendable in ConfigSetup
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # Focus on top performers
            },
            # Config states window size for downstream pipeline
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'prediction_days': self.config.prediction_days,
            # Config smooth day in FeatureEngineer, enhancing data stableness
            'smooth_window_size': self.config.smooth_window_size,
            # Config filter target indicator for experiment comparison
            'filter_ind': self.filter_ind,  # Default [], monitoring all indicators
            # Switch save/load cache file, enhancing pipeline performance
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'plot_feature_visualization': self.config.plot_feature_visualization,
            # Config cache name format
            'use_symbol_name': self.config.use_symbol_name,
            # Switch for interpretation strategy
            'bypass_interpretation': False,  # Default True
            # Switch for dynamic indicator threshold
            'use_dynamic_ind_threshold': True,  # Default True
            # Switch for signal strategy
            'use_signal_consistency_bonus': True,  # Depend on experiment purpose
            # Switch for senti/risk score and features
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            # Config for visualization
            'use_experiment_sequence': self.config.use_experiment_sequence,
            # Configuration for Initializing cache dir
            'CONFIG_CACHE_DIR': self.config_cache_dir,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_FEATURES_DIR': self.plot_features_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'PLOT_EXPER_DIR': self.plot_exper_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,
            'total_timesteps': 500000,  # Production-level training
            'reward_scaling': 1e-2,
            'cash_penalty_proportion': 0.0001,  # Minimal cash penalty
            'infusion_strength': 0.1,   # Maximum sentiment/risk influence
            'commission_rate': 0.0001,  # Realistic low costs
            'cvar_factor': 0.15,        # Strong risk management
            'slippage_rate': 0.0005     # Realistic slippage
        }
        
        model_params = {
            'PPO': {
                "n_steps": 2048,
                "ent_coef": 0.005,      # Very low entropy for stable policy
                "learning_rate": 0.0001, # Conservative learning rate
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
            'CPPO': {
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.0001,
                "batch_size": 64,
                "alpha": 0.05,
                "lambda_": 0.7,         # Balanced CVaR penalty
                "beta": 0.003
            }
        }
        
        return self._execute_experiment(
            experiment_id='full_exp_3',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Production-ready optimization with extensive validation",
            notes="Final optimized parameters for production deployment"
        )
    
    # ==================== UTILITY METHODS ====================
    def _execute_experiment(self, experiment_id: str, setup_config: Dict[str, Any], 
                          trading_config: Dict[str, Any], model_params: Dict[str, Any] = None,
                          description: str = "", notes: str = "") -> Dict[str, Any]:
        """
        Execute a complete experiment pipeline.
        
        Parameters
        ----------
        experiment_id : str
            Unique experiment identifier
        setup_config : dict
            Configuration setup parameters
        trading_config : dict
            Trading configuration parameters
        model_params : dict, optional
            Custom model parameters
        description : str
            Experiment description
        notes : str
            Additional notes
            
        Returns
        -------
        dict
            Experiment results
        """
        try:
            logging.info(f"ES Module - Executing experiment: {experiment_id}")
            
            # Initialize configuration
            config_setup = ConfigSetup(setup_config)
            logging.info(f"ES Module - ConfigSetup initialized with symbols: {config_setup.symbols}")
            
            # Fetch data
            data_resource = DataResource(config_setup)
            stock_data_dict = data_resource.fetch_stock_data()
            if not stock_data_dict:
                raise ValueError("No stock data fetched")
            logging.info(f"ES Module - Prepared stock data for {len(stock_data_dict)} symbols")

            # Generate news chunks path
            cache_path, filtered_cache_path = data_resource.cache_path_config()

            # Load news data
            news_chunks_gen = data_resource.load_news_data(cache_path, filtered_cache_path)
            logging.info("ES Module - News data loaded successfully")
            
            # Feature engineering
            feature_engineer = FeatureEngineer(config_setup)
            exper_data_dict = feature_engineer.generate_experiment_data(
                stock_data_dict, news_chunks_gen, exper_mode='rl_algorithm'
            )
            logging.info(f"ES Module - Generated experiment data for modes: {list(exper_data_dict.keys())}")

            print(f"ES Module - _execute_experiment - After feature_engineer, "
                  f"config_setup.test_start_date: {config_setup.test_start_date}"
                  f"config_setup.test_end_date: {config_setup.test_end_date}")
            
            # Process each experiment mode
            pipeline_results = {}
            
            for mode_name, mode_data in exper_data_dict.items():
                logging.info(f"ES Module - Processing mode: {mode_name}")
                
                # Create trading configuration
                mode_trading_config = trading_config.copy()
                if model_params and mode_name in model_params:
                    mode_trading_config['model_params'] = model_params[mode_name]
                
                config_trading = ConfigTrading(
                    custom_config=mode_trading_config,
                    upstream_config=config_setup,
                    model=mode_data.get('model_type', 'PPO')
                )
                print(f"ES Module - _execute_experiment - After Config_trading initialize, "
                  f"config_trading.test_start_date: {config_setup.test_start_date}"
                  f"config_trading.test_end_date: {config_setup.test_end_date}")
                # Process training, validation, and testing
                mode_results = self._process_mode(mode_name, mode_data, config_trading)
                print(f"ES Module - _execute_experiment - mode_results keys: {list(mode_results.keys())}")
                pipeline_results[mode_name] = mode_results
            
            # Log experiment
            experiment_config = {
                'experiment_id': experiment_id,
                'setup_config': setup_config,
                'trading_config': trading_config,
                'model_params': model_params,
                'description': description
            }
            
            self.experiment_tracker.log_experiment(
                experiment_id=experiment_id,
                config_params=experiment_config,
                results=pipeline_results,
                notes=notes
            )
            
            logging.info(f"ES Module - Experiment {experiment_id} completed successfully")
            return pipeline_results
            
        except Exception as e:
            logging.error(f"ES Module - Error executing experiment {experiment_id}: {e}")
            raise

    def _execute_experiment_with_visualization(self, experiment_id: str, setup_config: Dict[str, Any], 
                                    trading_config: Dict[str, Any], model_params: Dict[str, Any] = None,
                                    description: str = "", notes: str = "") -> Dict[str, Any]:
        """
        Execute experiment with immediate visualization after completion.
        
        This function orchestrates the experiment execution and then generates visualizations.
        It correctly structures the data for the visualization module and extracts benchmark
        data for comparison plots.

        Parameters
        ----------
        experiment_id : str
            Unique experiment identifier
        setup_config : dict
            Configuration setup parameters
        trading_config : dict
            Trading configuration parameters
        model_params : dict, optional
            Custom model parameters
        description : str
            Experiment description
        notes : str
            Additional notes
            
        Returns
        -------
        dict
            Experiment results including execution results and paths to generated visualizations.
            The structure is:
            {
                'mode_name_1': mode_results_dict, # From _execute_experiment
                'mode_name_2': mode_results_dict,
                ...
                'experiment_id': ...,
                'setup_config': ...,
                'immediate_visualizations': {
                    'asset_curve_comparison': 'path/to/plot.png',
                    'performance_heatmap': 'path/to/plot.png',
                    ...
                }
            }
        """
        try:
            logging.info(f"ES Module - Executing experiment with visualization: {experiment_id}")
            
            # --- Execute the core experiment ---
            exper_results = self._execute_experiment(experiment_id, setup_config, trading_config, 
                                            model_params, description, notes)
            
            # --- Prepare data for visualization ---
            # Restructure pipeline_results for visualization functions
            # Expected format: {mode_name: backtest_results_dict}
            pipeline_results_for_viz = {}
            
            # Extract benchmark data (from the first available mode's backtest results)
            benchmark_prices_series_for_viz = None
            benchmark_returns_array_for_viz = None

            # Iterate through exper_results to find mode-specific results
            # Exclude metadata keys that are not mode names
            reserved_keys = ['experiment_id', 'setup_config', 'trading_config', 'model_params', 'description', 'notes', 'immediate_visualizations']
            
            for key, value in exper_results.items():
                if key in reserved_keys:
                    continue # Skip metadata
                
                mode_name = key
                mode_results = value # This is the dict returned by _process_mode
                
                # Extract backtest_results which contains the data visualization needs
                backtest_data = mode_results.get('backtest_results', {})
                
                # Structure data for visualization: {mode_name: backtest_data}
                # Visualization functions expect to find 'asset_history', 'metrics', etc. inside backtest_data
                pipeline_results_for_viz[mode_name] = backtest_data 

                # Extract benchmark data from the first valid mode's backtest_data
                # Assumes all modes cover the same date range for benchmark comparison
                if benchmark_prices_series_for_viz is None:
                    benchmark_prices_series_for_viz = backtest_data.get('benchmark_prices_with_date', None)
                if benchmark_returns_array_for_viz is None:
                    benchmark_returns_array_for_viz = backtest_data.get('benchmark_returns', None)

            # --- Generate Visualizations ---
            visualization_results = {}
            try:
                # --- Enhanced visualizations with benchmark ---
                from .visualize.visualize_backtest import VisualizeBacktest, generate_all_visualizations_with_benchmark
                enhanced_visualizations = {}
                if not self.use_experiment_sequence:
                    # Pass the correctly structured data and extracted benchmark info
                    enhanced_visualizations = generate_all_visualizations_with_benchmark(
                        pipeline_results=pipeline_results_for_viz, # Correctly structured data
                        config_trading=self.config,                # Pass the config for directories etc.
                        benchmark_data=benchmark_prices_series_for_viz, # Benchmark prices with date index
                        benchmark_returns=benchmark_returns_array_for_viz, # Benchmark returns array
                        benchmark_name='QQQ' # Specify the benchmark name
                    )

                # Generate Full Comparison Visualization
                try:
                    visualizer_instance = VisualizeBacktest(self.config) # Create instance
                    full_comparison_plot_path = visualizer_instance.plot_full_comparison_visualization(
                        pipeline_results=pipeline_results_for_viz,
                        benchmark_name='QQQ',
                        experiment_name=experiment_id # Pass the experiment ID as the name
                    )
                    if full_comparison_plot_path:
                         enhanced_visualizations['full_comparison_visualization'] = full_comparison_plot_path
                         logging.info(f"ES Module - Generated full comparison visualization: {full_comparison_plot_path}")
                except Exception as full_viz_error:
                     logging.warning(f"ES Module - Could not generate full comparison visualization: {full_viz_error}", exc_info=True)
                
                visualization_results.update(enhanced_visualizations)
                
                logging.info(f"ES Module - Generated enhanced visualizations for {experiment_id}")
                for viz_name, viz_path in enhanced_visualizations.items():
                    if viz_path:
                        logging.info(f"  {viz_name}: {viz_path}")
                    
            except Exception as viz_error:
                logging.warning(f"ES Module - Could not generate enhanced visualizations: {viz_error}", exc_info=True)
                
                # --- Fallback to basic visualizations ---
                try:
                    from .visualize.visualize_backtest import generate_all_visualizations
                    # Use the same correctly structured data
                    basic_visualizations = generate_all_visualizations(
                        pipeline_results=pipeline_results_for_viz, # Correctly structured data
                        config_trading=self.config              # Pass the config
                    )
                    visualization_results.update(basic_visualizations)
                    logging.info("ES Module - Generated basic visualizations as fallback")
                except Exception as basic_viz_error:
                    logging.warning(f"ES Module - Could not generate basic visualizations: {basic_viz_error}", exc_info=True)
            
            # --- Attach visualization paths to the main results ---
            exper_results['immediate_visualizations'] = visualization_results
            
            return exper_results
            
        except Exception as e:
            logging.error(f"ES Module - Error executing experiment with visualization {experiment_id}: {e}", exc_info=True)
            raise
    
    def _process_mode(self, mode_name: str, mode_data: Dict[str, Any], 
                 config_trading: ConfigTrading) -> Dict[str, Any]:
        """
        Process a single experiment mode, including training, backtesting, and result preparation.
        
        This function orchestrates the full pipeline for one experiment mode:
        1. Creates train/valid/test environments.
        2. Initializes and trains a TradingAgent.
        3. Performs backtesting using TradingBacktest, which now includes fetching
        benchmark data aligned by date.
        4. Prepares results, including benchmark data for visualization.
        5. Generates reports and saves artifacts.

        Parameters
        ----------
        mode_name : str
            Name of the experiment mode (e.g., 'PPO_FinBERT').
        mode_data : dict
            Dictionary containing 'train', 'valid', and 'test' data splits.
        config_trading : ConfigTrading
            Configuration object holding trading parameters.

        Returns
        -------
        dict
            A dictionary containing results for this mode, including:
            - model_path (str): Path to the saved trained model.
            - results_path (str): Path to the saved backtest results.
            - metrics (dict): Computed performance metrics.
            - detailed_report (dict): A comprehensive report of the backtest.
            - backtest_results (dict): Raw backtest results from TradingBacktest.
            This now includes 'benchmark_prices_with_date' and 'benchmark_returns'.
            
        Raises
        ------
        Exception
            If any step in the process fails, an error is logged and re-raised.
        """
        try:
            import numpy as np
            logging.info(f"ES Module - Processing mode {mode_name}")
            
            # Extract data splits
            train_data = mode_data['train']
            valid_data = mode_data['valid']
            test_data = mode_data['test']

            logging.info(f"ES Module - Mode {mode_name} - Inspecting test_data structure:")
            if isinstance(test_data, list) and len(test_data) > 0:
                for idx, episode in enumerate(test_data):
                    start_date = episode.get('start_date', 'MISSING')
                    states_shape = episode.get('states', np.array([])).shape
                    targets_shape = episode.get('targets', np.array([])).shape if episode.get('targets') is not None else None
                    logging.info(f"ES Module - Mode {mode_name} - Test Episode {idx}: start_date={start_date}, states_shape={states_shape}, targets_shape={targets_shape}")
            else:
                logging.info(f"ES Module - Mode {mode_name} - test_data is not a list or is empty: {test_data}")
            
            logging.info(f"ES Module - Mode {mode_name} - Data splits: "
                        f"Train({len(train_data)}), Valid({len(valid_data)}), Test({len(test_data)})")
            
            # Create environments
            logging.info(f"ES Module - Mode {mode_name} - Creating trading environments")
            train_env = StockTradingEnv(config_trading, train_data, env_type='train')
            valid_env = StockTradingEnv(config_trading, valid_data, env_type='valid')
            test_env = StockTradingEnv(config_trading, test_data, env_type='test')

            print(f"ES Module - _process_mode - After Config_trading initialize, "
                  f"test_env.test_start_date: {test_env.test_start_date}"
                  f"test_env.test_end_date: {test_env.test_end_date}")
            
            # Create and train agent
            logging.info(f"ES Module - Mode {mode_name} - Initializing trading agent")
            agent = TradingAgent(config_trading)
            
            # Training
            logging.info(f"ES Module - Mode {mode_name} - Starting training")
            agent.train(
                train_env=train_env,
                valid_env=valid_env,
                total_timesteps=config_trading.total_timesteps,
                eval_freq=max(50000, config_trading.total_timesteps // 20),
                n_eval_episodes=5
            )
            
            # Save trained model
            model_save_path = agent.save_model()
            logging.info(f"ES Module - Mode {mode_name} - Model saved to: {model_save_path}")
            
            # Backtesting with benchmark support
            logging.info(f"ES Module - Mode {mode_name} - Starting backtesting with benchmark")
            backtester = TradingBacktest(config_trading)
            # Generate backtest results, including benchmark data
            backtest_results = backtester.run_backtest(agent, test_env, record_trades=True, use_benchmark=True)
            
            # Fetch benchmark data from results for visualization
            benchmark_prices_with_date = backtest_results.get('benchmark_prices_with_date', None)
            benchmark_returns = backtest_results.get('benchmark_returns', None)

            # Ensure benchmark data is 1D before putting it back or using it elsewhere
            # This prevents errors in np.polyfit and pd.Series construction in visualization
            if benchmark_prices_with_date is not None:
                if isinstance(benchmark_prices_with_date, np.ndarray) and benchmark_prices_with_date.ndim > 1:
                    logging.debug(f"ES Module - _process_mode - Flattening benchmark_prices_with_date from {benchmark_prices_with_date.shape} to 1D.")
                    backtest_results['benchmark_prices_with_date'] = benchmark_prices_with_date.ravel()
                # If it's a DataFrame/Series, ravel() usually works or is already 1D for Series values
                elif hasattr(benchmark_prices_with_date, 'values') and hasattr(benchmark_prices_with_date.values, 'ravel'):
                    flattened_prices = benchmark_prices_with_date.values.ravel()
                    # Create a new Series with the same index if it was a Series/DataFrame
                    if hasattr(benchmark_prices_with_date, 'index'):
                        backtest_results['benchmark_prices_with_date'] = pd.Series(flattened_prices, index=benchmark_prices_with_date.index)
                    else:
                         backtest_results['benchmark_prices_with_date'] = flattened_prices

            if benchmark_returns is not None:
                if isinstance(benchmark_returns, np.ndarray) and benchmark_returns.ndim > 1:
                    logging.debug(f"ES Module - _process_mode - Flattening benchmark_returns from {benchmark_returns.shape} to 1D.")
                    backtest_results['benchmark_returns'] = benchmark_returns.ravel()
                elif hasattr(benchmark_returns, 'values') and hasattr(benchmark_returns.values, 'ravel'):
                     flattened_returns = benchmark_returns.values.ravel()
                     # benchmark_returns is typically expected to be an array, so just assign the flattened array
                     backtest_results['benchmark_returns'] = flattened_returns

            # Log for debugging
            logging.info(f"ES Module - Mode {mode_name} - Benchmark data handling completed.")
            if benchmark_prices_with_date is not None:
                logging.warning(f"ES Module - _process_mode - Benchmark prices series shape: {getattr(benchmark_prices_with_date, 'shape', 'No shape attr')}")
            if benchmark_returns is not None:
                logging.warning(f"ES Module - _process_mode - Benchmark returns array shape: {getattr(benchmark_returns, 'shape', 'No shape attr')}")
            
            # Generate detailed report
            detailed_report = backtester.generate_detailed_report(backtest_results)

            # Generate trading history analysis
            symbols_list = getattr(config_trading, 'symbols', None)
            # Get initial asset value
            initial_value_list = backtest_results.get('asset_history')
            initial_value = initial_value_list[0] if initial_value_list else config_trading.initial_cash
            
            trading_analy_dict = analyze_trade_history(
                backtest_results.get('trade_history', []), 
                initial_asset_value=initial_value,
                symbols=symbols_list
            )
            # Add trading analysis to detailed report
            detailed_report['trading_analysis'] = trading_analy_dict
            
            # Save backtest results
            results_save_path = backtester.save_results(backtest_results)
            logging.info(f"ES Module - Mode {mode_name} - Backtest results saved to: {results_save_path}")
            
            # Mode results
            mode_results = {
                'model_path': model_save_path,
                'results_path': results_save_path,
                'metrics': backtest_results['metrics'],
                'detailed_report': detailed_report,
                'backtest_results': backtest_results,
            }
            
            # Log key metrics
            metrics = backtest_results['metrics']
            logging.info(f"ES Module - Mode {mode_name} - Key Results:")
            logging.info(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            logging.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logging.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            logging.info(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
            
            # Record benchmark metrics
            if 'benchmark_cagr' in metrics:
                logging.debug(f"  Benchmark CAGR: {metrics.get('benchmark_cagr', 0)*100:.2f}%")
                logging.debug(f"  Information Ratio: {metrics.get('information_ratio', 0):.4f}")
                logging.debug(f"  Alpha: {metrics.get('alpha', 0):.4f}")
            
            return mode_results
            
        except Exception as e:
            logging.error(f"ES Module - Error processing mode {mode_name}: {e}")
            raise

    def run_robustness_test(self, experiment_method, num_runs: int = 5, run_prefix: str = "robustness_run") -> Dict[str, Any]:
        """
        Run robustness test by executing the same experiment multiple times.

        Parameters
        ----------
        experiment_method : callable
            The experiment method to run, e.g., self.quick_exper_1
        num_runs : int, optional
            Number of times to run the experiment (default is 5).
        run_prefix : str, optional
            Prefix for individual run experiment IDs (default is "robustness_run").

        Returns
        -------
        dict
            A dictionary containing:
            - 'individual_results': List of results from each run.
            - 'aggregated_metrics': Statistical summary of key metrics.
            - 'robustness_report_path': Path to the generated summary report (txt/json).
            - 'robustness_visualization_path': Path to the generated visualization (png).
        """
        logging.info(f"ES Module - Starting robustness test: {experiment_method.__name__} for {num_runs} runs.")
        
        all_results = []
        all_metrics = {
            'cagr': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'final_return': [], # As a percentage, e.g., 1.5 for 150%
        }

        for i in range(num_runs):
            run_id = f"{run_prefix}_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"ES Module - Robustness Test - Running iteration {i+1}/{num_runs} with ID: {run_id}")
            
            try:
                # --- Execute the experiment ---
                # We directly call the method, not the wrapper that includes visualization
                # to avoid redundant plotting overhead during robustness test.
                run_result = experiment_method() # This calls e.g., self.quick_exper_1()
                
                # --- Extract and store metrics ---
                # Assuming the result structure from _execute_experiment
                # {mode_name: {metrics: {...}}, ...}
                run_metrics_summary = {}
                for mode_name, mode_data in run_result.items():
                    if mode_name in ['experiment_id', 'setup_config', 'trading_config', 'model_params', 'description', 'notes', 'immediate_visualizations']:
                        continue # Skip non-mode keys
                    metrics = mode_data.get('metrics', {})
                    run_metrics_summary[mode_name] = metrics
                    
                    # Collect metrics for statistical analysis (assuming first mode for simplicity)
                    # You can modify this to analyze all modes or a specific one
                    if i == 0: # Initialize lists for the first run
                        for key in all_metrics.keys():
                            all_metrics[key] = {mode_name: [] for mode_name in run_metrics_summary.keys()}
                    
                    for mode, mode_metrics in run_metrics_summary.items():
                        all_metrics['cagr'][mode].append(mode_metrics.get('cagr', 0) * 100) # Convert to %
                        all_metrics['sharpe_ratio'][mode].append(mode_metrics.get('sharpe_ratio', 0))
                        all_metrics['max_drawdown'][mode].append(mode_metrics.get('max_drawdown', 0) * 100) # Convert to %
                        initial_asset = mode_data.get('backtest_results', {}).get('asset_history', [1.0])[0]
                        final_asset = mode_metrics.get('final_asset', initial_asset)
                        final_return_pct = ((final_asset / initial_asset) - 1) * 100 if initial_asset > 0 else 0
                        all_metrics['final_return'][mode].append(final_return_pct)

                all_results.append({
                    'run_id': run_id,
                    'run_result': run_result,
                    'metrics_summary': run_metrics_summary
                })
                logging.info(f"ES Module - Robustness Test - Completed iteration {i+1}/{num_runs}.")

            except Exception as e:
                logging.error(f"ES Module - Robustness Test - Error in iteration {i+1}: {e}", exc_info=True)
                # Optionally, append a failure record
                all_results.append({
                    'run_id': run_id,
                    'error': str(e)
                })

        # --- Aggregate Metrics ---
        aggregated_stats = {}
        for mode_name in all_metrics['cagr'].keys(): # Iterate through modes
            aggregated_stats[mode_name] = {}
            for metric_name, values_list in all_metrics.items():
                values = values_list[mode_name]
                if not values:
                    aggregated_stats[mode_name][metric_name] = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'win_rate': np.nan}
                    continue
                
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Calculate win rate (positive final return)
                if metric_name == 'final_return':
                    win_rate = np.mean(np.array(values) > 0) * 100
                else:
                    win_rate = np.nan # Not applicable for other metrics

                aggregated_stats[mode_name][metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'win_rate': win_rate # Only meaningful for final_return
                }

        # --- Generate Report and Visualization ---
        report_path = ""
        viz_path = ""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"robustness_test_report_{experiment_method.__name__}_{timestamp}.txt"
            report_path = os.path.join(self.experiment_cache_dir, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(f"Robustness Test Report for {experiment_method.__name__}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Number of Runs: {num_runs}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                for mode_name, stats in aggregated_stats.items():
                    f.write(f"\n--- Metrics Summary for {mode_name} ---\n")
                    for metric, stat_dict in stats.items():
                        f.write(f"{metric.replace('_', ' ').title()}:\n")
                        f.write(f"  Mean: {stat_dict['mean']:.4f}\n")
                        f.write(f"  Std Dev: {stat_dict['std']:.4f}\n")
                        f.write(f"  Min: {stat_dict['min']:.4f}\n")
                        f.write(f"  Max: {stat_dict['max']:.4f}\n")
                        if not np.isnan(stat_dict['win_rate']):
                            f.write(f"  Win Rate (% of Positive Final Returns): {stat_dict['win_rate']:.2f}%\n")
            
            logging.info(f"ES Module - Robustness Test - Report saved to {report_path}")

            # --- Visualization ---
            viz_filename = f"robustness_test_viz_{experiment_method.__name__}_{timestamp}.png"
            viz_path = os.path.join(self.plot_exper_dir, viz_filename)
            
            num_modes = len(aggregated_stats)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10 * num_modes))
            if num_modes == 1:
                axes = axes[np.newaxis, :] # Make it 2D for consistent indexing
            
            metric_names = ['cagr', 'sharpe_ratio', 'max_drawdown', 'final_return']
            y_labels = ['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Final Return (%)']
            
            for i, mode_name in enumerate(aggregated_stats.keys()):
                mode_metrics_data = all_metrics # This is already mode-specific
                for j, (metric, ylabel) in enumerate(zip(metric_names, y_labels)):
                    ax = axes[i, j] if num_modes > 1 else axes[j]
                    data_to_plot = mode_metrics_data[metric][mode_name]
                    if data_to_plot:
                        ax.boxplot(data_to_plot, labels=[mode_name])
                        ax.set_title(f'{mode_name} - {ylabel} Distribution')
                        ax.set_ylabel(ylabel)
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{mode_name} - {ylabel} Distribution')

            plt.tight_layout()
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"ES Module - Robustness Test - Visualization saved to {viz_path}")

        except Exception as viz_error:
            logging.error(f"ES Module - Robustness Test - Error generating report/visualization: {viz_error}", exc_info=True)
            plt.close('all') # Ensure any open plots are closed on error


        logging.info(f"ES Module - Robustness test for {experiment_method.__name__} completed.")
        return {
            'individual_results': all_results,
            'aggregated_metrics': aggregated_stats,
            'robustness_report_path': report_path,
            'robustness_visualization_path': viz_path
        }

    def run_quick_exper_1_robustness(self, num_runs: int = 5) -> Dict[str, Any]:
        """
        Convenience method to run robustness test on quick_exper_1.
        """
        return self.run_robustness_test(self.quick_exper_1, num_runs, "quick_exper_1_run")

    def run_quick_exper_2_robustness(self, num_runs: int = 5) -> Dict[str, Any]:
        """
        Convenience method to run robustness test on quick_exper_2.
        """
        return self.run_robustness_test(self.quick_exper_2, num_runs, "quick_exper_2_run")

    def run_quick_exper_3_robustness(self, num_runs: int = 5) -> Dict[str, Any]:
        """
        Convenience method to run robustness test on quick_exper_3.
        """
        return self.run_robustness_test(self.quick_exper_3, num_runs, "quick_exper_3_run")

    def run_quick_exper_4_robustness(self, num_runs: int = 5) -> Dict[str, Any]:
        """
        Convenience method to run robustness test on quick_exper_4.
        """
        return self.run_robustness_test(self.quick_exper_4, num_runs, "quick_exper_4_run")

    def run_experiment_sequence(self, experiment_names: List[str], 
                          symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run a sequence of experiments.
        
        Parameters
        ----------
        experiment_names : list
            List of experiment names to run
        symbols : list, optional
            List of stock symbols
            
        Returns
        -------
        dict
            Results from all experiments
        """
        results = {}
        
        experiment_methods = {
            'quick_exper_1': self.quick_exper_1,
            'quick_exper_2': self.quick_exper_2,
            'quick_exper_3': self.quick_exper_3,
            'quick_exper_4': self.quick_exper_4,
            'full_exper_1': self.full_exper_1,
            'full_exper_2': self.full_exper_2,
            'full_exper_3': self.full_exper_3
        }
        
        for exp_name in experiment_names:
            if exp_name in experiment_methods:
                logging.info(f"ES Module - Running experiment sequence: {exp_name}")
                results[exp_name] = experiment_methods[exp_name](symbols)
            else:
                logging.warning(f"ES Module - Unknown experiment: {exp_name}")

        # Generate experiment comparison visualization
        try:
            # Collect all experiment records for comparison
            experiment_files = []
            for file in os.listdir(self.experiment_cache_dir):
                if file.startswith('experiment_log_') and file.endswith('.json'):
                    experiment_files.append(os.path.join(self.experiment_cache_dir, file))
            
            if experiment_files:
                comparison_report = self.experiment_visualizer.generate_experiment_comparison_report(experiment_files)
                optimization_path = self.experiment_visualizer.generate_optimization_path_visualization(experiment_files)
                
                logging.info(f"ES Module - Generated experiment comparison report: {comparison_report}")
                logging.info(f"ES Module - Generated optimization path visualization: {optimization_path}")
                
                # Add visualization results to overall results
                results['experiment_visualizations'] = {
                    'comparison_report': comparison_report,
                    'optimization_path': optimization_path
                }
        except Exception as e:
            logging.warning(f"ES Module - Could not generate experiment visualizations: {e}")
        
        return results
    
    def run_and_visualize_quick_experiments_sequence(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute all quick experiments in sequence and generate the comprehensive visualization report.

        This function runs quick_exper_1 to quick_exper_4, then creates a single,
        unified experiment comparison report using the refactored VisualizeExperiment module.

        Parameters
        ----------
        symbols : list of str, optional
            List of stock symbols to use for the experiments.

        Returns
        -------
        dict
            A dictionary containing experiment results and paths to the generated visualization.
            {
                'experiment_results': {...}, # Results from run_experiment_sequence
                'visualizations': {
                    'comprehensive_report': 'path/to/experiment_comprehensive_report_YYYYMMDD_HHMMSS.png',
                }
            }
        """
        logging.info("ES Module - Running and visualizing quick experiment sequence...")
        results = {}

        try:
            # Run all quick experiments
            all_quick_results = self.run_experiment_sequence(
                ['quick_exper_1', 'quick_exper_2', 'quick_exper_3', 'quick_exper_4'],
                symbols
            )
            results['experiment_results'] = all_quick_results
            logging.info("ES Module - All quick experiments completed.")

            # Generate the single comprehensive visualization report
            visualizations = {}
            
            # Get data from experiment log files (.json)
            experiment_files = []
            if os.path.exists(self.experiment_cache_dir):
                # Only from .json files
                for file in os.listdir(self.experiment_cache_dir):
                    if file.startswith('experiment_log_') and file.endswith('.json'):
                        experiment_files.append(os.path.join(self.experiment_cache_dir, file))
            
            logging.info(f"ES Module - Collected {len(experiment_files)} experiment log files for visualization.")

            if experiment_files:
                logging.info(f"ES Module - Found {len(experiment_files)} experiment records for visualization.")
                
                try:
                    from .visualize.visualize_experiment import generate_comprehensive_experiment_report
                    comprehensive_report_path = generate_comprehensive_experiment_report(self.config, experiment_files)
                    
                    # Record report
                    visualizations['comprehensive_report'] = comprehensive_report_path
                    logging.info(f"ES Module - Comprehensive experiment report generated: {comprehensive_report_path}")

                except Exception as e:
                    logging.error(f"ES Module - Failed to generate comprehensive experiment report: {e}", exc_info=True)
                    # visualizations['comprehensive_report_error'] = str(e)
                    
            else:
                logging.warning("ES Module - No experiment records found for visualization.")
                
            results['visualizations'] = visualizations

        except Exception as e:
            logging.error(f"ES Module - Error in run_and_visualize_quick_experiments_sequence: {e}", exc_info=True)
            results['error'] = str(e)

        logging.info("ES Module - Run and visualize quick experiment sequence completed.")
        return results

    def generate_experiment_visualizations_from_cache(self) -> Dict[str, str]:
        """
        Generate the comprehensive visualization report based on existing experiment logs in the cache.

        This is useful for re-analyzing results without re-running experiments.
        It only generates the single, unified comprehensive report.

        Returns
        -------
        dict
            A dictionary containing the path to the generated visualization file.
            {
                'comprehensive_report': 'path/to/experiment_comprehensive_report_YYYYMMDD_HHMMSS.png',
            }
        """
        logging.info("ES Module - Generating comprehensive experiment visualization from cache...")
        visualizations = {}

        try:
            # Get data from experiment log files (.json)
            experiment_files = []
            if os.path.exists(self.experiment_cache_dir):
                for file in os.listdir(self.experiment_cache_dir):
                    if file.startswith('experiment_log_') and file.endswith('.json'):
                        experiment_files.append(os.path.join(self.experiment_cache_dir, file))

            logging.info(f"ES Module - Collected {len(experiment_files)} experiment log files from cache for visualization.")

            if experiment_files:
                logging.info(f"ES Module - Found {len(experiment_files)} experiment records for visualization.")
                
                try:
                    from .visualize.visualize_experiment import generate_comprehensive_experiment_report
                    comprehensive_report_path = generate_comprehensive_experiment_report(self.config, experiment_files)
                    
                    visualizations['comprehensive_report'] = comprehensive_report_path
                    logging.info(f"ES Module - Comprehensive experiment report generated from cache: {comprehensive_report_path}")

                except Exception as e:
                    logging.error(f"ES Module - Failed to generate comprehensive experiment report from cache: {e}", exc_info=True)
                    
            else:
                logging.warning("ES Module - No experiment records found in cache for visualization.")
                
        except Exception as e:
            logging.error(f"ES Module - Error in generate_experiment_visualizations_from_cache: {e}", exc_info=True)
            visualizations['error'] = str(e)

        logging.info("ES Module - Comprehensive experiment visualization from cache generation completed.")
        return visualizations


# Utility functions for easy access
def create_experiment_scheme(config: ConfigSetup) -> ExperimentScheme:
    """
    Create and return an ExperimentScheme instance.
    
    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance
        
    Returns
    -------
    ExperimentScheme
        New ExperimentScheme instance
    """
    return ExperimentScheme(config)

def run_quick_experiment_sequence(config: ConfigSetup, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Run all quick experiments in sequence.
    
    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance
    symbols : list, optional
        List of stock symbols
        
    Returns
    -------
    dict
        Results from all quick experiments
    """
    scheme = ExperimentScheme(config)
    return scheme.run_experiment_sequence(
        ['quick_exper_1', 'quick_exper_2', 'quick_exper_3', 'quick_exper_4'], 
        symbols
    )

def run_full_experiment_sequence(config: ConfigSetup, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Run all full experiments in sequence.
    
    Parameters
    ----------
    config : ConfigSetup
        Configuration setup instance
    symbols : list, optional
        List of stock symbols
        
    Returns
    -------
    dict
        Results from all full experiments
    """
    scheme = ExperimentScheme(config)
    return scheme.run_experiment_sequence(
        ['full_exper_1', 'full_exper_2', 'full_exper_3'], 
        symbols
    )