# src/finbert_trader/exper_scheme.py
"""
Experiment Scheme Module for FinBERT-Driven Trading System
Purpose: Provide systematic experiment schemes for parameter tuning and validation
"""

import logging
import os
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
        
        # Use unified cache directories from ConfigSetup
        self.raw_data_dir = getattr(config, 'RAW_DATA_DIR', 'raw_data_cache')
        self.processed_news_dir = getattr(config, 'PROCESSED_NEWS_DIR', 'processed_news_cache')
        self.fused_data_dir = getattr(config, 'FUSED_DATA_DIR', 'fused_data_cache')
        self.exper_data_dir = getattr(config, 'EXPER_DATA_DIR', 'exper_data_cache')
        self.plot_cache_dir = getattr(config, 'PLOT_CACHE_DIR', 'plot_cache')
        self.plot_news_dir = getattr(config, 'PLOT_NEWS_DIR', 'plot_news_cache')
        self.results_cache_dir = getattr(config, 'RESULTS_CACHE_DIR', 'results_cache')
        self.experiment_cache_dir = getattr(config, 'EXPERIMENT_CACHE_DIR', 'exper_cache')
        self.scaler_cache_dir = getattr(config, 'SCALER_CACHE_DIR', 'scaler_cache')
        self.log_dir = getattr(config, 'LOG_SAVE_DIR', 'logs')
        
        # Ensure all directories exist
        for dir_path in [self.raw_data_dir, self.exper_data_dir, self.plot_cache_dir,
                        self.results_cache_dir, self.experiment_cache_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize experiment tracker with unified config
        self.experiment_tracker = ExperimentTracker(self.config)

        
        logging.info("ES Module - Initialized ExperimentScheme with unified configuration")
        logging.info(f"ES Module - Raw data directory: {self.raw_data_dir}")
        logging.info(f"ES Module - Experiment data directory: {self.exper_data_dir}")
        logging.info(f"ES Module - Plot cache directory: {self.plot_cache_dir}")
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
        
        # Minimal configuration for quick validation
        setup_config = {
            'symbols': self.symbols,
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            'exper_mode': {
                'rl_algorithm': ['PPO']  # Single algorithm for speed
            },
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'smooth_window_size': self.config.smooth_window_size,
            'filter_ind': self.filter_ind,
            'ind_mode': 'long',
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'use_senti_factor': False,
            'use_risk_factor': False,
            'use_senti_features': False,
            'use_risk_features': False,
            'use_senti_threshold': False,
            'use_risk_threshold': False,
            'use_dynamic_infusion': False,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_CACHE_DIR': self.plot_cache_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 100000,
            'total_timesteps': 200000,  
            'reward_scaling': 1e-3,
            'cash_penalty_proportion': 0.0005,
            'commission_rate': 0.001
        }
        
        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_1',
            setup_config=setup_config,
            trading_config=trading_config,
            description="Quick validation of basic parameters and pipeline",
            notes="Minimal configuration for rapid feedback on asset calculation and basic mechanics"
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
        
        # Minimal configuration for quick validation
        setup_config = {
            'symbols': self.symbols,
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            'exper_mode': {
                'rl_algorithm': ['PPO']  # Single algorithm for speed
            },
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'smooth_window_size': self.config.smooth_window_size,
            'filter_ind': self.filter_ind,
            'ind_mode': self.config.ind_mode,
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            'use_senti_threshold': False,
            'use_risk_threshold': False,
            'use_dynamic_infusion': False,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_CACHE_DIR': self.plot_cache_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,  # Larger capital for better signal-to-noise
            'total_timesteps': 200000,
            'reward_scaling': 1e-3,   # Enhanced reward scaling
            'cash_penalty_proportion': 0.0005,  # Reduced cash penalty
            'infusion_strength': 0.05,  # Increased sentiment/risk influence
            'commission_rate': 0.0001   # Lower transaction costs
        }
        
        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_2',
            setup_config=setup_config,
            trading_config=trading_config,
            description="Quick optimization of reward function parameters",
            notes="Enhanced reward signals with optimized penalty structure"
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
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            'exper_mode': {
                'rl_algorithm': ['PPO']  # Single algorithm for speed
            },
            'window_size': 50,  # Smaller window for faster processing
            'window_factor': 2,
            'window_extend': 50,
            'smooth_window_size': self.config.smooth_window_size,
            'filter_ind': self.filter_ind,
            'ind_mode': self.config.ind_mode,
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            'use_senti_threshold': True,
            'use_risk_threshold': True,
            'use_dynamic_infusion': False,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_CACHE_DIR': self.plot_cache_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 500000,
            'total_timesteps': 150000,
            'reward_scaling': 1e-2
        }
        
        # Custom model parameters for this experiment
        model_params = {
            'PPO': {
                "n_steps": 1024,
                "ent_coef": 0.01,       # Reduced entropy for more deterministic policy
                "learning_rate": 0.0003, # Slightly higher learning rate
                "batch_size": 32,       # Smaller batch size for more frequent updates
                "gamma": 0.99,
                "gae_lambda": 0.95
            }
        }
        
        return self._execute_experiment_with_visualization(
            experiment_id='quick_exper_3',
            setup_config=setup_config,
            trading_config=trading_config,
            model_params=model_params,
            description="Quick tuning of RL hyperparameters",
            notes="Optimized PPO parameters for better learning efficiency"
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
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO', 'A2C']  # All algorithms
            },
            'window_size': 50,
            'smooth_window_size': self.config.smooth_window_size,
            'filter_ind': self.filter_ind,
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_CACHE_DIR': self.plot_cache_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,
            'total_timesteps': 500000,  # Extended training
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
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO']  # Focus on top performers
            },
            'window_size': 50,
            'smooth_window_size': self.config.smooth_window_size,
            'filter_ind': self.filter_ind,
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_CACHE_DIR': self.plot_cache_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,
            'total_timesteps': 1000000,  # Long training for complex reward function
            'reward_scaling': 1e-2,
            'cash_penalty_proportion': 0.0005,
            'infusion_strength': 0.05,
            'commission_rate': 0.0005,
            'cvar_factor': 0.1,  # Enhanced CVaR weighting
            'risk_aversion': 0.1
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
            'start': self.config.start,
            'end': self.config.end,
            'train_start_date': self.config.train_start_date,
            'train_end_date': self.config.train_end_date,
            'valid_start_date': self.config.valid_start_date,
            'valid_end_date': self.config.valid_end_date,
            'test_start_date': self.config.test_start_date,
            'test_end_date': self.config.test_end_date,
            'exper_mode': {
                'rl_algorithm': ['PPO', 'CPPO']  # Production algorithms
            },
            'window_size': 50,
            'smooth_window_size': self.config.smooth_window_size,
            'filter_ind': self.filter_ind,
            'force_process_news': self.config.force_process_news,
            'force_fuse_data': self.config.force_fuse_data,
            'force_normalize_features': self.config.force_normalize_features,    # Ensure normalize target columns
            'use_senti_factor': True,
            'use_risk_factor': True,
            'use_senti_features': True,
            'use_risk_features': True,
            'use_senti_threshold': True,
            'use_risk_threshold': True,
            'RAW_DATA_DIR': self.raw_data_dir,
            'PROCESSED_NEWS_DIR': self.processed_news_dir,
            'FUSED_DATA_DIR': self.fused_data_dir,
            'EXPER_DATA_DIR': self.exper_data_dir,
            'PLOT_CACHE_DIR': self.plot_cache_dir,
            'PLOT_NEWS_DIR': self.plot_news_dir,
            'RESULTS_CACHE_DIR': self.results_cache_dir,
            'EXPERIMENT_CACHE_DIR': self.experiment_cache_dir,
            'SCALER_CACHE_DIR': self.scaler_cache_dir,
            'LOG_SAVE_DIR': self.log_dir
        }
        
        trading_config = {
            'initial_cash': 1000000,
            'total_timesteps': 2000000,  # Production-level training
            'reward_scaling': 1e-2,
            'cash_penalty_proportion': 0.0001,  # Minimal cash penalty
            'infusion_strength': 0.1,   # Maximum sentiment/risk influence
            'commission_rate': 0.0001,  # Realistic low costs
            'cvar_factor': 0.15,        # Strong risk management
            'risk_aversion': 0.15,
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
            
            # Step 1: Initialize configuration
            config_setup = ConfigSetup(setup_config)
            logging.info(f"ES Module - ConfigSetup initialized with symbols: {config_setup.symbols}")
            
            # Step 2: Fetch data
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
            
            # Step 3: Feature engineering
            feature_engineer = FeatureEngineer(config_setup)
            exper_data_dict = feature_engineer.generate_experiment_data(
                stock_data_dict, news_chunks_gen, exper_mode='rl_algorithm'
            )
            logging.info(f"ES Module - Generated experiment data for modes: {list(exper_data_dict.keys())}")
            
            # Step 4: Process each experiment mode
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
                
                # Process training, validation, and testing
                mode_results = self._process_mode(mode_name, mode_data, config_trading)
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
            Experiment results with visualization paths
        """
        try:
            logging.info(f"ES Module - Executing experiment with visualization: {experiment_id}")
            
            # Execute experiment
            results = self._execute_experiment(experiment_id, setup_config, trading_config, 
                                            model_params, description, notes)
            
            # Visualize in time
            visualization_results = {}
            try:
                # Generate asset curve
                asset_plot = self.experiment_visualizer.base_visualizer.generate_asset_curve_comparison(results)
                visualization_results['asset_curve_plot'] = asset_plot
                
                # Generate performance heatmap
                heatmap_plot = self.experiment_visualizer.base_visualizer.generate_performance_heatmap(results)
                visualization_results['heatmap_plot'] = heatmap_plot
                
                logging.info(f"ES Module - Generated immediate visualizations for {experiment_id}")
                logging.info(f"  Asset curve plot: {asset_plot}")
                logging.info(f"  Heatmap plot: {heatmap_plot}")
                
            except Exception as viz_error:
                logging.warning(f"ES Module - Could not generate immediate visualizations: {viz_error}")
            
            # Extend visualization to results
            results['immediate_visualizations'] = visualization_results
            
            return results
            
        except Exception as e:
            logging.error(f"ES Module - Error executing experiment with visualization {experiment_id}: {e}")
            raise

    
    def _process_mode(self, mode_name: str, mode_data: Dict[str, Any], 
                     config_trading: ConfigTrading) -> Dict[str, Any]:
        """
        Process a single experiment mode.
        
        Parameters
        ----------
        mode_name : str
            Name of the experiment mode
        mode_data : dict
            Data for this mode
        config_trading : ConfigTrading
            Trading configuration
            
        Returns
        -------
        dict
            Results for this mode
        """
        try:
            logging.info(f"ES Module - Processing mode {mode_name}")
            
            # Extract data splits
            train_data = mode_data['train']
            valid_data = mode_data['valid']
            test_data = mode_data['test']
            
            logging.info(f"ES Module - Mode {mode_name} - Data splits: "
                        f"Train({len(train_data)}), Valid({len(valid_data)}), Test({len(test_data)})")
            
            # Create environments
            logging.info(f"ES Module - Mode {mode_name} - Creating trading environments")
            train_env = StockTradingEnv(config_trading, train_data, env_type='train')
            valid_env = StockTradingEnv(config_trading, valid_data, env_type='valid')
            test_env = StockTradingEnv(config_trading, test_data, env_type='test')
            
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
            
            # Backtesting
            logging.info(f"ES Module - Mode {mode_name} - Starting backtesting")
            backtester = TradingBacktest(config_trading)
            backtest_results = backtester.run_backtest(agent, test_env, record_trades=True)
            
            # Generate detailed report
            detailed_report = backtester.generate_detailed_report(backtest_results)

            # Generate trading history analysis
            symbols_list = getattr(config_trading, 'symbols', None)
            initial_value = backtest_results.get('asset_history', [1.0])[0] if backtest_results.get('asset_history') else 1.0
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
                'backtest_results': backtest_results
            }
            
            # Log key metrics
            metrics = backtest_results['metrics']
            logging.info(f"ES Module - Mode {mode_name} - Key Results:")
            logging.info(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            logging.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logging.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            logging.info(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
            
            return mode_results
            
        except Exception as e:
            logging.error(f"ES Module - Error processing mode {mode_name}: {e}")
            raise
    
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
        ['quick_exper_1', 'quick_exper_2', 'quick_exper_3'], 
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