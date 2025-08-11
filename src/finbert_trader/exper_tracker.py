# finbert_trader/exper_tracker.py
"""
Experiment Tracker Module for FinBERT-Driven Trading System
Purpose: Track and log experiments for systematic parameter tuning and analysis
"""

import logging
import json
import pickle
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentTracker:
    """
    Class for tracking experiments and their results in the trading system.
    
    This tracker logs experiment configurations, results, and generates comprehensive reports
    to support systematic parameter tuning and analysis for the FinBERT trading system.
    """
    
    def __init__(self, config):
        """
        Initialize ExperimentTracker.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory to store experiment records (default: 'exper_cache')
        """
        self.config = config
        self.exper_cache_dir = getattr(self.config, 'EXPERIMENT_CACHE_DIR', 'exper_cache')
        self.experiments = []
        os.makedirs(self.exper_cache_dir, exist_ok=True)
        logging.info(f"ET Module - Initialized ExperimentTracker with cache dir: {self.exper_cache_dir}")
    
    def log_experiment(self, experiment_id: str, config_params: Dict[str, Any], 
                      results: Dict[str, Any], notes: str = "") -> str:
        """
        Log an experiment with its parameters and results.
        
        Parameters
        ----------
        experiment_id : str
            Unique identifier for the experiment
        config_params : dict
            Configuration parameters used in the experiment
        results : dict
            Experiment results
        notes : str, optional
            Additional notes about the experiment
            
        Returns
        -------
        str
            Path to saved experiment log file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            experiment_log = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'config_params': config_params,
                'results': results,
                'notes': notes,
                'metrics_summary': self._extract_metrics_summary(results)
            }
            
            self.experiments.append(experiment_log)
            
            # Save to JSON file for easy reading
            log_filename = f"experiment_log_{experiment_id}_{timestamp}.json"
            log_path = os.path.join(self.exper_cache_dir, log_filename)
            
            with open(log_path, 'w') as file:
                json.dump(experiment_log, file, indent=2, default=str)
                
            # Save to pickle file for complete data preservation
            pkl_filename = f"experiment_log_{experiment_id}_{timestamp}.pkl"
            pkl_path = os.path.join(self.exper_cache_dir, pkl_filename)
            
            with open(pkl_path, 'wb') as file:
                pickle.dump(experiment_log, file)
            
            logging.info(f"ET Module - Experiment {experiment_id} logged to: {log_path}")
            return log_path
            
        except Exception as e:
            logging.error(f"ET Module - Error logging experiment {experiment_id}: {e}")
            raise
    
    def _extract_metrics_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics summary from experiment results.
        
        Parameters
        ----------
        results : dict
            Experiment results
            
        Returns
        -------
        dict
            Key metrics summary
        """
        try:
            metrics_summary = {}
            
            if isinstance(results, dict):
                # Handle pipeline results with multiple algorithms
                if 'individual_results' in results:
                    for algo_name, algo_results in results['individual_results'].items():
                        if 'metrics' in algo_results:
                            metrics = algo_results['metrics']
                            metrics_summary[algo_name] = {
                                'cagr': metrics.get('cagr', 0),
                                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                                'max_drawdown': metrics.get('max_drawdown', 0),
                                'win_rate': metrics.get('win_rate', 0),
                                'final_asset': metrics.get('final_asset', 0)
                            }
                # Handle single algorithm results
                elif 'metrics' in results:
                    metrics = results['metrics']
                    metrics_summary = {
                        'cagr': metrics.get('cagr', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'final_asset': metrics.get('final_asset', 0)
                    }
            
            return metrics_summary
            
        except Exception as e:
            logging.error(f"ET Module - Error extracting metrics summary: {e}")
            return {}
    
    def load_experiment(self, experiment_path: str) -> Dict[str, Any]:
        """
        Load experiment from file.
        
        Parameters
        ----------
        experiment_path : str
            Path to experiment log file
            
        Returns
        -------
        dict
            Loaded experiment data
        """
        try:
            with open(experiment_path, 'r') as f:
                if experiment_path.endswith('.json'):
                    experiment_data = json.load(f)
                else:
                    raise ValueError("Unsupported file format. Use JSON files.")
            
            logging.info(f"ET Module - Loaded experiment from: {experiment_path}")
            return experiment_data
            
        except Exception as e:
            logging.error(f"ET Module - Error loading experiment from {experiment_path}: {e}")
            raise
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all logged experiments.
        
        Returns
        -------
        list
            List of all experiment records
        """
        return self.experiments.copy()
    
    def generate_experiment_report(self) -> str:
        """
        Generate comprehensive experiment report.
        
        Returns
        -------
        str
            Path to saved report file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create detailed report
            report = {
                'report_generated': timestamp,
                'total_experiments': len(self.experiments),
                'experiments_summary': []
            }
            
            for exper in self.experiments:
                exper_summary = {
                    'experiment_id': exper['experiment_id'],
                    'timestamp': exper['timestamp'],
                    'notes': exper['notes'],
                    'metrics_summary': exper.get('metrics_summary', {})
                }
                report['experiments_summary'].append(exper_summary)
            
            # Save report
            report_filename = f"experiment_report_{timestamp}.json"
            report_path = os.path.join(self.exper_cache_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logging.info(f"ET Module - Experiment report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"ET Module - Error generating experiment report: {e}")
            raise
    
    def clear_experiments(self):
        """Clear all stored experiments."""
        self.experiments.clear()
        logging.info("ET Module - Cleared all stored experiments")

# Utility functions
def load_experiment_tracker(config) -> ExperimentTracker:
    """
    Load existing ExperimentTracker with saved experiments.
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory containing experiment records
        
    Returns
    -------
    ExperimentTracker
        ExperimentTracker instance with loaded experiments
    """
    tracker = ExperimentTracker(config)
    
    # Load existing experiment files
    try:
        for file in os.listdir(tracker.exper_cache_dir):
            if file.startswith('experiment_log_') and file.endswith('.json'):
                file_path = os.path.join(tracker.exper_cache_dir, file)
                experiment_data = tracker.load_experiment(file_path)
                tracker.experiments.append(experiment_data)
        logging.info(f"ET Module - Loaded {len(tracker.experiments)} existing experiments")
    except Exception as e:
        logging.warning(f"ET Module - Could not load existing experiments: {e}")
    
    return tracker