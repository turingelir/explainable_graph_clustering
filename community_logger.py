# community_logger.py

import os
import logging
from datetime import datetime
import json

class CommunityLogger:
    """Logger utility for community detection experiments"""
    
    def __init__(self, dataset_name, log_dir='experiment_logs'):
        """
        Initialize logger for community detection experiments
        
        Args:
            dataset_name (str): Name of the dataset being used
            log_dir (str): Directory to store log files
        """
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'{dataset_name}_experiment_{timestamp}.log'
        self.log_path = os.path.join(log_dir, log_filename)
        
        # Configure logger
        self.logger = logging.getLogger(f'community_detection_{dataset_name}')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_config(self, config):
        """Log experiment configuration"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting new experiment on {self.dataset_name} dataset")
        self.logger.info(f"Configuration:\n{json.dumps(config, indent=2)}")
        self.logger.info(f"{'='*50}\n")
    
    def log_epoch(self, epoch, loss, metrics=None):
        """Log training epoch information"""
        log_msg = f"Epoch {epoch}: Loss = {loss:.4f}"
        if metrics:
            log_msg += f" | {' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}"
        self.logger.info(log_msg)
    
    def log_early_stopping(self, epoch):
        """Log early stopping event"""
        self.logger.info(f"\nEarly stopping triggered at epoch {epoch}")
    
    def log_metrics(self, metrics, section_name="Results"):
        """Log evaluation metrics"""
        self.logger.info(f"\n{'='*20} {section_name} {'='*20}")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric_name}: {value:.4f}")
            else:
                self.logger.info(f"{metric_name}: {value}")
        self.logger.info('='*50)
    
    def log_error(self, error_msg):
        """Log error messages"""
        self.logger.error(f"Error occurred: {error_msg}")
        
    def log_info(self, message):
        """Log general information"""
        self.logger.info(message)
        
    def log_warning(self, message):
        """Log warning messages"""
        self.logger.warning(message)
        
    def log_debug(self, message):
        """Log debug messages"""
        self.logger.debug(message)

    def log_model_summary(self, model):
        """Log model architecture summary"""
        self.logger.info("\nModel Architecture:")
        self.logger.info(str(model))
        
    def log_dataset_info(self, data_info):
        """Log dataset information"""
        self.logger.info("\nDataset Information:")
        for key, value in data_info.items():
            self.logger.info(f"{key}: {value}")