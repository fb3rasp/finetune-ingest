import os
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class TrainingLogger:
    """Enhanced logging system for training monitoring"""
    
    def __init__(self, log_dir: str, config: Optional[Dict[str, Any]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.log_level = self.config.get('level', 'INFO')
        self.log_to_file = self.config.get('log_to_file', True)
        self.log_hardware_info = self.config.get('log_hardware_info', True)
        self.log_training_metrics = self.config.get('log_training_metrics', True)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Training metrics storage
        self.training_metrics = []
        self.start_time = None
        
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with detailed format (if enabled)
        if self.log_to_file:
            log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.log_level.upper()))
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # Also create a latest.log symlink
            latest_log = self.log_dir / "latest.log"
            if latest_log.exists() or latest_log.is_symlink():
                latest_log.unlink()
            latest_log.symlink_to(log_file.name)
        
        # Suppress some noisy loggers
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
        logging.getLogger("transformers.generation.utils").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with comprehensive configuration"""
        self.start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 FINE-TUNING TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log directory: {self.log_dir}")
        
        # Log configuration
        self.logger.info("\n📋 TRAINING CONFIGURATION:")
        self._log_config_section("Model", config.get('model', {}))
        self._log_config_section("Training", config.get('training', {}))
        self._log_config_section("LoRA", config.get('lora', {}))
        self._log_config_section("Data", config.get('data', {}))
        
        if self.log_hardware_info:
            self.log_hardware_info_detailed()
    
    def _log_config_section(self, section_name: str, section_config: Dict[str, Any]):
        """Log a configuration section"""
        self.logger.info(f"\n  {section_name}:")
        for key, value in section_config.items():
            if isinstance(value, dict):
                self.logger.info(f"    {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"      {sub_key}: {sub_value}")
            else:
                self.logger.info(f"    {key}: {value}")
    
    def log_hardware_info_detailed(self):
        """Log comprehensive hardware information"""
        self.logger.info("\n🖥️  HARDWARE INFORMATION:")
        
        # Python and PyTorch versions
        self.logger.info(f"  Python version: {sys.version.split()[0]}")
        self.logger.info(f"  PyTorch version: {torch.__version__}")
        
        # CUDA information
        if torch.cuda.is_available():
            self.logger.info(f"  CUDA available: ✓")
            self.logger.info(f"  CUDA version: {torch.version.cuda}")
            self.logger.info(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_gb = gpu_props.total_memory / (1024**3)
                self.logger.info(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f} GB)")
            
            # Current GPU memory
            if torch.cuda.device_count() > 0:
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                self.logger.info(f"  GPU 0 memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        else:
            self.logger.info(f"  CUDA available: ✗ (using CPU)")
        
        # System information
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.logger.info(f"  System RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
            self.logger.info(f"  CPU cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
        except ImportError:
            self.logger.info("  System info: psutil not available")
    
    def log_dataset_info(self, dataset_stats: Dict[str, Any]):
        """Log dataset information"""
        self.logger.info("\n📊 DATASET INFORMATION:")
        self.logger.info(f"  Total examples: {dataset_stats.get('total_examples', 'N/A')}")
        self.logger.info(f"  Average text length: {dataset_stats.get('avg_text_length', 0):.1f} characters")
        self.logger.info(f"  Text length range: {dataset_stats.get('min_text_length', 0)} - {dataset_stats.get('max_text_length', 0)}")
        self.logger.info(f"  Template validation: {'✓ Passed' if dataset_stats.get('template_valid', False) else '✗ Failed'}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information"""
        self.logger.info("\n🤖 MODEL INFORMATION:")
        self.logger.info(f"  Base model: {model_info.get('base_model', 'N/A')}")
        self.logger.info(f"  Model size: {model_info.get('num_parameters', 'N/A')} parameters")
        self.logger.info(f"  Trainable parameters: {model_info.get('trainable_parameters', 'N/A')}")
        self.logger.info(f"  LoRA rank: {model_info.get('lora_rank', 'N/A')}")
        self.logger.info(f"  Target modules: {model_info.get('target_modules', 'N/A')}")
    
    def log_training_step(self, step: int, metrics: Dict[str, float]):
        """Log training step metrics"""
        if self.log_training_metrics:
            # Add timestamp and step info
            timestamped_metrics = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                **metrics
            }
            self.training_metrics.append(timestamped_metrics)
            
            # Log to console every few steps
            if step % 10 == 0:
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
                self.logger.info(f"Step {step}: {metrics_str}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log end of epoch metrics"""
        self.logger.info(f"\n📈 EPOCH {epoch} COMPLETED:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def log_training_complete(self, final_metrics: Dict[str, Any]):
        """Log training completion"""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 80)
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            self.logger.info(f"Total duration: {str(duration).split('.')[0]}")
        
        # Log final metrics
        if final_metrics:
            self.logger.info("\n📊 FINAL METRICS:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        # Save training metrics to file
        if self.log_training_metrics and self.training_metrics:
            self.save_training_metrics()
    
    def save_training_metrics(self):
        """Save training metrics to JSON file"""
        try:
            metrics_file = self.log_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            self.logger.info(f"📁 Training metrics saved to: {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save training metrics: {e}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error("=" * 50)
        self.logger.error(f"❌ ERROR: {context}")
        self.logger.error("=" * 50)
        self.logger.error(f"Error type: {type(error).__name__}")
        self.logger.error(f"Error message: {str(error)}")
        
        # Log stack trace in debug mode
        if self.logger.isEnabledFor(logging.DEBUG):
            import traceback
            self.logger.debug("Stack trace:")
            self.logger.debug(traceback.format_exc())
    
    def log_warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log warning with optional details"""
        self.logger.warning(f"⚠️  {message}")
        if details:
            for key, value in details.items():
                self.logger.warning(f"  {key}: {value}")
    
    def log_success(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log success message with optional details"""
        self.logger.info(f"✅ {message}")
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def create_progress_callback(self):
        """Create a callback function for training progress"""
        def progress_callback(step: int, total_steps: int, metrics: Dict[str, float]):
            percentage = (step / total_steps) * 100 if total_steps > 0 else 0
            self.log_training_step(step, {**metrics, "progress": percentage})
        
        return progress_callback
