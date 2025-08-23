import torch
import logging
from typing import Dict, Any, Tuple
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


class HardwareOptimizer:
    """Hardware-aware configuration optimizer for training"""
    
    def __init__(self):
        self.gpu_memory = self._get_gpu_memory()
        self.gpu_name = self._get_gpu_name()
        self.gpu_count = self._get_gpu_count()
        
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Get memory of the first GPU
            memory_bytes = torch.cuda.get_device_properties(0).total_memory
            return memory_bytes / (1024**3)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not get GPU memory: {e}")
            return 0.0
    
    def _get_gpu_name(self) -> str:
        """Get GPU name"""
        if not torch.cuda.is_available():
            return "CPU"
        
        try:
            return torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning(f"Could not get GPU name: {e}")
            return "Unknown GPU"
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs"""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    
    def get_optimal_batch_config(self, model_size_gb: float = 1.0) -> Tuple[int, int]:
        """
        Get optimal batch size and gradient accumulation steps
        
        Args:
            model_size_gb: Estimated model size in GB
            
        Returns:
            Tuple of (per_device_batch_size, gradient_accumulation_steps)
        """
        if self.gpu_memory <= 4:
            # Very low memory (old GPUs)
            return 1, 16
        elif self.gpu_memory <= 8:
            # Low memory (GTX 1080, RTX 3060, etc.)
            return 1, 8
        elif self.gpu_memory <= 12:
            # Medium memory (RTX 3060 Ti, RTX 4060 Ti)
            return 2, 4
        elif self.gpu_memory <= 16:
            # Good memory (RTX 3070, RTX 4070, RTX 5060 Ti)
            return 4, 2
        elif self.gpu_memory <= 24:
            # High memory (RTX 3090, RTX 4080)
            return 8, 1
        else:
            # Very high memory (RTX 4090, A100)
            return 16, 1
    
    def get_optimal_precision_config(self) -> Dict[str, bool]:
        """Get optimal precision configuration based on hardware"""
        # Check if BF16 is supported
        bf16_supported = (
            torch.cuda.is_available() and
            torch.cuda.is_bf16_supported()
        )
        
        if bf16_supported:
            return {"fp16": False, "bf16": True}
        elif torch.cuda.is_available():
            return {"fp16": True, "bf16": False}
        else:
            return {"fp16": False, "bf16": False}
    
    def get_optimal_memory_config(self) -> Dict[str, Any]:
        """Get optimal memory configuration"""
        if self.gpu_memory <= 8:
            return {
                "gradient_checkpointing": True,
                "dataloader_pin_memory": False,
                "dataloader_num_workers": 1,
                "remove_unused_columns": True
            }
        elif self.gpu_memory <= 16:
            return {
                "gradient_checkpointing": True,
                "dataloader_pin_memory": True,
                "dataloader_num_workers": 2,
                "remove_unused_columns": True
            }
        else:
            return {
                "gradient_checkpointing": False,
                "dataloader_pin_memory": True,
                "dataloader_num_workers": 4,
                "remove_unused_columns": True
            }
    
    def get_quantization_config(self, force_quantization: bool = False) -> Dict[str, Any]:
        """Get optimal quantization configuration"""
        if self.gpu_memory <= 12 or force_quantization:
            return {
                "enabled": True,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True
            }
        else:
            return {"enabled": False}
    
    def get_hardware_optimized_training_args(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get complete hardware-optimized training configuration"""
        
        # Get optimal configurations
        batch_size, grad_accum = self.get_optimal_batch_config()
        precision_config = self.get_optimal_precision_config()
        memory_config = self.get_optimal_memory_config()
        
        # Merge configurations
        optimized_config = {
            **base_config,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            **precision_config,
            **memory_config
        }
        
        # Add optimizer based on memory
        if self.gpu_memory <= 8:
            optimized_config["optim"] = "paged_adamw_8bit"
        else:
            optimized_config["optim"] = "adamw_torch"
        
        return optimized_config
    
    def log_hardware_info(self):
        """Log hardware information"""
        logger.info("=" * 50)
        logger.info("HARDWARE INFORMATION")
        logger.info("=" * 50)
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Count: {self.gpu_count}")
            logger.info(f"GPU Name: {self.gpu_name}")
            logger.info(f"GPU Memory: {self.gpu_memory:.1f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
        else:
            logger.info("Using CPU for training")
        logger.info("=" * 50)
