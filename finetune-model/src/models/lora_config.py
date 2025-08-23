import logging
from typing import Dict, Any, List
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class LoRAConfigManager:
    """Manager for LoRA configuration and model adaptation"""
    
    def __init__(self, lora_config: Dict[str, Any], model_config: Dict[str, Any]):
        self.lora_config = lora_config
        self.model_config = model_config
        self.base_model_name = model_config['base_model']
        
    def create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration with architecture-specific optimizations
        
        Returns:
            Configured LoraConfig object
        """
        # Get architecture-specific settings
        arch_config = self._get_architecture_config()
        
        # Merge with user config, prioritizing user settings
        target_modules = self.lora_config.get('target_modules') or arch_config['target_modules']
        lora_r = self.lora_config.get('r', arch_config.get('lora_r', 16))
        lora_alpha = self.lora_config.get('alpha', arch_config.get('lora_alpha', 32))
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=self.lora_config.get('dropout', 0.1),
            bias=self.lora_config.get('bias', "none"),
            task_type=self.lora_config.get('task_type', "CAUSAL_LM"),
            target_modules=target_modules
        )
        
        logger.info("LoRA Configuration:")
        logger.info(f"  Rank (r): {lora_r}")
        logger.info(f"  Alpha: {lora_alpha}")
        logger.info(f"  Dropout: {peft_config.lora_dropout}")
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Bias: {peft_config.bias}")
        
        return peft_config
    
    def _get_architecture_config(self) -> Dict[str, Any]:
        """Get architecture-specific LoRA configuration"""
        model_name_lower = self.base_model_name.lower()
        
        if "llama" in model_name_lower:
            return self._get_llama_config()
        elif "mistral" in model_name_lower:
            return self._get_mistral_config()
        elif "phi" in model_name_lower:
            return self._get_phi_config()
        elif "qwen" in model_name_lower:
            return self._get_qwen_config()
        elif "gemma" in model_name_lower:
            return self._get_gemma_config()
        else:
            logger.warning(f"Unknown architecture for {self.base_model_name}, using default config")
            return self._get_default_config()
    
    def _get_llama_config(self) -> Dict[str, Any]:
        """Configuration for Llama models"""
        return {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_r": 16,
            "lora_alpha": 32
        }
    
    def _get_mistral_config(self) -> Dict[str, Any]:
        """Configuration for Mistral models"""
        return {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_r": 64,
            "lora_alpha": 16
        }
    
    def _get_phi_config(self) -> Dict[str, Any]:
        """Configuration for Phi models"""
        return {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "dense",
                "fc1", "fc2"
            ],
            "lora_r": 32,
            "lora_alpha": 32
        }
    
    def _get_qwen_config(self) -> Dict[str, Any]:
        """Configuration for Qwen models"""
        return {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_r": 64,
            "lora_alpha": 16
        }
    
    def _get_gemma_config(self) -> Dict[str, Any]:
        """Configuration for Gemma models"""
        return {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_r": 16,
            "lora_alpha": 32
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for unknown architectures"""
        return {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ],
            "lora_r": 16,
            "lora_alpha": 32
        }
    
    def apply_lora_to_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Apply LoRA adaptation to the model
        
        Args:
            model: The base model to adapt
            
        Returns:
            LoRA-adapted model
        """
        logger.info("Applying LoRA adaptation to model...")
        
        # Create LoRA config
        peft_config = self.create_lora_config()
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters info
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"Trainable parameters: {trainable_params:,} / {all_param:,} "
                   f"({100 * trainable_params / all_param:.2f}%)")
        
        return model
    
    def get_optimal_rank_for_model_size(self, model_size_str: str) -> int:
        """
        Get optimal LoRA rank based on model size
        
        Args:
            model_size_str: Model size string (e.g., "1B", "7B", "13B")
            
        Returns:
            Optimal rank value
        """
        # Extract numeric value
        size_lower = model_size_str.lower()
        
        if "1b" in size_lower or "1.3b" in size_lower:
            return 8   # Smaller rank for 1B models
        elif "3b" in size_lower or "7b" in size_lower:
            return 16  # Medium rank for 3-7B models
        elif "13b" in size_lower or "15b" in size_lower:
            return 32  # Larger rank for 13-15B models
        elif "30b" in size_lower or "33b" in size_lower:
            return 64  # Large rank for 30B+ models
        elif "70b" in size_lower or "65b" in size_lower:
            return 128 # Very large rank for 70B+ models
        else:
            return 16  # Default rank
    
    def validate_target_modules(self, model: AutoModelForCausalLM, target_modules: List[str]) -> List[str]:
        """
        Validate and filter target modules based on actual model architecture
        
        Args:
            model: The model to validate against
            target_modules: List of target module names
            
        Returns:
            List of valid target modules
        """
        valid_modules = []
        model_modules = set()
        
        # Collect all module names from the model
        for name, _ in model.named_modules():
            if "." in name:
                module_type = name.split(".")[-1]
                model_modules.add(module_type)
        
        # Filter target modules to only include those that exist in the model
        for module in target_modules:
            if module in model_modules:
                valid_modules.append(module)
            else:
                logger.warning(f"Target module '{module}' not found in model architecture")
        
        if not valid_modules:
            logger.error("No valid target modules found! Using default 'q_proj' and 'v_proj'")
            valid_modules = ["q_proj", "v_proj"]
        
        logger.info(f"Valid target modules: {valid_modules}")
        return valid_modules
    
    def get_lora_scaling_factor(self) -> float:
        """
        Calculate LoRA scaling factor (alpha/r)
        
        Returns:
            Scaling factor
        """
        r = self.lora_config.get('r', 16)
        alpha = self.lora_config.get('alpha', 32)
        return alpha / r
    
    def estimate_memory_usage(self, base_model_params: int) -> Dict[str, float]:
        """
        Estimate additional memory usage from LoRA adaptation
        
        Args:
            base_model_params: Number of parameters in base model
            
        Returns:
            Dictionary with memory estimates in MB
        """
        r = self.lora_config.get('r', 16)
        target_modules = self.lora_config.get('target_modules', [])
        
        # Rough estimate: each target module adds 2 * r * original_dim parameters
        # Assuming average dimension of 4096 for transformer models
        avg_dim = 4096
        additional_params = len(target_modules) * 2 * r * avg_dim
        
        # Convert to memory (assuming float16)
        additional_memory_mb = additional_params * 2 / (1024 * 1024)
        
        return {
            "additional_parameters": additional_params,
            "additional_memory_mb": additional_memory_mb,
            "memory_overhead_percent": (additional_params / base_model_params) * 100
        }
