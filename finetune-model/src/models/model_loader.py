import os
import torch
import logging
from typing import Tuple, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


class ModelLoader:
    """Enhanced model loader with progressive fallback strategies"""
    
    def __init__(self, model_config: Dict[str, Any], hardware_config: Dict[str, Any]):
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.base_model = model_config['base_model']
        self.trust_remote_code = model_config.get('trust_remote_code', True)
        
        # Get HuggingFace token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
    def load_model_with_fallback(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model with progressive fallback strategies
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If all loading strategies fail
        """
        logger.info(f"Loading model: {self.base_model}")
        
        # Define loading strategies in order of preference
        strategies = [
            ("quantized (4-bit)", self._load_with_quantization),
            ("full precision", self._load_without_quantization),
            ("CPU only", self._load_cpu_only)
        ]
        
        last_error = None
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Attempting to load model with {strategy_name} strategy...")
                model, tokenizer = strategy_func()
                
                # Validate the loaded model
                self._validate_model(model, tokenizer)
                
                logger.info(f"✅ Model loaded successfully with {strategy_name} strategy")
                return model, tokenizer
                
            except Exception as e:
                logger.warning(f"❌ {strategy_name.capitalize()} loading failed: {e}")
                last_error = e
                continue
        
        # If we get here, all strategies failed
        raise RuntimeError(f"All model loading strategies failed. Last error: {last_error}")
    
    def _load_with_quantization(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Primary loading strategy with 4-bit quantization"""
        if not self.hardware_config.get('quantization', {}).get('enabled', True):
            raise ValueError("Quantization disabled in config")
        
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available for quantization")
        
        # Create quantization config
        bnb_config = self._create_quantization_config()
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            token=self.hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Load tokenizer
        tokenizer = self._load_tokenizer()
        
        return model, tokenizer
    
    def _load_without_quantization(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Fallback strategy without quantization"""
        # Determine dtype based on hardware
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        logger.info(f"Loading model with dtype: {torch_dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            token=self.hf_token,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True
        )
        
        tokenizer = self._load_tokenizer()
        
        return model, tokenizer
    
    def _load_cpu_only(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Last resort strategy - CPU only"""
        logger.warning("Loading model on CPU only - this will be slow!")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            token=self.hf_token,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True
        )
        
        tokenizer = self._load_tokenizer()
        
        return model, tokenizer
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            token=self.hf_token,
            trust_remote_code=self.trust_remote_code
        )
        
        # Configure padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                # Add a pad token if eos_token is also None
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Set padding side based on model architecture
        padding_side = self._get_padding_side()
        tokenizer.padding_side = padding_side
        
        logger.info(f"Tokenizer configured with padding side: {padding_side}")
        
        return tokenizer
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration"""
        quant_config = self.hardware_config.get('quantization', {})
        
        # Convert string dtype to torch dtype
        compute_dtype_str = quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
        if compute_dtype_str == 'bfloat16':
            compute_dtype = torch.bfloat16
        elif compute_dtype_str == 'float16':
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get('load_in_4bit', True),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
        )
        
        logger.info(f"Created quantization config: {bnb_config}")
        return bnb_config
    
    def _get_padding_side(self) -> str:
        """Get appropriate padding side for the model architecture"""
        model_name_lower = self.base_model.lower()
        
        if any(arch in model_name_lower for arch in ['llama', 'mistral', 'phi']):
            return "right"
        elif any(arch in model_name_lower for arch in ['gpt', 'opt']):
            return "left"
        else:
            # Default to right padding
            return "right"
    
    def _validate_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """Validate that the model and tokenizer are properly loaded"""
        try:
            # Check model is in expected state
            if model is None:
                raise ValueError("Model is None")
            
            # Check tokenizer
            if tokenizer is None:
                raise ValueError("Tokenizer is None")
            
            # Test basic tokenization
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text, return_tensors="pt")
            if tokens is None or tokens.shape[1] == 0:
                raise ValueError("Tokenizer failed to encode test text")
            
            # Test model forward pass (if on GPU and not too large)
            if torch.cuda.is_available() and tokens.shape[1] < 100:
                try:
                    with torch.no_grad():
                        if hasattr(model, 'device') and model.device.type == 'cuda':
                            tokens = tokens.to(model.device)
                        outputs = model(tokens)
                        if outputs is None or not hasattr(outputs, 'logits'):
                            raise ValueError("Model forward pass failed")
                except Exception as e:
                    logger.warning(f"Model validation forward pass failed (non-critical): {e}")
            
            logger.info("✅ Model and tokenizer validation passed")
            
        except Exception as e:
            raise ValueError(f"Model validation failed: {e}")
    
    def configure_model_for_training(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Configure model settings for training"""
        # Disable caching for training
        model.config.use_cache = False
        
        # Set pretraining_tp if it exists (for some models)
        if hasattr(model.config, 'pretraining_tp'):
            model.config.pretraining_tp = 1
        
        # Enable gradient checkpointing if specified
        if self.hardware_config.get('training', {}).get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        
        return model
    
    def get_model_info(self, model: AutoModelForCausalLM) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "base_model": self.base_model,
            "model_type": getattr(model.config, 'model_type', 'unknown'),
            "num_parameters": self._count_parameters(model),
            "device": str(model.device) if hasattr(model, 'device') else 'unknown',
            "dtype": str(model.dtype) if hasattr(model, 'dtype') else 'unknown',
            "is_quantized": self._is_quantized(model),
            "gradient_checkpointing": getattr(model.config, 'gradient_checkpointing', False)
        }
        
        return info
    
    def _count_parameters(self, model: AutoModelForCausalLM) -> str:
        """Count model parameters"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            def format_number(num):
                if num >= 1e9:
                    return f"{num/1e9:.1f}B"
                elif num >= 1e6:
                    return f"{num/1e6:.1f}M"
                elif num >= 1e3:
                    return f"{num/1e3:.1f}K"
                else:
                    return str(num)
            
            return f"{format_number(total_params)} total, {format_number(trainable_params)} trainable"
            
        except Exception as e:
            logger.warning(f"Could not count parameters: {e}")
            return "unknown"
    
    def _is_quantized(self, model: AutoModelForCausalLM) -> bool:
        """Check if model is quantized"""
        try:
            # Check for common quantization indicators
            if hasattr(model, 'hf_device_map'):
                return True
            
            # Check if any parameters are not float32/float16/bfloat16
            for param in model.parameters():
                if param.dtype in [torch.int8, torch.uint8]:
                    return True
            
            return False
            
        except Exception:
            return False
