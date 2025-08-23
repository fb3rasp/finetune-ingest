from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir: str = "./policy-chatbot-v1"
    max_seq_length: int = 512
    trust_remote_code: bool = True
    
    @classmethod
    def get_architecture_specific_config(cls, model_name: str) -> Dict[str, Any]:
        """Return architecture-specific configurations"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return {
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "lora_r": 16,
                "lora_alpha": 32,
                "padding_side": "right"
            }
        elif "mistral" in model_name_lower:
            return {
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_r": 64,
                "lora_alpha": 16,
                "padding_side": "right"
            }
        elif "phi" in model_name_lower:
            return {
                "target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
                "lora_r": 32,
                "lora_alpha": 32,
                "padding_side": "left"
            }
        else:
            # Default configuration
            return {
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_r": 16,
                "lora_alpha": 32,
                "padding_side": "right"
            }


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    save_steps: int = 25
    logging_steps: int = 5
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 3
    remove_unused_columns: bool = True
    gradient_checkpointing: bool = True
    
    # Will be set by hardware optimizer
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    dataloader_pin_memory: Optional[bool] = None


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None


@dataclass
class DataConfig:
    dataset_path: str = "training_dataset.jsonl"
    validation_split: float = 0.0
    max_text_length: int = 2048
    template_format: str = "alpaca"


@dataclass
class HardwareConfig:
    auto_optimize: bool = True
    force_cpu: bool = False
    mixed_precision: str = "bf16"
    quantization: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.quantization is None:
            self.quantization = {
                "enabled": True,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True
            }


@dataclass
class EvaluationConfig:
    run_evaluation: bool = True
    test_questions: List[str] = None
    
    def __post_init__(self):
        if self.test_questions is None:
            self.test_questions = [
                "What is the purpose of LINZ's fee waiver policy?",
                "Who has the authority to waive title fees?",
                "Can partial title fees be waived?",
                "What legal authority allows LINZ to waive title fees?"
            ]


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_hardware_info: bool = True
    log_training_metrics: bool = True
