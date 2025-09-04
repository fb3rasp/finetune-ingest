#!/usr/bin/env python3
"""
Step 6: Fine-tune base model using training prompts.
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from .base_step import BaseStep

# Import ML libraries with error handling
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer, 
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import SFTTrainer
    from datasets import load_dataset
    ML_LIBRARIES_AVAILABLE = True
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class FinetuneStep(BaseStep):
    """Step 6: Fine-tune base model using training prompts."""

    # Model-specific configurations
    MODEL_CONFIGS = {
        "llama": {
            "default_model": "meta-llama/Llama-3.2-1B-Instruct",
            "lora_config": {
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "r": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "training_args": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-4,
                "weight_decay": 0.001,
                "max_grad_norm": 0.3,
                "warmup_ratio": 0.03,
                "max_seq_length": 512
            },
            "model_kwargs": {},
            "test_prompt_template": "What is our company policy on remote work?"
        },
        "gemma": {
            "default_model": "google/gemma-2-2b-it",
            "lora_config": {
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "r": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "training_args": {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.1,
                "max_seq_length": 2048
            },
            "model_kwargs": {"attn_implementation": "eager"},
            "test_prompt_template": """<start_of_turn>user
What are the LINZ title fee policies for property transfers?<end_of_turn>
<start_of_turn>model
"""
        },
        "qwen": {
            "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "lora_config": {
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "r": 64,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "training_args": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.05,
                "max_seq_length": 2048
            },
            "model_kwargs": {"torch_dtype": torch.bfloat16},
            "test_prompt_template": """<|im_start|>user
What are the LINZ title fee policies for property transfers?<|im_end|>
<|im_start|>assistant
"""
        }
    }

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        if not ML_LIBRARIES_AVAILABLE:
            raise ImportError(f"Required ML libraries not available: {IMPORT_ERROR}")

    def detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type from model name"""
        model_name_lower = model_name.lower()
        if "llama" in model_name_lower:
            return "llama"
        elif "gemma" in model_name_lower:
            return "gemma"
        elif "qwen" in model_name_lower:
            return "qwen"
        else:
            # Default to llama if can't detect
            self.log(f"Cannot auto-detect model type from name: {model_name}. Defaulting to 'llama'", "warning")
            return "llama"

    def setup_model_and_tokenizer(self, model_name: str, model_type: str, hf_token: Optional[str], bnb_config):
        """Load model and tokenizer with type-specific configurations"""
        config = self.MODEL_CONFIGS[model_type]
        
        self.log(f"Loading {model_type} base model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **config["model_kwargs"]
            )
        except Exception as e:
            self.log(f"Error loading model with quantization: {e}", "warning")
            self.log("Trying without quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                **config["model_kwargs"]
            )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure model settings
        model.config.use_cache = False
        if hasattr(model.config, 'pretraining_tp'):
            model.config.pretraining_tp = 1

        self.log(f"Loading {model_type} tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        return model, tokenizer

    def setup_lora_config(self, model_type: str):
        """Setup LoRA configuration based on model type"""
        config = self.MODEL_CONFIGS[model_type]["lora_config"]
        return LoraConfig(
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            r=config["r"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config["target_modules"]
        )

    def setup_training_arguments(self, model_type: str, output_dir: str):
        """Setup training arguments based on model type and config.env"""
        config = self.MODEL_CONFIGS[model_type]["training_args"]
        
        # Helper function to parse boolean from env
        def get_bool_env(key: str, default: bool = False) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ('true', '1', 'yes', 'on')
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(os.getenv("NUM_TRAIN_EPOCHS", "3")),
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            optim=os.getenv("OPTIMIZER", "paged_adamw_8bit"),
            save_steps=int(os.getenv("SAVE_STEPS", "25")),
            logging_steps=int(os.getenv("LOGGING_STEPS", "5")),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            fp16=get_bool_env("USE_FP16", False),
            bf16=get_bool_env("USE_BF16", True),
            max_grad_norm=float(os.getenv("MAX_GRAD_NORM", str(config["max_grad_norm"]))),
            max_steps=-1,
            warmup_ratio=config["warmup_ratio"],
            group_by_length=get_bool_env("GROUP_BY_LENGTH", True),
            lr_scheduler_type=os.getenv("LR_SCHEDULER_TYPE", "cosine"),
            save_total_limit=int(os.getenv("SAVE_TOTAL_LIMIT", "3")),
            dataloader_pin_memory=get_bool_env("DATALOADER_PIN_MEMORY", False),
            remove_unused_columns=get_bool_env("REMOVE_UNUSED_COLUMNS", True),
            gradient_checkpointing=get_bool_env("GRADIENT_CHECKPOINTING", True),
            report_to=None if get_bool_env("DISABLE_WANDB", True) else []
        )

    def test_model(self, model_name: str, model_type: str, final_model_path: str, tokenizer, hf_token: Optional[str]):
        """Test the fine-tuned model with model-specific prompt format"""
        self.log("=" * 50)
        self.log(f"Testing the fine-tuned {model_type} model...")
        self.log("=" * 50)

        # Load base model for testing
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=os.getenv("DEVICE_MAP", "auto"),
            token=hf_token,
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
        )

        model = PeftModel.from_pretrained(base_model, final_model_path)
        model.eval()

        # Get model-specific test prompt
        test_prompt = self.MODEL_CONFIGS[model_type]["test_prompt_template"]
        
        # Fix tokenizer pad token if not set properly
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Use proper tokenizer call with attention mask
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Use config.env values for generation
        max_new_tokens = int(os.getenv("TEST_MAX_NEW_TOKENS", "200"))
        temperature = float(os.getenv("TEST_TEMPERATURE", "0.7"))
        top_p = float(os.getenv("TEST_TOP_P", "0.9"))

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.log(f"Test input: {test_prompt}")
        self.log(f"Model response: {response[len(test_prompt):]}")

    def check_prerequisites(self) -> bool:
        """Check if training files and required environment variables exist."""
        if not ML_LIBRARIES_AVAILABLE:
            self.log(f"Required ML libraries not available: {IMPORT_ERROR}", "error")
            return False

        train_dir = Path(self.config.qa_train_dir)
        if not train_dir.exists():
            self.log(f"Training directory does not exist: {train_dir}", "error")
            return False
        
        # Look for JSONL files in the training directory
        training_files = list(train_dir.glob("*.jsonl"))
        if not training_files:
            self.log(f"No JSONL training files found in {train_dir}", "error")
            return False
        
        if not os.getenv("FINETUNE_MODEL_NAME"):
            self.log("FINETUNE_MODEL_NAME environment variable not set", "error")
            return False
        
        return True

    def combine_training_data(self, train_dir: Path) -> tuple[str, int]:
        """Combine all JSONL training files into a single file"""
        training_files = list(train_dir.glob("*.jsonl"))
        self.log(f"Found {len(training_files)} training files to combine")
        
        combined_data = []
        total_prompts = 0
        
        for training_file in training_files:
            try:
                self.log(f"Loading training data from: {training_file.name}")
                
                with open(training_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                # Ensure we have the required 'text' field
                                if 'text' in data:
                                    combined_data.append(data)
                                    total_prompts += 1
                                else:
                                    self.log(f"Warning: Line {line_num} in {training_file.name} missing 'text' field", "warning")
                            except json.JSONDecodeError as e:
                                self.log(f"Warning: Invalid JSON on line {line_num} in {training_file.name}: {e}", "warning")
                
            except Exception as e:
                self.log(f"Error reading {training_file.name}: {e}", "error")
                continue
        
        if not combined_data:
            raise ValueError("No valid training data found")
        
        # Create combined training file
        combined_file = train_dir / "combined_training_data.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for data in combined_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        self.log(f"Created combined training file with {total_prompts} prompts: {combined_file}")
        return str(combined_file.absolute()), total_prompts

    def run(self, **kwargs) -> bool:
        """Run the fine-tuning step with combined training data."""
        self.log("Starting fine-tuning step...")
        
        if not self.check_prerequisites():
            return False
        
        try:
            # Ensure output directory exists
            model_dir = Path(self.config.training_model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all JSONL training files and combine them
            train_dir = Path(self.config.qa_train_dir)
            combined_file_path, total_prompts = self.combine_training_data(train_dir)
            
            # Get model configuration
            model_name = os.getenv("FINETUNE_MODEL_NAME")
            model_type = os.getenv("FINETUNE_MODEL_TYPE")
            if not model_type:
                model_type = self.detect_model_type(model_name)
            
            self.log(f"Using model: {model_name} (type: {model_type})")
            
            # Get HuggingFace token
            hf_token = os.getenv("HF_TOKEN")
            
            # Setup quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model and tokenizer
            self.log("Setting up model and tokenizer...")
            model, tokenizer = self.setup_model_and_tokenizer(model_name, model_type, hf_token, bnb_config)
            
            # Setup LoRA configuration
            self.log("Setting up LoRA configuration...")
            lora_config = self.setup_lora_config(model_type)
            model = get_peft_model(model, lora_config)
            
            # Load dataset
            self.log("Loading training dataset...")
            dataset = load_dataset('json', data_files=combined_file_path, split='train')
            
            # Setup training arguments
            self.log("Setting up training arguments...")
            training_args = self.setup_training_arguments(model_type, self.config.training_model_dir)
            
            # Initialize trainer
            self.log("Initializing SFT trainer...")
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                dataset_text_field="text",
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=self.MODEL_CONFIGS[model_type]["training_args"]["max_seq_length"],
                packing=False
            )
            
            # Start training
            self.log("Starting fine-tuning...")
            trainer.train()
            
            # Save model
            final_model_path = Path(self.config.training_model_dir) / "final"
            final_model_path.mkdir(exist_ok=True)
            
            self.log(f"Saving final model to: {final_model_path}")
            trainer.model.save_pretrained(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            
            # Save training completion info
            completion_info = {
                "model_name": model_name,
                "model_type": model_type,
                "training_completed": datetime.now().isoformat(),
                "total_prompts": total_prompts,
                "final_model_path": str(final_model_path)
            }
            
            completion_file = Path(self.config.training_model_dir) / "training_complete.json"
            save_json_atomic(completion_info, str(completion_file))
            
            # Test model if enabled
            if os.getenv("TEST_MODEL_AFTER_TRAINING", "false").lower() == "true":
                self.test_model(model_name, model_type, str(final_model_path), tokenizer, hf_token)
            
            self.log("Fine-tuning completed successfully!")
            self.log(f"Model saved to: {final_model_path}")
            self.log(f"Training data used: {total_prompts} prompts")
            
            return True
            
        except Exception as e:
            self.log(f"Fine-tuning failed: {str(e)}", "error")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "error")
            return False