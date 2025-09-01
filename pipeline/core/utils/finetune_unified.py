import torch
import os
import argparse
from dotenv import load_dotenv

# PYTHON_ARGCOMPLETE_OK
try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

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
        "default_model": "google/gemma-3-1b-it",
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

def detect_model_type(model_name):
    """Auto-detect model type from model name"""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return "llama"
    elif "gemma" in model_name_lower:
        return "gemma"
    elif "qwen" in model_name_lower:
        return "qwen"
    else:
        raise ValueError(f"Cannot auto-detect model type from name: {model_name}. Use --model-type argument.")

def setup_model_and_tokenizer(model_name, model_type, hf_token, bnb_config):
    """Load model and tokenizer with type-specific configurations"""
    config = MODEL_CONFIGS[model_type]
    
    print(f"Loading {model_type} base model: {model_name}")
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
        print(f"Error loading model with quantization: {e}")
        print("Trying without quantization...")
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

    print(f"Loading {model_type} tokenizer...")
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

def setup_lora_config(model_type):
    """Setup LoRA configuration based on model type"""
    config = MODEL_CONFIGS[model_type]["lora_config"]
    return LoraConfig(
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        r=config["r"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"]
    )

def setup_training_arguments(model_type, output_dir):
    """Setup training arguments based on model type and config.env"""
    config = MODEL_CONFIGS[model_type]["training_args"]
    
    # Helper function to parse boolean from env
    def get_bool_env(key, default=False):
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

def test_model(model_name, model_type, final_model_path, tokenizer, hf_token):
    """Test the fine-tuned model with model-specific prompt format"""
    print("\n" + "="*50)
    print(f"Testing the fine-tuned {model_type} model...")
    print("="*50)

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
    test_prompt = MODEL_CONFIGS[model_type]["test_prompt_template"]
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
    print(f"Test input: {test_prompt}")
    print(f"Model response: {response[len(test_prompt):]}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified fine-tuning script for Llama, Gemma, and Qwen models')
    parser.add_argument('--model-name', type=str, help='Model name to fine-tune (e.g., meta-llama/Llama-3.2-1B-Instruct)')
    parser.add_argument('--model-type', type=str, choices=['llama', 'gemma', 'qwen'], help='Model type (auto-detected if not specified)')
    parser.add_argument('--output-dir', type=str, help='Output directory for fine-tuned model')
    dataset_arg = parser.add_argument('--dataset', type=str, default=None, help='Dataset filename')
    parser.add_argument('--no-test', action='store_true', help='Skip model testing after training')
    
    # Enable argcomplete
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    load_dotenv("config.env")

    # Get HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file. Please add it as: HUGGINGFACE_TOKEN=your_token_here")

    # Helper function to parse boolean from env
    def get_bool_env(key, default=False):
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    # Determine model type first (from args or config default)
    model_type = args.model_type or os.getenv("DEFAULT_MODEL_TYPE", "llama")
    
    # Determine model name based on model type
    model_name = args.model_name
    if not model_name:
        # Use model-type specific default from config.env
        model_name_key = f"{model_type.upper()}_MODEL_NAME"
        model_name = os.getenv(model_name_key)
        if not model_name:
            # Fallback to hardcoded defaults
            model_name = MODEL_CONFIGS[model_type]["default_model"]
            print(f"No model specified, using default: {model_name}")
        else:
            print(f"Using model from config: {model_name}")

    # Validate model type (auto-detect if needed)
    try:
        if model_type not in MODEL_CONFIGS:
            model_type = detect_model_type(model_name)
    except ValueError:
        model_type = detect_model_type(model_name)
    
    print(f"Using model type: {model_type}")

    # Set output directory based on model type
    output_dir = args.output_dir
    if not output_dir:
        # Try model-type specific output dir from config.env
        output_dir_key = f"{model_type.upper()}_OUTPUT_DIR"
        output_dir = os.getenv(output_dir_key) or f"./policy-chatbot-{model_type}-v1"

    # Set dataset configuration
    dataset_path = os.getenv("DATASET_PATH", ".")
    dataset_file = args.dataset or os.getenv("DATASET", "training_dataset.jsonl")
    dataset_name = os.path.join(dataset_path, dataset_file)

    # Get other config defaults
    num_epochs = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
    skip_test = args.no_test or get_bool_env("SKIP_TESTING", False)

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Type: {model_type}")
    print(f"  Output: {output_dir}")
    print(f"  Dataset: {dataset_name}")

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, model_type, hf_token, bnb_config)

    # Configure and apply LoRA
    peft_config = setup_lora_config(model_type)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Setup training arguments
    training_arguments = setup_training_arguments(model_type, output_dir)

    # Load and prepare dataset
    print(f"Loading dataset from: {dataset_name}")
    dataset = load_dataset("json", data_files=dataset_name, split="train")
    print(f"Dataset loaded with {len(dataset)} examples")

    # Debug dataset structure
    print("Dataset structure:")
    print(f"Column names: {dataset.column_names}")
    if len(dataset) > 0:
        print(f"First example keys: {list(dataset[0].keys())}")
        print(f"First example text (first 100 chars): {str(dataset[0]['text'])[:100]}...")

    # Preprocess dataset
    def preprocess_function(examples):
        texts = []
        for text in examples["text"]:
            texts.append(str(text) if not isinstance(text, str) else text)
        return {"text": texts}

    dataset = dataset.map(preprocess_function, batched=True)

    # Initialize trainer
    max_seq_length = MODEL_CONFIGS[model_type]["training_args"]["max_seq_length"]
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # Start training
    print(f"Starting {model_type} fine-tuning process...")
    print(f"Training on {len(dataset)} examples for {training_arguments.num_train_epochs} epochs")
    print(f"Effective batch size: {training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps}")

    trainer.train()
    print("Fine-tuning complete!")

    # Save the final model
    final_model_path = f"{output_dir}/final"
    print(f"Saving model to {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Fine-tuned {model_type} model adapter saved to {final_model_path}")

    # Test the model unless skipped
    if not skip_test:
        test_model(model_name, model_type, final_model_path, tokenizer, hf_token)

if __name__ == "__main__":
    main()