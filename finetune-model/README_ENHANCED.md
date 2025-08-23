# Enhanced Fine-tuning Framework

A comprehensive, production-ready framework for fine-tuning large language models using LoRA (Low-Rank Adaptation) with advanced features including hardware optimization, robust error handling, and comprehensive monitoring.

## 🌟 Key Features

### ✅ **Hardware-Aware Optimization**

- Automatic GPU memory detection and optimization
- Progressive fallback strategies for model loading
- Quantization support with automatic fallbacks
- Optimal batch size and gradient accumulation

### ✅ **Robust Data Handling**

- Comprehensive dataset validation
- Template format validation (Alpaca, ChatML, Vicuna)
- Data quality checks and preprocessing
- Detailed dataset statistics and analysis

### ✅ **Enhanced Model Management**

- Progressive model loading with fallbacks (quantized → full precision → CPU)
- Architecture-specific LoRA configurations
- Automatic target module detection
- Memory usage estimation

### ✅ **Advanced Monitoring**

- Real-time training metrics and GPU monitoring
- Comprehensive logging with multiple levels
- Training callbacks with performance tracking
- Automatic checkpoint management

### ✅ **Model Evaluation**

- Automated response quality assessment
- Template adherence checking
- Performance benchmarking
- Consistency analysis

### ✅ **Configuration Management**

- YAML-based configuration with validation
- Command-line overrides
- Environment variable support
- Dry-run validation

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (optional)
cp config.env .env
# Edit .env to add your HUGGINGFACE_TOKEN if needed
```

### 2. Basic Usage

```bash
# Basic fine-tuning with default settings
python finetune.py

# Use custom configuration
python finetune.py --config my_config.yaml

# Validate configuration without training
python finetune.py --dry-run

# Resume from checkpoint
python finetune.py --resume-from ./policy-chatbot-v1/checkpoint-50
```

### 3. Configuration

Edit `config.yaml` to customize your training:

```yaml
model:
  base_model: "meta-llama/Llama-3.2-1B-Instruct"
  output_dir: "./my-custom-model"
  max_seq_length: 512

training:
  num_epochs: 3
  learning_rate: 2e-4
  save_steps: 25

data:
  dataset_path: "training_dataset.jsonl"
  template_format: "alpaca"

hardware:
  auto_optimize: true
  quantization:
    enabled: true
```

## 📁 Project Structure

```
finetune-model/
├── src/                          # Enhanced framework modules
│   ├── config/                   # Configuration management
│   │   ├── model_config.py       # Model and training configs
│   │   └── hardware_config.py    # Hardware optimization
│   ├── data/                     # Data handling
│   │   ├── dataset_loader.py     # Dataset loading and validation
│   │   └── preprocessor.py       # Template validation
│   ├── models/                   # Model management
│   │   ├── model_loader.py       # Robust model loading
│   │   └── lora_config.py        # LoRA configuration
│   ├── training/                 # Training components
│   │   ├── trainer.py           # Enhanced trainer
│   │   └── callbacks.py         # Training callbacks
│   └── utils/                    # Utilities
│       ├── logging.py           # Advanced logging
│       └── validation.py        # Model evaluation
├── finetune.py                   # Main training script
├── config.yaml                   # Configuration file
├── training_dataset.jsonl        # Training data
├── chat.py                       # Chat interface (unchanged)
└── requirements.txt              # Dependencies
```

## 🔧 Advanced Usage

### Custom Configuration

```yaml
# config.yaml - Complete example
model:
  base_model: "meta-llama/Llama-3.2-1B-Instruct"
  output_dir: "./policy-chatbot-v1"
  max_seq_length: 512
  trust_remote_code: true

training:
  num_epochs: 3
  learning_rate: 2e-4
  warmup_ratio: 0.03
  save_steps: 25
  logging_steps: 5
  weight_decay: 0.001
  max_grad_norm: 0.3
  lr_scheduler_type: "cosine"
  save_total_limit: 3
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"

data:
  dataset_path: "training_dataset.jsonl"
  validation_split: 0.0
  max_text_length: 2048
  template_format: "alpaca"

hardware:
  auto_optimize: true
  force_cpu: false
  mixed_precision: "bf16"
  quantization:
    enabled: true
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_use_double_quant: true

evaluation:
  run_evaluation: true
  test_questions:
    - "What is our company policy on remote work?"
    - "How do I request time off?"

logging:
  level: "INFO"
  log_to_file: true
  log_hardware_info: true
  log_training_metrics: true
```

### Command Line Options

```bash
# Dry run - validate config without training
python finetune.py --dry-run

# Override output directory
python finetune.py --output-dir ./my-model

# Disable evaluation
python finetune.py --no-evaluation

# Verbose logging
python finetune.py --verbose

# Resume training
python finetune.py --resume-from ./model/checkpoint-100
```

### Hardware Optimization

The framework automatically optimizes for your hardware:

- **< 8GB GPU**: 1 batch size, 8 grad accumulation, FP16
- **8-12GB GPU**: 2 batch size, 4 grad accumulation, BF16
- **12-16GB GPU**: 4 batch size, 2 grad accumulation, BF16
- **> 16GB GPU**: 8+ batch size, 1 grad accumulation, BF16

### Dataset Validation

Your training data is automatically validated for:

- Required format (JSONL with "text" field)
- Template adherence (Alpaca, ChatML, Vicuna)
- Data quality (empty texts, length limits)
- Encoding issues

## 📊 Monitoring and Logging

### Training Logs

Comprehensive logs are saved to `{output_dir}/logs/`:

- `latest.log` - Latest training log
- `training_metrics.json` - Detailed metrics
- `training_summary.json` - Training summary

### Real-time Monitoring

During training, monitor:

- Loss progression and learning rate
- GPU memory usage and utilization
- Training speed (steps/second)
- Hardware performance metrics

### Model Evaluation

Automatic evaluation includes:

- Response quality assessment
- Template adherence checking
- Inference performance metrics
- Memory usage analysis
- Consistency checking

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # The framework auto-optimizes, but you can force CPU:
   # In config.yaml:
   hardware:
     force_cpu: true
   ```

2. **Model Loading Fails**

   ```bash
   # Framework has automatic fallbacks:
   # Quantized → Full Precision → CPU
   # Check logs for which strategy worked
   ```

3. **Dataset Validation Errors**

   ```bash
   # Run dry run to see specific issues:
   python finetune.py --dry-run
   ```

4. **Template Format Issues**
   ```bash
   # Ensure your data follows the expected format:
   # For Alpaca (default):
   {
     "text": "Below is an instruction...\n\n### Instruction:\nYour question\n\n### Response:\nYour answer"
   }
   ```

### Performance Optimization

1. **For Faster Training**:

   - Increase batch size if you have GPU memory
   - Reduce sequence length if possible
   - Use BF16 instead of FP16 on modern GPUs

2. **For Better Quality**:

   - Increase LoRA rank (r parameter)
   - Train for more epochs
   - Use larger learning rate with warmup

3. **For Memory Efficiency**:
   - Enable gradient checkpointing
   - Use quantization
   - Reduce batch size

## 🔄 Migration from Original Script

To migrate from the original `finetune.py`:

1. **Backup your data**: Your `training_dataset.jsonl` works unchanged
2. **Update dependencies**: `pip install -r requirements.txt`
3. **Create config**: Use `config.yaml` instead of hardcoded values
4. **Run**: `python finetune.py` (same basic usage)

### Key Differences

| Original             | Enhanced                        |
| -------------------- | ------------------------------- |
| Hardcoded config     | YAML configuration              |
| Basic error handling | Robust fallbacks                |
| Minimal logging      | Comprehensive monitoring        |
| Manual optimization  | Auto hardware optimization      |
| No validation        | Dataset validation              |
| Basic training       | Advanced trainer with callbacks |

## 📈 Results and Output

After training completes, you'll find:

```
policy-chatbot-v1/
├── final/                    # Final model adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── training_config.json
├── logs/                     # Training logs
│   ├── latest.log
│   ├── training_metrics.json
│   └── training_summary.json
├── evaluation/               # Evaluation results (if enabled)
│   └── evaluation_results_*.json
└── checkpoint-*/            # Training checkpoints
```

## 🤝 Contributing

The enhanced framework is modular and extensible:

- Add new model architectures in `src/models/lora_config.py`
- Add new template formats in `src/data/preprocessor.py`
- Extend evaluation metrics in `src/utils/validation.py`
- Add hardware profiles in `src/config/hardware_config.py`

## 📄 License

Same as the original project - see main repository for license details.

---

**Need help?** Check the logs in `{output_dir}/logs/` for detailed information about any issues.
