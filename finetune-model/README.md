# Enhanced Fine-tuning Framework

A comprehensive, production-ready framework for fine-tuning large language models using LoRA (Low-Rank Adaptation) with advanced features including hardware optimization, robust error handling, and comprehensive monitoring.

## 🎯 **Purpose**

The Enhanced Fine-tuning Framework provides an industrial-strength solution for training domain-specific language models from generated training data. It combines automatic hardware optimization, robust error handling, and comprehensive monitoring to deliver production-ready fine-tuned models with minimal configuration.

## 🌟 **Key Features**

### **🔧 Hardware-Aware Optimization**

- **Automatic GPU Detection**: Intelligent memory detection and optimization
- **Progressive Fallback Strategies**: Quantized → Full Precision → CPU loading
- **Memory Management**: Optimal batch size and gradient accumulation
- **Quantization Support**: 4-bit, 8-bit, and mixed-precision training
- **Resource Monitoring**: Real-time GPU and memory usage tracking

### **📊 Robust Data Handling**

- **Comprehensive Validation**: Dataset format and quality verification
- **Template Support**: Alpaca, ChatML, Vicuna format validation
- **Quality Checks**: Empty text detection, length validation, encoding verification
- **Statistics Analysis**: Detailed dataset metrics and preprocessing insights
- **Error Recovery**: Graceful handling of malformed data

### **🤖 Enhanced Model Management**

- **Progressive Loading**: Multiple fallback strategies for model initialization
- **Architecture Detection**: Automatic LoRA target module identification
- **Memory Estimation**: Pre-training resource requirement analysis
- **Checkpoint Management**: Automatic saving and resumption capabilities
- **Model Evaluation**: Built-in quality assessment and benchmarking

### **📈 Advanced Monitoring**

- **Real-Time Metrics**: Training loss, learning rate, and performance tracking
- **GPU Monitoring**: Memory usage, utilization, and temperature tracking
- **Training Callbacks**: Custom callbacks for performance optimization
- **Comprehensive Logging**: Multi-level logging with file and console output
- **Progress Visualization**: Clear training progress and status reporting

### **⚙️ Configuration Management**

- **YAML Configuration**: Structured, version-controlled settings
- **Command-Line Overrides**: Flexible parameter modification
- **Environment Variables**: Secure credential and setting management
- **Validation**: Dry-run mode for configuration verification
- **Templates**: Pre-configured templates for common use cases

## 🚀 **Quick Start**

### **1. Installation**

```bash
cd finetune-model

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional)
cp ../config.env .env
# Edit .env to add your HUGGINGFACE_TOKEN if needed
```

### **2. Basic Fine-Tuning**

```bash
# Fine-tune with default settings
python finetune.py

# Use generated training data from pipeline
python finetune.py \
  --dataset-path "../data/training/NZISM-ISM Document-V.-3.9-April-2025_trainingdata.jsonl" \
  --output-dir "./nzism-chatbot-v1"

# Validate configuration without training
python finetune.py --dry-run
```

### **3. Resume Training**

```bash
# Resume from checkpoint
python finetune.py --resume-from ./nzism-chatbot-v1/checkpoint-50

# Resume with different parameters
python finetune.py \
  --resume-from ./nzism-chatbot-v1/checkpoint-50 \
  --learning-rate 1e-4
```

## 📁 **Project Structure**

```
finetune-model/
├── src/                          # Enhanced framework modules
│   ├── config/                   # Configuration management
│   │   ├── model_config.py       # Model and training configurations
│   │   └── hardware_config.py    # Hardware optimization profiles
│   ├── data/                     # Data handling
│   │   ├── dataset_loader.py     # Dataset loading and validation
│   │   └── preprocessor.py       # Template format validation
│   ├── models/                   # Model management
│   │   ├── model_loader.py       # Robust model loading with fallbacks
│   │   └── lora_config.py        # LoRA configuration and optimization
│   ├── training/                 # Training components
│   │   ├── trainer.py           # Enhanced trainer with monitoring
│   │   └── callbacks.py         # Custom training callbacks
│   └── utils/                    # Utilities
│       ├── logging.py           # Advanced logging system
│       └── validation.py        # Model evaluation and testing
├── finetune.py                   # Main training script
├── config.yaml                   # Default configuration
├── config-example.yaml           # Configuration template
├── PRD.md                        # Product requirements document
└── requirements.txt              # Python dependencies
```

## 🔧 **Configuration**

### **Basic Configuration**

```yaml
# config.yaml - Essential settings
model:
  base_model: "meta-llama/Llama-3.2-1B-Instruct"
  output_dir: "./my-custom-model"
  max_seq_length: 512

training:
  num_epochs: 3
  learning_rate: 2e-4
  save_steps: 25

data:
  dataset_path: "../data/training/my_dataset.jsonl"
  template_format: "alpaca"

hardware:
  auto_optimize: true
  quantization:
    enabled: true
```

### **Advanced Configuration**

```yaml
# Complete configuration example
model:
  base_model: "meta-llama/Llama-3.2-1B-Instruct"
  output_dir: "./nzism-chatbot-v1"
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
  r: 16 # LoRA rank (higher = more parameters)
  alpha: 32 # LoRA scaling factor
  dropout: 0.1 # LoRA dropout rate
  bias: "none" # Bias training strategy
  task_type: "CAUSAL_LM"

data:
  dataset_path: "../data/training/dataset.jsonl"
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
    - "What is the purpose of the NZISM?"
    - "Who is encouraged to use the NZISM?"
    - "What are baseline controls?"

logging:
  level: "INFO"
  log_to_file: true
  log_hardware_info: true
  log_training_metrics: true
```

### **Command Line Options**

```bash
# Configuration
--config CONFIG_FILE          Custom configuration file
--dataset-path PATH           Override dataset path
--output-dir DIR              Override output directory

# Training Control
--dry-run                     Validate configuration without training
--resume-from CHECKPOINT      Resume from specific checkpoint
--no-evaluation              Disable post-training evaluation

# Override Parameters
--learning-rate RATE          Override learning rate
--num-epochs N                Override number of epochs
--batch-size SIZE             Override batch size

# Logging and Monitoring
--verbose                     Enable verbose logging
--quiet                      Minimize output
--log-level LEVEL            Set logging level (DEBUG, INFO, WARN, ERROR)
```

## 🎯 **Use Cases**

### **Domain-Specific Chatbots**

```bash
# Train on policy documents
python finetune.py \
  --dataset-path "../data/training/policy_documents_trainingdata.jsonl" \
  --output-dir "./policy-chatbot-v1" \
  --config config-policy.yaml
```

### **Technical Documentation Assistant**

```bash
# Train on technical manuals
python finetune.py \
  --dataset-path "../data/training/tech_docs_trainingdata.jsonl" \
  --output-dir "./tech-assistant-v1" \
  --learning-rate 1e-4 \
  --num-epochs 5
```

### **Multi-Document Training**

```bash
# Train on combined datasets
python finetune.py \
  --dataset-path "../data/training/combined_trainingdata.jsonl" \
  --output-dir "./multi-domain-chatbot-v1" \
  --max-seq-length 1024
```

### **Production Training**

```bash
# High-quality training for production
python finetune.py \
  --config config-production.yaml \
  --dataset-path "../data/training/validated_dataset.jsonl" \
  --output-dir "./production-model-v1" \
  --verbose
```

## ⚡ **Hardware Optimization**

### **Automatic Optimization Profiles**

| GPU Memory  | Batch Size | Grad Accumulation | Precision | Quantization |
| ----------- | ---------- | ----------------- | --------- | ------------ |
| **< 8GB**   | 1          | 8                 | FP16      | 4-bit        |
| **8-12GB**  | 2          | 4                 | BF16      | 4-bit        |
| **12-16GB** | 4          | 2                 | BF16      | Optional     |
| **> 16GB**  | 8+         | 1                 | BF16      | Disabled     |

### **Manual Hardware Configuration**

```yaml
# Force specific hardware settings
hardware:
  auto_optimize: false
  force_cpu: false
  mixed_precision: "bf16"
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  quantization:
    enabled: true
    load_in_4bit: true
```

### **Memory Optimization Strategies**

```bash
# For low memory (< 8GB GPU)
python finetune.py \
  --config config-low-memory.yaml \
  --dataset-path "dataset.jsonl"

# For high memory (> 16GB GPU)
python finetune.py \
  --config config-high-performance.yaml \
  --dataset-path "dataset.jsonl"
```

## 📊 **Monitoring and Logging**

### **Training Logs Structure**

```
{output_dir}/logs/
├── latest.log                    # Latest training session
├── training_YYYYMMDD_HHMMSS.log  # Timestamped logs
├── training_metrics.json         # Detailed metrics
├── training_summary.json         # Training summary
└── hardware_info.json           # Hardware specifications
```

### **Real-Time Monitoring**

During training, monitor:

- **Training Progress**: Loss, learning rate, epoch progression
- **Hardware Usage**: GPU memory, utilization, temperature
- **Performance Metrics**: Steps/second, tokens/second
- **Quality Indicators**: Evaluation scores, checkpoint quality

### **Log Analysis**

```bash
# View training progress
tail -f ./nzism-chatbot-v1/logs/latest.log

# Check hardware optimization
cat ./nzism-chatbot-v1/logs/hardware_info.json

# Review training metrics
python -m json.tool ./nzism-chatbot-v1/logs/training_summary.json
```

## 🔍 **Model Evaluation**

### **Automatic Evaluation**

The framework automatically evaluates:

- **Response Quality**: Coherence, relevance, accuracy
- **Template Adherence**: Format consistency and structure
- **Performance Metrics**: Inference speed and memory usage
- **Consistency Analysis**: Response stability across similar inputs

### **Custom Evaluation**

```yaml
# Configure custom evaluation
evaluation:
  run_evaluation: true
  test_questions:
    - "What are the key security controls in NZISM?"
    - "How should agencies implement baseline controls?"
    - "What is the purpose of classification levels?"
  evaluation_template: "custom_eval_template.txt"
  save_responses: true
```

### **Evaluation Output**

```json
{
  "evaluation_summary": {
    "total_questions": 10,
    "average_response_time": 0.45,
    "average_response_length": 156,
    "template_adherence_score": 0.95
  },
  "detailed_results": [...]
}
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **CUDA Out of Memory**

   ```bash
   # Framework auto-optimizes, but you can force settings:
   python finetune.py --config config-low-memory.yaml
   ```

2. **Model Loading Failures**

   ```bash
   # Check progressive fallback in logs:
   # Quantized → Full Precision → CPU
   python finetune.py --verbose
   ```

3. **Dataset Validation Errors**

   ```bash
   # Run validation to identify issues:
   python finetune.py --dry-run
   ```

4. **Template Format Issues**

   ```bash
   # Ensure proper Alpaca format:
   {
     "text": "Below is an instruction...\n\n### Instruction:\nQuestion\n\n### Response:\nAnswer"
   }
   ```

### **Performance Optimization**

#### **For Faster Training**

- Increase batch size if GPU memory allows
- Reduce sequence length for your use case
- Use BF16 on modern GPUs (A100, RTX 30/40 series)
- Enable gradient checkpointing for memory efficiency

#### **For Better Quality**

- Increase LoRA rank (r parameter) for more model capacity
- Train for more epochs with lower learning rate
- Use larger datasets with diverse examples
- Implement validation split for monitoring

#### **For Memory Efficiency**

- Enable quantization (4-bit or 8-bit)
- Use gradient accumulation instead of larger batches
- Reduce sequence length if possible
- Enable gradient checkpointing

### **Debug Mode**

```bash
# Enable comprehensive debugging
python finetune.py \
  --verbose \
  --log-level DEBUG \
  --dry-run
```

## 🔄 **Integration with Pipeline**

### **Complete Workflow**

```bash
# Step 1: Generate training data
cd ../training_data_generator
python src/convert_qa_to_training.py \
  "data/qa_results/NZISM-ISM Document-V.-3.9-April-2025_qa.json" \
  --output-dir ../data/training

# Step 2: Fine-tune model
cd ../finetune-model
python finetune.py \
  --dataset-path "../data/training/NZISM-ISM Document-V.-3.9-April-2025_trainingdata.jsonl" \
  --output-dir "./nzism-chatbot-v1"

# Step 3: Evaluate results (optional)
cd ../fact_checker
python src/validate_qa.py \
  --input ../finetune-model/nzism-chatbot-v1/evaluation/
```

### **Programmatic Usage**

```python
from finetune_model.src.training.trainer import EnhancedTrainer
from finetune_model.src.config.model_config import ModelConfig

# Load configuration
config = ModelConfig.from_yaml("config.yaml")

# Initialize trainer
trainer = EnhancedTrainer(config)

# Start training
results = trainer.train()
print(f"Training completed. Final loss: {results['final_loss']}")
```

## 📈 **Output Structure**

After training completes:

```
{output_dir}/
├── final/                        # Final model adapter
│   ├── adapter_config.json       # LoRA configuration
│   ├── adapter_model.safetensors # Trained weights
│   ├── tokenizer.json           # Tokenizer configuration
│   └── training_config.json     # Training parameters
├── logs/                         # Training logs
│   ├── latest.log               # Latest session log
│   ├── training_metrics.json   # Detailed metrics
│   ├── training_summary.json   # Training summary
│   └── hardware_info.json      # Hardware specifications
├── evaluation/                   # Evaluation results (if enabled)
│   ├── evaluation_results.json # Evaluation metrics
│   └── sample_responses.json   # Example responses
└── checkpoint-*/                # Training checkpoints
    ├── adapter_model.safetensors
    ├── optimizer.pt
    ├── scheduler.pt
    └── training_args.bin
```

## 🔗 **Model Usage**

### **Load Trained Model**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./nzism-chatbot-v1/final")

# Use for inference
inputs = tokenizer("What is the NZISM?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### **Deployment Ready**

The trained model is ready for:

- **API Deployment**: Use with FastAPI, Flask, or similar frameworks
- **Chat Interfaces**: Integrate with web or mobile applications
- **Batch Processing**: Use for document analysis or Q&A systems
- **Further Training**: Use as base for additional fine-tuning

## 📚 **Dependencies**

```bash
# Core ML Framework
torch>=1.13.0
transformers>=4.21.0
peft>=0.3.0
datasets>=2.0.0

# Training Optimization
accelerate>=0.20.0
bitsandbytes>=0.39.0

# Data Processing
tokenizers>=0.13.0
pydantic>=2.0.0

# Configuration and Logging
pyyaml>=6.0
python-dotenv>=1.0.0

# Evaluation and Monitoring
wandb>=0.15.0          # Optional: experiment tracking
tensorboard>=2.0.0     # Optional: metrics visualization
```

## 🎉 **Benefits**

- **Production Ready**: Industrial-strength error handling and monitoring
- **Hardware Agnostic**: Automatic optimization for any GPU configuration
- **Quality Focused**: Comprehensive evaluation and validation
- **Easy Integration**: Seamless workflow with training data generation
- **Cost Effective**: Efficient training with LoRA and quantization
- **Maintainable**: Clean architecture with comprehensive logging

## 🤝 **Contributing**

The framework is modular and extensible:

- **Model Support**: Add new architectures in `src/models/lora_config.py`
- **Data Formats**: Extend template support in `src/data/preprocessor.py`
- **Evaluation**: Add metrics in `src/utils/validation.py`
- **Hardware**: Add profiles in `src/config/hardware_config.py`
- **Training**: Customize callbacks in `src/training/callbacks.py`

## 📄 **Migration Guide**

### **From Original Script**

1. **Backup Data**: Your `training_dataset.jsonl` works unchanged
2. **Update Dependencies**: `pip install -r requirements.txt`
3. **Create Configuration**: Replace hardcoded values with `config.yaml`
4. **Run Training**: `python finetune.py` (same basic usage)

### **Configuration Migration**

| Original (Hardcoded)   | Enhanced (config.yaml)          |
| ---------------------- | ------------------------------- |
| `model_name = "..."`   | `model: base_model: "..."`      |
| `learning_rate = 2e-4` | `training: learning_rate: 2e-4` |
| Manual batch sizing    | `hardware: auto_optimize: true` |
| Basic logging          | `logging: level: "INFO"`        |

---

**Need help?** Check the comprehensive logs in `{output_dir}/logs/` for detailed information about any issues, or refer to the troubleshooting section above.
