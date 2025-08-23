# Finetune Ingest

A comprehensive system for generating high-quality training data and fine-tuning large language models end-to-end.

## Project Structure

```bash
finetune-ingest/
├── document_chunker/           # Document processing and chunking
│   ├── src/
│   │   └── chunk_documents.py  # Document chunking script
│   └── README.md
├── training_data_generator/    # Q&A generation and data conversion
│   ├── src/
│   │   ├── generate_qa.py      # Q&A generation script (resumable)
│   │   ├── combine_qa_results.py # Combine individual Q&A files
│   │   └── convert_qa_to_training.py # Convert Q&A to training format
│   └── README.md
├── fact_checker/               # Q&A validation and quality assurance
│   ├── src/
│   │   └── validate_qa.py      # Validation script
│   └── README.md
├── finetune-model/             # 🆕 Complete fine-tuning framework
│   ├── src/                    # Modular training components
│   │   ├── config/             # Configuration management
│   │   ├── models/             # Model loading and LoRA setup
│   │   ├── training/           # Training pipeline and callbacks
│   │   └── utils/              # Logging and validation utilities
│   ├── finetune.py             # Main fine-tuning script
│   ├── config.yaml             # Training configuration
│   └── README_ENHANCED.md      # Detailed fine-tuning documentation
├── common/                     # Shared libraries and utilities
│   ├── document_processing/    # Document processing modules
│   │   ├── document_loaders.py # LangChain document loaders
│   │   └── text_splitters.py   # Enhanced text splitting
│   ├── llm/                    # LLM interface and chains
│   │   ├── llm_providers.py    # Unified LLM interface
│   │   └── qa_chains.py        # Q&A generation chains (enhanced)
│   └── utils/
│       └── helpers.py          # Common helper functions
├── config.env                  # Centralized configuration
└── run_split_workflow.sh       # Main workflow automation script
```

## 🚀 **Complete ML Pipeline**

### **Phase 1: Data Generation**

```bash
# 1. Document Processing
./run_split_workflow.sh --step 1

# 2. Q&A Generation (resumable)
./run_split_workflow.sh --step 2 --resume

# 3. Combine Results
./run_split_workflow.sh --step 3

# 4. Convert to Training Format
python3 training_data_generator/src/convert_qa_to_training.py \
  "data/document_training_data/NZISM-ISM Document-V.-3.9-April-2025_qa.json" \
  --output-dir data/document_training_data
```

### **Phase 2: Model Fine-Tuning**

```bash
# Fine-tune with generated data
cd finetune-model
python3 finetune.py \
  --dataset-path "../data/document_training_data/NZISM-ISM Document-V.-3.9-April-2025_trainingdata.jsonl" \
  --output-dir "./nzism-chatbot-v1"
```

### **Phase 3: Quality Assurance (Optional)**

```bash
# Validate training data quality
cd fact_checker
python src/validate_qa.py \
  --input /data/results/training_data.json \
  --output /data/results/validation_report.json
```

## ⚡ **Quick Start**

### 1. Setup Environment

```bash
# Activate conda environment
source /Users/rainer/.zshrc
conda activate finetune

# Copy and configure environment
cp config.env .env
# Edit .env with your API keys and settings
```

### 2. End-to-End Workflow

```bash
# Complete pipeline: data generation → fine-tuning
./run_split_workflow.sh
cd finetune-model && python3 finetune.py
```

### 3. Resume Interrupted Processing

```bash
# Resume Q&A generation from interruption
./run_split_workflow.sh --step 2 --resume

# Convert existing Q&A to training format
python3 training_data_generator/src/convert_qa_to_training.py data/qa_results
```

## Data Flow

### Complete Pipeline

```bash
📁 /data/incoming/          # Source documents (PDF, MD, HTML, DOCX, TXT)
    ↓ document_chunker/
📁 /data/chunks/            # document1_chunks.json, document2_chunks.json
    ↓ training_data_generator/ (resumable!)
📁 /data/qa_results/        # document1_qa.json, document2_qa.json
    ↓ convert_qa_to_training.py
📁 /data/document_training_data/ # document1_trainingdata.jsonl, document2_trainingdata.jsonl
    ↓ finetune-model/
📁 /finetune-model/output/  # Fine-tuned LoRA adapters
    ↓ fact_checker/ (optional)
📁 /data/results/           # validation_report.json, training_data_filtered.json
```

## Key Features

### End-to-End ML Pipeline

- **Complete Data Pipeline**: Document processing → Q&A generation → Training data
- **Advanced Fine-Tuning**: LoRA-based training with hardware optimization
- **Production-Ready**: Comprehensive error handling and monitoring
- **Format Conversion**: Automatic Q&A to Alpaca training format conversion

### **Enhanced Training Framework**

- **Hardware-Aware**: Automatic GPU detection and memory optimization
- **Progressive Loading**: Fallback strategies for model loading
- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation
- **Real-Time Monitoring**: Training metrics and performance tracking

### **Robust Data Generation**

- **Resumability**: Chunk-level tracking with interrupt safety
- **Error Isolation**: Failed chunks don't affect completed ones
- **Multiple LLM Providers**: OpenAI, Claude, Gemini, Local/Ollama
- **Quality Validation**: Optional fact-checking and filtering

### **Developer Experience**

- **Modular Architecture**: Run components independently
- **Comprehensive Logging**: Enhanced error tracking and debugging
- **Flexible Configuration**: YAML and environment-based settings
- **Clear Documentation**: Detailed guides for each component

## 🔧 **Configuration**

All settings are managed through `config.env` / `.env`:

```bash
# LLM Provider Settings
GENERATOR_PROVIDER=local
GENERATOR_MODEL=qwen3:14b
GENERATOR_TEMPERATURE=0.3
OLLAMA_BASE_URL=http://192.168.50.133:11434

# Directory Structure
GENERATOR_INCOMING_DIR=/data/incoming
GENERATOR_PROCESS_DIR=/data/chunks
GENERATOR_OUTPUT_DIR=/data/qa_results
GENERATOR_OUTPUT_FILE=/data/results/training_data.json

# Processing Settings
GENERATOR_CHUNK_SIZE=1000
GENERATOR_CHUNK_OVERLAP=200
GENERATOR_QUESTIONS_PER_CHUNK=5
GENERATOR_RESUME=true
GENERATOR_BATCH_PROCESSING=false

# Logging Settings
QA_LOG_DIR=/data/logs                    # Q&A generation failure logs

# Validation Settings
VALIDATOR_PROVIDER=local
VALIDATOR_MODEL=qwen3:14b
VALIDATOR_THRESHOLD=8.0
```

## 📚 **Module Documentation**

- **[Document Chunker](document_chunker/README.md)**: Document processing and chunking
- **[Training Data Generator](training_data_generator/README.md)**: Q&A generation and combining
- **[Fact Checker](fact_checker/README.md)**: Quality validation and filtering
- **[Fine-Tuning Framework](finetune-model/README_ENHANCED.md)**: Complete model training pipeline

## 🎯 **Use Cases**

### **Rapid Prototyping**

```bash
# Process small document set and train quickly
./run_split_workflow.sh --step 1
./run_split_workflow.sh --step 2
python3 training_data_generator/src/convert_qa_to_training.py data/qa_results
cd finetune-model && python3 finetune.py
```

### **Production Training**

```bash
# Full pipeline with validation
./run_split_workflow.sh
python3 training_data_generator/src/convert_qa_to_training.py data/qa_results
cd finetune-model && python3 finetune.py --config config-production.yaml
cd ../fact_checker && python src/validate_qa.py --input /data/results/training_data.json
```

### **Large Document Collections**

```bash
# Process in phases with resumability
./run_split_workflow.sh --step 1           # Chunk all documents
./run_split_workflow.sh --step 2 --resume  # Generate Q&A (resumable)
./run_split_workflow.sh --step 3           # Combine results

# Convert and train on specific documents
python3 training_data_generator/src/convert_qa_to_training.py \
  "data/qa_results/ImportantDoc_qa.json" \
  --output-dir data/training/priority/
```

### **Multi-Document Training**

```bash
# Convert multiple Q&A files to training format
python3 training_data_generator/src/convert_qa_to_training.py data/qa_results \
  --output-dir data/combined_training/

# Train on combined dataset
cd finetune-model
python3 finetune.py --dataset-path "../data/combined_training/*.jsonl"
```

## 🔍 **Monitoring Progress**

- **Data Generation**: Check `/data/qa_results/qa_generation_summary.json`
- **Training Progress**: Monitor `/finetune-model/*/logs/latest.log`
- **Individual Files**: Each `*_qa.json` shows completion status
- **Model Checkpoints**: Saved in `/finetune-model/output-dir/`
- **Error Logs**: Check `/data/logs/qa_failures_*.log`

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Import Errors**: Ensure conda environment is activated
2. **Permission Errors**: Scripts auto-fallback to local `./data/` directories
3. **Memory Issues**: Fine-tuning automatically optimizes for available hardware
4. **Training Failures**: Check dataset path in config and format validation

### **Error Recovery**

- **Failed Q&A Generation**: Use `--resume` flag to continue
- **Training Interruption**: Resume from latest checkpoint automatically
- **Data Format Issues**: Use conversion script to fix format
- **Hardware Limitations**: Framework automatically adjusts to available resources

## 🆕 **Recent Enhancements**

### **v2.0 Features**

- ✅ **Complete Fine-Tuning Pipeline**: End-to-end training framework
- ✅ **Enhanced Error Handling**: Improved LangChain compatibility and logging
- ✅ **Format Conversion**: Automatic Q&A to training data conversion
- ✅ **Hardware Optimization**: Automatic GPU detection and memory management
- ✅ **Modular Architecture**: Clear separation of concerns
- ✅ **Production Ready**: Comprehensive monitoring and error recovery

### **Performance Improvements**

- 🚀 **Faster Processing**: Optimized chunk processing and resume capabilities
- 🚀 **Better Resource Usage**: Hardware-aware training optimization
- 🚀 **Reduced Memory Footprint**: Efficient model loading strategies
- 🚀 **Improved Reliability**: Enhanced error isolation and recovery

## 🎉 **Complete ML Workflow**

This project now provides a **complete machine learning pipeline**:

1. **📄 Document Processing**: Multi-format document ingestion and chunking
2. **🤖 Q&A Generation**: AI-powered training data creation with resumability
3. **🔄 Format Conversion**: Automatic transformation to training formats
4. **🎯 Model Fine-Tuning**: LoRA-based training with hardware optimization
5. **✅ Quality Assurance**: Optional validation and filtering
6. **📊 Monitoring**: Comprehensive logging and progress tracking

Perfect for creating **domain-specific chatbots** and **specialized AI assistants** from your documents!
