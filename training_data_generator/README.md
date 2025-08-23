# Training Data Generator

AI-powered Q&A generation system for creating high-quality training datasets from document chunks with full resumability and format conversion capabilities.

## 🎯 **Purpose**

The Training Data Generator transforms chunked documents into structured question-answer pairs optimized for LLM fine-tuning. It provides robust, resumable Q&A generation with multiple output formats and comprehensive error handling for production-scale training data creation.

## ✨ **Features**

### **🤖 AI-Powered Q&A Generation**

- **Multi-Provider LLM Support**: OpenAI, Claude, Gemini, Local/Ollama models
- **Intelligent Question Creation**: Diverse question types (factual, analytical, explanatory, inferential)
- **Context-Aware Answers**: Comprehensive responses grounded in source text
- **Configurable Generation**: Questions per chunk, temperature, token limits

### **🔄 Industrial-Strength Resumability**

- **Chunk-Level Tracking**: Individual chunk completion status
- **Interrupt-Safe Processing**: Graceful handling of Ctrl+C interruptions
- **Atomic Progress Saves**: Immediate persistence after each chunk
- **Exact Resume Points**: Continue from precise stopping location
- **Status Persistence**: Complete state management across sessions

### **📊 Flexible Output Formats**

- **Individual Q&A Files**: Per-document processing results
- **Combined Training Data**: Unified dataset for training
- **Alpaca Format Conversion**: Ready-to-train JSONL format
- **Metadata Preservation**: Complete source traceability

### **⚡ Advanced Processing**

- **Sequential & Batch Modes**: Optimal processing strategies
- **Error Isolation**: Failed chunks don't affect completed work
- **Progress Monitoring**: Real-time processing status
- **Quality Assurance**: Comprehensive validation and statistics

## 🛠️ **Components**

### **1. Q&A Generation (`generate_qa.py`)**

Generates question-answer pairs from document chunks with full resumability.

#### **Basic Usage**

```bash
cd training_data_generator

# Generate Q&A with local model
python src/generate_qa.py \
  --provider local \
  --model qwen3:14b \
  --ollama-base-url http://192.168.50.133:11434 \
  --chunks-dir /data/chunks \
  --qa-dir /data/qa_results \
  --questions-per-chunk 5 \
  --temperature 0.3 \
  --resume
```

#### **Advanced Configuration**

```bash
# Custom prompts and advanced settings
python src/generate_qa.py \
  --provider openai \
  --model gpt-4 \
  --questions-per-chunk 7 \
  --temperature 0.2 \
  --max-tokens 3000 \
  --qa-extra-instructions "Focus on technical implementation details" \
  --qa-prompt-template-file custom_template.txt \
  --resume
```

### **2. Q&A Combiner (`combine_qa_results.py`)**

Combines individual Q&A files into unified training datasets.

#### **Basic Combining**

```bash
# Combine all Q&A files into single training dataset
python src/combine_qa_results.py \
  --qa-dir /data/qa_results \
  --output-file /data/results/training_data.json
```

#### **Custom Output Location**

```bash
# Combine with custom output path
python src/combine_qa_results.py \
  --qa-dir /data/qa_results \
  --output-file /data/training/nzism_training_data.json
```

### **3. Format Converter (`convert_qa_to_training.py`)**

Converts Q&A data to Alpaca-style training format for fine-tuning.

#### **Single File Conversion**

```bash
# Convert specific Q&A file to training format
python src/convert_qa_to_training.py \
  "data/qa_results/NZISM-ISM Document-V.-3.9-April-2025_qa.json" \
  --output-dir data/training
```

#### **Batch Conversion**

```bash
# Convert all Q&A files to training format
python src/convert_qa_to_training.py \
  data/qa_results \
  --output-dir data/training
```

## 🚀 **Quick Start Workflows**

### **Complete Pipeline**

```bash
# Step 1: Generate Q&A pairs
python src/generate_qa.py \
  --provider local \
  --model qwen3:14b \
  --chunks-dir /data/chunks \
  --qa-dir /data/qa_results \
  --resume

# Step 2: Combine results (optional)
python src/combine_qa_results.py \
  --qa-dir /data/qa_results \
  --output-file /data/results/training_data.json

# Step 3: Convert to training format
python src/convert_qa_to_training.py \
  data/qa_results \
  --output-dir data/training
```

### **Resume Interrupted Processing**

```bash
# Resume from exact stopping point
python src/generate_qa.py \
  --chunks-dir /data/chunks \
  --qa-dir /data/qa_results \
  --resume
```

### **High-Quality Generation**

```bash
# Use commercial models for highest quality
python src/generate_qa.py \
  --provider openai \
  --model gpt-4 \
  --questions-per-chunk 10 \
  --temperature 0.1 \
  --max-tokens 4000 \
  --qa-extra-instructions "Create comprehensive, detailed questions and answers"
```

## 🔧 **Configuration**

### **Environment Variables**

```bash
# LLM Provider Settings
GENERATOR_PROVIDER=local              # openai, claude, gemini, local
GENERATOR_MODEL=qwen3:14b            # Model name
GENERATOR_TEMPERATURE=0.3            # LLM temperature (0.0-1.0)
GENERATOR_MAX_TOKENS=2000           # Maximum response tokens

# Directory Structure
GENERATOR_PROCESS_DIR=/data/chunks          # Input chunks directory
GENERATOR_OUTPUT_DIR=/data/qa_results       # Q&A files output directory
GENERATOR_OUTPUT_FILE=/data/results/training_data.json  # Combined output file

# Generation Settings
GENERATOR_QUESTIONS_PER_CHUNK=5     # Questions to generate per chunk
GENERATOR_BATCH_PROCESSING=false    # Use batch processing mode
GENERATOR_RESUME=true               # Enable resume capability

# Local Model Settings (Ollama)
OLLAMA_BASE_URL=http://192.168.50.133:11434
OLLAMA_TOP_K=40
OLLAMA_TOP_P=0.9
OLLAMA_REPEAT_PENALTY=1.1
OLLAMA_NUM_PREDICT=512
```

### **Command Line Arguments**

#### **Q&A Generation**

```bash
# Directory Configuration
--chunks-dir          Input chunks directory
--qa-dir              Output Q&A files directory

# LLM Provider Configuration
--provider            LLM provider: openai, claude, gemini, local
--model               Model name (e.g., gpt-4, qwen3:14b)
--api-key             API key for commercial providers
--temperature         LLM temperature (0.0-1.0)
--max-tokens          Maximum response tokens

# Generation Configuration
--questions-per-chunk Number of questions per chunk (default: 3)
--qa-system-message   Custom system message
--qa-extra-instructions Additional instructions for Q&A generation
--qa-prompt-template-file Custom prompt template file

# Processing Options
--batch-processing    Enable batch processing mode
--resume              Resume from interruption

# Local Model Options (Ollama)
--ollama-base-url     Ollama server URL
--ollama-top-k        Top-K sampling parameter
--ollama-top-p        Top-P sampling parameter
--ollama-repeat-penalty Repeat penalty parameter
--ollama-num-predict  Maximum prediction tokens
--ollama-stop         Stop sequences (can be specified multiple times)
```

#### **Q&A Combining**

```bash
--qa-dir              Input Q&A files directory
--output-file         Output combined training data file
```

#### **Format Conversion**

```bash
input                 Input Q&A file or directory
--output-dir          Output directory for training files
--output-file         Specific output filename (for single file input)
```

## 📊 **Output Formats**

### **Individual Q&A Files**

Each document produces a `{document_name}_qa.json` file:

```json
{
  "metadata": {
    "file_name": "NZISM-ISM Document-V.-3.9-April-2025.pdf",
    "source_file": "/data/documents/NZISM-ISM Document-V.-3.9-April-2025.pdf",
    "file_type": ".pdf",
    "file_size": 5889299,
    "processed_at": "2025-08-23T10:30:45.123456"
  },
  "processing_info": {
    "splitting_strategy": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "total_chunks": 1951
  },
  "qa_generation_info": {
    "provider": "local",
    "model": "qwen3:14b",
    "questions_per_chunk": 5,
    "temperature": 0.3,
    "max_tokens": 2000,
    "generated_at": "2025-08-23T11:15:30.987654"
  },
  "training_pairs": [
    {
      "question": "What is the purpose of the NZISM?",
      "answer": "The NZISM details processes and controls essential for protecting all New Zealand Government information and systems.",
      "source_file": "/data/documents/NZISM-ISM Document-V.-3.9-April-2025.pdf",
      "file_name": "NZISM-ISM Document-V.-3.9-April-2025.pdf",
      "file_type": ".pdf",
      "chunk_id": "0_0",
      "chunk_index": 0,
      "chunk_start": 0,
      "chunk_end": 971,
      "source_text": "1 Version_3.9__April-2025..."
    }
  ],
  "status": "completed",
  "completed_chunks": [0, 1, 2, 5],
  "failed_chunks": [],
  "summary": {
    "total_chunks": 1951,
    "completed_chunks": 1916,
    "failed_chunks": 35,
    "total_qa_pairs": 5849
  },
  "completion_time": "2025-08-23T14:22:15.456789"
}
```

### **Combined Training Data**

The combiner creates a unified `training_data.json`:

```json
{
  "metadata": {
    "generated_by": "qa_combiner",
    "generated_at": "2025-08-23T14:25:00.123456",
    "total_qa_pairs": 12450,
    "num_documents": 5,
    "source_qa_files": [
      "NZISM-ISM Document-V.-3.9-April-2025_qa.json",
      "Policy-Manual-2024_qa.json"
    ],
    "processing_config": {
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "splitting_strategy": "recursive",
      "questions_per_chunk": 5,
      "temperature": 0.3,
      "max_tokens": 2000,
      "llm_provider": "local",
      "model_used": "qwen3:14b"
    }
  },
  "documents": [
    {
      "file_info": {...},
      "processing_info": {...},
      "qa_generation_info": {...},
      "summary": {...},
      "qa_pairs_count": 5849,
      "source_file": "NZISM-ISM Document-V.-3.9-April-2025_qa.json"
    }
  ],
  "training_pairs": [...]
}
```

### **Alpaca Training Format**

The converter creates `{document_name}_trainingdata.jsonl`:

```jsonl
{"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the purpose of the NZISM?\n\n### Response:\nThe NZISM details processes and controls essential for protecting all New Zealand Government information and systems."}
{"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWho is encouraged to use the NZISM?\n\n### Response:\nNew Zealand Government departments, agencies, Crown entities, local government, and private sector organisations are encouraged to use the NZISM."}
```

## 🎯 **Use Cases**

### **Production Training Data Generation**

```bash
# High-quality generation for production models
python src/generate_qa.py \
  --provider openai \
  --model gpt-4 \
  --questions-per-chunk 10 \
  --temperature 0.1 \
  --max-tokens 4000 \
  --qa-extra-instructions "Create comprehensive, production-quality Q&A pairs with detailed explanations"
```

### **Large Document Collections**

```bash
# Process large document sets with resumability
python src/generate_qa.py \
  --chunks-dir /data/chunks \
  --qa-dir /data/qa_results \
  --questions-per-chunk 7 \
  --batch-processing \
  --resume
```

### **Domain-Specific Fine-Tuning**

```bash
# Generate specialized training data
python src/generate_qa.py \
  --qa-extra-instructions "Focus on technical implementation details and best practices. Include code examples where applicable." \
  --questions-per-chunk 8 \
  --temperature 0.2
```

### **Multi-Format Output**

```bash
# Generate multiple output formats
python src/generate_qa.py --resume
python src/combine_qa_results.py
python src/convert_qa_to_training.py data/qa_results --output-dir data/training
```

## 📈 **Performance Optimization**

### **Generation Quality**

- **Higher Questions/Chunk**: More comprehensive coverage, longer processing
- **Lower Temperature**: More focused, consistent responses
- **Commercial Models**: Higher quality, additional cost
- **Custom Prompts**: Domain-specific optimization

### **Processing Speed**

- **Local Models**: Faster, unlimited usage
- **Batch Processing**: Parallel generation (experimental)
- **Resume Capability**: Efficient restart from interruptions
- **Chunk-Level Parallelism**: Independent chunk processing

### **Resource Management**

- **Memory Usage**: Sequential processing prevents memory issues
- **Disk Space**: Individual files allow incremental processing
- **Network Usage**: Configurable batch sizes and timeouts

## 🔍 **Monitoring and Quality**

### **Progress Tracking**

- **Real-Time Logging**: Chunk-by-chunk progress updates
- **Status Persistence**: Complete state tracking across sessions
- **Summary Reports**: Comprehensive processing statistics
- **Error Reporting**: Detailed failure analysis and recovery

### **Quality Metrics**

- **Generation Success Rate**: Percentage of successful chunks
- **Q&A Pair Count**: Total questions and answers generated
- **Processing Time**: Performance metrics per chunk/document
- **Error Analysis**: Failed chunk identification and causes

### **Resume Capabilities**

- **Exact Resumption**: Continue from precise stopping point
- **Status Validation**: Verify completion before resuming
- **Progress Preservation**: No lost work from interruptions
- **Flexible Restart**: Resume with different parameters

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Generation Failures**:

   - Check LLM provider configuration and API keys
   - Verify model availability and parameters
   - Review chunk content quality and size

2. **Resume Problems**:

   - Ensure Q&A files have correct status fields
   - Check file permissions and disk space
   - Verify chunk directory consistency

3. **Output Format Issues**:

   - Validate JSON structure in Q&A files
   - Check conversion script parameters
   - Verify training format requirements

4. **Performance Issues**:
   - Reduce questions per chunk for faster processing
   - Use local models for unlimited generation
   - Monitor memory usage with large documents

### **Quality Issues**

- **Poor Question Quality**: Adjust temperature, add custom instructions
- **Inconsistent Answers**: Lower temperature, use commercial models
- **Context Loss**: Increase chunk overlap in document chunking
- **Repetitive Content**: Vary question types, improve prompts

### **Error Recovery**

- **Failed Chunks**: Tracked separately, don't affect completed work
- **Interrupted Processing**: Resume exactly where stopped
- **Corrupted Files**: Automatic validation and recovery
- **Network Issues**: Retry mechanisms and graceful degradation

## 🔄 **Integration**

### **Pipeline Integration**

```bash
# Complete workflow integration
cd document_chunker && python src/chunk_documents.py
cd ../training_data_generator && python src/generate_qa.py --resume
python src/convert_qa_to_training.py data/qa_results --output-dir data/training
cd ../finetune-model && python finetune.py --dataset-path "../data/training/*.jsonl"
```

### **Custom Workflows**

The generator can be imported and used programmatically:

```python
from training_data_generator.src.generate_qa import QAGenerator

generator = QAGenerator(
    chunks_dir="/data/chunks",
    qa_dir="/data/qa_results",
    provider="openai",
    model="gpt-4",
    questions_per_chunk=8,
    temperature=0.2
)

summary = generator.generate_qa_for_all_documents(resume=True)
print(f"Generated Q&A for {summary['successful_files']} documents")
```

## 📚 **Dependencies**

**Core Requirements:**

- `langchain`: LLM integration framework
- `python-dotenv`: Environment configuration
- `pydantic`: Data validation and parsing

**LLM Providers:**

- `openai`: OpenAI API integration
- `anthropic`: Claude API integration
- `google-generativeai`: Gemini API integration
- `ollama`: Local model support

## 🎉 **Benefits**

- **Production Ready**: Industrial-strength resumability and error handling
- **Format Flexibility**: Multiple output formats for different use cases
- **Quality Assurance**: Comprehensive validation and monitoring
- **Cost Efficiency**: Local model support and intelligent resource usage
- **Scalability**: Handle single documents to large document collections
- **Transparency**: Complete traceability and provenance tracking

## 🔗 **Next Steps**

After generating training data:

1. **Quality Validation**: Use `fact_checker` to validate Q&A quality
2. **Format Conversion**: Convert to specific training formats as needed
3. **Model Fine-Tuning**: Use `finetune-model` for training
4. **Iterative Improvement**: Analyze results and refine generation parameters

The Training Data Generator provides the foundation for creating high-quality, domain-specific training datasets that enable effective LLM fine-tuning!
