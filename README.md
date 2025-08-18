# Finetune Ingest

A comprehensive system for generating and validating high-quality training data for LLM fine-tuning.

## 🏗️ **Project Structure**

```
finetune-ingest/
├── document_chunker/           # Document processing and chunking
│   ├── src/
│   │   └── chunk_documents.py  # Document chunking script
│   └── README.md
├── training_data_generator/    # Q&A generation and combining
│   ├── src/
│   │   ├── generate_qa.py      # Q&A generation script (resumable)
│   │   └── combine_qa_results.py # Combine individual Q&A files
│   └── README.md
├── fact_checker/               # Q&A validation and quality assurance
│   ├── src/
│   │   └── validate_qa.py      # Validation script
│   └── README.md
├── common/                     # Shared libraries and utilities
│   ├── document_processing/    # Document processing modules
│   │   ├── document_loaders.py # LangChain document loaders
│   │   └── text_splitters.py   # Enhanced text splitting
│   ├── llm/                    # LLM interface and chains
│   │   ├── llm_providers.py    # Unified LLM interface
│   │   └── qa_chains.py        # Q&A generation chains
│   └── utils/
│       └── helpers.py          # Common helper functions
├── config.env                  # Centralized configuration
└── run_split_workflow.sh       # Main workflow automation script
```

## 🚀 **Quick Start**

### 1. Setup Environment

```bash
# Activate conda environment
source /Users/rainer/.zshrc
conda activate finetune

# Copy and configure environment
cp config.env .env
# Edit .env with your API keys and settings
```

### 2. Run Split Workflow (Recommended)

```bash
# Complete workflow: chunking → Q&A generation → combining
./run_split_workflow.sh

# Resume interrupted workflow
./run_split_workflow.sh --resume

# Run specific steps
./run_split_workflow.sh --step 1  # Document chunking only
./run_split_workflow.sh --step 2  # Q&A generation only (resumable!)
./run_split_workflow.sh --step 3  # Combine results only
```

### 3. Validate Results (Optional)

```bash
cd fact_checker
python src/validate_qa.py \
  --input /data/results/training_data.json \
  --output /data/results/validation_report.json
```

## 📊 **Data Flow**

### Split Workflow (Recommended)

```
📁 /data/incoming/          # Source documents (PDF, MD, HTML, DOCX, TXT)
    ↓ document_chunker/
📁 /data/chunks/            # document1_chunks.json, document2_chunks.json, ...
    ↓ training_data_generator/ (resumable!)
📁 /data/qa_results/        # document1_qa.json, document2_qa.json, ...
    ↓ training_data_generator/
📁 /data/results/           # training_data.json (final combined file)
    ↓ fact_checker/ (optional)
📁 /data/results/           # validation_report.json, training_data_filtered.json
```

## 🛡️ **Key Features**

### Resumability

- **Chunk-level tracking**: Q&A generation saves progress after each chunk
- **Interrupt-safe**: Ctrl+C gracefully saves current state
- **Atomic operations**: No data corruption from interruptions
- **Exact resume**: Continue from the exact stopping point

### Reliability

- **Error isolation**: Failed chunks don't affect completed ones
- **Fallback mechanisms**: Robust error handling at all levels
- **Progress monitoring**: Real-time status tracking
- **Validation**: Optional quality assurance step

### Flexibility

- **Multiple LLM providers**: OpenAI, Claude, Gemini, Local/Ollama
- **Configurable processing**: Chunk sizes, overlap, questions per chunk
- **Multiple document formats**: PDF, Markdown, HTML, DOCX, TXT
- **Modular architecture**: Run components independently

## 🔧 **Configuration**

All settings are managed through `config.env` / `.env`:

```bash
# LLM Provider Settings
GENERATOR_PROVIDER=local
GENERATOR_MODEL=qwen3:14b
GENERATOR_TEMPERATURE=0.3
OLLAMA_BASE_URL=http://192.168.50.133:11434

# Directory Structure (Split Workflow)
GENERATOR_INCOMING_DIR=/data/incoming
GENERATOR_PROCESS_DIR=/data/chunks          # Document chunks
GENERATOR_OUTPUT_DIR=/data/qa_results       # Individual Q&A files
GENERATOR_OUTPUT_FILE=/data/results/training_data.json

# Processing Settings
GENERATOR_CHUNK_SIZE=1000
GENERATOR_CHUNK_OVERLAP=200
GENERATOR_QUESTIONS_PER_CHUNK=5
GENERATOR_RESUME=true                       # Enable resumability
GENERATOR_BATCH_PROCESSING=false            # Recommended for resumability

# Validation Settings
VALIDATOR_PROVIDER=local
VALIDATOR_MODEL=qwen3:14b
VALIDATOR_THRESHOLD=8.0
```

## 📚 **Module Documentation**

- **[Document Chunker](document_chunker/README.md)**: Document processing and chunking
- **[Training Data Generator](training_data_generator/README.md)**: Q&A generation and combining
- **[Fact Checker](fact_checker/README.md)**: Quality validation and filtering

## 🎯 **Use Cases**

### Development & Testing

```bash
# Process a few documents quickly
./run_split_workflow.sh --step 1
./run_split_workflow.sh --step 2
```

### Production Processing

```bash
# Full workflow with validation
./run_split_workflow.sh
cd fact_checker && python src/validate_qa.py --input /data/results/training_data.json
```

### Large Document Sets

```bash
# Process in phases, resume as needed
./run_split_workflow.sh --step 1           # Chunk all documents first
./run_split_workflow.sh --step 2 --resume  # Generate Q&A (resumable)
./run_split_workflow.sh --step 3           # Combine when ready
```

### Interruption Recovery

```bash
# If Q&A generation is interrupted, simply resume:
./run_split_workflow.sh --step 2 --resume
```

## 🔍 **Monitoring Progress**

- **Chunking**: Check `/data/chunks/chunking_summary.json`
- **Q&A Generation**: Check `/data/qa_results/qa_generation_summary.json`
- **Individual Files**: Each `*_qa.json` file shows completion status
- **Final Results**: Check `/data/results/training_data.json` for summary

## 🚨 **Troubleshooting**

### Common Issues

1. **Import Errors**: Ensure conda environment is activated
2. **Permission Errors**: Scripts auto-fallback to local `./data/` directories
3. **Interrupted Q&A**: Use `--resume` flag to continue from stopping point
4. **Missing Dependencies**: Check `requirements.txt` in each module

### Error Recovery

- **Failed chunks**: Tracked in `failed_chunks` array, don't stop processing
- **Incomplete files**: Status field shows `in_progress`, `interrupted`, `failed`
- **Resume from interruption**: Progress saved after each chunk completion

## 🔄 **Migration from Legacy**

If you were using the original `data_generator/src/main.py`:

### Old Command

```bash
cd data_generator
python src/main.py --provider local --model qwen3:14b --resume
```

### New Command

```bash
# From project root
./run_split_workflow.sh --resume
```

The new split workflow provides the same functionality with improved resumability and better error handling.

## 🧹 **Refactored Architecture**

The project has been completely refactored for better maintainability:

### What Changed

- **Removed Legacy Code**: Eliminated 400+ lines of unused code from `data_generator/`
- **Shared Modules**: All common functionality moved to `common/` folder
- **Logical Organization**: Document processing vs. LLM functionality clearly separated
- **Cleaner Imports**: No complex path manipulation needed
- **Single Source of Truth**: Each module has one authoritative location

### New Structure

- `common/document_processing/` - Document loading and text splitting
- `common/llm/` - LLM providers and Q&A generation chains
- `common/utils/` - Shared helper functions

## 🎉 **Benefits of Split Architecture**

1. **Never lose progress**: Chunk-level resumability
2. **Better debugging**: Isolate issues to specific components
3. **Resource management**: Process large datasets in phases
4. **Selective re-processing**: Re-run only failed or updated parts
5. **Parallel development**: Work on different components independently
6. **Clear separation**: Document processing vs. Q&A generation vs. validation
7. **Maintainable codebase**: No duplicated or legacy code
