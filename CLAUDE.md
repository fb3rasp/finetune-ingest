# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Setup and Installation

- Every time you start a new terminal, you need to source the .zshrc file to activate the conda environment.

```bash
source /Users/rainer/.zshrc
conda activate finetune
```

- Install the dependencies and copy the example environment file.

```bash
cd document-analysis-app
pip install -r requirements.txt
cp config_example.env .env
# Edit .env with your API keys
```

### Core Application Usage

#### Legacy Implementation (Original)
```bash
# Basic training data generation
python src/main.py --provider openai --model gpt-4

# List available providers and models
python src/main.py --list-providers
python src/main.py --provider openai --list-models

# Generate with custom parameters
python src/main.py --provider claude --questions-per-chunk 5 --chunk-size 800 --temperature 0.5
```

#### LangChain Implementation (Enhanced)
```bash
# Basic training data generation with LangChain
python src/main_langchain.py --provider openai --model gpt-4

# List available providers and models
python src/main_langchain.py --list-providers
python src/main_langchain.py --provider openai --list-models

# Advanced configuration with LangChain features
python src/main_langchain.py \
  --provider claude \
  --model claude-3-5-sonnet-20241022 \
  --chunk-size 1200 \
  --chunk-overlap 300 \
  --questions-per-chunk 5 \
  --splitting-strategy recursive \
  --batch-processing \
  --temperature 0.8

# Show current configuration
python src/main_langchain.py --show-config

# Run comprehensive demo
python example_usage.py
```

### MCP Server

```bash
# Start MCP server
chmod +x start_mcp_server.sh
./start_mcp_server.sh

# Or directly
python src/mcp_server/server.py
```

### Testing and Development

```bash
# Run the demo script (validates functionality)
python example_usage.py

# Validate setup
python -c "from src.data_processing.qa_generator import QAGenerator; print('Setup OK')"
```

### QA Validation (Quality Assurance)

```bash
# Validate training data for factual accuracy
python src/validate_qa.py --input training_data.json

# Validate with specific validator model
python src/validate_qa.py --input training_data.json --validator-provider claude --validator-model claude-3-5-sonnet-20241022

# Generate filtered training data (removes low-quality pairs)
python src/validate_qa.py --input training_data.json --filtered-output training_data_filtered.json --filter-threshold 7.0

# Show detailed validation results
python src/validate_qa.py --input training_data.json --show-details

# List available validator models
python src/validate_qa.py --list-providers --input dummy
```

### Automated Processing Pipeline

```bash
# Complete automated pipeline: process documents + validate with Claude Sonnet
./process_and_validate.sh

# Dry run to see what would be executed
./process_and_validate.sh --dry-run

# Custom configuration
./process_and_validate.sh --questions-per-chunk 7 --validation-threshold 7.5

# Skip validation (only process documents)
./process_and_validate.sh --skip-validation

# Quiet mode for automated workflows
./process_and_validate.sh --quiet

# Show help with all options
./process_and_validate.sh --help
```

## Architecture Overview

### Legacy Components (Original Implementation)

**DocumentProcessor** (`src/data_processing/preprocess.py`)

- Handles multi-format document extraction (PDF, MD, HTML, DOCX, TXT)
- Text chunking with configurable overlap
- Metadata extraction and section tracking

**QAGenerator** (`src/data_processing/qa_generator.py`)

- Multi-LLM provider support (OpenAI, Claude, Gemini, Local/Ollama)
- Q&A pair generation with source traceability
- Configurable parameters (temperature, questions per chunk)

### LangChain Components (Enhanced Implementation)

**LangChainDocumentLoader** (`src/langchain_processing/document_loaders.py`)

- Unified document loading using specialized LangChain loaders
- Enhanced metadata extraction and error handling
- Support for PyPDF, Unstructured Markdown/HTML/Word, and Text loaders

**EnhancedTextSplitter** (`src/langchain_processing/text_splitters.py`)

- Multiple splitting strategies: recursive, character, markdown, HTML
- Hierarchical splitting with semantic awareness
- Metadata preservation through splitting process

**UnifiedLLMProvider** (`src/langchain_processing/llm_providers.py`)

- Single interface for all LLM providers using LangChain wrappers
- Built-in retry logic, error handling, and rate limiting
- Support for streaming and batch processing

**QAGenerationChain** (`src/langchain_processing/qa_chains.py`)

- Advanced Q&A generation using LangChain chains and prompt templates
- Structured output parsing with Pydantic models
- Robust error handling with fallback parsing

**LangChainProcessor** (`src/langchain_processing/processors.py`)

- Main orchestrator coordinating all LangChain components
- Unified interface replacing legacy implementation
- Enhanced configuration and monitoring capabilities

**QAValidator** (`src/langchain_processing/qa_validator.py`)

- LangChain-based validation system for generated Q&A pairs
- Factual accuracy checking against source documents
- Automated scoring and quality filtering
- Comprehensive validation reporting and metrics

### Shared Components

**MCP Server** (`src/mcp_server/server.py`)

- Model Context Protocol server for MCP client integration
- Provides tools for document processing and Q&A generation
- Async server architecture

### Data Flow

1. Documents placed in `./incoming` directory
2. DocumentProcessor extracts text and creates chunks
3. QAGenerator processes chunks to create Q&A pairs with metadata
4. Output saved as structured JSON with full source traceability
5. **QAValidator validates generated pairs for factual accuracy** (optional)
6. **Filtered high-quality training data exported** (optional)

### Automated Pipeline Data Flow

1. **./process_and_validate.sh** → Complete automation
2. Documents in `./incoming/` → Processing
3. Raw training data → `./processing/raw_data/`
4. Validation with Claude Sonnet → `./processing/reports/`
5. Filtered high-quality data → `./processing/validated_data/`
6. Pipeline logs → `./processing/logs/`

### Directory Structure

```
document-analysis-app/
├── src/
│   ├── main.py                      # CLI entry point (legacy)
│   ├── main_langchain.py            # CLI entry point (LangChain)
│   ├── data_processing/             # Legacy processing modules
│   │   ├── preprocess.py            # Original document processor
│   │   └── qa_generator.py          # Original Q&A generator
│   ├── langchain_processing/        # Enhanced LangChain modules
│   │   ├── document_loaders.py      # LangChain document loaders
│   │   ├── text_splitters.py        # Enhanced text splitting
│   │   ├── llm_providers.py         # Unified LLM interface
│   │   ├── qa_chains.py             # Q&A generation chains
│   │   └── processors.py            # Main orchestrator
│   ├── mcp_server/                  # MCP server implementation
│   └── utils/                       # Helper functions
├── incoming/                        # Input documents (auto-created)
├── output/                         # Generated training data
└── requirements.txt                # Dependencies (includes LangChain)
```

## Environment Configuration

Required environment variables in `.env`:

- `OPENAI_API_KEY` - For OpenAI GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- `GOOGLE_API_KEY` - For Gemini models
- `OLLAMA_HOST` - For local models (default: <http://localhost:11434>)

Processing parameters:

- `DEFAULT_CHUNK_SIZE=1000`
- `DEFAULT_QUESTIONS_PER_CHUNK=3`
- `DEFAULT_TEMPERATURE=0.7`

## LLM Provider Support

The application supports multiple LLM providers with automatic fallback:

- **OpenAI**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude-3-opus, Claude-3-sonnet, Claude-3-haiku
- **Google**: Gemini-pro, Gemini-1.5-pro
- **Local**: Ollama-supported models (llama3, mistral, etc.)

## Key Integration Points

### MCP Tools Available

- `list_documents` - List files in incoming directory
- `configure_llm` - Set up LLM provider
- `process_document` - Extract text from single document
- `generate_qa_pairs` - Create Q&A pairs from document
- `generate_training_data` - Process all documents and create dataset

### Output Format

JSON structure with metadata, document info, and training pairs with full source traceability including chunk positions and file references.

## LangChain Integration Benefits

### Key Improvements Over Legacy Implementation

**Unified Interface**
- Single, consistent API for all LLM providers
- Standardized error handling and retry mechanisms
- Built-in rate limiting and request management

**Enhanced Document Processing**
- Specialized loaders for different file formats
- Better metadata extraction and preservation
- Improved error handling for corrupted documents

**Advanced Text Splitting**
- Multiple splitting strategies (recursive, semantic, format-aware)
- Hierarchical splitting with context preservation
- Configurable overlap and chunk size management

**Robust Q&A Generation**
- Structured output parsing with validation
- Fallback mechanisms for malformed responses
- Batch processing capabilities for efficiency

**Better Observability**
- Comprehensive logging and monitoring
- Configuration introspection and validation
- Quality metrics and processing statistics

### Migration Path

**For New Projects**: Use `main_langchain.py` for all new implementations
**For Existing Projects**: Legacy `main.py` remains fully functional
**Gradual Migration**: Components can be migrated incrementally

### LangChain-Specific Features

```bash
# Use different splitting strategies
python src/main_langchain.py --splitting-strategy markdown --chunk-size 1500

# Enable batch processing for better performance
python src/main_langchain.py --batch-processing --questions-per-chunk 5

# Monitor processing with detailed configuration
python src/main_langchain.py --show-config
```

## Development Notes

- No formal test suite - use `example_usage.py` for validation
- Error handling includes provider-specific troubleshooting messages
- Supports batch processing of multiple documents (enhanced in LangChain version)
- Automatic sample document creation when incoming directory is empty
- LangChain implementation provides superior error handling and observability
