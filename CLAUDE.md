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

```bash
# Basic training data generation
python src/main.py --provider openai --model gpt-4

# List available providers and models
python src/main.py --list-providers
python src/main.py --provider openai --list-models

# Generate with custom parameters
python src/main.py --provider claude --questions-per-chunk 5 --chunk-size 800 --temperature 0.5

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

## Architecture Overview

### Core Components

**DocumentProcessor** (`src/data_processing/preprocess.py`)

- Handles multi-format document extraction (PDF, MD, HTML, DOCX, TXT)
- Text chunking with configurable overlap
- Metadata extraction and section tracking

**QAGenerator** (`src/data_processing/qa_generator.py`)

- Multi-LLM provider support (OpenAI, Claude, Gemini, Local/Ollama)
- Q&A pair generation with source traceability
- Configurable parameters (temperature, questions per chunk)

**MCP Server** (`src/mcp_server/server.py`)

- Model Context Protocol server for MCP client integration
- Provides tools for document processing and Q&A generation
- Async server architecture

### Data Flow

1. Documents placed in `./incoming` directory
2. DocumentProcessor extracts text and creates chunks
3. QAGenerator processes chunks to create Q&A pairs with metadata
4. Output saved as structured JSON with full source traceability

### Directory Structure

```
document-analysis-app/
├── src/
│   ├── main.py              # CLI entry point
│   ├── data_processing/     # Core processing modules
│   ├── mcp_server/         # MCP server implementation
│   └── utils/              # Helper functions
├── incoming/               # Input documents (auto-created)
├── output/                # Generated training data
└── requirements.txt       # Dependencies
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

## Development Notes

- No formal test suite - use `example_usage.py` for validation
- Error handling includes provider-specific troubleshooting messages
- Supports batch processing of multiple documents
- Automatic sample document creation when incoming directory is empty
