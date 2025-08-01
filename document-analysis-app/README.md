# Document Analysis Q&A Generation System

A powerful, LangChain-based system for processing documents and generating high-quality question-answer pairs for LLM fine-tuning. This tool transforms various document formats into structured training data that can be used to fine-tune language models.

![Architecture](https://img.shields.io/badge/Architecture-LangChain-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

This system intelligently processes documents through a sophisticated pipeline that:

- **Extracts text** from multiple document formats (PDF, Markdown, HTML, DOCX, TXT)
- **Chunks documents** intelligently while preserving context
- **Generates Q&A pairs** using state-of-the-art LLMs
- **Validates output quality** with structured parsing and error recovery
- **Supports multiple LLM providers** (OpenAI, Anthropic Claude, Google Gemini, Local models)

## 🏗️ Architecture

```
Document Input → Document Loader → Text Splitter → LLM Provider → Q&A Chain → Training Data
```

### Core Components

- **LangChainProcessor**: Main orchestrator managing the entire pipeline
- **LangChainDocumentLoader**: Multi-format document loading with LangChain
- **EnhancedTextSplitter**: Intelligent text chunking with configurable strategies
- **UnifiedLLMProvider**: Standardized interface for multiple LLM providers
- **QAGenerationChain**: Sophisticated Q&A generation with validation

## 🚀 Features

### Document Processing

- ✅ **Multi-format support**: PDF, Markdown, HTML, DOCX, TXT
- ✅ **Intelligent chunking**: Preserves context with configurable overlap
- ✅ **Metadata preservation**: Tracks source information and processing details

### LLM Integration

- ✅ **Multi-provider support**: OpenAI, Anthropic, Google, Local (Ollama)
- ✅ **Flexible model selection**: Use any model supported by each provider
- ✅ **Cost optimization**: Switch providers based on cost/quality requirements
- ✅ **Robust error handling**: Automatic retries and fallback mechanisms

### Q&A Generation

- ✅ **High-quality prompts**: Engineered for diverse, educational Q&A pairs
- ✅ **Structured validation**: Pydantic models ensure data consistency
- ✅ **JSON output parsing**: Handles malformed LLM responses gracefully
- ✅ **Fallback parsing**: Multiple strategies to recover from parsing errors

### Quality Assurance

- ✅ **Source tracking**: Maintains provenance for every generated Q&A pair
- ✅ **Configurable parameters**: Fine-tune quality vs. quantity trade-offs
- ✅ **Validation pipeline**: Ensures training data meets quality standards

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended for environment management)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd document-analysis-app
   ```

2. **Create and activate conda environment**

   ```bash
   conda create -n finetune python=3.10
   conda activate finetune
   ```

3. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:

   ```env
   # OpenAI
   OPENAI_API_KEY=your_openai_api_key_here

   # Anthropic Claude
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Google Gemini
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **For local models (optional)**

   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Start Ollama service
   ollama serve

   # Pull a model (e.g., Llama 3)
   ollama pull llama3
   ```

## 🎮 Usage

### Basic Usage

```bash
# Activate the conda environment
conda activate finetune

# Process documents with OpenAI GPT-4
python src/main.py --provider openai --model gpt-4

# Process documents with Claude
python src/main.py --provider claude --model claude-3-5-sonnet-20241022

# Process documents with local model
python src/main.py --provider local --model llama3
```

### Advanced Configuration

```bash
# Customize processing parameters
python src/main.py \
    --provider claude \
    --model claude-3-5-sonnet-20241022 \
    --chunk-size 1500 \
    --chunk-overlap 300 \
    --questions-per-chunk 5 \
    --temperature 0.8 \
    --max-tokens 3000 \
    --incoming-dir ./my-docs \
    --output-file ./custom-training-data.json
```

### Utility Commands

```bash
# List available providers
python src/main.py --list-providers

# List models for a specific provider
python src/main.py --provider claude --list-models

# Show current configuration
python src/main.py --show-config

# Validate generated Q&A pairs
python src/validate_qa.py --input-file ./training_data.json
```

## 📁 File Organization

```
document-analysis-app/
├── incoming/                    # Place your documents here
│   ├── document1.pdf
│   ├── document2.md
│   └── document3.docx
├── src/
│   ├── main.py                  # Main entry point
│   ├── langchain_processing/    # LangChain-based components
│   │   ├── processors.py        # Main orchestrator
│   │   ├── document_loaders.py  # Multi-format document loading
│   │   ├── text_splitters.py    # Intelligent text chunking
│   │   ├── llm_providers.py     # Unified LLM interface
│   │   └── qa_chains.py         # Q&A generation chains
│   ├── utils/                   # Utility functions
│   └── validate_qa.py           # Q&A validation tools
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
└── README.md                    # This file
```

## ⚙️ Configuration Options

### Core Parameters

| Parameter        | Default                | Description                                  |
| ---------------- | ---------------------- | -------------------------------------------- |
| `--provider`     | `openai`               | LLM provider (openai, claude, gemini, local) |
| `--model`        | Provider default       | Specific model name                          |
| `--incoming-dir` | `./incoming`           | Directory containing source documents        |
| `--output-file`  | `./training_data.json` | Output file for training data                |

### Processing Parameters

| Parameter               | Default     | Description                               |
| ----------------------- | ----------- | ----------------------------------------- |
| `--chunk-size`          | `1000`      | Size of text chunks for processing        |
| `--chunk-overlap`       | `200`       | Overlap between consecutive chunks        |
| `--questions-per-chunk` | `3`         | Number of Q&A pairs to generate per chunk |
| `--splitting-strategy`  | `recursive` | Text splitting strategy                   |

### LLM Parameters

| Parameter       | Default  | Description                    |
| --------------- | -------- | ------------------------------ |
| `--temperature` | `0.7`    | LLM creativity (0.0-1.0)       |
| `--max-tokens`  | `2000`   | Maximum tokens in LLM response |
| `--api-key`     | From env | API key for the provider       |

## 📊 Output Format

The system generates structured training data in JSON format:

```json
{
  "metadata": {
    "generated_by": "finetune-ingest",
    "llm_provider": "claude",
    "model_used": "claude-3-5-sonnet-20241022",
    "num_documents": 3,
    "total_qa_pairs": 127,
    "generation_timestamp": "2024-01-15T10:30:00Z"
  },
  "training_pairs": [
    {
      "question": "What is the main purpose of machine learning?",
      "answer": "Machine learning enables computers to learn and improve from experience without being explicitly programmed, allowing them to make predictions and decisions based on data patterns.",
      "source_file": "ml_guide.pdf",
      "file_name": "ml_guide.pdf",
      "file_type": ".pdf",
      "chunk_id": 0,
      "chunk_start": 0,
      "chunk_end": 1000,
      "source_text": "Machine learning is a subset of artificial intelligence...",
      "generated_by": {
        "provider": "claude",
        "model": "claude-3-5-sonnet-20241022"
      }
    }
  ]
}
```

## 🛠️ Supported Formats

### Document Types

- **PDF**: Complex layouts, images, tables
- **Markdown**: Preserves structure and formatting
- **HTML**: Web pages and documentation
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files

### LLM Providers

#### OpenAI

- GPT-4, GPT-4 Turbo
- GPT-3.5 Turbo
- Custom fine-tuned models

#### Anthropic Claude

- Claude 3.5 Sonnet
- Claude 3 Opus
- Claude 3 Sonnet, Haiku

#### Google Gemini

- Gemini Pro
- Gemini Pro Vision

#### Local Models (via Ollama)

- Llama 3, Llama 2
- Mistral, CodeLlama
- Custom local models

## 🔧 Troubleshooting

### Common Issues

**Import Errors**

```bash
# Ensure you're in the right conda environment
conda activate finetune

# Reinstall dependencies
python -m pip install -r requirements.txt
```

**API Key Issues**

```bash
# Check if API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Or use --api-key argument
python src/main.py --provider openai --api-key your_key_here
```

**Model Not Found (404 Error)**

```bash
# List available models
python src/main.py --provider claude --list-models

# Use exact model name
python src/main.py --provider claude --model claude-3-5-sonnet-20241022
```

**Local Model Issues**

```bash
# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3

# Test connectivity
ollama list
```

### Performance Optimization

- **Chunk size**: Larger chunks = fewer API calls but potentially lower quality
- **Questions per chunk**: More questions = richer training data but higher costs
- **Temperature**: Lower = more consistent, Higher = more creative
- **Batch processing**: Enable for large document collections

## 🔄 Development Workflow

1. **Place documents** in the `incoming/` directory
2. **Configure parameters** via CLI arguments or environment variables
3. **Run processing** with your chosen LLM provider
4. **Validate output** using the validation tools
5. **Use training data** for your LLM fine-tuning pipeline

## 🚀 Future Enhancements

### Planned Features

- **Fact Checking**: RAG-based verification of generated Q&A pairs
- **Multi-agent Workflows**: Specialized agents for different validation tasks
- **External Knowledge Integration**: Connect to knowledge graphs and databases
- **Advanced Metrics**: Quality scoring and confidence ratings
- **Web Interface**: GUI for easier document processing and management

### Architecture Improvements

- **Streaming Processing**: Real-time document processing
- **Distributed Processing**: Scale across multiple machines
- **Caching Layer**: Reduce API calls for repeated processing
- **Plugin System**: Easy integration of custom components

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for robust LLM integration
- Document processing powered by specialized libraries (pdfplumber, python-docx, etc.)
- Inspired by the need for high-quality training data in LLM fine-tuning

---

**Need help?** Open an issue or check the troubleshooting section above.
