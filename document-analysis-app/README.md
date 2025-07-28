# Finetune-Ingest: Training Data Generation for LLM Fine-tuning

A comprehensive tool for generating high-quality training data from documents to fine-tune Large Language Models (LLMs). Extract insights from your documents and create Q&A pairs with full source traceability for building domain-specific chatbots.

## 🆕 **New: LangChain Integration**

This project now features an enhanced **LangChain-powered implementation** alongside the original system, providing superior document processing, advanced text splitting, unified LLM interfaces, and robust error handling.

## 🌟 Features

### Core Features
- **Multi-format Document Processing**: PDF, Markdown, HTML, DOCX, TXT
- **Multiple LLM Providers**: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini), Local (Ollama)
- **Source Traceability**: Full metadata linking Q&A pairs back to source documents
- **Intelligent Text Chunking**: Automatic text segmentation with configurable overlap
- **MCP Server Interface**: Model Context Protocol server for integration with MCP clients
- **Batch Processing**: Process multiple documents simultaneously
- **Configurable Parameters**: Customizable chunk sizes, question counts, temperature settings

### 🆕 Enhanced LangChain Features
- **Advanced Text Splitting**: Multiple strategies (recursive, semantic, markdown-aware, HTML-aware)
- **Unified LLM Interface**: Single API for all providers with built-in retry logic and error handling
- **Structured Output Parsing**: Robust JSON parsing with validation and fallback mechanisms
- **Enhanced Document Loading**: Specialized loaders for each format with better metadata extraction
- **Batch Processing**: Improved efficiency for large document sets
- **Quality Metrics**: Enhanced processing statistics and quality tracking
- **Configuration Introspection**: Detailed configuration monitoring and validation

## 🚀 Quick Start

### 1. Installation

```bash
cd document-analysis-app

# Install dependencies
pip install -r requirements.txt

# Setup environment configuration
cp config_example.env .env
# Edit .env with your API keys

# Optional: Install MCP server dependencies (if needed)
# pip install -r requirements-mcp.txt
```

### 2. Add Your Documents

Place your documents in the `./incoming` directory:

```bash
mkdir -p incoming
# Copy your PDFs, markdown files, etc. to ./incoming/
```

### 3. Generate Training Data

#### Option A: Auto-Select Implementation (Recommended)

```bash
# Automatically uses LangChain if available, otherwise falls back to legacy
python src/main_simple.py --provider openai --model gpt-4

# The system will automatically choose the best available implementation
python src/main_simple.py --provider claude --model claude-3-5-sonnet-20241022
```

#### Option B: Enhanced LangChain Implementation (Manual)

```bash
# Using OpenAI GPT-4 with LangChain
python src/main_langchain.py --provider openai --model gpt-4

# Using Claude with advanced features
python src/main_langchain.py --provider claude --model claude-3-5-sonnet-20241022

# Using Gemini with batch processing
python src/main_langchain.py --provider gemini --model gemini-pro --batch-processing

# Using local LLM with custom chunking
python src/main_langchain.py --provider local --model llama3 --splitting-strategy recursive
```

#### Option C: Original Implementation (Legacy)

```bash
# Using OpenAI GPT-4
python src/main.py --provider openai --model gpt-4

# Using Claude
python src/main.py --provider claude --model claude-3-sonnet-20240229

# Using Gemini
python src/main.py --provider gemini --model gemini-pro

# Using local LLM (requires Ollama)
python src/main.py --provider local --model llama3
```

## 📖 Usage Examples

### Enhanced LangChain Interface (Recommended)

```bash
# List available providers and models
python src/main_langchain.py --list-providers

# Show current configuration
python src/main_langchain.py --show-config

# Advanced configuration with multiple features
python src/main_langchain.py \
  --provider openai \
  --model gpt-4 \
  --questions-per-chunk 5 \
  --chunk-size 1200 \
  --chunk-overlap 300 \
  --splitting-strategy recursive \
  --batch-processing \
  --temperature 0.5 \
  --output-file enhanced_training_data.json

# Markdown-aware processing for documentation
python src/main_langchain.py \
  --provider claude \
  --splitting-strategy markdown \
  --chunk-size 1500 \
  --batch-processing

# Process with semantic chunking
python src/main_langchain.py \
  --provider gemini \
  --splitting-strategy recursive \
  --chunk-overlap 400
```

### Original Command Line Interface (Legacy)

```bash
# List available providers
python src/main.py --list-providers

# List models for a specific provider
python src/main.py --provider openai --list-models

# Generate with custom parameters
python src/main.py \
  --provider openai \
  --model gpt-4 \
  --questions-per-chunk 5 \
  --chunk-size 800 \
  --temperature 0.5 \
  --output-file custom_training_data.json

# Process only specific files
python src/main.py \
  --provider claude \
  --incoming-dir ./my_docs \
  --output-file claude_training.json
```

### MCP Server Mode

Start the MCP server for integration with MCP-compatible clients:

```bash
# Make executable and start
chmod +x start_mcp_server.sh
./start_mcp_server.sh
```

The MCP server provides these tools:

- `list_documents` - List files in incoming directory
- `configure_llm` - Set up LLM provider
- `process_document` - Extract text from single document
- `generate_qa_pairs` - Create Q&A pairs from document
- `generate_training_data` - Process all documents and create dataset
- `get_provider_info` - Get available providers and models
- `validate_setup` - Check configuration and dependencies

### Python API Usage

#### Enhanced LangChain API (Recommended)

```python
from src.langchain_processing import LangChainProcessor, LLMProvider

# Initialize enhanced processor
processor = LangChainProcessor(
    incoming_dir="./incoming",
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    chunk_size=1200,
    chunk_overlap=300,
    questions_per_chunk=5,
    splitting_strategy="recursive",
    use_batch_processing=True,
    temperature=0.7
)

# Generate training data with enhanced features
training_data = processor.generate_qa_training_data(
    output_file="./enhanced_training_data.json"
)

print(f"Generated {training_data['metadata']['total_qa_pairs']} Q&A pairs")
print(f"Quality metrics: {training_data['metadata'].get('quality_metrics', {})}")
```

#### Original API (Legacy)

```python
from src.data_processing.preprocess import DocumentProcessor
from src.data_processing.qa_generator import QAGenerator, LLMProvider

# Initialize components
processor = DocumentProcessor("./incoming")
qa_generator = QAGenerator(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    temperature=0.7
)

# Process documents
documents = processor.get_documents()
processed_docs = []

for doc_path in documents:
    processed_doc = processor.preprocess_document(doc_path)
    if processed_doc:
        processed_docs.append(processed_doc)

# Generate training data
training_data = qa_generator.generate_training_data(
    processed_docs,
    "./training_data.json"
)

print(f"Generated {len(training_data['training_pairs'])} Q&A pairs")
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your API keys:

```env
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Local LLM Configuration
OLLAMA_HOST=http://localhost:11434

# Processing Parameters
DEFAULT_CHUNK_SIZE=1000
DEFAULT_QUESTIONS_PER_CHUNK=3
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2000
```

### Supported LLM Providers

| Provider           | Models                                         | API Key Required  |
| ------------------ | ---------------------------------------------- | ----------------- |
| **OpenAI**         | gpt-4, gpt-4-turbo, gpt-3.5-turbo              | OPENAI_API_KEY    |
| **Anthropic**      | claude-3-opus, claude-3-sonnet, claude-3-haiku | ANTHROPIC_API_KEY |
| **Google**         | gemini-pro, gemini-1.5-pro                     | GOOGLE_API_KEY    |
| **Local (Ollama)** | llama3, llama2, mistral, codellama             | None (local)      |

### Document Formats

| Format       | Extension   | Features                     |
| ------------ | ----------- | ---------------------------- |
| **PDF**      | .pdf        | Page-based section tracking  |
| **Markdown** | .md         | Header-based sections        |
| **HTML**     | .html, .htm | Element-based structure      |
| **Word**     | .docx       | Paragraph and style tracking |
| **Text**     | .txt        | Simple section detection     |

## 📊 Output Format

The generated training data is saved as JSON with this structure:

```json
{
  "metadata": {
    "generated_by": "finetune-ingest",
    "llm_provider": "openai",
    "model_used": "gpt-4",
    "num_documents": 3,
    "total_qa_pairs": 45
  },
  "documents": [
    {
      "file_info": {
        "source_file": "./incoming/doc1.pdf",
        "file_name": "doc1.pdf",
        "file_type": ".pdf"
      },
      "qa_pairs_count": 15,
      "word_count": 2500
    }
  ],
  "training_pairs": [
    {
      "question": "What is machine learning?",
      "answer": "Machine learning is a subset of artificial intelligence...",
      "source_file": "./incoming/doc1.pdf",
      "file_name": "doc1.pdf",
      "chunk_id": 0,
      "chunk_start": 0,
      "chunk_end": 1000,
      "source_text": "Machine learning is...",
      "generated_by": {
        "provider": "openai",
        "model": "gpt-4"
      }
    }
  ]
}
```

## 🧪 Demo and Testing

Run the comprehensive demo to test all features:

```bash
python example_usage.py
```

This demo will:

- Create example documents
- Test document processing
- Try multiple LLM providers (if configured)
- Demonstrate advanced features
- Show usage examples

## 🔍 Advanced Features

### Enhanced LangChain Features

```bash
# Multiple text splitting strategies
python src/main_langchain.py --splitting-strategy recursive    # Hierarchical (default)
python src/main_langchain.py --splitting-strategy markdown     # Markdown-aware
python src/main_langchain.py --splitting-strategy html         # HTML-aware
python src/main_langchain.py --splitting-strategy character    # Simple character-based

# Custom chunk parameters with overlap
python src/main_langchain.py --chunk-size 1500 --chunk-overlap 400

# Batch processing for efficiency
python src/main_langchain.py --batch-processing --questions-per-chunk 5

# Configuration monitoring
python src/main_langchain.py --show-config
```

### Original Advanced Features

```bash
# Different chunk sizes
python src/main.py --chunk-size 500    # Smaller chunks
python src/main.py --chunk-size 1500   # Larger chunks

# More questions per chunk
python src/main.py --questions-per-chunk 5
```

### Temperature and Creativity Control

```bash
# More conservative (factual) - works with both implementations
python src/main_langchain.py --temperature 0.3
python src/main.py --temperature 0.3

# More creative - works with both implementations
python src/main_langchain.py --temperature 1.0
python src/main.py --temperature 1.0
```

### Local LLM Setup (Ollama)

1. Install Ollama: <https://ollama.ai>
2. Pull a model: `ollama pull llama3`
3. Start Ollama: `ollama serve`
4. Use with either implementation:
   - Enhanced: `python src/main_langchain.py --provider local --model llama3`
   - Original: `python src/main.py --provider local --model llama3`

## 📋 API Reference

### Enhanced LangChain API

#### LangChainProcessor
- `generate_qa_training_data(processed_documents, output_file)` - Generate complete dataset
- `process_single_document(file_path)` - Process single document with enhanced features
- `process_all_documents()` - Process all documents in incoming directory
- `update_configuration(**kwargs)` - Update processor settings dynamically
- `get_processing_info()` - Get comprehensive configuration information

#### LangChainDocumentLoader
- `load_document(file_path)` - Load document with specialized LangChain loaders
- `load_documents_from_directory()` - Load all documents from directory
- `extract_enhanced_metadata(documents)` - Extract comprehensive document metadata

#### EnhancedTextSplitter
- `split_documents(documents, strategy)` - Split with multiple strategies
- `split_text_adaptive(text, file_type)` - Adaptive splitting based on file type
- `get_chunk_statistics(chunks)` - Get detailed chunk statistics

#### UnifiedLLMProvider
- `generate_response(prompt, system_message)` - Generate single response
- `generate_streaming_response(prompt, system_message)` - Generate streaming response
- `batch_generate(prompts, system_message)` - Generate multiple responses efficiently

### Original API (Legacy)

#### DocumentProcessor
- `get_documents()` - Find all supported documents
- `preprocess_document(file_path)` - Extract and process single document
- `chunk_text(text, chunk_size, overlap)` - Split text into chunks

#### QAGenerator
- `generate_qa_pairs(chunk, metadata, num_questions)` - Generate Q&A from chunk
- `generate_training_data(processed_docs, output_file)` - Create complete dataset
- `get_available_providers()` - List supported LLM providers
- `get_provider_models(provider)` - List models for provider

## 🛠 Troubleshooting

### Common Issues

#### "No API key configured"

- Ensure your `.env` file contains the correct API key
- Check the key is valid and has sufficient credits

#### "No documents found"

- Verify files are in the `./incoming` directory
- Check file formats are supported (PDF, MD, HTML, DOCX, TXT)

#### "Failed to initialize provider"

- Install missing dependencies: `pip install -r requirements.txt`
- For local provider, ensure Ollama is running

#### "ImportError: No module named 'mcp'"

- The MCP package may not be available in your pip index
- MCP server functionality is optional - the main application works without it
- Try installing from alternative sources: `pip install -r requirements-mcp.txt`
- Use `python src/main_langchain.py` or `python src/main.py` for core functionality

### Dependency Issues

```bash
# Reinstall all dependencies (includes LangChain packages)
pip install --force-reinstall -r requirements.txt

# Install specific providers
pip install openai anthropic google-generativeai ollama

# Install LangChain packages
pip install langchain langchain-community langchain-openai langchain-anthropic langchain-google-genai langchain-ollama unstructured

# Install document processing
pip install pdfplumber python-docx beautifulsoup4 markdown

# Optional: MCP server functionality (if available)
pip install -r requirements-mcp.txt
```

### LangChain-Specific Issues

#### "ModuleNotFoundError: No module named 'langchain_*'"
- Install missing LangChain package: `pip install langchain-openai langchain-anthropic langchain-google-genai langchain-ollama`
- **Easy Solution**: Use `python src/main_simple.py` which automatically falls back to legacy implementation

#### "1 validation error for LLMChain"
- This indicates an issue with LangChain chain configuration
- **Easy Solution**: Use `python src/main_simple.py` for automatic fallback

#### Enhanced vs Legacy Implementation
- **Recommended**: Use `python src/main_simple.py` for automatic implementation selection
- Use `python src/main_langchain.py --show-config` to verify LangChain setup
- Fall back to `python src/main.py` if LangChain issues persist

#### Python Environment Issues
- If you see Python version mismatches, try:
  ```bash
  # Check which Python you're using
  which python && python --version
  
  # Install packages for the correct Python version
  python -m pip install -r requirements.txt
  ```

## 🤝 Integration

### MCP Client Integration

The app can be used as an MCP server with any MCP-compatible client:

```json
{
  "command": [
    "python",
    "/path/to/document-analysis-app/src/mcp_server/server.py"
  ],
  "name": "finetune-ingest",
  "description": "Generate training data for LLM fine-tuning"
}
```

### Programmatic Usage

#### Enhanced LangChain Integration
```python
# Custom document processing pipeline with LangChain
from src.langchain_processing import LangChainProcessor, LLMProvider

# Initialize with advanced features
processor = LangChainProcessor(
    incoming_dir="custom_dir",
    provider=LLMProvider.CLAUDE,
    splitting_strategy="recursive",
    use_batch_processing=True,
    temperature=0.5
)

# Generate training data with enhanced processing
training_data = processor.generate_qa_training_data("./output.json")
```

#### Legacy Integration
```python
# Custom document processing pipeline (original)
from src.data_processing.preprocess import DocumentProcessor
from src.data_processing.qa_generator import QAGenerator

# Your custom logic here
processor = DocumentProcessor("custom_dir")
qa_gen = QAGenerator(provider="claude", temperature=0.5)

# Process and generate training data
# ... your code
```

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the example usage script
3. Validate your setup:
   - Enhanced: `python src/main_langchain.py --show-config`
   - Legacy: `python src/main.py --list-providers`

## 🎯 Which Implementation Should I Use?

### Use **Enhanced LangChain Implementation** (`main_langchain.py`) for:
- ✅ New projects requiring robust document processing
- ✅ Complex documents with varied formats (Markdown, HTML with headers)
- ✅ Large document sets requiring batch processing
- ✅ Projects needing advanced text splitting strategies
- ✅ Applications requiring detailed configuration monitoring
- ✅ Production environments needing robust error handling

### Use **Original Implementation** (`main.py`) for:
- ✅ Simple, straightforward document processing
- ✅ Existing projects that work well with current setup
- ✅ Environments where LangChain dependencies are problematic
- ✅ Lightweight processing with minimal overhead

---

**Ready to generate high-quality training data for your LLM fine-tuning projects with both original and enhanced LangChain capabilities!** 🚀
