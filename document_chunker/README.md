# Document Chunker

Intelligent document processing and chunking system for preparing documents for Q&A generation and training data creation.

## 🎯 **Purpose**

The Document Chunker processes various document formats and intelligently breaks them into manageable, semantically meaningful chunks optimized for Q&A generation. It serves as the foundation of the training data pipeline by ensuring documents are properly segmented while preserving context and metadata.

## ✨ **Features**

### **Multi-Format Document Support**

- **PDF Files**: Extract text while preserving structure
- **Markdown**: Respect heading hierarchy and formatting
- **HTML**: Parse web content and documentation
- **DOCX**: Microsoft Word document processing
- **TXT**: Plain text files with encoding detection

### **Intelligent Text Splitting**

- **Recursive Splitting**: Hierarchical chunking that respects natural boundaries
- **Character-Based**: Fixed-size chunking with overlap control
- **Markdown-Aware**: Preserves markdown structure and headers
- **HTML-Aware**: Maintains HTML element boundaries
- **Configurable Overlap**: Prevent context loss at chunk boundaries

### **Robust Processing**

- **Resume Capability**: Skip already processed documents
- **Error Handling**: Graceful fallbacks for problematic documents
- **Metadata Preservation**: Track source files, processing parameters, and statistics
- **Progress Tracking**: Detailed logging and summary reports

### **Output Optimization**

- **Q&A Generation Ready**: Structured output optimized for downstream processing
- **Atomic Operations**: Safe concurrent processing with file locking
- **Comprehensive Metadata**: Full document and chunk provenance tracking

## 🚀 **Quick Start**

### **Basic Document Chunking**

```bash
cd document_chunker

# Process all documents in incoming directory
python src/chunk_documents.py \
  --incoming-dir /data/incoming \
  --chunks-dir /data/chunks \
  --chunk-size 1000 \
  --chunk-overlap 200
```

### **Advanced Configuration**

```bash
# Custom chunking with markdown-aware splitting
python src/chunk_documents.py \
  --incoming-dir /data/incoming \
  --chunks-dir /data/chunks \
  --chunk-size 1500 \
  --chunk-overlap 300 \
  --splitting-strategy markdown \
  --resume
```

### **Resume Processing**

```bash
# Resume interrupted chunking (skip completed documents)
python src/chunk_documents.py \
  --incoming-dir /data/incoming \
  --chunks-dir /data/chunks \
  --resume
```

## 📊 **Chunking Strategies**

### **Recursive (Default)**

- **Best for**: General documents, mixed content
- **Behavior**: Respects paragraph and sentence boundaries
- **Advantages**: Maintains semantic coherence
- **Use case**: Most document types, policy documents, manuals

### **Character-Based**

- **Best for**: Uniform text, data consistency
- **Behavior**: Fixed character-count chunks with overlap
- **Advantages**: Predictable chunk sizes
- **Use case**: Technical documentation, structured data

### **Markdown-Aware**

- **Best for**: Markdown documents, documentation
- **Behavior**: Preserves heading hierarchy and structure
- **Advantages**: Maintains document organization
- **Use case**: README files, documentation sites, wikis

### **HTML-Aware**

- **Best for**: Web content, HTML documentation
- **Behavior**: Respects HTML element boundaries
- **Advantages**: Preserves semantic HTML structure
- **Use case**: Web pages, HTML exports, technical docs

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Directory Settings
GENERATOR_INCOMING_DIR=/data/incoming    # Source documents directory
GENERATOR_PROCESS_DIR=/data/chunks       # Output chunks directory

# Processing Settings
GENERATOR_CHUNK_SIZE=1000                # Characters per chunk
GENERATOR_CHUNK_OVERLAP=200              # Overlap between chunks
GENERATOR_SPLITTING_STRATEGY=recursive   # Chunking strategy
GENERATOR_RESUME=true                    # Skip completed documents
```

### **Command Line Arguments**

```bash
# Directory Configuration
--incoming-dir          Source documents directory
--chunks-dir            Output chunks directory

# Text Processing
--chunk-size            Characters per chunk (default: 1000)
--chunk-overlap         Overlap between chunks (default: 200)
--splitting-strategy    Chunking method: recursive, character, markdown, html

# Processing Options
--resume                Skip already processed documents
```

### **Optimal Chunk Size Guidelines**

| Use Case               | Chunk Size | Overlap | Strategy  |
| ---------------------- | ---------- | ------- | --------- |
| **General Documents**  | 1000-1500  | 200-300 | recursive |
| **Technical Manuals**  | 800-1200   | 150-250 | markdown  |
| **Policy Documents**   | 1200-1800  | 250-350 | recursive |
| **Web Content**        | 1000-1500  | 200-300 | html      |
| **Short Form Content** | 600-1000   | 100-200 | character |

## 📋 **Output Structure**

### **Chunk File Format**

Each document produces a `{document_name}_chunks.json` file:

```json
{
  "status": "completed",
  "metadata": {
    "file_name": "NZISM-ISM Document-V.-3.9-April-2025.pdf",
    "source_file": "/data/incoming/NZISM-ISM Document-V.-3.9-April-2025.pdf",
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
  "chunks": [
    {
      "text": "1 Version_3.9__April-2025\n1. About information security...",
      "start_char": 0,
      "end_char": 971,
      "chunk_id": "0_0",
      "chunk_index": 0,
      "document_index": 0,
      "word_count": 165,
      "char_count": 971,
      "metadata": {...}
    }
  ],
  "qa_generation_status": {
    "completed_chunks": [],
    "total_chunks": 1951,
    "last_processed_chunk": -1,
    "is_complete": false
  }
}
```

### **Summary Report**

Processing creates a `chunking_summary.json` file:

```json
{
  "total_documents": 5,
  "successful_chunks": 5,
  "failed_chunks": 0,
  "chunk_files_created": [
    "NZISM-ISM Document-V.-3.9-April-2025_chunks.json",
    "Policy-Manual-2024_chunks.json"
  ],
  "chunks_directory": "/data/chunks",
  "processed_at": "2025-08-23T10:35:15.789012"
}
```

## 🎯 **Use Cases**

### **Single Document Processing**

```bash
# Process one document with custom settings
python src/chunk_documents.py \
  --incoming-dir /data/incoming \
  --chunks-dir /data/chunks \
  --chunk-size 1500 \
  --chunk-overlap 300 \
  --splitting-strategy markdown
```

### **Batch Document Processing**

```bash
# Process multiple documents with resume capability
python src/chunk_documents.py \
  --incoming-dir /data/incoming \
  --chunks-dir /data/chunks \
  --resume
```

### **Custom Chunk Sizes for Different Content**

```bash
# Large chunks for comprehensive content
python src/chunk_documents.py \
  --chunk-size 2000 \
  --chunk-overlap 400 \
  --splitting-strategy recursive

# Small chunks for detailed analysis
python src/chunk_documents.py \
  --chunk-size 800 \
  --chunk-overlap 150 \
  --splitting-strategy character
```

### **Document Type Optimization**

```bash
# Markdown documentation
python src/chunk_documents.py \
  --splitting-strategy markdown \
  --chunk-size 1200 \
  --chunk-overlap 250

# HTML content
python src/chunk_documents.py \
  --splitting-strategy html \
  --chunk-size 1000 \
  --chunk-overlap 200
```

## 📈 **Performance Optimization**

### **Chunk Size Considerations**

- **Larger Chunks (1500+)**: Better context, fewer chunks, longer processing
- **Smaller Chunks (800-)**: More granular, faster processing, potential context loss
- **Optimal Range**: 1000-1500 characters for most use cases

### **Overlap Guidelines**

- **Minimum**: 15-20% of chunk size to prevent context loss
- **Recommended**: 20-25% of chunk size for good context preservation
- **Maximum**: 30-35% of chunk size (diminishing returns beyond this)

### **Memory Management**

- Documents processed one at a time to manage memory usage
- Automatic cleanup of intermediate processing objects
- Efficient text splitting with minimal memory overhead

## 🔍 **Monitoring and Debugging**

### **Progress Tracking**

- Real-time logging of document processing
- Summary statistics for batch operations
- Error reporting with specific failure details

### **Quality Assurance**

- Chunk size validation and statistics
- Character and word count tracking
- Metadata consistency verification

### **Error Handling**

- Graceful fallbacks for unsupported document formats
- Automatic retry with alternative processing methods
- Detailed error logging for troubleshooting

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Unsupported Document Format**:

   - Check file extension and format
   - Verify document is not corrupted
   - Try converting to supported format

2. **Memory Issues with Large Documents**:

   - Reduce chunk size
   - Process documents individually
   - Check available system memory

3. **Permission Errors**:

   - Verify read access to incoming directory
   - Ensure write access to chunks directory
   - Check file ownership and permissions

4. **Encoding Problems**:
   - Documents automatically processed with UTF-8
   - Manual encoding conversion may be needed
   - Check for special characters or symbols

### **Performance Issues**

- **Slow Processing**: Reduce chunk size or switch to character-based splitting
- **Large Output Files**: Increase chunk size to reduce number of chunks
- **Memory Usage**: Process fewer documents concurrently

### **Quality Issues**

- **Context Loss**: Increase chunk overlap
- **Poor Semantic Boundaries**: Switch to recursive or markdown-aware splitting
- **Inconsistent Chunks**: Use character-based splitting for uniformity

## 🔄 **Integration**

### **Pipeline Integration**

```bash
# Complete workflow: chunking → Q&A generation → training
cd document_chunker
python src/chunk_documents.py --incoming-dir /data/incoming --chunks-dir /data/chunks

cd ../training_data_generator
python src/generate_qa.py --chunks-dir /data/chunks --qa-dir /data/qa_results

cd ../finetune-model
python finetune.py --dataset-path "/data/training/combined_dataset.jsonl"
```

### **Custom Processing Workflows**

The chunker can be imported and used programmatically:

```python
from document_chunker.src.chunk_documents import DocumentChunker

chunker = DocumentChunker(
    incoming_dir="/data/incoming",
    chunks_dir="/data/chunks",
    chunk_size=1200,
    chunk_overlap=250,
    splitting_strategy="recursive"
)

summary = chunker.chunk_all_documents(resume=True)
print(f"Processed {summary['successful_chunks']} documents")
```

## 📚 **Dependencies**

**Core Requirements:**

- `langchain`: Advanced text splitting capabilities
- `python-dotenv`: Environment configuration
- `pathlib`: Cross-platform path handling

**Document Processing:**

- PDF processing requires `PyPDF2` or `pdfplumber`
- DOCX processing requires `python-docx`
- HTML processing uses built-in `html.parser`

## 🎉 **Benefits**

- **Semantic Preservation**: Intelligent chunking maintains document meaning
- **Format Flexibility**: Support for all major document formats
- **Processing Efficiency**: Resume capability and batch processing
- **Quality Assurance**: Comprehensive metadata and error handling
- **Pipeline Ready**: Optimized output for downstream Q&A generation
- **Scalable**: Handles single documents to large document collections

## 🔗 **Next Steps**

After chunking documents:

1. **Q&A Generation**: Use `training_data_generator` to create question-answer pairs
2. **Quality Validation**: Run `fact_checker` to validate generated content
3. **Model Training**: Use `finetune-model` to train on the generated data

The chunked documents serve as the foundation for creating high-quality, domain-specific training datasets!
