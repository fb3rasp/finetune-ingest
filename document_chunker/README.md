# Document Chunker

This module handles document chunking - processing documents and breaking them down into manageable chunks for Q&A generation.

## Usage

```bash
python src/chunk_documents.py \
  --incoming-dir /data/incoming \
  --chunks-dir /data/chunks \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --splitting-strategy recursive \
  --resume
```

## Features

- **Multiple document formats**: PDF, Markdown, HTML, DOCX, TXT
- **Configurable chunking**: Adjustable chunk size and overlap
- **Multiple strategies**: Recursive, character, markdown, html splitting
- **Resume capability**: Skip already processed documents
- **Metadata preservation**: Full document metadata tracking
- **Error handling**: Robust fallback mechanisms

## Output

Creates chunk files in the format `{document_name}_chunks.json` containing:

```json
{
  "status": "completed",
  "metadata": { "file_name": "...", "source_file": "..." },
  "processing_info": { "chunk_size": 1000, "total_chunks": 15 },
  "chunks": [{ "text": "...", "chunk_id": "0_0" }],
  "qa_generation_status": {
    "completed_chunks": [],
    "total_chunks": 15,
    "is_complete": false
  }
}
```

## Configuration

Uses environment variables from project root `config.env`:

- `GENERATOR_INCOMING_DIR`: Source documents directory
- `GENERATOR_PROCESS_DIR`: Output chunks directory
- `GENERATOR_CHUNK_SIZE`: Text chunk size
- `GENERATOR_CHUNK_OVERLAP`: Overlap between chunks
- `GENERATOR_SPLITTING_STRATEGY`: Text splitting strategy
- `GENERATOR_RESUME`: Resume processing flag
