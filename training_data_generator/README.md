# Training Data Generator

This module handles Q&A generation and combining - processing document chunks to create training data for LLM fine-tuning.

## Components

### 1. Q&A Generation (`generate_qa.py`)

Generates question-answer pairs from document chunks with full resumability.

```bash
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

### 2. Q&A Combiner (`combine_qa_results.py`)

Combines individual Q&A files into a single training data file.

```bash
python src/combine_qa_results.py \
  --qa-dir /data/qa_results \
  --output-file /data/results/training_data.json
```

## Features

### Q&A Generation

- **Full resumability**: Interrupt and resume at any chunk
- **Multiple LLM providers**: OpenAI, Claude, Gemini, Local/Ollama
- **Configurable generation**: Questions per chunk, temperature, etc.
- **Progress tracking**: Chunk-level completion status
- **Error recovery**: Failed chunks don't affect others
- **Atomic saves**: Immediate progress persistence

### Q&A Combining

- **Validation**: Only includes completed Q&A files
- **Metadata preservation**: Full traceability maintained
- **Statistics**: Comprehensive summary generation
- **Format compatibility**: Standard training data output

## Output Formats

### Q&A Files (`{document_name}_qa.json`)

```json
{
  "metadata": { "file_name": "...", "source_file": "..." },
  "qa_generation_info": { "provider": "local", "model": "qwen3:14b" },
  "training_pairs": [{ "question": "...", "answer": "..." }],
  "status": "completed|in_progress|interrupted|failed",
  "completed_chunks": [0, 1, 2, 5],
  "summary": { "total_qa_pairs": 45 }
}
```

### Combined Training Data (`training_data.json`)

```json
{
  "metadata": {
    "generated_by": "qa_combiner",
    "total_qa_pairs": 150,
    "num_documents": 3,
    "processing_config": { "llm_provider": "local", "model_used": "qwen3:14b" }
  },
  "documents": [{ "file_info": {}, "summary": {} }],
  "training_pairs": [
    { "question": "...", "answer": "...", "source_file": "..." }
  ]
}
```

## Configuration

Uses environment variables from project root `config.env`:

- `GENERATOR_PROCESS_DIR`: Input chunks directory
- `GENERATOR_OUTPUT_DIR`: Output Q&A files directory
- `GENERATOR_OUTPUT_FILE`: Final combined training data file
- `GENERATOR_PROVIDER`: LLM provider (openai, claude, gemini, local)
- `GENERATOR_MODEL`: Specific model name
- `GENERATOR_QUESTIONS_PER_CHUNK`: Questions to generate per chunk
- `GENERATOR_TEMPERATURE`: LLM temperature setting
- `GENERATOR_RESUME`: Resume capability flag

## Resumability

The Q&A generation process is designed to be fully resumable:

1. **Chunk-level tracking**: Each chunk completion is saved immediately
2. **Status persistence**: Current state always saved to disk
3. **Interrupt handling**: Graceful handling of Ctrl+C interruptions
4. **Automatic resume**: `--resume` flag continues from exact stopping point
5. **Progress visibility**: Clear indication of completion status

This makes it safe to interrupt long-running Q&A generation processes and resume them later without losing progress.
