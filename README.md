![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Simplified Training Data Pipeline

This document describes the new, unified approach to generating training data from documents.

## Overview

The new pipeline consolidates the entire workflow into a single CLI tool (`run.py`) that orchestrates all steps while keeping them independent for reliability and resumability.

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up Configuration

Copy your API keys and settings to `config.env`:

```bash
cp config.env.example config.env
# Edit config.env with your API keys
```

### Run Individual Steps

1. **Chunk Documents**

   ```bash
   python run.py chunk --input-dir data/documents --output-dir data/document_chunks
   ```

2. **Generate Q&A Pairs**

   ```bash
   python run.py generate-qa --provider openai --questions-per-chunk 3
   ```

3. **Validate Quality**

   ```bash
   python run.py validate --verbose --resume
   ```

4. **Format for Training**

   ```bash
   python run.py format --template alpaca --threshold 8.0
   ```

### Run Complete Pipeline

```bash
python run.py pipeline --provider openai --questions-per-chunk 3 --format-threshold 8.0
```

## Command Reference

### `python run.py chunk`

Splits documents into manageable chunks for processing.

**Options:**

- `--input-dir`: Directory containing source documents (default: `data/documents`)
- `--output-dir`: Directory to save chunks (default: `data/document_chunks`)
- `--chunk-size`: Size of each chunk (default: `1000`)
- `--chunk-overlap`: Overlap between chunks (default: `200`)
- `--resume`: Resume from existing progress
- `--verbose`: Enable detailed output

### `python run.py generate-qa`

Creates question-answer pairs from document chunks.

**Options:**

- `--input-dir`: Directory containing chunks (default: `data/document_chunks`)
- `--output-dir`: Directory to save Q&A pairs (default: `data/document_training_data`)
- `--provider`: LLM provider (`openai`, `claude`, `gemini`, `local`)
- `--model`: Specific model name
- `--questions-per-chunk`: Number of Q&A pairs per chunk (default: `3`)
- `--temperature`: LLM temperature (default: `0.7`)
- `--resume`: Resume from existing progress
- `--verbose`: Enable detailed output

### `python run.py validate`

Validates Q&A pairs for quality and accuracy. **No filtering is applied during validation** - all Q&A pairs are validated and scored. A detailed score distribution summary is provided at the end.

**Options:**

- `--input`: Input training data file (default: `data/document_training_data/training_data.json`)
- `--output`: Validation report file (default: `data/document_training_data/validation_report.json`)
- `--provider`: LLM provider for validation
- `--model`: Model for validation
- `--resume`: Resume from existing progress
- `--verbose`: Enable detailed output

### `python run.py format`

Formats validated Q&A pairs for model training. **Threshold filtering is applied during formatting** - you can specify a quality threshold to filter out low-quality Q&A pairs.

**Options:**

- `--input`: Input validation report file (default: `data/document_training_data/validation_report.json`)
- `--output`: Final training data file (default: `data/document_training_data/training_data_final.jsonl`)
- `--template`: Training format template (default: `alpaca`)
- `--threshold`: Quality threshold for filtering (0.0 = no filtering, default: `0.0`)
- `--verbose`: Enable detailed output

### `python run.py pipeline`

Runs the complete pipeline in sequence.

**Options:** Combines options from all individual steps, plus:

- `--format-threshold`: Quality threshold for formatting (0.0 = no filtering)

## File Structure

```bash
data/
├── documents/                    # Source documents (input)
├── document_chunks/             # Chunked documents (intermediate)
├── document_training_data/      # Generated training data (output)
│   ├── training_data.json       # Raw Q&A pairs
│   ├── validation_report.json   # Quality validation report with score distribution
│   └── training_data_final.jsonl    # Formatted for training (filtered by threshold)
└── ...

pipeline/                        # New unified pipeline code
├── config.py                   # Configuration management
├── steps/                      # Individual pipeline steps
│   ├── chunk_step.py
│   ├── generate_qa_step.py
│   ├── validate_step.py
│   └── format_step.py
└── ...

run.py                          # Main CLI entry point
```

## Benefits

1. **Single Entry Point**: One `run.py` command for everything
2. **Independent Steps**: Each step can be run separately for robustness
3. **Resume Capability**: Failed steps can be resumed without starting over
4. **Consistent Interface**: Same command structure for all operations
5. **Simplified Project**: Fewer directories and scripts to manage
6. **Flexible Filtering**: Validate once, format with different thresholds
7. **Transparent Quality**: Complete score distribution to understand data quality

## Migration from Old Structure

The old directories (`document_chunker/`, `training_data_generator/`, `fact_checker/`) are deprecated but still functional. You can gradually migrate by:

1. Testing the new pipeline with your existing data
2. Comparing results to ensure quality
3. Removing the old directories once confident

## Configuration

You can set default values using environment variables in `config.env`:

```bash
# Pipeline settings
PIPELINE_INPUT_DIR=data/documents
PIPELINE_CHUNKS_DIR=data/document_chunks
PIPELINE_QA_DIR=data/document_training_data

# LLM settings
PIPELINE_LLM_PROVIDER=openai
PIPELINE_LLM_MODEL=gpt-4
PIPELINE_QUESTIONS_PER_CHUNK=3

# Validation settings
PIPELINE_VALIDATION_THRESHOLD=8.0
PIPELINE_FILTER_THRESHOLD=7.0  # Deprecated - use format threshold instead
```

## Examples

### Basic workflow

```bash
# Step by step
python run.py chunk --verbose
python run.py generate-qa --provider openai --questions-per-chunk 5
python run.py validate --verbose --resume
python run.py format --template alpaca --threshold 8.0

# Or all at once
python run.py pipeline --provider openai --questions-per-chunk 5 --format-threshold 8.0 --verbose
```

### Resume interrupted work

```bash
python run.py chunk --resume
python run.py generate-qa --resume --provider openai
```

### Use local models

```bash
python run.py generate-qa --provider local --model llama2:7b
python run.py validate --provider local --model llama2:7b
```

### Flexible threshold filtering

```bash
# Validate all Q&A pairs (no filtering)
python run.py validate --verbose --resume

# Format with different thresholds
python run.py format --threshold 8.0    # Only include Q&A pairs with score >= 8.0
python run.py format --threshold 7.5    # Only include Q&A pairs with score >= 7.5
python run.py format --threshold 0.0    # Include all Q&A pairs (no filtering)
```

## New Workflow: Validate Once, Format Multiple Times

The new approach separates validation from filtering:

1. **Validation Step**: Validates all Q&A pairs and provides a detailed score distribution
2. **Format Step**: Applies threshold filtering during formatting

This allows you to:

- Run validation once and see the complete quality distribution
- Try different thresholds without re-validating
- Make informed decisions about quality thresholds based on the score distribution

### Score Distribution Example

After validation, you'll see output like:

```bash
==================================================
VALIDATION SUMMARY
==================================================
Total Q&A pairs: 5826
PASS: 5712 (98.0%)
NEEDS_REVIEW: 98 (1.7%)
FAIL: 16 (0.3%)

SCORE DISTRIBUTION:
Overall: 0-1: 0 (0.0%)
Overall: 1-2: 0 (0.0%)
Overall: 2-3: 0 (0.0%)
Overall: 3-4: 0 (0.0%)
Overall: 4-5: 0 (0.0%)
Overall: 5-6: 0 (0.0%)
Overall: 6-7: 16 (0.3%)
Overall: 7-8: 98 (1.7%)
Overall: 8-9: 1250 (21.5%)
Overall: 9-10: 4462 (76.6%)
```

This new structure dramatically simplifies your workflow while preserving the ability to run steps independently and resume failed operations.
