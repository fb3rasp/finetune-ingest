![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Simplified Training Data Pipeline

This document describes the new, unified approach to generating training data from documents.

## Overview

This pipeline provides a complete end-to-end solution for creating fine-tuned language models from documents. The unified CLI tool (`run.py`) orchestrates a 7-step process that transforms raw documents into deployable Ollama models:

1. **Document Processing** → Split documents into manageable chunks
2. **Q&A Generation** → Create question-answer pairs from content
3. **Quality Validation** → Assess and score Q&A pair quality
4. **Format Conversion** → Transform data for training formats
5. **Training Preparation** → Convert Q&A pairs to training prompts
6. **Model Fine-tuning** → Fine-tune base models with your data
7. **Model Export** → Export trained models to Ollama format

Each step can be run independently for reliability and resumability, or as part of complete pipeline workflows.

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

3. **Combine Q&A Files** (if needed)

   ```bash
   python run.py combine --input-dir data/qa_results --output data/results/training_data.json
   ```

4. **Validate Quality**

   ```bash
   python run.py validate --verbose --resume
   ```

5. **Format for Training**

   ```bash
   python run.py format --template alpaca --threshold 8.0
   ```

### Run Complete Pipeline (Steps 1-4)

```bash
python run.py pipeline --provider openai --questions-per-chunk 3 --format-threshold 8.0
```

### Run Full Training Pipeline (Steps 1-7)

```bash
# Run the complete data preparation pipeline (steps 1-4)
python run.py pipeline --provider openai --questions-per-chunk 3 --format-threshold 8.0

# Convert Q&A pairs to training prompts (step 5)
python run.py qa-train --verbose

# Fine-tune the model (step 6)
python run.py finetune

# Export to Ollama (step 7)
python run.py export
```

## Pipeline Steps

This pipeline consists of 7 main steps that can be run individually or as a complete sequence:

1. **chunk** - Split documents into manageable chunks
2. **generate-qa** - Generate Q&A pairs from chunks
3. **validate** - Validate Q&A quality and accuracy
4. **format** - Format validated pairs for training
5. **qa-train** - Convert Q&A pairs to training prompts
6. **finetune** - Fine-tune model using training data
7. **export** - Export fine-tuned model to Ollama

## Command Reference

### `python run.py chunk`

**Step 1:** Splits documents into manageable chunks for processing.

**Options:**
- `--input-dir, -i`: Directory containing source documents (overrides .env setting)
- `--output-dir, -o`: Directory to save chunks (overrides .env setting)
- `--chunk-size`: Size of each text chunk (overrides .env setting)
- `--chunk-overlap`: Overlap between chunks (overrides .env setting)
- `--resume`: Resume from existing progress (overrides .env setting)
- `--verbose, -v`: Enable verbose output (overrides .env setting)

### `python run.py generate-qa`

**Step 2:** Creates question-answer pairs from document chunks.

**Options:**
- `--input-dir, -i`: Directory containing chunks (overrides .env setting)
- `--output-dir, -o`: Directory to save Q&A pairs (overrides .env setting)
- `--provider`: LLM provider (`openai`, `claude`, `gemini`, `local`) (overrides .env setting)
- `--model`: Specific model name (overrides .env setting)
- `--questions-per-chunk`: Number of Q&A pairs per chunk (overrides .env setting)
- `--temperature`: LLM temperature (overrides .env setting)
- `--resume`: Resume from existing progress (overrides .env setting)
- `--verbose, -v`: Enable verbose output (overrides .env setting)

### `python run.py validate`

**Step 3:** Validates Q&A pairs for quality and accuracy. **No filtering is applied during validation** - all Q&A pairs are validated and scored. A detailed score distribution summary is provided at the end.

**Options:**
- `--input, -i`: Input training data JSON file (overrides .env setting)
- `--output, -o`: Output validation report (overrides .env setting)
- `--filtered-output`: Output filtered training data (overrides .env setting)
- `--provider`: LLM provider for validation (overrides .env setting)
- `--model`: Model name for validation (overrides .env setting)
- `--threshold`: Pass/fail threshold (overrides .env setting)
- `--filter-threshold`: Filtering threshold (overrides .env setting)
- `--resume`: Resume from existing progress (overrides .env setting)
- `--verbose, -v`: Enable verbose output (overrides .env setting)

### `python run.py format`

**Step 4:** Formats validated Q&A pairs for model training. **Threshold filtering is applied during formatting** - you can specify a quality threshold to filter out low-quality Q&A pairs.

**Options:**
- `--input, -i`: Input validation report (overrides .env setting)
- `--output, -o`: Output formatted training data (overrides .env setting)
- `--template`: Training format template (`alpaca`, `chatml`, etc.) (overrides .env setting)
- `--threshold`: Quality threshold for filtering (0.0 = no filtering) (overrides .env setting)
- `--verbose, -v`: Enable verbose output (overrides .env setting)

### `python run.py qa-train`

**Step 5:** Converts validated Q&A pairs to model-specific training prompts.

**Options:**
- `--verbose, -v`: Enable verbose output

**Environment Variables Required:**
- Uses configuration from `.env` file for input/output paths and model type

### `python run.py finetune`

**Step 6:** Fine-tunes base model using training prompts.

**Environment Variables Required:**
- `FINETUNE_MODEL_NAME`: Name of the base model to fine-tune
- `FINETUNE_MODEL_TYPE`: Type of model (optional)
- `FINETUNE_OUTPUT_DIR`: Directory to save fine-tuned model (optional)

### `python run.py export`

**Step 7:** Exports fine-tuned LoRA model to Ollama format.

**Environment Variables Required:**
- `EXPORT_MODEL_PATH`: Path to the fine-tuned model
- `EXPORT_MODEL_NAME`: Name for the exported Ollama model
- `EXPORT_OUTPUT_DIR`: Output directory (default: `./merged_models`)

### `python run.py combine`

**Utility:** Combines individual Q&A files into a single training data file.

**Options:**
- `--input-dir, -i`: Directory containing Q&A files to combine (default: `_data/qa_results`)
- `--output, -o`: Output combined training data file (default: `_data/results/training_data.json`)
- `--verbose, -v`: Enable verbose output

### `python run.py pipeline`

**Complete Pipeline:** Runs steps 1-4 in sequence (chunk → generate-qa → validate → format).

**Options:** Combines options from individual steps:
- `--input-dir`: Directory containing source documents (overrides .env setting)
- `--chunk-size`: Size of each text chunk (overrides .env setting)
- `--provider`: LLM provider (overrides .env setting)
- `--questions-per-chunk`: Number of Q&A pairs per chunk (overrides .env setting)
- `--validation-threshold`: Quality threshold for validation (overrides .env setting)
- `--format-threshold`: Quality threshold for formatting (0.0 = no filtering) (overrides .env setting)
- `--resume`: Resume from existing progress (overrides .env setting)
- `--verbose, -v`: Enable verbose output (overrides .env setting)

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

1. **Complete End-to-End Pipeline**: From raw documents to deployable Ollama models
2. **Single Entry Point**: One `run.py` command for all operations
3. **Independent Steps**: Each step can be run separately for robustness and debugging
4. **Resume Capability**: Failed steps can be resumed without starting over
5. **Consistent Interface**: Same command structure for all operations
6. **Simplified Project**: Fewer directories and scripts to manage
7. **Flexible Filtering**: Validate once, format with different thresholds
8. **Transparent Quality**: Complete score distribution to understand data quality
9. **Model Agnostic**: Support for OpenAI, Claude, Gemini, and local models
10. **Production Ready**: Includes fine-tuning and model export capabilities

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

# Fine-tuning settings (for steps 5-7)
FINETUNE_MODEL_NAME=microsoft/DialoGPT-medium
FINETUNE_MODEL_TYPE=mistral
FINETUNE_OUTPUT_DIR=./fine_tuned_models

# Export settings (for step 7)
EXPORT_MODEL_PATH=./fine_tuned_models/adapter_model
EXPORT_MODEL_NAME=my-custom-model
EXPORT_OUTPUT_DIR=./merged_models
```

## Examples

### Basic Data Preparation Workflow (Steps 1-4)

```bash
# Step by step approach
python run.py chunk --verbose
python run.py generate-qa --provider openai --questions-per-chunk 5
python run.py combine --verbose  # Optional: if you have separate Q&A files
python run.py validate --verbose --resume
python run.py format --template alpaca --threshold 8.0

# Or run data preparation pipeline all at once
python run.py pipeline --provider openai --questions-per-chunk 5 --format-threshold 8.0 --verbose
```

### Complete Training Workflow (Steps 1-7)

```bash
# Data preparation (steps 1-4)
python run.py pipeline --provider openai --questions-per-chunk 5 --format-threshold 8.0 --verbose

# Training preparation and execution (steps 5-7)
python run.py qa-train --verbose
python run.py finetune
python run.py export

# Or run each step individually for more control:
# Step 1: Chunk documents
python run.py chunk --input-dir ./my_docs --chunk-size 1500 --verbose

# Step 2: Generate Q&A pairs
python run.py generate-qa --provider openai --model gpt-4 --questions-per-chunk 3 --verbose

# Step 3: Validate quality
python run.py validate --provider openai --model gpt-4 --verbose --resume

# Step 4: Format for training
python run.py format --template alpaca --threshold 8.0 --verbose

# Step 5: Convert to training prompts
python run.py qa-train --verbose

# Step 6: Fine-tune model
python run.py finetune

# Step 7: Export to Ollama
python run.py export
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
