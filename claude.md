# Claude Code Configuration

This file contains settings and commands for Claude Code to better assist with this project.

## Project Overview
This is a Training Data Pipeline for ML fine-tuning that processes documents into high-quality Q&A training data.

## Common Commands

### Development
- `python run.py --help` - Show all available commands
- `python run.py pipeline --help` - Show full pipeline options

### Pipeline Steps
- `python run.py chunk --verbose` - Chunk documents
- `python run.py generate-qa --provider openai --questions-per-chunk 3` - Generate Q&A pairs
- `python run.py validate --verbose --resume` - Validate Q&A quality
- `python run.py format --template alpaca --threshold 8.0` - Format for training

### Full Pipeline
- `python run.py pipeline --provider openai --questions-per-chunk 3 --format-threshold 8.0 --verbose` - Run complete pipeline

### Testing
- `python -m pytest` - Run tests (if available)
- `python -c "from pipeline.config import PipelineConfig; print('Config loaded successfully')"` - Test config

### Linting/Formatting
- `black .` - Format code
- `flake8 .` - Lint code
- `mypy .` - Type checking

## Project Structure
- `run.py` - Main CLI entry point
- `pipeline/` - Core pipeline modules
- `pipeline/config.py` - Configuration management
- `pipeline/steps/` - Individual pipeline steps
- `pipeline/core/` - Core utilities (LLM providers, document processing)
- `data/` - Data directories (documents, chunks, training data)

## Environment Setup
- Copy `config.env.example` to `config.env`
- Set API keys for LLM providers
- Install dependencies: `pip install -r requirements.txt`