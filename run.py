#!/usr/bin/env python3
"""
Unified Training Data Pipeline
A simplified interface for the entire training data generation workflow.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import typer
from dotenv import load_dotenv

# Import pipeline modules
from pipeline.steps.chunk_step import ChunkStep
from pipeline.steps.generate_qa_step import GenerateQAStep
from pipeline.steps.validate_step import ValidateStep
from pipeline.steps.format_step import FormatStep
from pipeline.config import PipelineConfig

app = typer.Typer(
    name="run",
    help="Unified Training Data Pipeline - Generate high-quality training data from documents",
    add_completion=False
)

# Load environment variables
load_dotenv(PROJECT_ROOT / "config.env")

@app.command("chunk")
def chunk_documents(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Directory containing source documents. Overrides .env setting."),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save document chunks. Overrides .env setting."),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Size of each text chunk. Overrides .env setting."),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Overlap between chunks. Overrides .env setting."),
    resume: bool = typer.Option(None, "--resume", help="Resume from existing progress. Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 1: Chunk documents into smaller pieces for processing.
    
    Takes documents from the input directory and splits them into manageable chunks,
    saving the results to the output directory for Q&A generation.
    """
    # Load config from .env file first
    config = PipelineConfig.from_env()
    
    # Override with command-line arguments if they are provided
    if input_dir is not None: config.input_dir = input_dir
    if output_dir is not None: config.chunks_dir = output_dir
    if chunk_size is not None: config.chunk_size = chunk_size
    if chunk_overlap is not None: config.chunk_overlap = chunk_overlap
    if verbose is not None: config.verbose = verbose
    
    # The 'resume' flag is a special case, it's not in PipelineConfig by default
    # but the step's run method accepts it. We'll handle it directly.
    # If the flag is not used, resume_val will be None, so we default to False.
    resume_val = resume if resume is not None else False

    step = ChunkStep(config)
    success = step.run(resume=resume_val)
    
    if success:
        typer.echo("✅ Document chunking completed successfully!")
    else:
        typer.echo("❌ Document chunking failed!", err=True)
        raise typer.Exit(1)

@app.command("generate-qa")
def generate_qa(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Directory containing document chunks. Overrides .env setting."),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save Q&A pairs. Overrides .env setting."),
    provider: str = typer.Option(None, "--provider", help="LLM provider (openai, claude, gemini, local). Overrides .env setting."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name. Overrides .env setting."),
    questions_per_chunk: int = typer.Option(None, "--questions-per-chunk", help="Number of Q&A pairs per chunk. Overrides .env setting."),
    temperature: float = typer.Option(None, "--temperature", help="LLM temperature. Overrides .env setting."),
    resume: bool = typer.Option(None, "--resume", help="Resume from existing progress. Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 2: Generate question-answer pairs from document chunks.
    
    Processes each chunk to create training data in the form of questions and answers
    based on the content of each chunk.
    """
    # Load config from .env file first
    config = PipelineConfig.from_env()

    # Override with command-line arguments if they are provided
    if input_dir is not None: config.chunks_dir = input_dir
    if output_dir is not None: config.qa_dir = output_dir
    if provider is not None: config.llm_provider = provider
    if model is not None: config.llm_model = model
    if questions_per_chunk is not None: config.questions_per_chunk = questions_per_chunk
    if temperature is not None: config.temperature = temperature
    if verbose is not None: config.verbose = verbose
    
    resume_val = resume if resume is not None else False
    
    step = GenerateQAStep(config)
    success = step.run(resume=resume_val)
    
    if success:
        typer.echo("✅ Q&A generation completed successfully!")
    else:
        typer.echo("❌ Q&A generation failed!", err=True)
        raise typer.Exit(1)

@app.command("validate")
def validate_qa(
    input_file: str = typer.Option(None, "--input", "-i", help="Input training data JSON file. Overrides .env setting."),
    output_file: str = typer.Option(None, "--output", "-o", help="Output validation report. Overrides .env setting."),
    filtered_output: Optional[str] = typer.Option(None, "--filtered-output", help="Output filtered training data. Overrides .env setting."),
    provider: str = typer.Option(None, "--provider", help="LLM provider for validation. Overrides .env setting."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for validation. Overrides .env setting."),
    threshold: float = typer.Option(None, "--threshold", help="Pass/fail threshold. Overrides .env setting."),
    filter_threshold: float = typer.Option(None, "--filter-threshold", help="Filtering threshold. Overrides .env setting."),
    resume: bool = typer.Option(None, "--resume", help="Resume from existing progress. Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 3: Validate Q&A pairs for quality and accuracy.
    
    Uses an LLM to score each Q&A pair for factual accuracy, completeness, and consistency.
    Generates a detailed report and optionally filters out low-quality pairs.
    """
    config = PipelineConfig.from_env()

    if input_file is not None: config.training_data_file = input_file
    if output_file is not None: config.validation_report_file = output_file
    if filtered_output is not None: config.filtered_training_data_file = filtered_output
    if provider is not None: config.validator_provider = provider
    if model is not None: config.validator_model = model
    if threshold is not None: config.validation_threshold = threshold
    if filter_threshold is not None: config.filter_threshold = filter_threshold
    if verbose is not None: config.verbose = verbose
    
    resume_val = resume if resume is not None else False
    
    step = ValidateStep(config)
    success = step.run(resume=resume_val)
    
    if success:
        typer.echo("✅ Q&A validation completed successfully!")
    else:
        typer.echo("❌ Q&A validation failed!", err=True)
        raise typer.Exit(1)

@app.command("format")
def format_training_data(
    input_file: str = typer.Option(None, "--input", "-i", help="Input filtered training data. Overrides .env setting."),
    output_file: str = typer.Option(None, "--output", "-o", help="Output formatted training data. Overrides .env setting."),
    template: str = typer.Option(None, "--template", help="Training format template (alpaca, chatml, etc.). Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 4: Format validated Q&A pairs for model training.
    
    Converts the validated Q&A pairs into the specific format required for fine-tuning,
    such as Alpaca format or other training templates.
    """
    config = PipelineConfig.from_env()

    if input_file is not None: config.filtered_training_data_file = input_file
    if output_file is not None: config.final_training_data_file = output_file
    if template is not None: config.training_template = template
    if verbose is not None: config.verbose = verbose
    
    step = FormatStep(config)
    success = step.run()
    
    if success:
        typer.echo("✅ Training data formatting completed successfully!")
    else:
        typer.echo("❌ Training data formatting failed!", err=True)
        raise typer.Exit(1)

@app.command("pipeline")
def run_full_pipeline(
    input_dir: str = typer.Option(None, "--input-dir", help="Directory containing source documents. Overrides .env setting."),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Size of each text chunk. Overrides .env setting."),
    provider: str = typer.Option(None, "--provider", help="LLM provider. Overrides .env setting."),
    questions_per_chunk: int = typer.Option(None, "--questions-per-chunk", help="Number of Q&A pairs per chunk. Overrides .env setting."),
    validation_threshold: float = typer.Option(None, "--validation-threshold", help="Quality threshold for validation. Overrides .env setting."),
    resume: bool = typer.Option(None, "--resume", help="Resume from existing progress. Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Run the complete pipeline: chunk -> generate-qa -> validate -> format.
    
    Executes all steps in sequence with consistent configuration.
    If any step fails, the pipeline stops and reports the error.
    """
    typer.echo("🚀 Starting complete training data pipeline...")
    
    # Step 1: Chunk documents
    typer.echo("\n📄 Step 1: Chunking documents...")
    chunk_documents(
        input_dir=input_dir,
        chunk_size=chunk_size,
        resume=resume,
        verbose=verbose
    )
    
    # Step 2: Generate Q&A
    typer.echo("\n❓ Step 2: Generating Q&A pairs...")
    generate_qa(
        provider=provider,
        questions_per_chunk=questions_per_chunk,
        resume=resume,
        verbose=verbose
    )
    
    # Step 3: Validate
    typer.echo("\n✅ Step 3: Validating Q&A pairs...")
    validate_qa(
        provider=provider,
        threshold=validation_threshold,
        resume=resume,
        verbose=verbose
    )
    
    # Step 4: Format
    typer.echo("\n📝 Step 4: Formatting training data...")
    format_training_data(verbose=verbose)
    
    typer.echo("\n🎉 Complete pipeline finished successfully!")
    typer.echo("Your training data is ready at: data/document_training_data/training_data_final.jsonl")

if __name__ == "__main__":
    app()
