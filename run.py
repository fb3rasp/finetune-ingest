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
from pipeline.steps.validate_qa_step import ValidateQAStep
from pipeline.steps.filter_qa_step import FilterQAStep
from pipeline.config import PipelineConfig
from pipeline.steps.qa_training_step import QATrainingStep
from pipeline.steps.export_step import ExportStep
from pipeline.steps.finetune_step import FinetuneStep

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

@app.command("validate-qa")
def validate_qa_data(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Directory containing Q&A JSON files. Overrides .env setting."),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save validation reports. Overrides .env setting."),
    provider: str = typer.Option(None, "--provider", help="LLM provider for validation. Overrides .env setting."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for validation. Overrides .env setting."),
    threshold: float = typer.Option(None, "--threshold", help="Pass/fail threshold. Overrides .env setting."),
    resume: bool = typer.Option(None, "--resume", help="Resume from existing progress. Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 3: Validate Q&A pairs for quality and accuracy.
    
    Uses an LLM to score each Q&A pair for factual accuracy, completeness, and consistency.
    Processes all JSON files in the input directory and generates individual validation reports.
    """
    config = PipelineConfig.from_env()

    if input_dir is not None: config.qa_dir = input_dir
    if output_dir is not None: config.validate_qa_dir = output_dir
    if provider is not None: config.validator_provider = provider
    if model is not None: config.validator_model = model
    if threshold is not None: config.validation_threshold = threshold
    if verbose is not None: config.verbose = verbose
    
    resume_val = resume if resume is not None else False
    
    step = ValidateQAStep(config)
    success = step.run(resume=resume_val)
    
    if success:
        typer.echo("✅ Q&A validation completed successfully!")
    else:
        typer.echo("❌ Q&A validation failed!", err=True)
        raise typer.Exit(1)

@app.command("filter-qa")
def filter_qa_data(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Directory containing validation JSON files. Overrides .env setting."),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save filtered data. Overrides .env setting."),
    threshold: float = typer.Option(None, "--threshold", help="Quality threshold for filtering (0.0 = no filtering). Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 4: Filter validated Q&A pairs based on quality threshold.
    
    Processes all validation JSON files from the input directory and creates
    filtered versions in the output directory with '_filtered.json' suffix.
    """
    config = PipelineConfig.from_env()

    if input_dir is not None: config.validate_qa_dir = input_dir
    if output_dir is not None: config.filter_qa_dir = output_dir
    if verbose is not None: config.verbose = verbose
    
    # Use threshold from command line or default to 0.0 (no filtering)
    threshold_val = threshold if threshold is not None else 0.0
    
    step = FilterQAStep(config)
    success = step.run(threshold=threshold_val)
    
    if success:
        typer.echo("✅ Q&A filtering completed successfully!")
    else:
        typer.echo("❌ Q&A filtering failed!", err=True)
        raise typer.Exit(1)

@app.command("combine")
def combine_qa_files_cmd(
    input_dir: str = typer.Option("_data/generate-qa", "--input-dir", "-i", help="Directory containing Q&A files to combine"),
    output_file: str = typer.Option("_data/combine/training_data.json", "--output", "-o", help="Output combined training data file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Step 4-a: Combine individual Q&A files into a single training data file.
    
    Takes individual Q&A result files from the Q&A generation step and combines
    them into the unified format expected by the validation step.
    """
    from combine_qa_files import combine_qa_files
    try:
        success = combine_qa_files(qa_dir=input_dir, output_file=output_file)
        
        if success:
            typer.echo("✅ Q&A files combined successfully!")
        else:
            typer.echo("❌ No Q&A files found to combine!", err=True)
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Failed to combine Q&A files: {e}", err=True)
        raise typer.Exit(1)


@app.command("qa-train")
def qa_to_training(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Directory containing filtered Q&A files. Overrides .env setting."),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save training prompts. Overrides .env setting."),
    template: str = typer.Option(None, "--template", help="Training template (alpaca, llama, etc.). Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Step 5: Convert filtered Q&A pairs to training prompts.
    
    Processes all filtered JSON files from the input directory and creates
    training prompt files in the output directory with JSONL format.
    """
    config = PipelineConfig.from_env()
    
    if input_dir is not None: config.filter_qa_dir = input_dir
    if output_dir is not None: config.qa_train_dir = output_dir
    if template is not None: config.training_template = template
    if verbose is not None: config.verbose = verbose
    
    step = QATrainingStep(config)
    success = step.run()
    if success:
        typer.echo("✅ QA to training prompts conversion completed successfully!")
    else:
        typer.echo("❌ QA to training prompts conversion failed!", err=True)
        raise typer.Exit(1)

@app.command("finetune")
def finetune_model():
    """
    Step 6: Fine-tune base model using training prompts.
    
    Required Environment Variables:
        FINETUNE_MODEL_NAME: Base model name (e.g., meta-llama/Llama-3.2-1B-Instruct)
    
    Optional Environment Variables:
        FINETUNE_MODEL_TYPE: Model type (llama, gemma, qwen) - auto-detected if not set
        HF_TOKEN: HuggingFace token for accessing gated models
        
        # Training Configuration
        NUM_TRAIN_EPOCHS: Number of training epochs (default: 3)
        OPTIMIZER: Optimizer type (default: paged_adamw_8bit)
        SAVE_STEPS: Save checkpoint every N steps (default: 25)
        LOGGING_STEPS: Log every N steps (default: 5)
        MAX_GRAD_NORM: Max gradient norm for clipping (model-specific default)
        SAVE_TOTAL_LIMIT: Max number of checkpoints to keep (default: 3)
        
        # Precision & Performance
        USE_FP16: Use 16-bit precision (default: false)
        USE_BF16: Use bfloat16 precision (default: true)
        GROUP_BY_LENGTH: Group by sequence length (default: true)
        LR_SCHEDULER_TYPE: Learning rate scheduler (default: cosine)
        GRADIENT_CHECKPOINTING: Use gradient checkpointing (default: true)
        DATALOADER_PIN_MEMORY: Pin memory for dataloader (default: false)
        REMOVE_UNUSED_COLUMNS: Remove unused columns (default: true)
        DISABLE_WANDB: Disable Weights & Biases logging (default: true)
        
        # Testing
        TEST_MODEL_AFTER_TRAINING: Test model after training (default: false)
        TEST_MAX_NEW_TOKENS: Max tokens for test generation (default: 200)
        TEST_TEMPERATURE: Temperature for test generation (default: 0.7)
        TEST_TOP_P: Top-p for test generation (default: 0.9)
        DEVICE_MAP: Device mapping strategy (default: auto)
        TRUST_REMOTE_CODE: Trust remote code in models (default: true)
    
    Examples:
        # Basic fine-tuning with Llama
        export FINETUNE_MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
        python run.py finetune
        
        # Fine-tuning with custom settings
        export FINETUNE_MODEL_NAME="google/gemma-2-2b-it"
        export FINETUNE_MODEL_TYPE="gemma"
        export NUM_TRAIN_EPOCHS="5"
        export TEST_MODEL_AFTER_TRAINING="true"
        python run.py finetune
    """
    config = PipelineConfig.from_env()
    step = FinetuneStep(config)
    success = step.run()
    if success:
        typer.echo("✅ Fine-tuning completed successfully!")
    else:
        typer.echo("❌ Fine-tuning failed!", err=True)
        raise typer.Exit(1)

@app.command("export")
def export_model(
    model_name: str = typer.Option(None, "--model-name", help="Name for the exported Ollama model. Overrides .env setting."),
    system_prompt: str = typer.Option(None, "--system-prompt", help="Custom system prompt for the exported model. Overrides .env setting.")
):
    """
    Step 7: Export fine-tuned LoRA model to Ollama format.
    
    Required Environment Variables:
        EXPORT_MODEL_PATH: Path to the fine-tuned LoRA adapter directory
        EXPORT_MODEL_NAME: Name for the exported Ollama model (can be overridden by --model-name)
    
    Optional Environment Variables:
        EXPORT_OUTPUT_DIR: Output directory for merged model (default: _data/export)
        EXPORT_SYSTEM_PROMPT: Custom system prompt for the model (can be overridden by --system-prompt)
        EXPORT_JOB_ID: Job ID to load system prompt from job config
        EXPORT_SKIP_MERGE: Skip model merging, use existing merged model (default: false)
        
        # Ollama Creation Settings
        OLLAMA_CREATE_TIMEOUT_MINUTES: Timeout for ollama create command (default: 20)
        OLLAMA_SHOW_REALTIME_OUTPUT: Show real-time ollama output (default: false)
        
        # HuggingFace Settings
        HF_TOKEN: HuggingFace token for accessing gated models
    
    Process Overview:
        1. Load LoRA adapter and base model
        2. Merge LoRA weights with base model
        3. Save merged model to disk
        4. Create Ollama Modelfile with appropriate template
        5. Run 'ollama create' to register the model
    
    Model Type Detection:
        - Automatically detects model type (llama, gemma, qwen) from config.json
        - Uses appropriate chat template and stop tokens for each model type
        - Falls back to llama format if detection fails
    
    Examples:
        # Basic export
        export EXPORT_MODEL_PATH="_data/06-finetune-model/final"
        export EXPORT_MODEL_NAME="my-custom-model"
        python run.py export
        
        # Export with custom system prompt
        export EXPORT_MODEL_PATH="_data/06-finetune-model/final"
        export EXPORT_MODEL_NAME="policy-bot"
        export EXPORT_SYSTEM_PROMPT="You are a helpful policy assistant."
        python run.py export
        
        # Export with debugging (see ollama create progress)
        export EXPORT_MODEL_PATH="_data/06-finetune-model/final"
        export EXPORT_MODEL_NAME="debug-model"
        export OLLAMA_SHOW_REALTIME_OUTPUT="true"
        export OLLAMA_CREATE_TIMEOUT_MINUTES="30"
        python run.py export
        
        # Skip merge step (if model already merged)
        export EXPORT_MODEL_PATH="_data/06-finetune-model/final"
        export EXPORT_MODEL_NAME="existing-model"
        export EXPORT_SKIP_MERGE="true"
        python run.py export
    
    Troubleshooting:
        - If command hangs, set OLLAMA_SHOW_REALTIME_OUTPUT=true
        - For large models, increase OLLAMA_CREATE_TIMEOUT_MINUTES
        - Ensure Ollama server is running and accessible
        - Check that EXPORT_MODEL_PATH contains valid LoRA adapter files
    """
    config = PipelineConfig.from_env()
    step = ExportStep(config)
    success = step.run(model_name=model_name, system_prompt=system_prompt)
    if success:
        typer.echo("✅ Model export completed successfully!")
    else:
        typer.echo("❌ Model export failed!", err=True)
        raise typer.Exit(1)

@app.command("pipeline")
def run_full_pipeline(
    input_dir: str = typer.Option(None, "--input-dir", help="Directory containing source documents. Overrides .env setting."),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Size of each text chunk. Overrides .env setting."),
    provider: str = typer.Option(None, "--provider", help="LLM provider. Overrides .env setting."),
    questions_per_chunk: int = typer.Option(None, "--questions-per-chunk", help="Number of Q&A pairs per chunk. Overrides .env setting."),
    validation_threshold: float = typer.Option(None, "--validation-threshold", help="Quality threshold for validation. Overrides .env setting."),
    format_threshold: float = typer.Option(None, "--format-threshold", help="Quality threshold for formatting (0.0 = no filtering). Overrides .env setting."),
    resume: bool = typer.Option(None, "--resume", help="Resume from existing progress. Overrides .env setting."),
    verbose: bool = typer.Option(None, "--verbose", "-v", help="Enable verbose output. Overrides .env setting.")
):
    """
    Steps 1-4: Run the complete data preparation pipeline (chunk -> generate-qa -> validate-qa -> filter-qa).
    
    Executes all steps in sequence with consistent configuration.
    If any step fails, the pipeline stops and reports the error.
    """
    typer.echo("🚀 Starting complete training data pipeline...")
    
    # Step 1: Chunk documents
    typer.echo("\n📄 Step 1: Chunking documents...")
    chunk_documents(
        input_dir=input_dir,
        output_dir=None,
        chunk_size=chunk_size,
        chunk_overlap=None,
        resume=resume,
        verbose=verbose
    )
    
    # Step 2: Generate Q&A
    typer.echo("\n❓ Step 2: Generating Q&A pairs...")
    generate_qa(
        input_dir=None,
        output_dir=None,
        provider=provider,
        model=None,
        questions_per_chunk=questions_per_chunk,
        temperature=None,
        resume=resume,
        verbose=verbose
    )
    
    # Step 3: Validate Q&A
    typer.echo("\n✅ Step 3: Validating Q&A pairs...")
    validate_qa_data(
        input_dir=None,
        output_dir=None,
        provider=provider,
        model=None,
        threshold=validation_threshold,
        resume=resume,
        verbose=verbose
    )
    
    # Step 4: Filter Q&A
    typer.echo("\n🔍 Step 4: Filtering Q&A data...")
    filter_qa_data(
        input_dir=None,
        output_dir=None,
        threshold=format_threshold,
        verbose=verbose
    )
    
    typer.echo("\n🎉 Complete pipeline finished successfully!")
    typer.echo("Your training data is ready at: data/document_training_data/training_data_final.jsonl")

if __name__ == "__main__":
    app()
