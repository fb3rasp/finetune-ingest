"""
Configuration management for the training data pipeline.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Central configuration for all pipeline steps."""
    
    # Input/Output directories - step-based folder structure
    input_dir: str = "data/documents"
    chunks_dir: str = "_data/chunk"
    qa_dir: str = "_data/generate-qa"
    validate_qa_dir: str = "_data/validate-qa"
    filter_qa_dir: str = "_data/filter-qa"
    combine_dir: str = "_data/combine"
    qa_train_dir: str = "_data/qa-train"
    training_model_dir: str = "_data/finetune-model"
    export_dir: str = "_data/export"
    
    # File paths based on step directories
    training_data_file: str = "_data/combine/training_data.json"
    validation_report_file: str = "_data/validate-qa/validation_report.json"
    filtered_training_data_file: str = "_data/filter-qa/training_data_filtered.json"
    final_training_data_file: str = "_data/qa-train/training_data_final.jsonl"
    
    # Document chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitting_strategy: str = "recursive"
    
    # Q&A Generation
    llm_provider: str = "local"
    llm_model: Optional[str] = "llama3"
    questions_per_chunk: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000
    reasoning: bool = False
    
    # Validation
    validator_provider: str = "openai"
    validator_model: Optional[str] = None
    validation_threshold: float = 8.0
    filter_threshold: float = 7.0
    validator_reasoning: bool = False
    
    # Training format
    training_template: str = "alpaca"
    
    # Export settings
    export_model_name: Optional[str] = None
    export_system_prompt: Optional[str] = "You are a helpful assistant specialized in AWS AppStream 2.0 and Well-Architected Framework principles. You provide accurate, clear information based on AWS documentation and best practices."
    
    # General settings
    verbose: bool = False
    batch_size: int = 10
    
    def __post_init__(self):
        """Configuration post-initialization."""
        # Note: Directories are created on-demand by individual steps when needed
        pass
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls(
            input_dir=os.getenv("PIPELINE_INPUT_DIR", "data/documents"),
            chunks_dir=os.getenv("PIPELINE_CHUNKS_DIR", "_data/chunk"),
            qa_dir=os.getenv("PIPELINE_QA_DIR", "_data/generate-qa"),
            validate_qa_dir=os.getenv("PIPELINE_VALIDATION_QA_DIR", "_data/validate-qa"),
            filter_qa_dir=os.getenv("PIPELINE_FILTER_QA_DIR", "_data/filter-qa"),
            combine_dir=os.getenv("PIPELINE_COMBINE_DIR", "_data/combine"),
            qa_train_dir=os.getenv("PIPELINE_QA_TRAIN_DIR", "_data/qa-train"),
            training_model_dir=os.getenv("PIPELINE_TRAINING_MODEL_DIR", "_data/finetune-model"),
            export_dir=os.getenv("PIPELINE_EXPORT_DIR", "_data/export"),
            
            # File paths
            training_data_file=os.getenv("PIPELINE_TRAINING_DATA_FILE", "_data/combine/training_data.json"),
            validation_report_file=os.getenv("PIPELINE_VALIDATION_REPORT_FILE", "_data/validate-qa/validation_report.json"),
            filtered_training_data_file=os.getenv("PIPELINE_FILTERED_TRAINING_DATA_FILE", "_data/filter-qa/training_data_filtered.json"),
            final_training_data_file=os.getenv("PIPELINE_FINAL_TRAINING_DATA_FILE", "_data/qa-train/training_data_final.jsonl"),
            
            chunk_size=int(os.getenv("PIPELINE_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("PIPELINE_CHUNK_OVERLAP", "200")),
            splitting_strategy=os.getenv("PIPELINE_SPLITTING_STRATEGY", "recursive"),
            
            llm_provider=os.getenv("PIPELINE_LLM_PROVIDER", "local"),
            llm_model=os.getenv("PIPELINE_LLM_MODEL", "llama3"),
            questions_per_chunk=int(os.getenv("PIPELINE_QUESTIONS_PER_CHUNK", "3")),
            temperature=float(os.getenv("PIPELINE_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("PIPELINE_MAX_TOKENS", "2000")),
            reasoning=os.getenv("PIPELINE_REASONING", "false").lower() in ("true", "1", "yes"),
            
            validator_provider=os.getenv("PIPELINE_VALIDATOR_PROVIDER", "openai"),
            validator_model=os.getenv("PIPELINE_VALIDATOR_MODEL"),
            validation_threshold=float(os.getenv("PIPELINE_VALIDATION_THRESHOLD", "8.0")),
            filter_threshold=float(os.getenv("PIPELINE_FILTER_THRESHOLD", "7.0")),
            validator_reasoning=os.getenv("PIPELINE_VALIDATOR_REASONING", "false").lower() in ("true", "1", "yes"),
            
            training_template=os.getenv("PIPELINE_TRAINING_TEMPLATE", "alpaca"),
            export_model_name=os.getenv("EXPORT_MODEL_NAME"),
            export_system_prompt=os.getenv("EXPORT_SYSTEM_PROMPT", "You are a helpful assistant specialized in AWS AppStream 2.0 and Well-Architected Framework principles. You provide accurate, clear information based on AWS documentation and best practices."),
            verbose=os.getenv("PIPELINE_VERBOSE", "false").lower() in ("true", "1", "yes"),
            batch_size=int(os.getenv("PIPELINE_BATCH_SIZE", "10")),
        )
