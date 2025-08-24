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
    
    # Input/Output directories
    input_dir: str = "data/documents"
    chunks_dir: str = "data/documents_chunks"
    qa_dir: str = "data/documents_training_data"
    
    # File paths
    training_data_file: str = "data/documents_training_data/training_data.json"
    validation_report_file: str = "data/documents_training_data/validation_report.json"
    filtered_training_data_file: str = "data/documents_training_data/training_data_filtered.json"
    final_training_data_file: str = "data/documents_training_data/training_data_final.jsonl"
    
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
    
    # General settings
    verbose: bool = False
    batch_size: int = 10
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for dir_path in [self.chunks_dir, self.qa_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls(
            input_dir=os.getenv("PIPELINE_INPUT_DIR", "data/documents"),
            chunks_dir=os.getenv("PIPELINE_CHUNKS_DIR", "data/document_chunks"),
            qa_dir=os.getenv("PIPELINE_QA_DIR", "data/document_training_data"),
            
            chunk_size=int(os.getenv("PIPELINE_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("PIPELINE_CHUNK_OVERLAP", "200")),
            
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
            verbose=os.getenv("PIPELINE_VERBOSE", "false").lower() in ("true", "1", "yes"),
        )
