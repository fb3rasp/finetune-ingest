"""
Configuration management for the training data pipeline.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

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
    qa_combine_dir: str = "_data/05-a-combine"
    qa_combine_filename: str = "training_data.json"
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
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Create configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f) or {}
        
        # Create config with YAML values, using defaults for missing values
        return cls(
            input_dir=yaml_data.get("input_dir", "data/documents"),
            chunks_dir=yaml_data.get("chunks_dir", "_data/chunk"),
            qa_dir=yaml_data.get("qa_dir", "_data/generate-qa"),
            validate_qa_dir=yaml_data.get("validate_qa_dir", "_data/validate-qa"),
            filter_qa_dir=yaml_data.get("filter_qa_dir", "_data/filter-qa"),
            combine_dir=yaml_data.get("combine_dir", "_data/combine"),
            qa_combine_dir=yaml_data.get("qa_combine_dir", "_data/05-a-combine"),
            qa_combine_filename=yaml_data.get("qa_combine_filename", "training_data.json"),
            qa_train_dir=yaml_data.get("qa_train_dir", "_data/qa-train"),
            training_model_dir=yaml_data.get("training_model_dir", "_data/finetune-model"),
            export_dir=yaml_data.get("export_dir", "_data/export"),
            
            # File paths
            training_data_file=yaml_data.get("training_data_file", "_data/combine/training_data.json"),
            validation_report_file=yaml_data.get("validation_report_file", "_data/validate-qa/validation_report.json"),
            filtered_training_data_file=yaml_data.get("filtered_training_data_file", "_data/filter-qa/training_data_filtered.json"),
            final_training_data_file=yaml_data.get("final_training_data_file", "_data/qa-train/training_data_final.jsonl"),
            
            chunk_size=yaml_data.get("chunk_size", 1000),
            chunk_overlap=yaml_data.get("chunk_overlap", 200),
            splitting_strategy=yaml_data.get("splitting_strategy", "recursive"),
            
            llm_provider=yaml_data.get("llm_provider", "local"),
            llm_model=yaml_data.get("llm_model", "llama3"),
            questions_per_chunk=yaml_data.get("questions_per_chunk", 3),
            temperature=yaml_data.get("temperature", 0.7),
            max_tokens=yaml_data.get("max_tokens", 2000),
            reasoning=yaml_data.get("reasoning", False),
            
            validator_provider=yaml_data.get("validator_provider", "openai"),
            validator_model=yaml_data.get("validator_model"),
            validation_threshold=yaml_data.get("validation_threshold", 8.0),
            filter_threshold=yaml_data.get("filter_threshold", 7.0),
            validator_reasoning=yaml_data.get("validator_reasoning", False),
            
            training_template=yaml_data.get("training_template", "alpaca"),
            export_model_name=yaml_data.get("export_model_name"),
            export_system_prompt=yaml_data.get("export_system_prompt", "You are a helpful assistant specialized in AWS AppStream 2.0 and Well-Architected Framework principles. You provide accurate, clear information based on AWS documentation and best practices."),
            verbose=yaml_data.get("verbose", False),
            batch_size=yaml_data.get("batch_size", 10),
        )
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables with YAML override support."""
        project_root = Path(__file__).parent.parent
        
        # Step 1: Check for YAML configuration files (highest priority)
        yaml_files = [
            project_root / "config.yaml",
            project_root / "config.yml",
            project_root / "pipeline.yaml",
            project_root / "pipeline.yml"
        ]
        
        for yaml_file in yaml_files:
            if yaml_file.exists() and YAML_AVAILABLE:
                # Load YAML config first, then merge with environment variables
                config = cls.from_yaml(str(yaml_file))
                
                # Override YAML values with environment variables if they exist
                # This allows env vars to override YAML on a per-field basis
                config_file = project_root / "config.env"
                if config_file.exists():
                    load_dotenv(config_file)
                
                # Override specific fields with env vars if present
                if os.getenv("EXPORT_MODEL_NAME"):
                    config.export_model_name = os.getenv("EXPORT_MODEL_NAME")
                if os.getenv("EXPORT_SYSTEM_PROMPT"):
                    config.export_system_prompt = os.getenv("EXPORT_SYSTEM_PROMPT")
                
                return config
        
        # Step 2: Fall back to environment-only configuration
        # Load environment variables from config.env file
        config_file = project_root / "config.env"

        if config_file.exists():
            load_dotenv(config_file)
        
        return cls(
            input_dir=os.getenv("PIPELINE_INPUT_DIR", "data/documents"),
            chunks_dir=os.getenv("PIPELINE_CHUNKS_DIR", "_data/chunk"),
            qa_dir=os.getenv("PIPELINE_QA_DIR", "_data/generate-qa"),
            validate_qa_dir=os.getenv("PIPELINE_VALIDATION_QA_DIR", "_data/validate-qa"),
            filter_qa_dir=os.getenv("PIPELINE_FILTER_QA_DIR", "_data/filter-qa"),
            combine_dir=os.getenv("PIPELINE_COMBINE_DIR", "_data/combine"),
            qa_combine_dir=os.getenv("PIPELINE_QA_COMBINE_DIR", "_data/05-a-combine"),
            qa_combine_filename=os.getenv("PIPELINE_QA_COMBINE_FILENAME", "training_data.json"),
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
