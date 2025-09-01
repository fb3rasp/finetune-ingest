"""
Q&A validation step for the training data pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.llm import QAValidator
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class ValidateQAStep(BaseStep):
    """Step 3: Validate Q&A pairs for quality and accuracy."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.validator = QAValidator(
            provider=config.validator_provider,
            model=config.validator_model,
            validation_threshold=config.validation_threshold,
            verbose=config.verbose,
            reasoning=config.validator_reasoning
        )
    
    def check_prerequisites(self) -> bool:
        """Check if Q&A files exist."""
        qa_dir = Path(self.config.qa_dir)
        if not qa_dir.exists():
            self.log(f"Q&A directory does not exist: {qa_dir}", "error")
            return False
        
        qa_files = list(qa_dir.glob("*.json"))
        if not qa_files:
            self.log(f"No JSON files found in {qa_dir}", "error")
            return False
        
        return True
    
    def run(self, resume: bool = False) -> bool:
        """Run the Q&A validation step."""
        self.log("Starting Q&A validation step...")
        
        if not self.check_prerequisites():
            return False

        # Ensure output directory exists
        validate_dir = Path(self.config.validate_qa_dir)
        validate_dir.mkdir(parents=True, exist_ok=True)
        
        # Get Q&A files
        qa_dir = Path(self.config.qa_dir)
        qa_files = list(qa_dir.glob("*.json"))
        
        self.log(f"Found {len(qa_files)} Q&A files to validate")
        
        total_processed = 0
        total_skipped = 0
        
        for qa_file in qa_files:
            try:
                # Create output file path
                validation_file = validate_dir / f"{qa_file.stem}_validation.json"
                
                # Check if already processed
                if resume and validation_file.exists():
                    existing_report = load_json_if_exists(str(validation_file))
                    if existing_report and existing_report.get('status') == 'completed':
                        self.log(f"Skipping {qa_file.name} - already validated")
                        total_skipped += 1
                        continue
                
                # Log processing
                self.log(f"Validating: {qa_file.name}")
                
                # Run validation for this file
                report = self.validator.validate_training_data(
                    training_data_path=str(qa_file),
                    output_path=str(validation_file),
                    resume=resume
                )
                
                if report and 'validation_results' in report:
                    total_processed += 1
                    self.log(f"Completed {qa_file.name}: {len(report.get('validation_results', []))} Q&A pairs validated")
                    self.log(f"Validation report saved to: {validation_file}")
                else:
                    self.log(f"No validation results found for {qa_file.name}", "warning")
                
            except Exception as e:
                self.log(f"Error validating {qa_file.name}: {e}", "error")
                continue
        
        # Summary
        self.log("=" * 50)
        self.log("Validation Summary:")
        self.log(f"Files processed: {total_processed}, skipped: {total_skipped}")
        
        return total_processed > 0
