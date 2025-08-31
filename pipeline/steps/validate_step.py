"""
Q&A validation step for the training data pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.llm import QAValidator
from pipeline.core.utils.helpers import log_message
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class ValidateStep(BaseStep):
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
        """Check if training data file exists."""
        training_file = Path(self.config.training_data_file)
        if not training_file.exists():
            self.log(f"Training data file does not exist: {training_file}", "error")
            return False
        return True
    
    def run(self, resume: bool = False) -> bool:
        """Run the Q&A validation step."""
        self.log("Starting Q&A validation step...")
        
        if not self.check_prerequisites():
            return False

        # Log the exact files being used for validation
        self.log(f"Reading training data from: {self.config.training_data_file}")
        
        try:
            # Run validation
            report = self.validator.validate_training_data(
                training_data_path=self.config.training_data_file,
                output_path=self.config.validation_report_file,
                resume=resume
            )
            
            # Summary
            if report and 'summary_statistics' in report:
                stats = report['summary_statistics']
                self.log("=" * 50)
                self.log("Validation Summary:")
                self.log(f"Total Q&A pairs: {stats.get('total_qa_pairs', 0)}")
                self.log(f"Passed validation: {stats.get('pass_count', 0)}")
                self.log(f"Need review: {stats.get('needs_review_count', 0)}")
                self.log(f"Failed: {stats.get('fail_count', 0)}")
                self.log(f"Pass rate: {stats.get('pass_rate', 0):.1%}")
                
                # Validation does not perform filtering; just log the report path
                self.log(f"Validation report saved to: {self.config.validation_report_file}")
            
            return True
            
        except Exception as e:
            self.log(f"Validation failed: {e}", "error")
            return False
