"""
Training data formatting step for the pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.utils.helpers import log_message
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class FormatStep(BaseStep):
    """Step 4: Format validated Q&A pairs for model training."""
    
    def check_prerequisites(self) -> bool:
        """Check if filtered training data exists."""
        filtered_file = Path(self.config.filtered_training_data_file)
        if not filtered_file.exists():
            self.log(f"Filtered training data file does not exist: {filtered_file}", "error")
            return False
        return True
    
    def format_alpaca(self, qa_pairs: List[Dict]) -> List[str]:
        """Format Q&A pairs in Alpaca style."""
        formatted_lines = []
        
        for pair in qa_pairs:
            if "question" not in pair or "answer" not in pair:
                continue
            
            # Skip low-quality pairs if they have validation scores
            if "validation_score" in pair:
                overall_score = pair["validation_score"].get("overall_score", 0)
                if overall_score < self.config.filter_threshold:
                    continue
            
            # Create Alpaca-style prompt
            alpaca_text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{pair['question']}\n\n"
                f"### Response:\n{pair['answer']}"
            )
            
            # Create JSONL entry
            jsonl_entry = {"text": alpaca_text}
            formatted_lines.append(json.dumps(jsonl_entry, ensure_ascii=False))
        
        return formatted_lines
    
    def run(self) -> bool:
        """Run the formatting step."""
        self.log("Starting training data formatting step...")
        
        if not self.check_prerequisites():
            return False
        
        try:
            # Load filtered training data
            with open(self.config.filtered_training_data_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            qa_pairs = training_data.get('training_pairs', [])
            if not qa_pairs:
                self.log("No Q&A pairs found in filtered training data", "error")
                return False
            
            # Format based on template
            if self.config.training_template == "alpaca":
                formatted_lines = self.format_alpaca(qa_pairs)
            else:
                self.log(f"Unsupported training template: {self.config.training_template}", "error")
                return False
            
            if not formatted_lines:
                self.log("No training examples generated after formatting", "error")
                return False
            
            # Save formatted training data
            output_file = Path(self.config.final_training_data_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in formatted_lines:
                    f.write(line + '\n')
            
            self.log("=" * 50)
            self.log(f"Formatting completed: {len(formatted_lines)} training examples")
            self.log(f"Final training data saved to: {output_file}")
            
            return True
            
        except Exception as e:
            self.log(f"Formatting failed: {e}", "error")
            return False
