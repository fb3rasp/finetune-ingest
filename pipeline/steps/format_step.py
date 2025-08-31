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

from pipeline.core.utils.helpers import log_message
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class FormatStep(BaseStep):
    """Step 4: Format validated Q&A pairs for model training."""
    
    def check_prerequisites(self) -> bool:
        """Check if validation report exists."""
        validation_file = Path(self.config.validation_report_file)
        if not validation_file.exists():
            self.log(f"Validation report file does not exist: {validation_file}", "error")
            return False
        return True
    
    def format_alpaca(self, qa_pairs: List[Dict], threshold: float = 0.0) -> List[str]:
        """Format Q&A pairs in Alpaca style."""
        formatted_lines = []
        filtered_count = 0
        
        for pair in qa_pairs:
            if "question" not in pair or "answer" not in pair:
                continue
            
            # Skip low-quality pairs if they have validation scores and threshold is set
            if "validation_score" in pair and threshold > 0:
                overall_score = pair["validation_score"].get("overall_score", 0)
                if overall_score < threshold:
                    filtered_count += 1
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
        
        if threshold > 0:
            self.log(f"Filtered out {filtered_count} Q&A pairs below threshold {threshold}")
        
        return formatted_lines
    
    def run(self, threshold: float = 0.0) -> bool:
        """Run formatting: filter validated Q&A and write JSON with selected fields."""
        self.log("Starting training data formatting step...")

        if not self.check_prerequisites():
            return False

        try:
            # Load validation report
            with open(self.config.validation_report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)

            qa_pairs = report.get('validation_results', [])
            if not qa_pairs:
                self.log("No Q&A pairs found in validation report", "error")
                return False

            # Filter by validation_score threshold and collect desired fields
            filtered = []
            for pair in qa_pairs:
                score = (pair.get('validation_score') or {}).get('overall_score', 0)
                if threshold and score < threshold:
                    continue
                filtered.append({
                    'question': pair.get('question'),
                    'answer': pair.get('answer'),
                    'validation_score': pair.get('validation_score'),
                    'source_file': pair.get('source_file')
                })

            # Write out filtered JSON
            output_path = Path(self.config.filtered_training_data_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(filtered, out_f, ensure_ascii=False, indent=2)

            self.log(f"Filtered training data ({len(filtered)} entries) saved to: {output_path}")
            return True

        except Exception as e:
            self.log(f"Formatting failed: {e}", "error")
            return False
