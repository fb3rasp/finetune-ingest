"""
Q&A filtering step for the pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class FilterQAStep(BaseStep):
    """Step 4: Filter validated Q&A pairs based on quality threshold."""
    
    def check_prerequisites(self) -> bool:
        """Check if validation files exist."""
        validation_dir = Path(self.config.validate_qa_dir)
        if not validation_dir.exists():
            self.log(f"Validation directory does not exist: {validation_dir}", "error")
            return False
        
        validation_files = list(validation_dir.glob("*.json"))
        if not validation_files:
            self.log(f"No validation JSON files found in {validation_dir}", "error")
            return False
        
        return True
    
    
    def run(self, threshold: float = 0.0) -> bool:
        """Run filtering: filter validated Q&A files and write filtered JSON files."""
        self.log("Starting Q&A filtering step...")

        if not self.check_prerequisites():
            return False

        # Ensure output directory exists
        filter_dir = Path(self.config.filter_qa_dir)
        filter_dir.mkdir(parents=True, exist_ok=True)
        
        # Get validation files
        validation_dir = Path(self.config.validate_qa_dir)
        validation_files = list(validation_dir.glob("*.json"))
        
        self.log(f"Found {len(validation_files)} validation files to filter")
        
        total_processed = 0
        total_skipped = 0
        total_filtered_pairs = 0
        
        for validation_file in validation_files:
            try:
                # Create output file path with _filtered suffix
                filtered_file = filter_dir / f"{validation_file.stem}_filtered.json"
                
                # Log processing
                self.log(f"Filtering: {validation_file.name}")
                
                # Load validation data
                validation_data = load_json_if_exists(str(validation_file))
                if not validation_data:
                    self.log(f"Could not load validation data from {validation_file.name}", "warning")
                    continue
                
                qa_pairs = validation_data.get('validation_results', [])
                if not qa_pairs:
                    self.log(f"No Q&A pairs found in {validation_file.name}", "warning")
                    continue

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

                # Create filtered data structure
                filtered_data = {
                    "metadata": {
                        "source_validation_file": str(validation_file),
                        "filter_threshold": threshold,
                        "original_qa_count": len(qa_pairs),
                        "filtered_qa_count": len(filtered),
                        "filtered_at": str(datetime.now())
                    },
                    "filtered_pairs": filtered
                }

                # Write out filtered JSON
                save_json_atomic(filtered_data, str(filtered_file))
                
                total_processed += 1
                total_filtered_pairs += len(filtered)
                self.log(f"Completed {validation_file.name}: {len(filtered)} pairs (from {len(qa_pairs)} original)")
                self.log(f"Filtered data saved to: {filtered_file}")
                
            except Exception as e:
                self.log(f"Error filtering {validation_file.name}: {e}", "error")
                continue
        
        # Summary
        self.log("=" * 50)
        self.log("Filtering Summary:")
        self.log(f"Files processed: {total_processed}")
        self.log(f"Total filtered Q&A pairs: {total_filtered_pairs}")
        self.log(f"Filter threshold: {threshold}")
        
        return total_processed > 0
