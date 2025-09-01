#!/usr/bin/env python3
"""
Step 5: Convert validated Q&A pairs to model-specific training prompts.
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from pipeline.core.utils.prompt_adapter import PromptAdapter
from .base_step import BaseStep

class QATrainingStep(BaseStep):
    """Step 5: Convert validated Q&A pairs to training prompts."""

    def check_prerequisites(self) -> bool:
        """Check if filtered Q&A files exist."""
        filter_dir = Path(self.config.filter_qa_dir)
        if not filter_dir.exists():
            self.log(f"Filter directory does not exist: {filter_dir}", "error")
            return False
        
        filtered_files = list(filter_dir.glob("*.json"))
        if not filtered_files:
            self.log(f"No filtered JSON files found in {filter_dir}", "error")
            return False
        
        return True

    def run(self, **kwargs) -> bool:
        """Convert filtered Q&A files to training prompts."""
        self.log("Starting QA to training prompts conversion step...")
        
        if not self.check_prerequisites():
            return False
        
        # Ensure output directory exists
        train_dir = Path(self.config.qa_train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Get filtered files
        filter_dir = Path(self.config.filter_qa_dir)
        filtered_files = list(filter_dir.glob("*.json"))
        
        self.log(f"Found {len(filtered_files)} filtered files to process")
        
        # Initialize prompt adapter
        adapter = PromptAdapter()
        
        total_processed = 0
        total_training_prompts = 0
        all_training_prompts = []
        
        for filtered_file in filtered_files:
            try:
                # Create output file path
                output_file = train_dir / f"{filtered_file.stem}_training.jsonl"
                
                # Log processing
                self.log(f"Processing: {filtered_file.name}")
                
                # Load filtered data
                filtered_data = load_json_if_exists(str(filtered_file))
                if not filtered_data:
                    self.log(f"Could not load filtered data from {filtered_file.name}", "warning")
                    continue
                
                qa_pairs = filtered_data.get('filtered_pairs', [])
                if not qa_pairs:
                    self.log(f"No filtered pairs found in {filtered_file.name}", "warning")
                    continue
                
                # Convert Q&A pairs to training prompts
                training_prompts = []
                for pair in qa_pairs:
                    question = pair.get('question', '')
                    answer = pair.get('answer', '')
                    
                    if question and answer:
                        # Create training prompt using adapter
                        prompt = adapter.create_training_prompt(
                            instruction=question,
                            response=answer,
                            system_prompt="You are a helpful assistant that answers questions accurately and concisely.",
                            model_type=self.config.training_template
                        )
                        
                        training_prompts.append({
                            'text': prompt,
                            'question': question,
                            'answer': answer,
                            'source_file': pair.get('source_file'),
                            'validation_score': pair.get('validation_score')
                        })
                
                # Save individual training file as JSONL
                with open(output_file, 'w', encoding='utf-8') as f:
                    for prompt in training_prompts:
                        f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
                
                # Add to collection for combined file
                all_training_prompts.extend(training_prompts)
                
                total_processed += 1
                total_training_prompts += len(training_prompts)
                self.log(f"Completed {filtered_file.name}: {len(training_prompts)} training prompts")
                self.log(f"Training prompts saved to: {output_file}")
                
            except Exception as e:
                self.log(f"Error processing {filtered_file.name}: {e}", "error")
                continue
        
        # Create combined final training file
        if all_training_prompts:
            final_output = Path(self.config.final_training_data_file)
            final_output.parent.mkdir(parents=True, exist_ok=True)
            
            with open(final_output, 'w', encoding='utf-8') as f:
                for prompt in all_training_prompts:
                    f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
            
            self.log(f"Combined training data saved to: {final_output}")
        
        # Summary
        self.log("=" * 50)
        self.log("Training Conversion Summary:")
        self.log(f"Files processed: {total_processed}")
        self.log(f"Total training prompts: {total_training_prompts}")
        self.log(f"Template used: {self.config.training_template}")
        
        return total_processed > 0
