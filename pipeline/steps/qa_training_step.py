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
        """Check if combined Q&A file exists."""
        combined_file = Path(self.config.qa_combine_dir) / self.config.qa_combine_filename
        if not combined_file.exists():
            self.log(f"Combined Q&A file does not exist: {combined_file}", "error")
            return False
        
        return True

    def run(self, **kwargs) -> bool:
        """Convert combined Q&A file to training prompts."""
        self.log("Starting QA to training prompts conversion step...")
        
        if not self.check_prerequisites():
            return False
        
        # Ensure output directory exists
        train_dir = Path(self.config.qa_train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Get combined file
        combined_file = Path(self.config.qa_combine_dir) / self.config.qa_combine_filename
        
        self.log(f"Processing combined file: {combined_file}")
        
        # Initialize prompt adapter
        adapter = PromptAdapter()
        
        try:
            # Load combined data
            combined_data = load_json_if_exists(str(combined_file))
            if not combined_data:
                self.log(f"Could not load combined data from {combined_file}", "error")
                return False
            
            # Get training pairs from combined file
            qa_pairs = combined_data.get('training_pairs', [])
            if not qa_pairs:
                self.log(f"No training pairs found in {combined_file}", "error")
                return False
            
            self.log(f"Found {len(qa_pairs)} Q&A pairs to convert")
            
            # Convert Q&A pairs to training prompts
            all_training_prompts = []
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
                    
                    all_training_prompts.append({
                        'text': prompt,
                        'question': question,
                        'answer': answer,
                        'source_file': pair.get('source_file'),
                        'validation_score': pair.get('validation_score')
                    })
            
            total_training_prompts = len(all_training_prompts)
            
            # Create final training file in the configured qa_train_dir
            final_output = Path(self.config.qa_train_dir) / "training_data_final.jsonl"
            final_output.parent.mkdir(parents=True, exist_ok=True)
            
            with open(final_output, 'w', encoding='utf-8') as f:
                for prompt in all_training_prompts:
                    f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
            
            self.log(f"Training data saved to: {final_output}")
            
            # Summary
            self.log("=" * 50)
            self.log("Training Conversion Summary:")
            self.log(f"Combined file processed: {combined_file.name}")
            self.log(f"Total training prompts: {total_training_prompts}")
            self.log(f"Template used: {self.config.training_template}")
            
            return True
        
        except Exception as e:
            self.log(f"Error processing combined file: {e}", "error")
            return False
