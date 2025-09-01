#!/usr/bin/env python3
"""
Step 7: Fine-tune base model using training prompts.
"""
import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from .base_step import BaseStep

class FinetuneStep(BaseStep):
    """Step 7: Fine-tune base model using training prompts."""

    def check_prerequisites(self) -> bool:
        """Check if training files and required environment variables exist."""
        train_dir = Path(self.config.qa_train_dir)
        if not train_dir.exists():
            self.log(f"Training directory does not exist: {train_dir}", "error")
            return False
        
        # Look for JSONL files in the training directory
        training_files = list(train_dir.glob("*.jsonl"))
        if not training_files:
            self.log(f"No JSONL training files found in {train_dir}", "error")
            return False
        
        if not os.getenv("FINETUNE_MODEL_NAME"):
            self.log("FINETUNE_MODEL_NAME environment variable not set", "error")
            return False
        
        return True

    def run(self, **kwargs) -> bool:
        """Run the fine-tuning step with combined training data."""
        self.log("Starting fine-tuning step...")
        
        if not self.check_prerequisites():
            return False
        
        # Ensure output directory exists
        model_dir = Path(self.config.training_model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all JSONL training files
        train_dir = Path(self.config.qa_train_dir)
        training_files = list(train_dir.glob("*.jsonl"))
        
        self.log(f"Found {len(training_files)} training files to combine")
        
        # Combine all training data into a single file
        combined_data = []
        total_prompts = 0
        
        for training_file in training_files:
            try:
                self.log(f"Loading training data from: {training_file.name}")
                
                with open(training_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                # Ensure we have the required 'text' field
                                if 'text' in data:
                                    combined_data.append(data)
                                    total_prompts += 1
                                else:
                                    self.log(f"Warning: Line {line_num} in {training_file.name} missing 'text' field", "warning")
                            except json.JSONDecodeError as e:
                                self.log(f"Warning: Invalid JSON on line {line_num} in {training_file.name}: {e}", "warning")
                
            except Exception as e:
                self.log(f"Error reading {training_file.name}: {e}", "error")
                continue
        
        if not combined_data:
            self.log("No valid training data found", "error")
            return False
        
        # Create combined training file
        combined_file = train_dir / "combined_training_data.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for data in combined_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        self.log(f"Created combined training file with {total_prompts} prompts: {combined_file}")
        
        # Prepare finetune command
        script_path = Path(__file__).parent.parent.parent / "pipeline" / "core" / "utils" / "finetune_unified.py"
        model_name = os.getenv("FINETUNE_MODEL_NAME")
        model_type = os.getenv("FINETUNE_MODEL_TYPE")
        output_dir = self.config.training_model_dir
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-name", model_name,
            "--dataset", str(combined_file.absolute()),
            "--output-dir", output_dir
        ]
        
        if model_type:
            cmd.extend(["--model-type", model_type])
        
        self.log(f"Running fine-tuning command: {' '.join(cmd)}")
        
        # Run finetune script
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.log(f"Fine-tuning failed: {result.stderr}", "error")
            if result.stdout:
                self.log(f"Stdout: {result.stdout}", "info")
            return False
        
        self.log(f"Fine-tuning completed successfully!")
        self.log(f"Model saved to: {output_dir}")
        self.log(f"Training data used: {total_prompts} prompts from {len(training_files)} files")
        
        if result.stdout:
            self.log(f"Fine-tuning output: {result.stdout}")
        
        return True
