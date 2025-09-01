#!/usr/bin/env python3
"""
Step 6: Export fine-tuned LoRA model to Ollama format.
"""
import sys
import os
import subprocess
from pathlib import Path
from pipeline.config import PipelineConfig
from .base_step import BaseStep

class ExportStep(BaseStep):
    """Step 6: Export LoRA model to Ollama format."""

    def check_prerequisites(self) -> bool:
        adapter_path = os.getenv("EXPORT_MODEL_PATH", "")
        if not adapter_path or not Path(adapter_path).exists():
            self.log(f"Export model path does not exist: {adapter_path}", "error")
            return False
        if not os.getenv("EXPORT_MODEL_NAME"):
            self.log("EXPORT_MODEL_NAME environment variable not set", "error")
            return False
        return True

    def run(self, **kwargs) -> bool:
        self.log("Starting model export step...")
        if not self.check_prerequisites():
            return False
        script_path = Path(__file__).parent.parent.parent / "pipeline" / "core" / "utils" / "export_to_ollama.py"
        model_path = os.getenv("EXPORT_MODEL_PATH")
        model_name = os.getenv("EXPORT_MODEL_NAME")
        output_dir = os.getenv("EXPORT_OUTPUT_DIR", self.config.export_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--model-path", model_path,
            "--model-name", model_name,
            "--output-dir", output_dir
        ]
        if self.config.verbose:
            cmd.append("--skip-merge=false")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.log(f"Model export failed: {result.stderr}", "error")
            return False
        self.log(f"Model export completed. Ollama model name: {model_name}")
        return True
