#!/usr/bin/env python3
"""
Step 7: Export fine-tuned LoRA model to Ollama format.
"""
import sys
import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from .base_step import BaseStep

# Import ML libraries with error handling
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    ML_LIBRARIES_AVAILABLE = True
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class ExportStep(BaseStep):
    """Step 7: Export LoRA model to Ollama format."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        if not ML_LIBRARIES_AVAILABLE:
            raise ImportError(f"Required ML libraries not available: {IMPORT_ERROR}")

    def detect_model_type(self, merged_model_path: str) -> str:
        """
        Detect the model type from the model config or path.
        
        Args:
            merged_model_path: Path to the merged model directory
            
        Returns:
            Model type string (llama, gemma, qwen, or alpaca)
        """
        # Try to detect from config.json
        config_file = os.path.join(merged_model_path, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                model_name = config.get("_name_or_path", "").lower()
                architectures = config.get("architectures", [])
                
                # Check architectures first
                for arch in architectures:
                    arch_lower = arch.lower()
                    if "llama" in arch_lower:
                        return "llama"
                    elif "gemma" in arch_lower:
                        return "gemma"
                    elif "qwen" in arch_lower:
                        return "qwen"
                
                # Check model name
                if "llama" in model_name:
                    return "llama"
                elif "gemma" in model_name:
                    return "gemma"
                elif "qwen" in model_name:
                    return "qwen"
                    
            except Exception as e:
                self.log(f"Warning: Could not read config.json: {e}", "warning")
        
        # Fallback to path-based detection
        path_lower = merged_model_path.lower()
        if "llama" in path_lower:
            return "llama"
        elif "gemma" in path_lower:
            return "gemma"
        elif "qwen" in path_lower:
            return "qwen"
        
        # Default fallback
        self.log("Warning: Could not detect model type, defaulting to 'llama'", "warning")
        return "llama"

    def get_model_specific_ollama_template(self, model_type: str) -> str:
        """
        Get Ollama template that matches the training format for each model type.
        
        Args:
            model_type: Model type (llama, gemma, qwen, alpaca)
            
        Returns:
            Ollama-compatible template string
        """
        templates = {
            "llama": """{{- if .Messages }}
{{- range .Messages }}
{{- if eq .Role "system" }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .Content }}<|eot_id|>{{- else if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>{{- end }}
{{- end }}
{{- else }}<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{- end }}<|start_header_id|>assistant<|end_header_id|>

""",
            
            "gemma": """{{- if .Messages }}
{{- range .Messages }}
{{- if eq .Role "user" }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}<end_of_turn>
{{- else if eq .Role "system" }}{{ .Content }}

{{- end }}
{{- end }}
{{- else }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{- end }}<start_of_turn>model
""",
            
            "qwen": """{{- if .Messages }}
{{- range .Messages }}
{{- if eq .Role "system" }}<|im_start|>system
{{ .Content }}<|im_end|>
{{- else if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{- end }}
{{- end }}
{{- else }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}<|im_start|>assistant
""",
            
            "alpaca": """{{- if .Messages }}
{{- range .Messages }}
{{- if eq .Role "system" }}{{ .Content }}

{{- else if eq .Role "user" }}### Instruction:
{{ .Content }}

{{- else if eq .Role "assistant" }}### Response:
{{ .Content }}

{{- end }}
{{- end }}
{{- else }}### Instruction:
{{ .Prompt }}

{{- end }}### Response:
"""
        }
        
        return templates.get(model_type, templates["llama"])

    def get_model_stop_tokens(self, model_type: str) -> List[str]:
        """
        Get model-specific stop tokens for generation.
        
        Args:
            model_type: Model type (llama, gemma, qwen, alpaca)
            
        Returns:
            List of stop tokens
        """
        stop_tokens_map = {
            "llama": ["<|eot_id|>", "<|end_of_text|>"],
            "gemma": ["<end_of_turn>", "<start_of_turn>"],
            "qwen": ["<|im_end|>", "<|endoftext|>"],
            "alpaca": ["### Instruction:", "### Response:"]
        }
        
        return stop_tokens_map.get(model_type, ["<|eot_id|>", "<|end_of_text|>"])

    def get_default_system_prompt(self) -> str:
        """Get default system prompt for models without specific prompts"""
        return self.config.export_system_prompt

    def merge_lora_model(self, adapter_path: str, output_path: str) -> bool:
        """
        Merge LoRA adapter with base model and save the merged model.
        
        Args:
            adapter_path: Path to the LoRA adapter directory
            output_path: Path to save the merged model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.log(f"Loading adapter from: {adapter_path}")
            
            # Load the adapter config to get base model info
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                self.log(f"Error: adapter_config.json not found in {adapter_path}", "error")
                return False
            
            # Read adapter config to get base model name
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config['base_model_name_or_path']
            
            self.log(f"Base model: {base_model_name}")
            
            # Load base model
            self.log("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.log("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            
            # Load and merge the LoRA adapter
            self.log("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            
            self.log("Merging LoRA weights with base model...")
            merged_model = model.merge_and_unload()
            
            # Save the merged model
            self.log(f"Saving merged model to: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            merged_model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
            
            self.log("✅ Model merging completed successfully!")
            return True
            
        except Exception as e:
            self.log(f"❌ Error during model merging: {str(e)}", "error")
            return False

    def create_modelfile(self, model_name: str, merged_model_path: str, 
                        system_prompt: Optional[str] = None, job_id: Optional[str] = None) -> str:
        """
        Create an Ollama Modelfile for the merged model.
        
        Args:
            model_name: Name for the Ollama model
            merged_model_path: Path to the merged model directory
            system_prompt: Custom system prompt for the model
            job_id: Job ID to load system prompt from (if system_prompt not provided)
            
        Returns:
            Content of the Modelfile
        """
        
        # Convert to absolute path for Ollama
        abs_model_path = os.path.abspath(merged_model_path)
        
        # Determine system prompt
        if system_prompt:
            # Use provided system prompt
            final_system_prompt = system_prompt
        elif job_id:
            # Try to load system prompt from job configuration
            try:
                from job_config_manager import JobConfigManager
                job_manager = JobConfigManager()
                job_config = job_manager.load_job_config(job_id)
                final_system_prompt = job_config.system_prompt
            except Exception as e:
                self.log(f"Warning: Could not load system prompt from job {job_id}: {e}", "warning")
                final_system_prompt = self.get_default_system_prompt()
        else:
            # Try to load from training completion info
            try:
                completion_file = Path(merged_model_path).parent / "training_complete.json"
                if completion_file.exists():
                    with open(completion_file, 'r') as f:
                        completion_info = json.load(f)
                    final_system_prompt = completion_info.get("system_prompt", self.get_default_system_prompt())
                else:
                    final_system_prompt = self.get_default_system_prompt()
            except Exception:
                final_system_prompt = self.get_default_system_prompt()
        
        # Detect model type and use appropriate template
        model_type = self.detect_model_type(merged_model_path)
        self.log(f"Detected model type: {model_type}")
        
        # Use model-specific template
        chat_template = self.get_model_specific_ollama_template(model_type)
        self.log(f"Using {model_type}-specific template for Ollama")
        
        # Get model-specific stop tokens
        stop_tokens = self.get_model_stop_tokens(model_type)
        stop_parameters = "\n".join([f'PARAMETER stop "{token}"' for token in stop_tokens])
        
        modelfile_content = f'''FROM {abs_model_path}

SYSTEM """{final_system_prompt}"""

TEMPLATE """{chat_template}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
{stop_parameters}
'''
        
        return modelfile_content

    def validate_ollama_model_name(self, model_name: str) -> str:
        """
        Validate and fix Ollama model name format.
        Ollama model names must be lowercase and can only contain alphanumeric chars, dots, dashes, and underscores.
        """
        import re
        # Convert to lowercase and replace invalid characters
        valid_name = re.sub(r'[^a-z0-9._-]', '-', model_name.lower())
        # Remove consecutive dashes
        valid_name = re.sub(r'-+', '-', valid_name)
        # Remove leading/trailing dashes
        valid_name = valid_name.strip('-')
        return valid_name

    def create_ollama_model(self, model_name: str, modelfile_content: str, temp_dir: str) -> bool:
        """
        Create an Ollama model using the Modelfile.
        
        Args:
            model_name: Name for the Ollama model
            modelfile_content: Content of the Modelfile
            temp_dir: Temporary directory for the Modelfile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate and fix model name
            valid_model_name = self.validate_ollama_model_name(model_name)
            if valid_model_name != model_name:
                self.log(f"Model name adjusted for Ollama compatibility: '{model_name}' → '{valid_model_name}'")
            
            # Write Modelfile to temp directory
            modelfile_path = os.path.join(temp_dir, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)
            
            self.log(f"Creating Ollama model: {valid_model_name}")
            self.log(f"Using Modelfile: {modelfile_path}")
            if self.config.verbose:
                self.log("--- Modelfile contents ---")
                self.log(modelfile_content)
                self.log("--- End Modelfile ---")
            
            # Run ollama create command with timeout and better feedback
            self.log("⏳ Running ollama create command (this may take several minutes for large models)...")
            
            # Get timeout from environment or use default (20 minutes)
            timeout_minutes = int(os.getenv("OLLAMA_CREATE_TIMEOUT_MINUTES", "20"))
            timeout_seconds = timeout_minutes * 60
            
            # Check if we want real-time output
            show_realtime = os.getenv("OLLAMA_SHOW_REALTIME_OUTPUT", "false").lower() == "true"
            
            try:
                if show_realtime:
                    # Stream output in real-time
                    self.log("📺 Streaming ollama create output...")
                    process = subprocess.Popen([
                        "ollama", "create", valid_model_name, "-f", modelfile_path
                    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                       text=True, cwd=temp_dir, bufsize=1, universal_newlines=True)
                    
                    # Stream output line by line
                    for line in process.stdout:
                        if line.strip():
                            self.log(f"[OLLAMA] {line.strip()}")
                    
                    process.wait(timeout=timeout_seconds)
                    result_code = process.returncode
                    stdout_content = ""
                    stderr_content = ""
                else:
                    # Capture output (existing behavior with timeout)
                    result = subprocess.run([
                        "ollama", "create", valid_model_name, "-f", modelfile_path
                    ], capture_output=True, text=True, cwd=temp_dir, timeout=timeout_seconds)
                    
                    result_code = result.returncode
                    stdout_content = result.stdout
                    stderr_content = result.stderr
                    
            except subprocess.TimeoutExpired:
                self.log(f"❌ Ollama create command timed out after {timeout_minutes} minutes", "error")
                self.log("💡 Try increasing OLLAMA_CREATE_TIMEOUT_MINUTES environment variable", "info")
                self.log("💡 Or set OLLAMA_SHOW_REALTIME_OUTPUT=true to see progress", "info")
                return False
            except Exception as e:
                self.log(f"❌ Error running ollama create command: {str(e)}", "error")
                return False
            
            if result_code == 0:
                self.log(f"✅ Ollama model '{valid_model_name}' created successfully!")
                self.log(f"You can now use your model with: ollama run {valid_model_name}")
                return True
            else:
                self.log(f"❌ Error creating Ollama model (exit code: {result_code})", "error")
                if stdout_content:
                    self.log(f"STDOUT: {stdout_content}", "error")
                if stderr_content:
                    self.log(f"STDERR: {stderr_content}", "error")
                return False
                
        except FileNotFoundError:
            self.log("❌ Error: 'ollama' command not found. Please install Ollama first.", "error")
            self.log("Visit: https://ollama.ai/download", "info")
            return False
        except Exception as e:
            self.log(f"❌ Error creating Ollama model: {str(e)}", "error")
            return False

    def check_prerequisites(self) -> bool:
        """Check if basic ML libraries are available."""
        if not ML_LIBRARIES_AVAILABLE:
            self.log(f"Required ML libraries not available: {IMPORT_ERROR}", "error")
            return False
        return True

    def check_export_prerequisites(self, model_path: str, model_name: Optional[str]) -> bool:
        """Check if export prerequisites are met."""
        if not self.check_prerequisites():
            return False

        if not model_path or not Path(model_path).exists():
            self.log(f"Export model path does not exist: {model_path}", "error")
            return False
            
        if not model_name:
            self.log("Model name not provided. Use --model-name argument, EXPORT_MODEL_NAME environment variable, or config file", "error")
            return False
            
        return True

    def run(self, model_name: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs) -> bool:
        """Run the model export step."""
        self.log("Starting model export step...")
        
        # Get configuration
        model_path = os.getenv("EXPORT_MODEL_PATH")
        output_dir = os.getenv("EXPORT_OUTPUT_DIR", self.config.export_dir)
        
        # Priority for model name: CLI argument > environment variable > config default
        final_model_name = model_name or os.getenv("EXPORT_MODEL_NAME") or self.config.export_model_name
        
        # Priority for system prompt: CLI argument > environment variable > config default
        final_system_prompt = system_prompt or os.getenv("EXPORT_SYSTEM_PROMPT")
        job_id = os.getenv("EXPORT_JOB_ID")
        skip_merge = os.getenv("EXPORT_SKIP_MERGE", "false").lower() == "true"
        
        # Check prerequisites after we have all the configuration
        if not self.check_export_prerequisites(model_path, final_model_name):
            return False
        
        try:
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Set up paths
            merged_model_path = os.path.join(output_dir, f"{final_model_name}_merged")
            
            self.log(f"🚀 Starting export process for: {final_model_name}")
            self.log(f"Adapter path: {model_path}")
            self.log(f"Merged model path: {merged_model_path}")
            
            # Step 1: Merge LoRA with base model (unless skipped)
            if not skip_merge:
                if not self.merge_lora_model(model_path, merged_model_path):
                    return False
            else:
                if not os.path.exists(merged_model_path):
                    self.log(f"❌ Error: Merged model not found at {merged_model_path} (use EXPORT_SKIP_MERGE only if model already exists)", "error")
                    return False
                self.log(f"⏭️  Skipping merge step, using existing model at: {merged_model_path}")
            
            # Step 2: Create Ollama model
            with tempfile.TemporaryDirectory() as temp_dir:
                # Read chat template if available
                chat_template = None
                template_path = os.path.join(model_path, "chat_template.jinja")
                if os.path.exists(template_path):
                    with open(template_path, "r") as f:
                        chat_template = f.read().strip()
                    self.log(f"📝 Using chat template from: {template_path}")
                
                # Create Modelfile
                modelfile_content = self.create_modelfile(
                    final_model_name, 
                    merged_model_path, 
                    system_prompt=final_system_prompt,
                    job_id=job_id
                )
                
                # Create Ollama model
                if self.create_ollama_model(final_model_name, modelfile_content, temp_dir):
                    self.log(f"🎉 Export completed successfully!")
                    self.log(f"Your fine-tuned model is now available in Ollama as: {final_model_name}")
                    return True
                else:
                    return False
                    
        except Exception as e:
            self.log(f"Model export failed: {str(e)}", "error")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "error")
            return False