#!/usr/bin/env python3
"""
Export fine-tuned LoRA models to Ollama format.

This script merges LoRA adapters with their base models and creates
Ollama-compatible models that can be run locally.

Usage:
    python export_to_ollama.py --model-path policy-chatbot-v1/final --model-name linz-policy-llama
    python export_to_ollama.py --model-path policy-chatbot-gemma3-v1/final --model-name linz-policy-gemma
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# PYTHON_ARGCOMPLETE_OK
try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False


def model_path_completer(prefix, parsed_args, **kwargs):
    """Custom completer for model paths"""
    paths = []
    
    # Add common model directories
    for base_dir in ['.', './fine_tuned_model', './_jobs']:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if 'adapter_config.json' in files:
                    paths.append(root)
                # Also add directories that might contain models
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if os.path.exists(os.path.join(full_path, 'adapter_config.json')):
                        paths.append(full_path)
    
    return [p for p in paths if p.startswith(prefix)]


def system_prompt_file_completer(prefix, parsed_args, **kwargs):
    """Custom completer for system prompt files"""
    paths = []
    
    # Look for JSON and YAML files in common locations
    for base_dir in ['_data', '_jobs/templates', '.']:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if f.endswith(('.json', '.yaml', '.yml')):
                        full_path = os.path.join(root, f)
                        paths.append(full_path)
    
    return [p for p in paths if p.startswith(prefix)]


def job_id_completer(prefix, parsed_args, **kwargs):
    """Custom completer for job IDs"""
    job_ids = []
    
    jobs_dir = '_jobs'
    if os.path.exists(jobs_dir):
        for item in os.listdir(jobs_dir):
            job_path = os.path.join(jobs_dir, item)
            if os.path.isdir(job_path) and item not in ['templates']:
                # Check if it has job config or final model
                if (os.path.exists(os.path.join(job_path, 'job_config.yaml')) or
                    os.path.exists(os.path.join(job_path, 'final'))):
                    job_ids.append(item)
    
    return [jid for jid in job_ids if jid.startswith(prefix)]


def merge_lora_model(adapter_path: str, output_path: str) -> bool:
    """
    Merge LoRA adapter with base model and save the merged model.
    
    Args:
        adapter_path: Path to the LoRA adapter directory
        output_path: Path to save the merged model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Loading adapter from: {adapter_path}")
        
        # Load the adapter config to get base model info
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print(f"Error: adapter_config.json not found in {adapter_path}")
            return False
        
        # Read adapter config to get base model name
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config['base_model_name_or_path']
        
        print(f"Base model: {base_model_name}")
        
        # Load base model
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load and merge the LoRA adapter
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        
        # Save the merged model
        print(f"Saving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        print("✅ Model merging completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model merging: {str(e)}")
        return False


def fix_chat_template_for_ollama(template: str) -> str:
    """
    Fix HuggingFace chat template to be compatible with Ollama.
    
    Args:
        template: Original HuggingFace Jinja2 template
        
    Returns:
        Ollama-compatible template
    """
    # Remove unsupported functions like bos_token, eos_token
    template = template.replace("{{ bos_token }}", "")
    template = template.replace("{{ eos_token }}", "")
    template = template.replace("{{bos_token}}", "")
    template = template.replace("{{eos_token}}", "")
    
    # Convert to Ollama template format
    # This is a basic conversion - complex templates may need manual adjustment
    ollama_template = """{{- if .Messages }}
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
"""
    
    return ollama_template


def create_modelfile(model_name: str, merged_model_path: str, chat_template: Optional[str] = None, 
                    system_prompt: Optional[str] = None, job_id: Optional[str] = None) -> str:
    """
    Create an Ollama Modelfile for the merged model.
    
    Args:
        model_name: Name for the Ollama model
        merged_model_path: Path to the merged model directory
        chat_template: Optional custom chat template
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
            print(f"Warning: Could not load system prompt from job {job_id}: {e}")
            final_system_prompt = get_default_system_prompt()
    else:
        # Try to load from training completion info
        try:
            completion_file = Path(merged_model_path).parent / "training_complete.json"
            if completion_file.exists():
                with open(completion_file, 'r') as f:
                    completion_info = json.load(f)
                final_system_prompt = completion_info.get("system_prompt", get_default_system_prompt())
            else:
                final_system_prompt = get_default_system_prompt()
        except Exception:
            final_system_prompt = get_default_system_prompt()
    
    # Use a basic chat template if none provided, or fix the provided template
    if not chat_template:
        chat_template = """{{- if .Messages }}
{{- range .Messages }}
{{- if eq .Role "system" }}System: {{ .Content }}

{{- else if eq .Role "user" }}Human: {{ .Content }}

{{- else if eq .Role "assistant" }}Assistant: {{ .Content }}

{{- end }}
{{- end }}
{{- else }}Human: {{ .Prompt }}

{{- end }}Assistant: """
    else:
        # Fix HuggingFace template for Ollama compatibility
        chat_template = fix_chat_template_for_ollama(chat_template)
    
    modelfile_content = f'''FROM {abs_model_path}

SYSTEM """{final_system_prompt}"""

TEMPLATE """{chat_template}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "Human:"
PARAMETER stop "System:"
'''
    
    return modelfile_content


def get_default_system_prompt() -> str:
    """Get default system prompt for models without specific prompts"""
    return """You are a helpful assistant specialized in Land Information New Zealand (LINZ) title fee policies. You provide accurate, clear information about title fees, registration processes, and related policies. Always base your responses on official LINZ policies and be precise with fee amounts and procedures."""


def validate_ollama_model_name(model_name: str) -> str:
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


def create_ollama_model(model_name: str, modelfile_content: str, temp_dir: str) -> bool:
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
        valid_model_name = validate_ollama_model_name(model_name)
        if valid_model_name != model_name:
            print(f"Model name adjusted for Ollama compatibility: '{model_name}' → '{valid_model_name}'")
        
        # Write Modelfile to temp directory
        modelfile_path = os.path.join(temp_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        print(f"Creating Ollama model: {valid_model_name}")
        print(f"Using Modelfile: {modelfile_path}")
        print("\n--- Modelfile contents ---")
        print(modelfile_content)
        print("--- End Modelfile ---\n")
        
        # Run ollama create command
        result = subprocess.run([
            "ollama", "create", valid_model_name, "-f", modelfile_path
        ], capture_output=True, text=True, cwd=temp_dir)
        
        if result.returncode == 0:
            print(f"✅ Ollama model '{valid_model_name}' created successfully!")
            print("\nYou can now use your model with:")
            print(f"  ollama run {valid_model_name}")
            return True
        else:
            print(f"❌ Error creating Ollama model:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Error: 'ollama' command not found. Please install Ollama first.")
        print("Visit: https://ollama.ai/download")
        return False
    except Exception as e:
        print(f"❌ Error creating Ollama model: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned LoRA models to Ollama format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export Llama model
  python export_to_ollama.py --model-path policy-chatbot-v1/final --model-name linz-policy-llama
  
  # Export Gemma model
  python export_to_ollama.py --model-path policy-chatbot-gemma3-v1/final --model-name linz-policy-gemma
  
  # Export with custom output directory
  python export_to_ollama.py --model-path policy-chatbot-v1/final --model-name linz-policy-llama --output-dir ./merged_models
        """
    )
    
    model_path_arg = parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the LoRA adapter directory (e.g., policy-chatbot-v1/final)"
    )
    if ARGCOMPLETE_AVAILABLE:
        model_path_arg.completer = model_path_completer
    
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name for the Ollama model (e.g., linz-policy-llama)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./merged_models",
        help="Directory to save merged models (default: ./merged_models)"
    )
    
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip model merging step (use existing merged model)"
    )
    
    job_id_arg = parser.add_argument(
        "--job-id",
        type=str,
        help="Job ID to load system prompt from"
    )
    if ARGCOMPLETE_AVAILABLE:
        job_id_arg.completer = job_id_completer
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the model"
    )
    
    system_prompt_file_arg = parser.add_argument(
        "--system-prompt-file",
        type=str,
        help="JSON/YAML file containing system prompt and metadata"
    )
    if ARGCOMPLETE_AVAILABLE:
        system_prompt_file_arg.completer = system_prompt_file_completer
    
    # Enable argcomplete
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print(f"❌ Error: Not a valid LoRA adapter directory (missing adapter_config.json): {args.model_path}")
        sys.exit(1)
    
    # Set up paths
    merged_model_path = os.path.join(args.output_dir, f"{args.model_name}_merged")
    
    print(f"🚀 Starting export process for: {args.model_name}")
    print(f"Adapter path: {args.model_path}")
    print(f"Merged model path: {merged_model_path}")
    
    # Step 1: Merge LoRA with base model (unless skipped)
    if not args.skip_merge:
        if not merge_lora_model(args.model_path, merged_model_path):
            sys.exit(1)
    else:
        if not os.path.exists(merged_model_path):
            print(f"❌ Error: Merged model not found at {merged_model_path} (use --skip-merge only if model already exists)")
            sys.exit(1)
        print(f"⏭️  Skipping merge step, using existing model at: {merged_model_path}")
    
    # Step 2: Create Ollama model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Read chat template if available
        chat_template = None
        template_path = os.path.join(args.model_path, "chat_template.jinja")
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                chat_template = f.read().strip()
            print(f"📝 Using chat template from: {template_path}")
        
        # Create Modelfile
        modelfile_content = create_modelfile(
            args.model_name, 
            merged_model_path, 
            chat_template,
            system_prompt=args.system_prompt,
            job_id=args.job_id
        )
        
        # Create Ollama model
        if create_ollama_model(args.model_name, modelfile_content, temp_dir):
            print(f"\n🎉 Export completed successfully!")
            print(f"Your fine-tuned model is now available in Ollama as: {args.model_name}")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()