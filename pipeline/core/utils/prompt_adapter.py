#!/usr/bin/env python3
"""
Prompt Adapter for Multi-Model System Prompt Integration

Adapts system prompts and training data formats for different model types
(Llama, Gemma, Qwen) with their specific chat templates and formatting.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelPromptConfig:
    """Configuration for model-specific prompt formatting"""
    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    bos_token: str = ""
    eos_token: str = ""
    supports_system_role: bool = True


class PromptAdapter:
    """Adapts prompts and training data for different model architectures"""
    
    def __init__(self):
        self.model_configs = self._initialize_model_configs()
    
    def _initialize_model_configs(self) -> Dict[str, ModelPromptConfig]:
        """Initialize model-specific prompt configurations"""
        return {
            "llama": ModelPromptConfig(
                system_prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                system_suffix="<|eot_id|>",
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                user_suffix="<|eot_id|>",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                assistant_suffix="<|eot_id|>",
                supports_system_role=True
            ),
            "gemma": ModelPromptConfig(
                system_prefix="",  # Gemma doesn't have explicit system role
                system_suffix="",
                user_prefix="<start_of_turn>user\n",
                user_suffix="<end_of_turn>\n",
                assistant_prefix="<start_of_turn>model\n",
                assistant_suffix="<end_of_turn>\n",
                supports_system_role=False
            ),
            "qwen": ModelPromptConfig(
                system_prefix="<|im_start|>system\n",
                system_suffix="<|im_end|>\n",
                user_prefix="<|im_start|>user\n",
                user_suffix="<|im_end|>\n",
                assistant_prefix="<|im_start|>assistant\n",
                assistant_suffix="<|im_end|>\n",
                supports_system_role=True
            ),
            "alpaca": ModelPromptConfig(
                system_prefix="",  # Alpaca uses simple instruction format
                system_suffix="",
                user_prefix="### Instruction:\n",
                user_suffix="\n\n",
                assistant_prefix="### Response:\n",
                assistant_suffix="",
                supports_system_role=False
            )
        }
    
    def format_system_prompt(self, system_prompt: str, model_type: str) -> str:
        """Format system prompt for specific model type"""
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        config = self.model_configs[model_type]
        
        if not config.supports_system_role:
            # For models without system role, prepend to user message
            return system_prompt
        
        return f"{config.system_prefix}{system_prompt}{config.system_suffix}"
    
    def create_training_prompt(self, instruction: str, response: str, 
                             system_prompt: str, model_type: str) -> str:
        """Create formatted training prompt for specific model"""
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        config = self.model_configs[model_type]
        
        if config.supports_system_role:
            # Standard system/user/assistant format
            formatted_prompt = (
                f"{config.system_prefix}{system_prompt}{config.system_suffix}"
                f"{config.user_prefix}{instruction}{config.user_suffix}"
                f"{config.assistant_prefix}{response}{config.assistant_suffix}"
            )
        else:
            # For models without system role, include system prompt in first user message
            combined_instruction = f"{system_prompt}\n\n{instruction}"
            formatted_prompt = (
                f"{config.user_prefix}{combined_instruction}{config.user_suffix}"
                f"{config.assistant_prefix}{response}{config.assistant_suffix}"
            )
        
        return formatted_prompt
    
    def extract_instruction_response(self, text: str) -> Dict[str, Optional[str]]:
        """Extract instruction and response from existing training text"""
        # Try to parse standard instruction-response format
        patterns = [
            # Standard format with markers
            (r"### Instruction:\s*(.*?)\s*### Response:\s*(.*)", "instruction_response"),
            # Below is an instruction format
            (r"Below is an instruction.*?\n\n### Instruction:\s*(.*?)\s*### Response:\s*(.*)", "instruction_response"),
            # Simple Q&A format
            (r"Question:\s*(.*?)\s*Answer:\s*(.*)", "qa"),
            # User/Assistant format
            (r"User:\s*(.*?)\s*Assistant:\s*(.*)", "user_assistant"),
        ]
        
        import re
        for pattern, format_type in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return {
                    "instruction": match.group(1).strip(),
                    "response": match.group(2).strip(),
                    "format": format_type
                }
        
        # If no pattern matches, try to split on common separators
        separators = ["\n\nResponse:", "\nAnswer:", "\nAssistant:"]
        for sep in separators:
            if sep in text:
                parts = text.split(sep, 1)
                if len(parts) == 2:
                    instruction = parts[0].strip()
                    response = parts[1].strip()
                    
                    # Clean up instruction
                    for prefix in ["Instruction:", "Question:", "User:", "Below is an instruction"]:
                        if instruction.startswith(prefix):
                            instruction = instruction[len(prefix):].strip()
                    
                    return {
                        "instruction": instruction,
                        "response": response,
                        "format": "inferred"
                    }
        
        # If all else fails, use the whole text as instruction with empty response
        return {
            "instruction": text.strip(),
            "response": None,
            "format": "fallback"
        }
    
    def convert_dataset_format(self, input_path: Path, output_path: Path, 
                             system_prompt: str, model_type: str) -> Dict[str, Any]:
        """Convert dataset to model-specific format with system prompt"""
        
        if not input_path.exists():
            return {"success": False, "error": f"Input file not found: {input_path}"}
        
        converted_examples = []
        conversion_stats = {
            "total_processed": 0,
            "successful_conversions": 0,
            "parsing_errors": 0,
            "format_distribution": {}
        }
        
        try:
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    conversion_stats["total_processed"] += 1
                    
                    try:
                        data = json.loads(line)
                        
                        if 'text' not in data:
                            conversion_stats["parsing_errors"] += 1
                            continue
                        
                        # Extract instruction and response
                        parsed = self.extract_instruction_response(data['text'])
                        
                        if parsed['response'] is None:
                            # Skip examples without response
                            conversion_stats["parsing_errors"] += 1
                            continue
                        
                        # Track format distribution
                        format_type = parsed['format']
                        conversion_stats["format_distribution"][format_type] = \
                            conversion_stats["format_distribution"].get(format_type, 0) + 1
                        
                        # Create model-specific training prompt
                        formatted_text = self.create_training_prompt(
                            instruction=parsed['instruction'],
                            response=parsed['response'],
                            system_prompt=system_prompt,
                            model_type=model_type
                        )
                        
                        # Preserve other fields from original data
                        converted_data = {'text': formatted_text}
                        for key, value in data.items():
                            if key != 'text':
                                converted_data[key] = value
                        
                        converted_examples.append(converted_data)
                        conversion_stats["successful_conversions"] += 1
                        
                    except json.JSONDecodeError:
                        conversion_stats["parsing_errors"] += 1
                        continue
                    except Exception as e:
                        conversion_stats["parsing_errors"] += 1
                        continue
            
            # Write converted dataset
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for example in converted_examples:
                    outfile.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            return {
                "success": True,
                "output_path": str(output_path),
                "examples_converted": len(converted_examples),
                "stats": conversion_stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Conversion failed: {str(e)}",
                "stats": conversion_stats
            }
    
    def validate_system_prompt(self, system_prompt: str, model_type: str) -> Dict[str, Any]:
        """Validate system prompt for model compatibility"""
        
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "estimated_tokens": len(system_prompt.split()) * 1.3  # Rough estimation
        }
        
        # Check length
        if len(system_prompt) > 2000:
            validation["warnings"].append("System prompt is very long (>2000 chars)")
        
        if len(system_prompt.split()) > 500:
            validation["warnings"].append("System prompt may exceed token limits")
        
        # Check for empty prompt
        if not system_prompt.strip():
            validation["valid"] = False
            validation["errors"].append("System prompt cannot be empty")
        
        # Model-specific validations
        if model_type == "gemma":
            if "system" in system_prompt.lower() and "role" in system_prompt.lower():
                validation["warnings"].append("Gemma models don't have explicit system role - prompt will be integrated into user messages")
        
        # Check for potentially problematic content
        problematic_patterns = [
            ("You are an AI", "Consider using more specific role descriptions"),
            ("I am", "Use third person ('You are') instead of first person"),
            ("<|", "Avoid using model-specific tokens in system prompt")
        ]
        
        for pattern, suggestion in problematic_patterns:
            if pattern.lower() in system_prompt.lower():
                validation["warnings"].append(f"Contains '{pattern}': {suggestion}")
        
        return validation
    
    def create_test_prompt(self, system_prompt: str, test_question: str, model_type: str) -> str:
        """Create a test prompt for model evaluation"""
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        config = self.model_configs[model_type]
        
        if config.supports_system_role:
            test_prompt = (
                f"{config.system_prefix}{system_prompt}{config.system_suffix}"
                f"{config.user_prefix}{test_question}{config.user_suffix}"
                f"{config.assistant_prefix}"
            )
        else:
            # For models without system role
            combined_input = f"{system_prompt}\n\n{test_question}"
            test_prompt = (
                f"{config.user_prefix}{combined_input}{config.user_suffix}"
                f"{config.assistant_prefix}"
            )
        
        return test_prompt
    
    def get_model_chat_template(self, model_type: str) -> Optional[str]:
        """Get Jinja2 chat template for model type"""
        
        templates = {
            "llama": """{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{- '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- elif message['role'] == 'user' %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}""",
            
            "gemma": """{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'system' %}
        {{- message['content'] + '\n\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<start_of_turn>model\n' }}
{%- endif %}""",
            
            "qwen": """{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' %}
        {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""
        }
        
        return templates.get(model_type)


def main():
    """Example usage"""
    adapter = PromptAdapter()
    
    # Test system prompt formatting
    system_prompt = "You are a helpful cybersecurity expert."
    
    for model_type in ["llama", "gemma", "qwen"]:
        print(f"\n{model_type.upper()} Format:")
        print("-" * 40)
        
        formatted = adapter.format_system_prompt(system_prompt, model_type)
        print(f"System: {repr(formatted)}")
        
        # Test full training prompt
        training_prompt = adapter.create_training_prompt(
            instruction="What is phishing?",
            response="Phishing is a cybersecurity attack...",
            system_prompt=system_prompt,
            model_type=model_type
        )
        print(f"Training: {repr(training_prompt[:100])}...")
        
        # Validate system prompt
        validation = adapter.validate_system_prompt(system_prompt, model_type)
        print(f"Validation: {validation}")


if __name__ == "__main__":
    main()