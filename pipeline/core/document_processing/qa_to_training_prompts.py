#!/usr/bin/env python3
"""
Q&A to Training Prompts Converter

Converts Q&A JSON files to model-specific training prompt formats.
Supports Llama, Gemma, and Qwen models with their specific chat templates.

Usage:
    python qa_to_training_prompts.py --input input.json --output output.json --model-type llama
    python qa_to_training_prompts.py --input data.jsonl --output training.json --model-type gemma --system-prompt "You are helpful."
"""

import json
import argparse
import sys
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from prompt_adapter import PromptAdapter

# PYTHON_ARGCOMPLETE_OK
try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False


def input_file_completer(prefix, parsed_args, **kwargs):
    """Custom completer for input files"""
    files = []
    
    # Look for JSON/JSONL files in common locations
    for base_dir in ['_data', '.']:
        if os.path.exists(base_dir):
            for root, dirs, file_list in os.walk(base_dir):
                for f in file_list:
                    if f.endswith(('.json', '.jsonl')):
                        full_path = os.path.join(root, f)
                        files.append(full_path)
    
    return [f for f in files if f.startswith(prefix)]


def system_prompt_file_completer(prefix, parsed_args, **kwargs):
    """Custom completer for system prompt files"""
    files = []
    
    # Look for JSON and YAML files in common locations
    for base_dir in ['_data', '_jobs/templates', '.']:
        if os.path.exists(base_dir):
            for root, dirs, file_list in os.walk(base_dir):
                for f in file_list:
                    if f.endswith(('.json', '.yaml', '.yml')):
                        full_path = os.path.join(root, f)
                        files.append(full_path)
    
    return [f for f in files if f.startswith(prefix)]


def model_type_completer(prefix, parsed_args, **kwargs):
    """Custom completer for model types"""
    model_types = ['llama', 'gemma', 'qwen']
    return [mt for mt in model_types if mt.startswith(prefix)]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert Q&A JSON files to model-specific training prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input questions.json --output training.json --model-type llama
  %(prog)s --input data.jsonl --output prompts.json --model-type gemma --system-prompt "You are a helpful assistant."
  %(prog)s --input nzism.json --output nzism-training.json --model-type llama --system-prompt-file _data/nzism_metadata.json
  %(prog)s --input qa.jsonl --output formatted.json --model-type qwen --system-prompt-file metadata.json
        """
    )
    
    input_arg = parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input Q&A JSON file (supports .json or .jsonl)'
    )
    if ARGCOMPLETE_AVAILABLE:
        input_arg.completer = input_file_completer
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file (model name will be appended, e.g., output-llama.json)'
    )
    
    model_type_arg = parser.add_argument(
        '--model-type', '-m',
        type=str,
        required=True,
        choices=['llama', 'gemma', 'qwen'],
        help='Target model type for prompt formatting'
    )
    if ARGCOMPLETE_AVAILABLE:
        model_type_arg.completer = model_type_completer
    
    parser.add_argument(
        '--system-prompt', '-s',
        type=str,
        help='System prompt to include in training data (default: "You are a helpful and knowledgeable assistant.")'
    )
    
    system_prompt_file_arg = parser.add_argument(
        '--system-prompt-file', '-f',
        type=str,
        help='JSON file containing system prompt and dataset metadata (supports "system-prompt" and "document-summary" fields)'
    )
    if ARGCOMPLETE_AVAILABLE:
        system_prompt_file_arg.completer = system_prompt_file_completer
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Enable argcomplete
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)
    
    return parser.parse_args()


def load_system_prompt_from_file(file_path: str, verbose: bool = False) -> str:
    """Load system prompt from JSON or YAML file"""
    prompt_file = Path(file_path)
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt file not found: {file_path}")
    
    if verbose:
        print(f"Loading system prompt from: {file_path}")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            if file_path.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Try to extract system prompt from different possible structures
        system_prompt = None
        
        if isinstance(data, dict):
            # Check for common system prompt keys (prioritize hyphenated version)
            for key in ['system-prompt', 'system_prompt', 'systemPrompt', 'prompt', 'system']:
                if key in data and data[key]:
                    system_prompt = data[key]
                    break
            
            # If no direct system prompt, try to build from description and datasets
            if not system_prompt:
                if 'document-summary' in data:
                    system_prompt = data['document-summary']
                elif 'description' in data:
                    system_prompt = data['description']
                elif 'datasets' in data:
                    # Build system prompt from dataset descriptions
                    topics = []
                    for dataset in data['datasets']:
                        if 'description' in dataset:
                            topics.append(dataset['description'])
                    if topics:
                        system_prompt = f"You are a knowledgeable assistant specializing in: {', '.join(topics)}"
        
        if not system_prompt:
            raise ValueError("Could not find system prompt in file. Expected 'system-prompt', 'system_prompt', 'prompt', 'document-summary', or 'description' field.")
        
        if verbose:
            print(f"Extracted system prompt: {system_prompt[:100]}...")
            
        return system_prompt.strip()
        
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid file format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading system prompt file: {e}")


def generate_output_filename(output_path: str, model_type: str) -> str:
    """Generate output filename with model type suffix"""
    path = Path(output_path)
    stem = path.stem
    suffix = path.suffix
    
    # Add model type to filename
    new_stem = f"{stem}-{model_type}"
    return str(path.parent / f"{new_stem}{suffix}")


def load_qa_data(input_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """Load Q&A data from JSON or JSONL file"""
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    data = []
    
    if verbose:
        print(f"Loading data from: {input_path}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Try JSONL format first (one JSON object per line)
        try:
            f.seek(0)  # Reset to beginning
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # If first line fails, might be regular JSON
                    if line_num == 1:
                        break
                    else:
                        if verbose:
                            print(f"Warning: Skipping invalid JSON on line {line_num}")
                        continue
            
            # If we successfully loaded data as JSONL, we're done
            if data:
                if verbose:
                    print(f"Detected JSONL format")
                return data
        except:
            pass
        
        # Try as regular JSON format
        try:
            f.seek(0)  # Reset to beginning
            content = json.load(f)
            if isinstance(content, list):
                data = content
            elif isinstance(content, dict):
                data = [content]
            else:
                raise ValueError("JSON must contain a list of objects or a single object")
            if verbose:
                print(f"Detected JSON format")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON/JSONL format: {e}")
    
    if verbose:
        print(f"Loaded {len(data)} entries")
    
    return data


def convert_qa_to_training_prompts(
    qa_data: List[Dict[str, Any]], 
    model_type: str, 
    system_prompt: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """Convert Q&A data to model-specific training prompts"""
    
    adapter = PromptAdapter()
    converted_prompts = []
    conversion_stats = {
        "total_processed": 0,
        "successful_conversions": 0,
        "parsing_errors": 0,
        "format_distribution": {}
    }
    
    if verbose:
        print(f"Converting to {model_type} format with system prompt: {system_prompt[:50]}...")
    
    for entry_num, entry in enumerate(qa_data, 1):
        conversion_stats["total_processed"] += 1
        
        try:
            # Extract text field (required)
            if 'text' not in entry:
                if verbose:
                    print(f"Warning: Entry {entry_num} missing 'text' field, skipping")
                conversion_stats["parsing_errors"] += 1
                continue
            
            # Parse instruction and response
            parsed = adapter.extract_instruction_response(entry['text'])
            
            if parsed['response'] is None:
                if verbose:
                    print(f"Warning: Entry {entry_num} could not extract response, skipping")
                conversion_stats["parsing_errors"] += 1
                continue
            
            # Track format distribution
            format_type = parsed['format']
            conversion_stats["format_distribution"][format_type] = \
                conversion_stats["format_distribution"].get(format_type, 0) + 1
            
            # Create model-specific training prompt
            formatted_prompt = adapter.create_training_prompt(
                instruction=parsed['instruction'],
                response=parsed['response'],
                system_prompt=system_prompt,
                model_type=model_type
            )
            
            # Create output entry
            output_entry = {
                'text': formatted_prompt
            }
            
            # Preserve other fields from original entry
            for key, value in entry.items():
                if key != 'text':
                    output_entry[key] = value
            
            # Add metadata
            output_entry['_conversion_info'] = {
                'source_format': format_type,
                'model_type': model_type,
                'system_prompt_hash': hash(system_prompt)
            }
            
            converted_prompts.append(output_entry)
            conversion_stats["successful_conversions"] += 1
            
            if verbose and entry_num % 10 == 0:
                print(f"Processed {entry_num}/{len(qa_data)} entries...")
                
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing entry {entry_num}: {e}")
            conversion_stats["parsing_errors"] += 1
            continue
    
    return {
        "prompts": converted_prompts,
        "stats": conversion_stats
    }


def save_training_prompts(prompts: List[Dict[str, Any]], output_path: str, verbose: bool = False) -> None:
    """Save training prompts to output file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Saving {len(prompts)} training prompts to: {output_path}")
    
    if output_path.endswith('.jsonl'):
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    else:
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"Successfully saved training prompts")


def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        # Validate arguments
        if args.system_prompt and args.system_prompt_file:
            print("Error: Cannot specify both --system-prompt and --system-prompt-file")
            sys.exit(1)
        
        # Determine system prompt
        if args.system_prompt_file:
            system_prompt = load_system_prompt_from_file(args.system_prompt_file, args.verbose)
        elif args.system_prompt:
            system_prompt = args.system_prompt
        else:
            system_prompt = "You are a helpful and knowledgeable assistant."
        
        # Generate output filename with model type
        output_path = generate_output_filename(args.output, args.model_type)
        
        if args.verbose:
            print(f"Q&A to Training Prompts Converter")
            print(f"Input: {args.input}")
            print(f"Output: {output_path}")
            print(f"Model Type: {args.model_type}")
            if args.system_prompt_file:
                print(f"System Prompt File: {args.system_prompt_file}")
            print(f"System Prompt: {system_prompt}")
            print("-" * 50)
        
        # Load Q&A data
        qa_data = load_qa_data(args.input, args.verbose)
        
        if not qa_data:
            print("Error: No valid Q&A data found in input file")
            sys.exit(1)
        
        # Convert to training prompts
        result = convert_qa_to_training_prompts(
            qa_data, 
            args.model_type, 
            system_prompt,
            args.verbose
        )
        
        # Save results
        save_training_prompts(result["prompts"], output_path, args.verbose)
        
        # Print statistics
        stats = result["stats"]
        print(f"\nConversion Summary:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Successfully converted: {stats['successful_conversions']}")
        print(f"  Parsing errors: {stats['parsing_errors']}")
        print(f"  Success rate: {stats['successful_conversions']/stats['total_processed']*100:.1f}%")
        
        if stats["format_distribution"]:
            print(f"  Format distribution:")
            for fmt, count in stats["format_distribution"].items():
                print(f"    {fmt}: {count}")
        
        print(f"\nOutput saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()