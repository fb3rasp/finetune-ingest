"""
Enhanced main entry point using LangChain-based processing.

This provides a modern alternative to main.py using the new LangChain
architecture while maintaining compatibility with existing CLI arguments.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_processing import LangChainProcessor, UnifiedLLMProvider, LLMProvider
from utils.helpers import log_message, save_results


def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description='Generate training data for LLM fine-tuning using LangChain'
    )
    
    # Directory and file arguments
    parser.add_argument('--incoming-dir', default='./incoming', 
                       help='Directory containing source documents')
    parser.add_argument('--output-file', default='./training_data.json',
                       help='Output file for training data')
    
    # LLM Provider Configuration
    parser.add_argument('--provider', 
                       choices=['openai', 'claude', 'gemini', 'local'],
                       default='openai',
                       help='LLM provider to use')
    parser.add_argument('--model', 
                       help='Specific model name (uses provider default if not specified)')
    parser.add_argument('--api-key', 
                       help='API key for the provider (uses env vars if not specified)')
    
    # Ollama-specific options (for --provider local)
    parser.add_argument('--ollama-base-url',
                        help='Base URL for Ollama server (e.g., http://192.168.1.10:11434)')
    parser.add_argument('--ollama-top-k', type=int,
                        help='Ollama top_k')
    parser.add_argument('--ollama-top-p', type=float,
                        help='Ollama top_p')
    parser.add_argument('--ollama-repeat-penalty', type=float,
                        help='Ollama repeat_penalty')
    parser.add_argument('--ollama-num-predict', type=int,
                        help='Ollama num_predict')
    parser.add_argument('--ollama-stop', action='append',
                        help='Ollama stop sequence (can be set multiple times)')
    
    # Text Processing Parameters
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Size of text chunks for processing')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap between text chunks')
    parser.add_argument('--splitting-strategy', 
                       choices=['recursive', 'character', 'markdown', 'html'],
                       default='recursive',
                       help='Text splitting strategy')
    
    # Q&A Generation Parameters
    parser.add_argument('--questions-per-chunk', type=int, default=3,
                       help='Number of questions to generate per text chunk')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for LLM generation')
    parser.add_argument('--max-tokens', type=int, default=2000,
                       help='Maximum tokens for LLM response')
    
    # Prompt customization
    parser.add_argument('--qa-system-message',
                       help='Optional system message to steer the Q&A generation style')
    parser.add_argument('--qa-extra-instructions',
                       help='Extra instructions appended to the Q&A prompt (e.g., tone, format)')
    parser.add_argument('--qa-prompt-template-file',
                       help='Path to a custom prompt template file (must include {text} and {num_questions})')
    
    # Processing Options
    parser.add_argument('--batch-processing', action='store_true',
                       help='Use batch processing for better efficiency')
    parser.add_argument('--structured-output', action='store_true',
                       help='Use structured output parsing (experimental)')
    
    # Utility options
    parser.add_argument('--list-models', action='store_true',
                       help='List available models for the specified provider')
    parser.add_argument('--list-providers', action='store_true',
                       help='List all available LLM providers')
    parser.add_argument('--show-config', action='store_true',
                       help='Show current processor configuration')
    
    args = parser.parse_args()
    
    # Handle list providers command
    if args.list_providers:
        providers = UnifiedLLMProvider.get_available_providers()
        print("Available LLM providers (LangChain):")
        for provider in providers:
            print(f"  - {provider}")
            models = UnifiedLLMProvider.get_provider_models(provider)
            for model in models[:3]:  # Show first 3 models
                print(f"    • {model}")
            if len(models) > 3:
                print(f"    • ... and {len(models) - 3} more")
        return
    
    # Handle list models command
    if args.list_models:
        models = UnifiedLLMProvider.get_provider_models(args.provider)
        print(f"Available models for {args.provider} (LangChain):")
        for model in models:
            print(f"  - {model}")
        return
    
    log_message("Initializing LangChain Document Analysis App...")
    log_message(f"Using provider: {args.provider} with LangChain integration")
    
    # Build provider-specific kwargs safely
    provider_kwargs = {}
    if args.provider == 'local':
        # Allow .env OLLAMA_BASE_URL as a fallback if flag not supplied
        base_url = args.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
        if base_url:
            provider_kwargs['base_url'] = base_url
        if args.ollama_top_k is not None:
            provider_kwargs['top_k'] = args.ollama_top_k
        if args.ollama_top_p is not None:
            provider_kwargs['top_p'] = args.ollama_top_p
        if args.ollama_repeat_penalty is not None:
            provider_kwargs['repeat_penalty'] = args.ollama_repeat_penalty
        if args.ollama_num_predict is not None:
            provider_kwargs['num_predict'] = args.ollama_num_predict
        if args.ollama_stop:
            provider_kwargs['stop'] = args.ollama_stop

    # Load optional custom prompt template
    custom_prompt_template_str = None
    if args.qa_prompt_template_file:
        p = Path(args.qa_prompt_template_file)
        if p.exists():
            custom_prompt_template_str = p.read_text(encoding='utf-8')
        else:
            log_message(f"Custom prompt template not found: {args.qa_prompt_template_file}")
    
    # Initialize LangChain processor
    try:
        processor = LangChainProcessor(
            incoming_dir=args.incoming_dir,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            questions_per_chunk=args.questions_per_chunk,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            splitting_strategy=args.splitting_strategy,
            use_batch_processing=args.batch_processing,
            # Prompt customization
            qa_system_message=args.qa_system_message,
            qa_additional_instructions=args.qa_extra_instructions,
            qa_custom_template=custom_prompt_template_str,
            # Provider-specific kwargs (e.g., Ollama)
            **provider_kwargs
        )
        
        log_message(f"Initialized LangChain processor with model: {processor.llm_provider.model}")
        
    except Exception as e:
        log_message(f"Error initializing LangChain processor: {e}")
        print("\nTroubleshooting:")
        print(f"1. Make sure you have the API key set for {args.provider}")
        print("   Set it in .env file or use --api-key argument")
        print(f"2. Check that the model name is correct (use --list-models)")
        print(f"3. Verify LangChain packages are installed:")
        print("   pip install langchain langchain-community langchain-openai langchain-anthropic")
        
        if args.provider == 'local':
            print("4. For local provider, make sure Ollama is running (or reachable via IP):")
            print("   ollama serve")
            print(f"   ollama pull {args.model or 'llama3'}")
            if not (args.ollama_base_url or os.getenv('OLLAMA_BASE_URL')):
                print("   Optional: set --ollama-base-url http://<host>:11434 for remote Ollama")
        return
    
    # Show configuration if requested
    if args.show_config:
        config = processor.get_processing_info()
        print("\nCurrent LangChain Processor Configuration:")
        print(f"  Document Loader:")
        print(f"    - Incoming Directory: {config['document_loader']['incoming_dir']}")
        print(f"    - Supported Extensions: {config['document_loader']['supported_extensions']}")
        print(f"    - Available Documents: {config['document_loader']['available_documents']}")
        print(f"  Text Splitter:")
        print(f"    - Chunk Size: {config['text_splitter']['chunk_size']}")
        print(f"    - Chunk Overlap: {config['text_splitter']['chunk_overlap']}")
        print(f"    - Strategy: {config['text_splitter']['splitting_strategy']}")
        print(f"  LLM Provider:")
        print(f"    - Provider: {config['llm_provider']['provider']}")
        print(f"    - Model: {config['llm_provider']['model']}")
        print(f"    - Temperature: {config['llm_provider']['temperature']}")
        print(f"  Q&A Chain:")
        print(f"    - Questions per Chunk: {config['qa_chain']['questions_per_chunk']}")
        print(f"  Processing Options:")
        print(f"    - Batch Processing: {config['processing_options']['use_batch_processing']}")
        return
    
    # Check for documents
    document_files = processor.document_loader.get_documents()
    
    if not document_files:
        log_message("No documents found in incoming directory")
        log_message(f"Place your documents (.pdf, .md, .html, .docx, .txt) in: {args.incoming_dir}")
        
        # Create sample document for demonstration (reuse from original main.py)
        incoming_path = Path(args.incoming_dir)
        incoming_path.mkdir(parents=True, exist_ok=True)
        
        sample_doc = incoming_path / "sample_langchain_guide.md"
        if not sample_doc.exists():
            sample_content = """# LangChain Integration Guide

## Introduction
LangChain is a powerful framework for developing applications powered by language models. This enhanced version of the document analysis app leverages LangChain's robust architecture for superior document processing and Q&A generation.

## Key Improvements

### Unified LLM Interface
The new implementation provides a single, consistent interface for multiple LLM providers:
- OpenAI GPT models
- Anthropic Claude models  
- Google Gemini models
- Local models via Ollama

### Enhanced Document Loading
LangChain's specialized document loaders provide:
- Better metadata extraction
- Improved error handling
- Consistent document structure
- Support for complex document formats

### Advanced Text Splitting
Multiple splitting strategies are now available:
- **Recursive Character Splitter**: Hierarchical splitting for optimal chunks
- **Markdown Header Splitter**: Preserves document structure
- **HTML Header Splitter**: Maintains semantic sections
- **Character Splitter**: Simple splitting for basic needs

### Intelligent Q&A Generation
The new Q&A generation system features:
- Structured output parsing with Pydantic models
- Fallback parsing for robust error handling
- Batch processing for improved efficiency
- Enhanced quality metrics and metadata

## Benefits

### Reliability
- Built-in retry mechanisms
- Robust error handling
- Standardized interfaces
- Battle-tested components

### Scalability  
- Batch processing capabilities
- Efficient memory management
- Streaming response support
- Concurrent request handling

### Extensibility
- Easy to add new providers
- Modular architecture
- Plugin-friendly design
- Community-driven improvements

## Usage Examples

### Basic Usage
```bash
python src/main_langchain.py --provider openai --model gpt-4
```

### Advanced Configuration
```bash
python src/main_langchain.py \\
  --provider claude \\
  --model claude-3-5-sonnet-20241022 \\
  --chunk-size 1200 \\
  --chunk-overlap 300 \\
  --questions-per-chunk 5 \\
  --splitting-strategy recursive \\
  --batch-processing \\
  --temperature 0.8
```

### Local Model Usage
```bash
python src/main_langchain.py \\
  --provider local \\
  --model llama3 \\
  --batch-processing
```

## Conclusion

The LangChain integration represents a significant upgrade to the document analysis capabilities, providing a more robust, scalable, and maintainable solution for training data generation.
"""
            sample_doc.write_text(sample_content)
            log_message(f"Created LangChain sample document: {sample_doc}")
            log_message("Run the command again to process this sample document")
        
        return
    
    # Generate training data
    log_message("Generating Q&A training data with LangChain...")
    log_message(f"Configuration: {args.questions_per_chunk} questions per chunk, "
               f"{args.temperature} temperature, {args.splitting_strategy} splitting")
    
    training_data = processor.generate_qa_training_data(output_file=args.output_file)
    
    if not training_data:
        log_message("No training data was generated")
        return
    
    # Generate summary report
    summary = {
        'configuration': {
            'framework': 'langchain',
            'provider': args.provider,
            'model': processor.llm_provider.model,
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap,
            'splitting_strategy': args.splitting_strategy,
            'questions_per_chunk': args.questions_per_chunk,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'batch_processing': args.batch_processing
        },
        'results': {
            'total_documents': training_data['metadata']['num_documents'],
            'total_qa_pairs': training_data['metadata']['total_qa_pairs'],
            'documents_processed': [doc['file_info']['file_name'] 
                                  for doc in training_data['documents']],
            'output_file': args.output_file,
            'quality_metrics': training_data['metadata'].get('quality_metrics', {})
        }
    }
    
    summary_file = args.output_file.replace('.json', '_summary.json')
    save_results(summary, summary_file)
    
    log_message(f"LangChain training data generation complete!")
    log_message(f"Generated {summary['results']['total_qa_pairs']} Q&A pairs "
               f"from {summary['results']['total_documents']} documents")
    log_message(f"Training data saved to: {args.output_file}")
    log_message(f"Summary saved to: {summary_file}")
    
    # Show sample Q&A pairs
    if training_data['training_pairs']:
        log_message("\nSample Q&A pairs:")
        for i, qa in enumerate(training_data['training_pairs'][:2]):
            print(f"\nSample {i+1}:")
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer'][:150]}{'...' if len(qa['answer']) > 150 else ''}")
            print(f"Source: {qa['file_name']} (chunk {qa['chunk_id']})")
            if qa.get('section_header'):
                print(f"Section: {qa['section_header']}")
    
    # Show quality metrics if available
    if 'quality_metrics' in training_data['metadata']:
        metrics = training_data['metadata']['quality_metrics']
        log_message("\nQuality Metrics:")
        print(f"  Average Question Length: {metrics.get('avg_question_length', 0):.1f} chars")
        print(f"  Average Answer Length: {metrics.get('avg_answer_length', 0):.1f} chars")
        print(f"  Unique Source Files: {metrics.get('unique_source_files', 0)}")


if __name__ == "__main__":
    main()