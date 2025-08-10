import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure local src path
# Ensure local src path and project root for importing `common`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from langchain_processing import LangChainProcessor
from common.utils.helpers import log_message, save_results


def main():
    # Load environment from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to default behavior

    parser = argparse.ArgumentParser(description='Generate training data for LLM fine-tuning')

    # Helper function to get env defaults with type conversion
    def get_env_default(key: str, default, convert_type=None):
        value = os.getenv(key, default)
        if convert_type and value != default:
            try:
                if convert_type == bool:
                    return value.lower() in ('true', '1', 'yes', 'on')
                return convert_type(value)
            except (ValueError, TypeError):
                return default
        return value

    # Directory arguments with env defaults
    parser.add_argument('--incoming-dir', 
                       default=get_env_default('GENERATOR_INCOMING_DIR', '/data/incoming'),
                       help='Directory containing source documents')
    parser.add_argument('--process-dir', 
                       default=get_env_default('GENERATOR_PROCESS_DIR', '/data/process'),
                       help='Directory to store caches and intermediate data')
    parser.add_argument('--output-file', 
                       default=get_env_default('GENERATOR_OUTPUT_FILE', '/data/results/training_data.json'),
                       help='Output file for training data')

    # LLM provider arguments with env defaults
    parser.add_argument('--provider', 
                       choices=['openai', 'claude', 'gemini', 'local'], 
                       default=get_env_default('GENERATOR_PROVIDER', 'openai'))
    parser.add_argument('--model', 
                       default=get_env_default('GENERATOR_MODEL', None))
    parser.add_argument('--api-key')
    parser.add_argument('--ollama-base-url', 
                       default=get_env_default('OLLAMA_BASE_URL', None))
    parser.add_argument('--ollama-top-k', type=int, 
                       default=get_env_default('OLLAMA_TOP_K', None, int))
    parser.add_argument('--ollama-top-p', type=float,
                       default=get_env_default('OLLAMA_TOP_P', None, float))
    parser.add_argument('--ollama-repeat-penalty', type=float,
                       default=get_env_default('OLLAMA_REPEAT_PENALTY', None, float))
    parser.add_argument('--ollama-num-predict', type=int,
                       default=get_env_default('OLLAMA_NUM_PREDICT', None, int))
    parser.add_argument('--ollama-stop', action='append')

    # Text processing arguments with env defaults
    parser.add_argument('--chunk-size', type=int, 
                       default=get_env_default('GENERATOR_CHUNK_SIZE', 1000, int))
    parser.add_argument('--chunk-overlap', type=int, 
                       default=get_env_default('GENERATOR_CHUNK_OVERLAP', 200, int))
    parser.add_argument('--splitting-strategy', 
                       choices=['recursive', 'character', 'markdown', 'html'], 
                       default=get_env_default('GENERATOR_SPLITTING_STRATEGY', 'recursive'))
    parser.add_argument('--questions-per-chunk', type=int, 
                       default=get_env_default('GENERATOR_QUESTIONS_PER_CHUNK', 3, int))
    parser.add_argument('--temperature', type=float, 
                       default=get_env_default('GENERATOR_TEMPERATURE', 0.7, float))
    parser.add_argument('--max-tokens', type=int, 
                       default=get_env_default('GENERATOR_MAX_TOKENS', 2000, int))

    # Advanced prompt arguments
    parser.add_argument('--qa-system-message')
    parser.add_argument('--qa-extra-instructions')
    parser.add_argument('--qa-prompt-template-file')

    # Processing mode arguments with env defaults
    parser.add_argument('--batch-processing', action='store_true',
                       default=get_env_default('GENERATOR_BATCH_PROCESSING', False, bool))
    parser.add_argument('--resume', action='store_true',
                       default=get_env_default('GENERATOR_RESUME', False, bool))

    args = parser.parse_args()

    provider_kwargs = {}
    if args.provider == 'local':
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

    custom_prompt = None
    if args.qa_prompt_template_file:
        p = Path(args.qa_prompt_template_file)
        if p.exists():
            custom_prompt = p.read_text(encoding='utf-8')

    # If default /data paths are not writable, auto-fallback to local ./data
    def ensure_writable_dir(path_str: str) -> str:
        p = Path(path_str)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return path_str
        except Exception:
            local = Path.cwd() / 'data' / p.name
            local.mkdir(parents=True, exist_ok=True)
            return str(local)

    if args.incoming_dir.startswith('/data'):
        args.incoming_dir = ensure_writable_dir(args.incoming_dir)
    if args.process_dir.startswith('/data'):
        args.process_dir = ensure_writable_dir(args.process_dir)
    out_parent = str(Path(args.output_file).parent)
    if out_parent.startswith('/data'):
        out_parent = ensure_writable_dir(out_parent)
        args.output_file = str(Path(out_parent) / Path(args.output_file).name)

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
        process_dir=args.process_dir,
        results_file=args.output_file,
        qa_system_message=args.qa_system_message,
        qa_additional_instructions=args.qa_extra_instructions,
        qa_custom_template=custom_prompt,
        **provider_kwargs,
    )

    document_files = processor.document_loader.get_documents()
    if not document_files:
        log_message(f"No documents found in {args.incoming_dir}")
        return

    log_message("Generating Q&A training data...")
    data = processor.generate_qa_training_data(output_file=args.output_file, resume=args.resume)
    if not data:
        log_message("No training data generated")
        return

    summary = {
        'configuration': {
            'provider': args.provider,
            'model': processor.llm_provider.model,
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap,
            'splitting_strategy': args.splitting_strategy,
            'questions_per_chunk': args.questions_per_chunk,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'batch_processing': args.batch_processing,
        },
        'results': {
            'total_documents': data['metadata']['num_documents'],
            'total_qa_pairs': data['metadata']['total_qa_pairs'],
            'documents_processed': [doc['file_info']['file_name'] for doc in data['documents']],
            'output_file': args.output_file,
            'quality_metrics': data['metadata'].get('quality_metrics', {}),
        },
    }
    summary_file = args.output_file.replace('.json', '_summary.json')
    save_results(summary, summary_file)
    log_message(f"Training data saved to: {args.output_file}")
    log_message(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()


