import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from langchain_processing.qa_validator import QAValidator
from common.llm.llm_providers import UnifiedLLMProvider
from common.utils.helpers import log_message


def main():
    # Load environment from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to default behavior
    
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
    
    parser = argparse.ArgumentParser(description='Validate Q&A training data for factual accuracy')
    
    # Input/Output arguments with env defaults
    parser.add_argument('--input', '-i', 
                       default=get_env_default('VALIDATOR_INPUT', '/data/results/training_data.json'),
                       help='Input training data JSON file to validate')
    parser.add_argument('--output', '-o', 
                       default=get_env_default('VALIDATOR_OUTPUT', '/data/results/training_data_validation_report.json'),
                       help='Output validation report JSON file')
    parser.add_argument('--filtered-output', 
                       default=get_env_default('VALIDATOR_FILTERED_OUTPUT', None),
                       help='Output filtered training data (removes low-scoring pairs)')
    
    # Validator configuration with env defaults
    parser.add_argument('--validator-provider', choices=['openai', 'claude', 'gemini', 'local'], 
                       default=get_env_default('VALIDATOR_PROVIDER', 'openai'))
    parser.add_argument('--validator-model',
                       default=get_env_default('VALIDATOR_MODEL', None))
    parser.add_argument('--validator-api-key')
    parser.add_argument('--temperature', type=float, 
                       default=get_env_default('VALIDATOR_TEMPERATURE', 0.1, float))
    parser.add_argument('--ollama-base-url',
                       default=get_env_default('OLLAMA_BASE_URL', None))
    parser.add_argument('--threshold', type=float, 
                       default=get_env_default('VALIDATOR_THRESHOLD', 8.0, float))
    parser.add_argument('--filter-threshold', type=float, 
                       default=get_env_default('VALIDATOR_FILTER_THRESHOLD', 7.0, float))
    parser.add_argument('--batch-size', type=int, 
                       default=get_env_default('VALIDATOR_BATCH_SIZE', 10, int))
    parser.add_argument('--quiet', '-q', action='store_true',
                       default=get_env_default('VALIDATOR_QUIET', False, bool))
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging showing progress per request')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    provider_kwargs = {}
    if args.validator_provider == 'local':
        base_url = args.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
        if base_url:
            provider_kwargs['base_url'] = base_url

    validator = QAValidator(
        provider=args.validator_provider,
        model=args.validator_model,
        api_key=args.validator_api_key,
        temperature=args.temperature,
        validation_threshold=args.threshold,
        batch_size=args.batch_size,
        verbose=args.verbose,
        **provider_kwargs,
    )

    report = validator.validate_training_data(training_data_path=args.input, output_path=args.output)
    if not args.quiet:
        stats = report['summary_statistics']
        log_message(f"Validated {stats['total_qa_pairs']} pairs; PASS: {stats['pass_count']} ({stats['pass_rate']:.1%})")
        log_message(f"Report saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


