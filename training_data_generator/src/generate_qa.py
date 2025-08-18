import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Optional

# Ensure local src path and project root for importing `common`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from common.llm import UnifiedLLMProvider, LLMProvider, QAGenerationChain
from common.utils.helpers import log_message, save_json_atomic, load_json_if_exists


class QAGenerator:
    def __init__(
        self,
        chunks_dir: str = "/data/chunks",
        qa_dir: str = "/data/qa_results",
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        questions_per_chunk: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_batch_processing: bool = False,
        qa_system_message: Optional[str] = None,
        qa_additional_instructions: Optional[str] = None,
        qa_custom_template: Optional[str] = None,
        **kwargs,
    ):
        self.chunks_dir = Path(chunks_dir)
        self.qa_dir = Path(qa_dir)
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self.use_batch_processing = use_batch_processing

        # Initialize LLM provider
        self.llm_provider = UnifiedLLMProvider(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Initialize QA chain
        self.qa_chain = QAGenerationChain(
            llm_provider=self.llm_provider,
            questions_per_chunk=questions_per_chunk,
            system_message=qa_system_message,
            extra_instructions=qa_additional_instructions,
            custom_template=qa_custom_template,
        )

    def get_chunk_files(self) -> List[Path]:
        """Get all chunk files from the chunks directory."""
        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        log_message(f"Found {len(chunk_files)} chunk files to process")
        return chunk_files

    def generate_qa_for_document(self, chunk_file: Path, resume: bool = False) -> bool:
        """Generate Q&A pairs for a single document's chunks."""
        log_message(f"Processing chunk file: {chunk_file.name}")

        # Load chunk data
        chunk_data = load_json_if_exists(str(chunk_file))
        if not chunk_data:
            log_message(f"Failed to load chunk file: {chunk_file.name}")
            return False

        if chunk_data.get('status') != 'completed':
            log_message(f"Chunk file {chunk_file.name} is not marked as completed")
            return False

        # Create output file path
        qa_file = self.qa_dir / chunk_file.name.replace("_chunks.json", "_qa.json")

        # Load existing Q&A data if resuming
        qa_data = None
        if resume and qa_file.exists():
            qa_data = load_json_if_exists(str(qa_file))

        # Initialize Q&A data structure
        if qa_data is None:
            qa_data = {
                'metadata': chunk_data['metadata'].copy(),
                'processing_info': chunk_data['processing_info'].copy(),
                'qa_generation_info': {
                    'provider': self.llm_provider.provider.value,
                    'model': self.llm_provider.model,
                    'questions_per_chunk': self.qa_chain.questions_per_chunk,
                    'temperature': self.llm_provider.temperature,
                    'max_tokens': self.llm_provider.max_tokens,
                    'generated_at': datetime.now().isoformat(),
                },
                'chunks': [],
                'training_pairs': [],
                'status': 'in_progress',
                'completed_chunks': [],
                'failed_chunks': [],
            }

        chunks = chunk_data['chunks']
        total_chunks = len(chunks)
        completed_chunks = set(qa_data.get('completed_chunks', []))
        
        log_message(f"Processing {total_chunks} chunks ({len(completed_chunks)} already completed)")

        try:
            if self.use_batch_processing:
                # Batch processing mode
                chunks_to_process = [
                    chunk for i, chunk in enumerate(chunks) 
                    if i not in completed_chunks
                ]
                
                if chunks_to_process:
                    def on_chunk_done(pairs_for_chunk: List[Dict], chunk_idx: int):
                        qa_data['training_pairs'].extend(pairs_for_chunk)
                        qa_data['completed_chunks'].append(chunk_idx)
                        # Save progress after each chunk
                        save_json_atomic(qa_data, str(qa_file))
                        log_message(f"Completed chunk {chunk_idx + 1}/{total_chunks}")

                    # Process remaining chunks in batch
                    for i, chunk in enumerate(chunks_to_process):
                        original_idx = chunks.index(chunk)
                        if original_idx in completed_chunks:
                            continue
                            
                        try:
                            pairs = self.qa_chain.generate_qa_pairs(chunk, qa_data['metadata'])
                            if pairs:
                                on_chunk_done(pairs, original_idx)
                            else:
                                qa_data['failed_chunks'].append(original_idx)
                                log_message(f"No Q&A pairs generated for chunk {original_idx + 1}")
                        except Exception as e:
                            log_message(f"Error processing chunk {original_idx + 1}: {e}")
                            qa_data['failed_chunks'].append(original_idx)
                            
            else:
                # Sequential processing mode
                for i, chunk in enumerate(chunks):
                    if i in completed_chunks:
                        continue

                    try:
                        log_message(f"Processing chunk {i + 1}/{total_chunks}")
                        pairs = self.qa_chain.generate_qa_pairs(chunk, qa_data['metadata'])
                        
                        if pairs:
                            qa_data['training_pairs'].extend(pairs)
                            qa_data['completed_chunks'].append(i)
                            log_message(f"Generated {len(pairs)} Q&A pairs for chunk {i + 1}")
                        else:
                            qa_data['failed_chunks'].append(i)
                            log_message(f"No Q&A pairs generated for chunk {i + 1}")

                        # Save progress after each chunk (crucial for resume functionality)
                        save_json_atomic(qa_data, str(qa_file))

                    except KeyboardInterrupt:
                        log_message(f"Interrupted during chunk {i + 1}. Progress saved.")
                        qa_data['status'] = 'interrupted'
                        save_json_atomic(qa_data, str(qa_file))
                        return False
                    except Exception as e:
                        log_message(f"Error processing chunk {i + 1}: {e}")
                        qa_data['failed_chunks'].append(i)
                        # Continue with next chunk
                        continue

            # Mark as completed
            qa_data['status'] = 'completed'
            qa_data['completion_time'] = datetime.now().isoformat()
            qa_data['summary'] = {
                'total_chunks': total_chunks,
                'completed_chunks': len(qa_data['completed_chunks']),
                'failed_chunks': len(qa_data['failed_chunks']),
                'total_qa_pairs': len(qa_data['training_pairs']),
            }

            save_json_atomic(qa_data, str(qa_file))
            log_message(f"Completed Q&A generation for {chunk_file.name}")
            log_message(f"Generated {len(qa_data['training_pairs'])} Q&A pairs from {len(qa_data['completed_chunks'])}/{total_chunks} chunks")
            return True

        except Exception as e:
            log_message(f"Fatal error processing {chunk_file.name}: {e}")
            qa_data['status'] = 'failed'
            qa_data['error'] = str(e)
            save_json_atomic(qa_data, str(qa_file))
            return False

    def generate_qa_for_all_documents(self, resume: bool = False) -> Dict:
        """Generate Q&A pairs for all chunk files."""
        chunk_files = self.get_chunk_files()
        
        if not chunk_files:
            log_message(f"No chunk files found in {self.chunks_dir}")
            return {
                'total_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'qa_files_created': []
            }

        successful = 0
        failed = 0
        qa_files = []

        for chunk_file in chunk_files:
            try:
                if self.generate_qa_for_document(chunk_file, resume=resume):
                    successful += 1
                    qa_files.append(chunk_file.name.replace("_chunks.json", "_qa.json"))
                else:
                    failed += 1
            except KeyboardInterrupt:
                log_message("Process interrupted by user")
                break
            except Exception as e:
                log_message(f"Error processing {chunk_file.name}: {e}")
                failed += 1

        summary = {
            'total_files': len(chunk_files),
            'successful_files': successful,
            'failed_files': failed,
            'qa_files_created': qa_files,
            'qa_directory': str(self.qa_dir),
            'processed_at': datetime.now().isoformat(),
        }

        # Save summary
        summary_file = self.qa_dir / "qa_generation_summary.json"
        save_json_atomic(summary, str(summary_file))
        
        log_message(f"Q&A generation complete: {successful}/{len(chunk_files)} files processed")
        log_message(f"Q&A files saved to: {self.qa_dir}")
        log_message(f"Summary saved to: {summary_file}")
        
        return summary


def main():
    # Load environment from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to default behavior

    parser = argparse.ArgumentParser(description='Generate Q&A training data from document chunks')

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
    parser.add_argument('--chunks-dir', 
                       default=get_env_default('GENERATOR_PROCESS_DIR', '/data/chunks'),
                       help='Directory containing chunk files')
    parser.add_argument('--qa-dir', 
                       default=get_env_default('GENERATOR_OUTPUT_DIR', '/data/qa_results'),
                       help='Directory to store Q&A result files')

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

    # Q&A generation arguments with env defaults
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
                       default=get_env_default('GENERATOR_RESUME', False, bool),
                       help='Resume Q&A generation from where it was interrupted')

    args = parser.parse_args()

    # Configure provider-specific kwargs
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

    # Load custom prompt template if provided
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

    if args.chunks_dir.startswith('/data'):
        args.chunks_dir = ensure_writable_dir(args.chunks_dir)
    if args.qa_dir.startswith('/data'):
        args.qa_dir = ensure_writable_dir(args.qa_dir)

    # Create Q&A generator
    generator = QAGenerator(
        chunks_dir=args.chunks_dir,
        qa_dir=args.qa_dir,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        questions_per_chunk=args.questions_per_chunk,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_batch_processing=args.batch_processing,
        qa_system_message=args.qa_system_message,
        qa_additional_instructions=args.qa_extra_instructions,
        qa_custom_template=custom_prompt,
        **provider_kwargs,
    )

    # Generate Q&A pairs
    summary = generator.generate_qa_for_all_documents(resume=args.resume)
    
    log_message("Q&A generation completed!")
    log_message(f"Configuration: provider={args.provider}, model={generator.llm_provider.model}")
    log_message(f"Results: {summary['successful_files']}/{summary['total_files']} files processed")


if __name__ == "__main__":
    main()
