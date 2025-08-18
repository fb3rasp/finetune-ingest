import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List

# Ensure local src path and project root for importing `common`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from common.document_processing import LangChainDocumentLoader, EnhancedTextSplitter
from common.utils.helpers import log_message, save_json_atomic, load_json_if_exists


class DocumentChunker:
    def __init__(
        self,
        incoming_dir: str = "/data/incoming",
        chunks_dir: str = "/data/chunks",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitting_strategy: str = "recursive",
    ):
        self.incoming_dir = incoming_dir
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.splitting_strategy = splitting_strategy
        
        self.document_loader = LangChainDocumentLoader(incoming_dir)
        self.text_splitter = EnhancedTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def chunk_single_document(self, file_path: Path, resume: bool = False) -> bool:
        """Process a single document and create its chunk file."""
        log_message(f"Chunking document: {file_path.name}")
        
        # Create output file path
        chunk_file = self.chunks_dir / f"{file_path.stem}_chunks.json"
        
        # Check if already processed
        if resume and chunk_file.exists():
            existing_data = load_json_if_exists(str(chunk_file))
            if existing_data and existing_data.get('status') == 'completed':
                log_message(f"Skipping {file_path.name} - already chunked")
                return True

        try:
            # Load document
            documents_obj, metadata = self.document_loader.load_document(file_path)
            if not documents_obj:
                log_message(f"Failed to load document: {file_path.name}")
                return False

            # Enhanced metadata
            enhanced_metadata = {
                'file_name': file_path.name,
                'source_file': str(file_path),
                'file_type': file_path.suffix.lower(),
                'file_size': file_path.stat().st_size,
                'processed_at': datetime.now().isoformat(),
            }

            # Split documents into chunks
            try:
                chunks = self.text_splitter.split_documents(
                    documents_obj, 
                    strategy=self.splitting_strategy
                )
            except Exception as e:
                log_message(f"Error splitting document {file_path.name}: {e}")
                # Fallback to simple text splitting
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                except ImportError:
                    # If LangChain is not available, create simple chunks
                    log_message("LangChain not available, using simple chunking")
                    text = "\n\n".join(d.page_content for d in documents_obj)
                    chunk_size = self.text_splitter.chunk_size
                    chunks = []
                    for i in range(0, len(text), chunk_size):
                        chunk_text = text[i:i + chunk_size]
                        chunks.append({
                            'text': chunk_text,
                            'start_char': i,
                            'end_char': i + len(chunk_text),
                            'chunk_id': f"0_{len(chunks)}",
                            'chunk_index': len(chunks),
                            'document_index': 0,
                            'metadata': enhanced_metadata.copy(),
                            'word_count': len(chunk_text.split()),
                            'char_count': len(chunk_text),
                        })
                else:
                    fallback_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.text_splitter.chunk_size,
                        chunk_overlap=self.text_splitter.chunk_overlap
                    )
                    text = "\n\n".join(d.page_content for d in documents_obj)
                    text_chunks = fallback_splitter.split_text(text)
                    chunks = []
                    for i, chunk_text in enumerate(text_chunks):
                        chunks.append({
                            'text': chunk_text,
                            'start_char': i * (self.text_splitter.chunk_size - self.text_splitter.chunk_overlap),
                            'end_char': i * (self.text_splitter.chunk_size - self.text_splitter.chunk_overlap) + len(chunk_text),
                            'chunk_id': f"0_{i}",
                            'chunk_index': i,
                            'document_index': 0,
                            'metadata': enhanced_metadata.copy(),
                            'word_count': len(chunk_text.split()),
                            'char_count': len(chunk_text),
                        })

            # Create document chunk data
            chunk_data = {
                'status': 'completed',
                'metadata': enhanced_metadata,
                'processing_info': {
                    'splitting_strategy': self.splitting_strategy,
                    'chunk_size': self.text_splitter.chunk_size,
                    'chunk_overlap': self.text_splitter.chunk_overlap,
                    'total_chunks': len(chunks),
                },
                'chunks': chunks,
                'qa_generation_status': {
                    'completed_chunks': [],
                    'total_chunks': len(chunks),
                    'last_processed_chunk': -1,
                    'is_complete': False,
                }
            }

            # Save chunk file
            save_json_atomic(chunk_data, str(chunk_file))
            log_message(f"Created {len(chunks)} chunks for {file_path.name}")
            return True

        except Exception as e:
            log_message(f"Error processing {file_path.name}: {e}")
            return False

    def chunk_all_documents(self, resume: bool = False) -> Dict:
        """Process all documents in the incoming directory."""
        document_files = self.document_loader.get_documents()
        
        if not document_files:
            log_message(f"No documents found in {self.incoming_dir}")
            return {
                'total_documents': 0,
                'successful_chunks': 0,
                'failed_chunks': 0,
                'chunk_files_created': []
            }

        successful = 0
        failed = 0
        chunk_files = []

        for file_path in document_files:
            if self.chunk_single_document(file_path, resume=resume):
                successful += 1
                chunk_files.append(f"{file_path.stem}_chunks.json")
            else:
                failed += 1

        summary = {
            'total_documents': len(document_files),
            'successful_chunks': successful,
            'failed_chunks': failed,
            'chunk_files_created': chunk_files,
            'chunks_directory': str(self.chunks_dir),
            'processed_at': datetime.now().isoformat(),
        }

        # Save summary
        summary_file = self.chunks_dir / "chunking_summary.json"
        save_json_atomic(summary, str(summary_file))
        
        log_message(f"Chunking complete: {successful}/{len(document_files)} documents processed")
        log_message(f"Chunk files saved to: {self.chunks_dir}")
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

    parser = argparse.ArgumentParser(description='Chunk documents for Q&A training data generation')

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
    parser.add_argument('--chunks-dir', 
                       default=get_env_default('GENERATOR_PROCESS_DIR', '/data/chunks'),
                       help='Directory to store chunk files')

    # Text processing arguments with env defaults
    parser.add_argument('--chunk-size', type=int, 
                       default=get_env_default('GENERATOR_CHUNK_SIZE', 1000, int))
    parser.add_argument('--chunk-overlap', type=int, 
                       default=get_env_default('GENERATOR_CHUNK_OVERLAP', 200, int))
    parser.add_argument('--splitting-strategy', 
                       choices=['recursive', 'character', 'markdown', 'html'], 
                       default=get_env_default('GENERATOR_SPLITTING_STRATEGY', 'recursive'))

    # Processing mode arguments
    parser.add_argument('--resume', action='store_true',
                       default=get_env_default('GENERATOR_RESUME', False, bool),
                       help='Resume chunking, skip already processed documents')

    args = parser.parse_args()

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
    if args.chunks_dir.startswith('/data'):
        args.chunks_dir = ensure_writable_dir(args.chunks_dir)

    # Create chunker
    chunker = DocumentChunker(
        incoming_dir=args.incoming_dir,
        chunks_dir=args.chunks_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitting_strategy=args.splitting_strategy,
    )

    # Process documents
    summary = chunker.chunk_all_documents(resume=args.resume)
    
    log_message("Document chunking completed!")
    log_message(f"Configuration: chunk_size={args.chunk_size}, overlap={args.chunk_overlap}, strategy={args.splitting_strategy}")
    log_message(f"Results: {summary['successful_chunks']}/{summary['total_documents']} documents chunked")


if __name__ == "__main__":
    main()
