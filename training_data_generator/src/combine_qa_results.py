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

from common.utils.helpers import log_message, save_json_atomic, load_json_if_exists


class QACombiner:
    def __init__(self, qa_dir: str = "/data/qa_results", output_file: str = "/data/results/training_data.json"):
        self.qa_dir = Path(qa_dir)
        self.output_file = output_file

    def get_qa_files(self) -> List[Path]:
        """Get all Q&A result files."""
        qa_files = list(self.qa_dir.glob("*_qa.json"))
        log_message(f"Found {len(qa_files)} Q&A files to combine")
        return qa_files

    def combine_qa_files(self) -> Dict:
        """Combine all individual Q&A files into a single training data file."""
        qa_files = self.get_qa_files()
        
        if not qa_files:
            log_message(f"No Q&A files found in {self.qa_dir}")
            return {}

        # Initialize combined data structure
        combined_data = {
            'metadata': {
                'generated_by': 'qa_combiner',
                'generated_at': datetime.now().isoformat(),
                'total_qa_pairs': 0,
                'num_documents': 0,
                'source_qa_files': [],
            },
            'documents': [],
            'training_pairs': [],
        }

        all_training_pairs = []
        documents_info = []

        for qa_file in qa_files:
            log_message(f"Processing {qa_file.name}")
            
            qa_data = load_json_if_exists(str(qa_file))
            if not qa_data:
                log_message(f"Failed to load {qa_file.name}")
                continue

            if qa_data.get('status') != 'completed':
                log_message(f"Skipping {qa_file.name} - not completed (status: {qa_data.get('status')})")
                continue

            # Add training pairs
            pairs = qa_data.get('training_pairs', [])
            all_training_pairs.extend(pairs)

            # Add document info
            document_info = {
                'file_info': qa_data.get('metadata', {}),
                'processing_info': qa_data.get('processing_info', {}),
                'qa_generation_info': qa_data.get('qa_generation_info', {}),
                'summary': qa_data.get('summary', {}),
                'qa_pairs_count': len(pairs),
                'source_file': qa_file.name,
            }
            documents_info.append(document_info)
            combined_data['metadata']['source_qa_files'].append(qa_file.name)

        # Update combined data
        combined_data['training_pairs'] = all_training_pairs
        combined_data['documents'] = documents_info
        combined_data['metadata']['total_qa_pairs'] = len(all_training_pairs)
        combined_data['metadata']['num_documents'] = len(documents_info)

        # Add processing configuration from first document (assuming consistent config)
        if documents_info:
            first_doc = documents_info[0]
            combined_data['metadata']['processing_config'] = {
                'chunk_size': first_doc.get('processing_info', {}).get('chunk_size'),
                'chunk_overlap': first_doc.get('processing_info', {}).get('chunk_overlap'),
                'splitting_strategy': first_doc.get('processing_info', {}).get('splitting_strategy'),
                'questions_per_chunk': first_doc.get('qa_generation_info', {}).get('questions_per_chunk'),
                'temperature': first_doc.get('qa_generation_info', {}).get('temperature'),
                'max_tokens': first_doc.get('qa_generation_info', {}).get('max_tokens'),
                'llm_provider': first_doc.get('qa_generation_info', {}).get('provider'),
                'model_used': first_doc.get('qa_generation_info', {}).get('model'),
            }

        # Save combined file
        save_json_atomic(combined_data, self.output_file)
        
        log_message(f"Combined training data saved to: {self.output_file}")
        log_message(f"Total Q&A pairs: {len(all_training_pairs)}")
        log_message(f"Total documents: {len(documents_info)}")

        return combined_data


def main():
    # Load environment from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to default behavior

    parser = argparse.ArgumentParser(description='Combine individual Q&A files into a single training data file')

    # Helper function to get env defaults
    def get_env_default(key: str, default):
        return os.getenv(key, default)

    # Directory arguments with env defaults
    parser.add_argument('--qa-dir', 
                       default=get_env_default('GENERATOR_OUTPUT_DIR', '/data/qa_results'),
                       help='Directory containing Q&A result files')
    parser.add_argument('--output-file', 
                       default=get_env_default('GENERATOR_OUTPUT_FILE', '/data/results/training_data.json'),
                       help='Output file for combined training data')

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

    if args.qa_dir.startswith('/data'):
        args.qa_dir = ensure_writable_dir(args.qa_dir)
    
    output_parent = str(Path(args.output_file).parent)
    if output_parent.startswith('/data'):
        output_parent = ensure_writable_dir(output_parent)
        args.output_file = str(Path(output_parent) / Path(args.output_file).name)

    # Create combiner and process
    combiner = QACombiner(qa_dir=args.qa_dir, output_file=args.output_file)
    result = combiner.combine_qa_files()
    
    if result:
        log_message("Q&A combination completed successfully!")
    else:
        log_message("No data to combine.")


if __name__ == "__main__":
    main()
