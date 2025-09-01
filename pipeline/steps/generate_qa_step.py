"""
Q&A generation step for the training data pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.llm import UnifiedLLMProvider, QAGenerationChain
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class GenerateQAStep(BaseStep):
    """Step 2: Generate Q&A pairs from document chunks."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.llm_provider = UnifiedLLMProvider(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning=config.reasoning
        )
        self.qa_chain = QAGenerationChain(
            llm_provider=self.llm_provider,
            questions_per_chunk=config.questions_per_chunk
        )
    
    def check_prerequisites(self) -> bool:
        """Check if chunk files exist."""
        chunks_dir = Path(self.config.chunks_dir)
        if not chunks_dir.exists():
            self.log(f"Chunks directory does not exist: {chunks_dir}", "error")
            return False
        
        chunk_files = list(chunks_dir.glob("*.json"))
        if not chunk_files:
            self.log(f"No JSON files found in {chunks_dir}", "error")
            return False
        
        return True
    
    def run(self, resume: bool = False) -> bool:
        """Run the Q&A generation step."""
        self.log("Starting Q&A generation step...")
        
        if not self.check_prerequisites():
            return False
        
        # Ensure output directory exists
        qa_dir = Path(self.config.qa_dir)
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        # Get chunk files
        chunks_dir = Path(self.config.chunks_dir)
        chunk_files = list(chunks_dir.glob("*.json"))
        
        self.log(f"Found {len(chunk_files)} JSON files to process")
        
        all_qa_pairs = []
        total_processed = 0
        total_skipped = 0
        
        for chunk_file in chunk_files:
            try:
                # Create output file path
                qa_file = qa_dir / f"{chunk_file.stem}_qa.json"
                
                # Check if already processed
                if resume and qa_file.exists():
                    existing_data = load_json_if_exists(str(qa_file))
                    if existing_data and existing_data.get('status') == 'completed':
                        self.log(f"Skipping {chunk_file.name} - already processed")
                        # Add existing Q&A pairs to collection
                        if 'training_pairs' in existing_data:
                            all_qa_pairs.extend(existing_data['training_pairs'])
                        total_skipped += 1
                        continue
                
                # Load chunk data
                self.log(f"Processing: {chunk_file.name}")
                chunk_data = load_json_if_exists(str(chunk_file))
                
                if not chunk_data or 'chunks' not in chunk_data:
                    self.log(f"Invalid chunk data in {chunk_file.name}")
                    continue
                
                # Initialize or load existing Q&A data for this file
                qa_data = None
                if resume and qa_file.exists():
                    qa_data = load_json_if_exists(str(qa_file))
                
                if qa_data is None:
                    qa_data = {
                        "metadata": {
                            "source_chunk_file": str(chunk_file),
                            "llm_provider": self.config.llm_provider,
                            "llm_model": self.config.llm_model,
                            "questions_per_chunk": self.config.questions_per_chunk,
                            "processed_at": str(datetime.now()),
                            "status": "in_progress"
                        },
                        "training_pairs": [],
                        "completed_chunks": [],
                        "failed_chunks": []
                    }
                
                # Generate Q&A pairs for each chunk
                total_chunks = len(chunk_data['chunks'])
                self.log(f"Processing {total_chunks} chunks from {chunk_file.name}")
                
                for chunk_idx, chunk in enumerate(chunk_data['chunks']):
                    # Skip if already completed
                    if chunk_idx in qa_data.get('completed_chunks', []):
                        self.log(f"Chunk {chunk_idx + 1}/{total_chunks}: Already completed, skipping")
                        continue
                    
                    try:
                        # Pass the entire chunk object and the document's metadata
                        pairs = self.qa_chain.generate_qa_pairs(chunk, chunk_data['metadata'])
                        
                        # Log progress for each chunk
                        chunk_num = chunk_idx + 1
                        self.log(f"Chunk {chunk_num}/{total_chunks}: Generated {len(pairs)} Q&A pairs")
                        
                        # Add chunk metadata to pairs
                        for pair in pairs:
                            pair['chunk_id'] = chunk['chunk_id']
                            pair['source_file'] = chunk_data['metadata']['source_file']
                        
                        # Add pairs to the collection
                        qa_data['training_pairs'].extend(pairs)
                        qa_data['completed_chunks'].append(chunk_idx)
                        
                        # Save progress after each chunk
                        save_json_atomic(qa_data, str(qa_file))
                        
                    except Exception as e:
                        chunk_num = chunk_idx + 1
                        self.log(f"Chunk {chunk_num}/{total_chunks}: Error generating Q&A - {e}")
                        qa_data['failed_chunks'].append(chunk_idx)
                        # Save progress even on error
                        save_json_atomic(qa_data, str(qa_file))
                        continue
                
                # Mark as completed
                qa_data['metadata']['status'] = 'completed'
                qa_data['metadata']['completion_time'] = str(datetime.now())
                qa_data['metadata']['qa_count'] = len(qa_data['training_pairs'])
                qa_data['metadata']['completed_chunks_count'] = len(qa_data['completed_chunks'])
                qa_data['metadata']['failed_chunks_count'] = len(qa_data['failed_chunks'])
                
                # Final save
                save_json_atomic(qa_data, str(qa_file))
                all_qa_pairs.extend(qa_data['training_pairs'])
                
                self.log(f"Completed {chunk_file.name}: {len(qa_data['training_pairs'])} total Q&A pairs generated")
                total_processed += 1
                
            except Exception as e:
                self.log(f"Error processing {chunk_file.name}: {e}", "error")
                continue
        
        
        # Summary
        self.log("=" * 50)
        self.log(f"Q&A generation completed: {total_processed} processed, {total_skipped} skipped")
        self.log(f"Total Q&A pairs generated: {len(all_qa_pairs)}")
        
        return len(all_qa_pairs) > 0
