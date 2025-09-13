"""
Q&A generation step for the training data pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.llm import UnifiedLLMProvider, QAGenerationChain
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists, load_yaml_if_exists
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
        # Don't create qa_chain here - we'll create it per prompt configuration
        self.incoming_dir = Path("_incoming")
    
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
    
    def _load_definition_file(self, chunk_file: Path) -> Optional[Dict]:
        """Load definition file for a given chunk file based on its source document."""
        try:
            # Load chunk data to get source file information
            chunk_data = load_json_if_exists(str(chunk_file))
            if not chunk_data or 'metadata' not in chunk_data:
                return None
            
            source_file = chunk_data['metadata'].get('source_file', '')
            if not source_file:
                return None
            
            # Get the source document name without extension
            source_path = Path(source_file)
            base_name = f"{source_path.stem}-definition"
            
            # Try YAML first (preferred format), then JSON for backward compatibility
            for extension, loader in [('.yaml', load_yaml_if_exists), ('.yml', load_yaml_if_exists), ('.json', load_json_if_exists)]:
                definition_file = self.incoming_dir / f"{base_name}{extension}"
                if definition_file.exists():
                    definition_data = loader(str(definition_file))
                    if definition_data and 'training_prompts' in definition_data:
                        self.log(f"Found definition file: {definition_file.name}")
                        return definition_data
                    else:
                        self.log(f"Definition file {definition_file.name} missing 'training_prompts'", "warning")
            
            return None
            
        except Exception as e:
            self.log(f"Error loading definition file: {e}", "warning")
            return None
    
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
                # Load definition file to check for multiple prompts
                definition = self._load_definition_file(chunk_file)
                
                if definition and 'training_prompts' in definition:
                    # Multiple prompt processing
                    self.log(f"Processing {chunk_file.name} with {len(definition['training_prompts'])} prompts")
                    success = self._process_multi_prompt(chunk_file, definition, qa_dir, resume)
                    if success:
                        total_processed += 1
                    # Load all generated QA pairs for tracking
                    for prompt_idx, prompt_config in enumerate(definition['training_prompts']):
                        qa_file = qa_dir / f"{chunk_file.stem}_qa_prompt_{prompt_idx+1:02d}.json"
                        if qa_file.exists():
                            existing_data = load_json_if_exists(str(qa_file))
                            if existing_data and 'training_pairs' in existing_data:
                                all_qa_pairs.extend(existing_data['training_pairs'])
                else:
                    # Single prompt processing (backward compatibility)
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
                        elif existing_data and existing_data.get('status') == 'partial':
                            failed_count = len(existing_data.get('failed_chunks', []))
                            completed_count = len(existing_data.get('completed_chunks', []))
                            self.log(f"Resuming {chunk_file.name} - {completed_count} completed, {failed_count} failed chunks to retry")
                    
                    # Process with single default prompt
                    success = self._process_single_prompt(chunk_file, qa_file, resume)
                    if success:
                        total_processed += 1
                        # Load generated QA pairs
                        existing_data = load_json_if_exists(str(qa_file))
                        if existing_data and 'training_pairs' in existing_data:
                            all_qa_pairs.extend(existing_data['training_pairs'])
                
            except Exception as e:
                self.log(f"Error processing {chunk_file.name}: {e}", "error")
                continue
        
        # Summary
        self.log("=" * 50)
        self.log(f"Q&A generation completed: {total_processed} processed, {total_skipped} skipped")
        self.log(f"Total Q&A pairs generated: {len(all_qa_pairs)}")
        
        return len(all_qa_pairs) > 0

    def _process_multi_prompt(self, chunk_file: Path, definition: Dict, qa_dir: Path, resume: bool) -> bool:
        """Process chunk file with multiple prompts from definition."""
        chunk_data = load_json_if_exists(str(chunk_file))
        if not chunk_data or 'chunks' not in chunk_data:
            self.log(f"Invalid chunk data in {chunk_file.name}")
            return False
        
        training_prompts = definition.get('training_prompts', [])
        system_prompt = definition.get('model_system_prompt')
        
        success_count = 0
        for prompt_idx, prompt_config in enumerate(training_prompts):
            try:
                # Create output file for this prompt
                qa_file = qa_dir / f"{chunk_file.stem}_qa_prompt_{prompt_idx+1:02d}.json"
                
                # Check if already processed
                if resume and qa_file.exists():
                    existing_data = load_json_if_exists(str(qa_file))
                    if existing_data and existing_data.get('status') == 'completed':
                        self.log(f"Skipping prompt {prompt_idx+1} for {chunk_file.name} - already processed")
                        success_count += 1
                        continue
                    elif existing_data and existing_data.get('status') == 'partial':
                        failed_count = len(existing_data.get('failed_chunks', []))
                        completed_count = len(existing_data.get('completed_chunks', []))
                        self.log(f"Resuming prompt {prompt_idx+1} for {chunk_file.name} - {completed_count} completed, {failed_count} failed chunks to retry")
                
                # Create QA chain with custom prompt
                qa_chain = QAGenerationChain(
                    llm_provider=self.llm_provider,
                    questions_per_chunk=prompt_config.get('num_questions', self.config.questions_per_chunk),
                    system_message=system_prompt,
                    custom_prompt=prompt_config.get('prompt')
                )
                
                # Process this prompt
                self.log(f"Processing prompt {prompt_idx+1}/{len(training_prompts)} for {chunk_file.name}")
                success = self._process_with_qa_chain(chunk_file, chunk_data, qa_file, qa_chain, prompt_idx+1, len(training_prompts), resume)
                if success:
                    success_count += 1
                    
            except Exception as e:
                self.log(f"Error processing prompt {prompt_idx+1} for {chunk_file.name}: {e}", "error")
                continue
        
        return success_count > 0

    def _process_single_prompt(self, chunk_file: Path, qa_file: Path, resume: bool) -> bool:
        """Process chunk file with single default prompt."""
        chunk_data = load_json_if_exists(str(chunk_file))
        if not chunk_data or 'chunks' not in chunk_data:
            self.log(f"Invalid chunk data in {chunk_file.name}")
            return False
        
        # Create default QA chain
        qa_chain = QAGenerationChain(
            llm_provider=self.llm_provider,
            questions_per_chunk=self.config.questions_per_chunk
        )
        
        return self._process_with_qa_chain(chunk_file, chunk_data, qa_file, qa_chain, 1, 1, resume)

    def _process_with_qa_chain(self, chunk_file: Path, chunk_data: Dict, qa_file: Path, qa_chain: QAGenerationChain, prompt_num: int, total_prompts: int, resume: bool) -> bool:
        """Process chunks with given QA chain."""
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
                    "questions_per_chunk": qa_chain.questions_per_chunk,
                    "processed_at": str(datetime.now()),
                    "status": "in_progress",
                    "prompt_number": prompt_num,
                    "total_prompts": total_prompts
                },
                "training_pairs": [],
                "completed_chunks": [],
                "failed_chunks": []
            }
        
        # Generate Q&A pairs for each chunk
        total_chunks = len(chunk_data['chunks'])
        
        for chunk_idx, chunk in enumerate(chunk_data['chunks']):
            # Skip if already completed
            if chunk_idx in qa_data.get('completed_chunks', []):
                continue
            
            # If this is a failed chunk during resume, remove it from failed list to retry
            if chunk_idx in qa_data.get('failed_chunks', []):
                qa_data['failed_chunks'].remove(chunk_idx)
                self.log(f"Retrying previously failed chunk {chunk_idx + 1}")
            
            try:
                # Pass the entire chunk object and the document's metadata
                pairs = qa_chain.generate_qa_pairs(chunk, chunk_data['metadata'])
                
                # Log progress for each chunk
                chunk_num = chunk_idx + 1
                prompt_info = f"Prompt {prompt_num}/{total_prompts}, " if total_prompts > 1 else ""
                self.log(f"{prompt_info}Chunk {chunk_num}/{total_chunks}: Generated {len(pairs)} Q&A pairs")
                
                # Add chunk metadata to pairs
                for pair in pairs:
                    pair['chunk_id'] = chunk.get('chunk_id')
                    pair['source_file'] = chunk_data['metadata'].get('source_file')
                
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
        
        # Mark as completed only if no chunks failed
        failed_count = len(qa_data['failed_chunks'])
        if failed_count == 0:
            qa_data['metadata']['status'] = 'completed'
        else:
            qa_data['metadata']['status'] = 'partial'
            
        qa_data['metadata']['completion_time'] = str(datetime.now())
        qa_data['metadata']['qa_count'] = len(qa_data['training_pairs'])
        qa_data['metadata']['completed_chunks_count'] = len(qa_data['completed_chunks'])
        qa_data['metadata']['failed_chunks_count'] = failed_count
        
        # Final save
        save_json_atomic(qa_data, str(qa_file))
        
        self.log(f"Completed processing: {len(qa_data['training_pairs'])} total Q&A pairs generated")
        return qa_data['metadata']['status'] == 'completed'
