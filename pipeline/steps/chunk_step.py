"""
Document chunking step for the training data pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.document_processing import LangChainDocumentLoader, EnhancedTextSplitter
from pipeline.core.utils.helpers import log_message, save_json_atomic, load_json_if_exists
from pipeline.config import PipelineConfig
from .base_step import BaseStep


class ChunkStep(BaseStep):
    """Step 1: Chunk documents into smaller pieces."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.document_loader = LangChainDocumentLoader(config.input_dir)
        self.text_splitter = EnhancedTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def check_prerequisites(self) -> bool:
        """Check if input directory exists and contains documents."""
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            self.log(f"Input directory does not exist: {input_path}", "error")
            return False
        
        # Check for supported document files
        supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
        has_documents = any(
            f.suffix.lower() in supported_extensions 
            for f in input_path.iterdir() 
            if f.is_file()
        )
        
        if not has_documents:
            self.log(f"No supported documents found in {input_path}", "error")
            return False
        
        return True
    
    def run(self, resume: bool = False) -> bool:
        """Run the document chunking step."""
        self.log("Starting document chunking step...")
        
        if not self.check_prerequisites():
            return False
        
        # Ensure output directory exists
        chunks_dir = Path(self.config.chunks_dir)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of documents to process
        input_path = Path(self.config.input_dir)
        supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
        document_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not document_files:
            self.log("No documents to process", "error")
            return False
        
        self.log(f"Found {len(document_files)} documents to process")
        
        total_processed = 0
        total_skipped = 0
        
        for doc_file in document_files:
            try:
                # Create output file path
                chunk_file = chunks_dir / f"{doc_file.stem}_chunks.json"
                
                # Check if already processed
                if resume and chunk_file.exists():
                    existing_data = load_json_if_exists(str(chunk_file))
                    if existing_data and existing_data.get('status') == 'completed':
                        self.log(f"Skipping {doc_file.name} - already chunked")
                        total_skipped += 1
                        continue
                
                # Load and chunk the document
                self.log(f"Processing: {doc_file.name}")
                try:
                    documents, _ = self.document_loader.load_document(doc_file)
                    self.log(f"Successfully loaded {len(documents) if documents else 0} document sections")
                except Exception as e:
                    log_message(f"Error loading {doc_file.name}: {e} - skipping this document", "error")
                    continue
                
                if not documents:
                    self.log(f"No content extracted from {doc_file.name}")
                    continue
                
                # Split into chunks
                self.log(f"Starting to split {len(documents)} documents into chunks...")
                try:
                    chunks = self.text_splitter.split_documents(documents, strategy=self.config.splitting_strategy)
                    self.log(f"Successfully created {len(chunks)} chunks")
                except Exception as e:
                    self.log(f"Error splitting document {doc_file.name}: {e}", "error")
                    continue
                
                # Prepare chunk data
                chunk_data = {
                    "metadata": {
                        "source_file": str(doc_file),
                        "chunk_count": len(chunks),
                        "chunk_size": self.config.chunk_size,
                        "chunk_overlap": self.config.chunk_overlap,
                        "processed_at": str(datetime.now()),
                        "splitting_strategy": self.config.splitting_strategy
                    },
                    "chunks": [
                        {
                            "chunk_id": f"{doc_file.stem}_chunk_{i}",
                            "content": chunk["text"],  # ← Changed from chunk.page_content to chunk["text"]
                            "metadata": chunk["metadata"]  # ← Changed from chunk.metadata to chunk["metadata"]
                        }
                        for i, chunk in enumerate(chunks)
                    ],
                    "status": "completed"
                }
                
                # Save chunk data
                save_json_atomic(chunk_data, str(chunk_file))
                self.log(f"Created {len(chunks)} chunks for {doc_file.name}")
                total_processed += 1
                
            except Exception as e:
                self.log(f"Error processing {doc_file.name}: {e}", "error")
                continue
        
        # Summary
        self.log("=" * 50)
        self.log(f"Chunking completed: {total_processed} processed, {total_skipped} skipped")
        
        return total_processed > 0
