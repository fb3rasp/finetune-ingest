"""
Main processor orchestrator using LangChain components.

This module coordinates all LangChain components to provide a unified
interface for document processing and Q&A generation, replacing the
legacy implementation with modern LangChain patterns.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from .document_loaders import LangChainDocumentLoader
from .text_splitters import EnhancedTextSplitter
from .llm_providers import UnifiedLLMProvider, LLMProvider
from .qa_chains import QAGenerationChain
from utils.helpers import log_message, save_json_atomic, load_json_if_exists


class LangChainProcessor:
    """Main orchestrator for LangChain-based document processing."""
    
    def __init__(
        self,
        incoming_dir: str = "./incoming",
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        questions_per_chunk: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        splitting_strategy: str = "recursive",
        use_batch_processing: bool = False,
        # New: prompt customization
        qa_system_message: Optional[str] = None,
        qa_additional_instructions: Optional[str] = None,
        qa_custom_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LangChain processor.
        
        Args:
            incoming_dir: Directory containing source documents
            provider: LLM provider to use
            model: Specific model name
            api_key: API key for the provider
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            questions_per_chunk: Number of Q&A pairs per chunk
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM responses
            splitting_strategy: Text splitting strategy
            use_batch_processing: Whether to use batch processing for Q&A generation
            **kwargs: Additional parameters
        """
        self.incoming_dir = incoming_dir
        self.splitting_strategy = splitting_strategy
        self.use_batch_processing = use_batch_processing
        self.qa_system_message = qa_system_message
        self.qa_additional_instructions = qa_additional_instructions
        self.qa_custom_template = qa_custom_template
        
        # Initialize components
        log_message("Initializing LangChain processor components...")
        
        # Document loader
        self.document_loader = LangChainDocumentLoader(incoming_dir)
        
        # Text splitter
        self.text_splitter = EnhancedTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # LLM provider
        self.llm_provider = UnifiedLLMProvider(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Q&A generation chain
        self.qa_chain = QAGenerationChain(
            llm_provider=self.llm_provider,
            questions_per_chunk=questions_per_chunk,
            system_message=self.qa_system_message,
            extra_instructions=self.qa_additional_instructions,
            custom_template=self.qa_custom_template
        )
        
        log_message("LangChain processor initialized successfully")
    
    def _cache_dir_for(self, file_path: Path) -> Path:
        """Return cache directory for a given source file (under incoming/.cache)."""
        base_dir = Path(self.incoming_dir) / ".cache"
        safe_name = file_path.name + ".cache"
        cache_dir = base_dir / safe_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def process_single_document(self, file_path: Path, resume: bool = False) -> Optional[Dict]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed document dictionary or None if processing failed
        """
        try:
            log_message(f"Processing document: {file_path.name}")
            
            cache_dir = self._cache_dir_for(file_path)
            docs_cache_path = cache_dir / "documents.json"
            chunks_cache_path = cache_dir / "chunks.json"

            documents = None
            enhanced_metadata = None
            chunks = None

            if resume:
                cached_docs = load_json_if_exists(str(docs_cache_path))
                cached_chunks = load_json_if_exists(str(chunks_cache_path))
                if cached_docs and cached_chunks:
                    log_message(f"Resuming: using cached load/splits for {file_path.name}")
                    documents = cached_docs.get("documents")
                    enhanced_metadata = cached_docs.get("metadata")
                    chunks = cached_chunks

            if documents is None or enhanced_metadata is None:
                # Load document fresh
                documents_obj, base_metadata = self.document_loader.load_document(file_path)
                if not documents_obj:
                    log_message(f"No content loaded from {file_path.name}")
                    return None
                enhanced_metadata = self.document_loader.extract_enhanced_metadata(documents_obj)
                # Convert LangChain Document objects to serializable format for cache
                documents = [
                    {
                        "page_content": d.page_content,
                        "metadata": dict(getattr(d, "metadata", {}))
                    }
                    for d in documents_obj
                ]
                save_json_atomic({"documents": documents, "metadata": enhanced_metadata}, str(docs_cache_path))

            if chunks is None:
                # Reconstruct LangChain docs if needed
                try:
                    from langchain.schema import Document as LCDocument
                    documents_reconstructed = [
                        LCDocument(page_content=d["page_content"], metadata=d.get("metadata", {}))
                        for d in documents
                    ]
                    # Split documents into chunks
                    chunks = self.text_splitter.split_documents(
                        documents_reconstructed, 
                        strategy=self.splitting_strategy
                    )
                except Exception:
                    # Fallback: split concatenated text adaptively
                    log_message("Falling back to adaptive text splitting for cached documents")
                    concatenated = "\n\n".join(d.get("page_content", "") for d in documents)
                    file_type = (enhanced_metadata or {}).get('file_type', '.txt')
                    chunks = self.text_splitter.split_text_adaptive(concatenated, file_type=file_type)

                save_json_atomic(chunks, str(chunks_cache_path))
            
            if not chunks:
                log_message(f"No chunks created from {file_path.name}")
                return None
            
            log_message(f"Created {len(chunks)} chunks from {file_path.name}")
            
            # Get chunk statistics
            chunk_stats = self.text_splitter.get_chunk_statistics(chunks)
            
            return {
                'metadata': enhanced_metadata,
                'chunks': chunks,
                'chunk_statistics': chunk_stats,
                'processing_info': {
                    'splitting_strategy': self.splitting_strategy,
                    'chunk_size': self.text_splitter.chunk_size,
                    'chunk_overlap': self.text_splitter.chunk_overlap,
                    'processed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            log_message(f"Error processing document {file_path}: {str(e)}")
            return None
    
    def process_all_documents(self, resume: bool = False) -> List[Dict]:
        """
        Process all documents in the incoming directory.
        
        Returns:
            List of processed document dictionaries
        """
        log_message("Processing all documents...")
        
        document_files = self.document_loader.get_documents()
        if not document_files:
            log_message("No documents found to process")
            return []
        
        processed_documents = []
        
        for file_path in document_files:
            processed_doc = self.process_single_document(file_path, resume=resume)
            if processed_doc:
                processed_documents.append(processed_doc)
        
        log_message(f"Successfully processed {len(processed_documents)} documents")
        return processed_documents
    
    def generate_qa_training_data(
        self, 
        processed_documents: Optional[List[Dict]] = None,
        output_file: str = "./training_data.json",
        resume: bool = False
    ) -> Dict:
        """
        Generate complete Q&A training dataset.
        
        Args:
            processed_documents: Pre-processed documents (will process all if None)
            output_file: Output file path for training data
            
        Returns:
            Training data dictionary
        """
        if processed_documents is None:
            processed_documents = self.process_all_documents(resume=resume)
        
        if not processed_documents:
            log_message("No processed documents available for Q&A generation")
            return {}
        
        log_message("Generating Q&A training data...")
        
        # Initialize training data structure
        training_data = {
            'metadata': {
                'generated_by': 'finetune-ingest-langchain',
                'generated_at': datetime.now().isoformat(),
                'llm_provider': self.llm_provider.provider.value,
                'model_used': self.llm_provider.model,
                'processing_config': {
                    'chunk_size': self.text_splitter.chunk_size,
                    'chunk_overlap': self.text_splitter.chunk_overlap,
                    'splitting_strategy': self.splitting_strategy,
                    'questions_per_chunk': self.qa_chain.questions_per_chunk,
                    'temperature': self.llm_provider.temperature,
                    'max_tokens': self.llm_provider.max_tokens
                },
                'num_documents': len(processed_documents),
                'total_qa_pairs': 0
            },
            'documents': [],
            'training_pairs': []
        }
        
        all_qa_pairs = []
        # Resume support: load existing output if present
        existing = load_json_if_exists(output_file) if resume else None
        processed_chunk_keys = set()
        if existing and isinstance(existing, dict):
            # preserve previously written pairs and docs metadata
            all_qa_pairs = existing.get('training_pairs', [])
            # collect chunk_ids to skip
            for qa in all_qa_pairs:
                cid = qa.get('chunk_id')
                fname = qa.get('file_name')
                if cid is not None and fname:
                    processed_chunk_keys.add(f"{fname}::{cid}")
            # Merge doc-level summaries later; keep existing as base
            training_data['documents'] = existing.get('documents', [])
            training_data['metadata'].update({
                'generated_by': existing.get('metadata', {}).get('generated_by', training_data['metadata']['generated_by']),
                'generated_at': existing.get('metadata', {}).get('generated_at', training_data['metadata']['generated_at'])
            })
        
        # Process each document
        for doc in processed_documents:
            log_message(f"Generating Q&A pairs for {doc['metadata']['file_name']}")
            
            doc_qa_pairs = []
            
            if self.use_batch_processing:
                # Use batch processing for better efficiency
                def _on_chunk_done(pairs_for_chunk: List[Dict]):
                    # Append and persist incrementally for resume support
                    nonlocal all_qa_pairs, training_data
                    all_qa_pairs.extend(pairs_for_chunk)
                    training_data['training_pairs'] = all_qa_pairs
                    training_data['metadata']['total_qa_pairs'] = len(all_qa_pairs)
                    save_json_atomic(training_data, output_file)

                # If resuming, skip already-processed chunks
                file_name = doc['metadata'].get('file_name')
                chunks_to_process = [
                    c for c in doc['chunks']
                    if not (resume and f"{file_name}::{c.get('chunk_id')}" in processed_chunk_keys)
                ]

                batch_qa_pairs = self.qa_chain.generate_batch_qa_pairs(
                    chunks_to_process, 
                    doc['metadata'],
                    on_chunk_done=_on_chunk_done
                )
                doc_qa_pairs.extend(batch_qa_pairs)
            else:
                # Process chunks individually
                file_name = doc['metadata'].get('file_name')
                for chunk in doc['chunks']:
                    chunk_id = chunk.get('chunk_id')
                    if resume and f"{file_name}::{chunk_id}" in processed_chunk_keys:
                        continue
                    chunk_qa_pairs = self.qa_chain.generate_qa_pairs(
                        chunk, 
                        doc['metadata']
                    )
                    doc_qa_pairs.extend(chunk_qa_pairs)
                    # Append incrementally and persist so we can resume
                    if chunk_qa_pairs:
                        all_qa_pairs.extend(chunk_qa_pairs)
                        training_data['training_pairs'] = all_qa_pairs
                        training_data['metadata']['total_qa_pairs'] = len(all_qa_pairs)
                        save_json_atomic(training_data, output_file)
            
            if not self.use_batch_processing:
                # already extended incrementally
                pass
            else:
                # already extended incrementally via on_chunk_done callback
                pass
            
            # Add/update document summary in training data
            # Avoid duplicate document summaries on resume and keep counts accurate
            file_name = doc['metadata'].get('file_name')
            existing_docs = {d.get('file_info', {}).get('file_name') for d in training_data['documents']}
            # Compute total pairs for this document across all_qa_pairs
            total_pairs_for_doc = sum(1 for qa in all_qa_pairs if qa.get('file_name') == file_name)
            if file_name not in existing_docs:
                training_data['documents'].append({
                    'file_info': doc['metadata'],
                    'chunk_count': len(doc['chunks']),
                    'qa_pairs_count': total_pairs_for_doc,
                    'chunk_statistics': doc.get('chunk_statistics', {}),
                    'processing_info': doc.get('processing_info', {})
                })
            else:
                # Update existing entry's qa_pairs_count
                for d in training_data['documents']:
                    if d.get('file_info', {}).get('file_name') == file_name:
                        d['qa_pairs_count'] = total_pairs_for_doc
                        break
            
            log_message(f"Generated {len(doc_qa_pairs)} Q&A pairs for {doc['metadata']['file_name']}")
        
        # Finalize training data
        training_data['training_pairs'] = all_qa_pairs
        training_data['metadata']['total_qa_pairs'] = len(all_qa_pairs)
        
        # Calculate quality metrics
        if all_qa_pairs:
            avg_question_length = sum(len(qa['question']) for qa in all_qa_pairs) / len(all_qa_pairs)
            avg_answer_length = sum(len(qa['answer']) for qa in all_qa_pairs) / len(all_qa_pairs)
            
            training_data['metadata']['quality_metrics'] = {
                'avg_question_length': avg_question_length,
                'avg_answer_length': avg_answer_length,
                'total_questions': len(all_qa_pairs),
                'unique_source_files': len(set(qa['file_name'] for qa in all_qa_pairs))
            }
        
        # Save training data
        try:
            save_json_atomic(training_data, output_file, indent=2, ensure_ascii=False)
            log_message(f"Training data saved to {output_file}")
        except Exception as e:
            log_message(f"Error saving training data: {str(e)}")
        
        log_message(f"Generated {len(all_qa_pairs)} total Q&A pairs from {len(processed_documents)} documents")
        return training_data
    
    def update_configuration(self, **kwargs):
        """
        Update processor configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        updated_params = []
        
        # Update chunk size and overlap
        if 'chunk_size' in kwargs or 'chunk_overlap' in kwargs:
            new_chunk_size = kwargs.get('chunk_size', self.text_splitter.chunk_size)
            new_overlap = kwargs.get('chunk_overlap', self.text_splitter.chunk_overlap)
            self.text_splitter.update_chunk_size(new_chunk_size, new_overlap)
            updated_params.extend(['chunk_size', 'chunk_overlap'])
        
        # Update splitting strategy
        if 'splitting_strategy' in kwargs:
            self.splitting_strategy = kwargs['splitting_strategy']
            updated_params.append('splitting_strategy')
        
        # Update Q&A generation parameters
        if 'questions_per_chunk' in kwargs:
            self.qa_chain.update_questions_per_chunk(kwargs['questions_per_chunk'])
            updated_params.append('questions_per_chunk')
        
        # Update LLM parameters
        llm_params = {k: v for k, v in kwargs.items() 
                     if k in ['temperature', 'max_tokens']}
        if llm_params:
            self.llm_provider.update_parameters(**llm_params)
            updated_params.extend(llm_params.keys())
        
        # Update batch processing
        if 'use_batch_processing' in kwargs:
            self.use_batch_processing = kwargs['use_batch_processing']
            updated_params.append('use_batch_processing')
        
        if updated_params:
            log_message(f"Updated configuration parameters: {', '.join(updated_params)}")
    
    def get_processing_info(self) -> Dict:
        """Get comprehensive information about the processor configuration."""
        return {
            'document_loader': {
                'incoming_dir': str(self.document_loader.incoming_dir),
                'supported_extensions': list(self.document_loader.supported_extensions),
                'available_documents': len(self.document_loader.get_documents())
            },
            'text_splitter': {
                'chunk_size': self.text_splitter.chunk_size,
                'chunk_overlap': self.text_splitter.chunk_overlap,
                'splitting_strategy': self.splitting_strategy
            },
            'llm_provider': self.llm_provider.get_model_info(),
            'qa_chain': self.qa_chain.get_chain_info(),
            'processing_options': {
                'use_batch_processing': self.use_batch_processing
            }
        }
    
    @classmethod
    def create_from_config(cls, config: Dict) -> 'LangChainProcessor':
        """
        Create processor instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured LangChainProcessor instance
        """
        return cls(**config)