"""
LangChain-based document loaders for various file formats.

This module provides a unified interface for loading different document types
using LangChain's specialized loaders with enhanced metadata extraction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredMarkdownLoader,
        UnstructuredHTMLLoader,
        UnstructuredWordDocumentLoader,
        TextLoader
    )
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain packages not available: {e}")
    print("Install with: pip install langchain langchain-community")
    print("Falling back to basic document loading...")
    LANGCHAIN_AVAILABLE = False
    
    # Create dummy classes to prevent import errors
    class PyPDFLoader: pass
    class UnstructuredMarkdownLoader: pass
    class UnstructuredHTMLLoader: pass
    class UnstructuredWordDocumentLoader: pass
    class TextLoader: pass
    class Document: pass
from utils.helpers import log_message


class LangChainDocumentLoader:
    """Unified document loader using LangChain components."""
    
    def __init__(self, incoming_dir: str = "./incoming"):
        self.incoming_dir = Path(incoming_dir)
        self.supported_extensions = {'.pdf', '.md', '.html', '.htm', '.docx', '.txt'}
        
        # Mapping of file extensions to LangChain loaders
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.md': UnstructuredMarkdownLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.txt': TextLoader
        }
    
    def load_document(self, file_path: Path) -> Tuple[List[Document], Dict]:
        """
        Load a document using appropriate LangChain loader.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (documents list, metadata dict)
        """
        log_message(f"Loading {file_path.name} with LangChain")
        
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Base metadata
        metadata = {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_type': file_extension,
            'file_size': file_path.stat().st_size,
        }
        
        try:
            # Get the appropriate loader class
            loader_class = self.loader_map[file_extension]
            
            # Handle different loader initialization patterns
            if file_extension == '.txt':
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            
            # Load documents
            documents = loader.load()
            
            # Enhance metadata for each document
            for doc in documents:
                doc.metadata.update(metadata)
                # Add document-specific metadata if available
                if hasattr(doc.metadata, 'page') and doc.metadata.get('page') is not None:
                    doc.metadata['section_type'] = 'page'
                    doc.metadata['section_number'] = doc.metadata['page']
                elif hasattr(doc.metadata, 'Header 1') and doc.metadata.get('Header 1'):
                    doc.metadata['section_type'] = 'header'
                    doc.metadata['section_title'] = doc.metadata['Header 1']
            
            log_message(f"Loaded {len(documents)} document sections from {file_path.name}")
            return documents, metadata
            
        except Exception as e:
            log_message(f"Error loading {file_path}: {str(e)}")
            return [], metadata
    
    def load_documents_from_directory(self) -> List[Tuple[List[Document], Dict]]:
        """
        Load all supported documents from the incoming directory.
        
        Returns:
            List of (documents, metadata) tuples
        """
        document_files = self.get_documents()
        loaded_documents = []
        
        for file_path in document_files:
            docs, metadata = self.load_document(file_path)
            if docs:
                loaded_documents.append((docs, metadata))
        
        return loaded_documents
    
    def get_documents(self) -> List[Path]:
        """Get all supported documents from the incoming directory."""
        if not self.incoming_dir.exists():
            log_message(f"Creating incoming directory: {self.incoming_dir}")
            self.incoming_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        documents = []
        for file_path in self.incoming_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                documents.append(file_path)
        
        log_message(f"Found {len(documents)} documents to process")
        return documents
    
    def extract_enhanced_metadata(self, documents: List[Document]) -> Dict:
        """
        Extract enhanced metadata from loaded documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Enhanced metadata dictionary
        """
        if not documents:
            return {}
        
        # Aggregate metadata from all document sections
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Extract section information
        sections = []
        for i, doc in enumerate(documents):
            section_info = {
                'section_id': i,
                'char_count': len(doc.page_content),
                'word_count': len(doc.page_content.split()),
            }
            
            # Add section-specific metadata
            if doc.metadata.get('section_type'):
                section_info['type'] = doc.metadata['section_type']
            if doc.metadata.get('section_number'):
                section_info['number'] = doc.metadata['section_number']
            if doc.metadata.get('section_title'):
                section_info['title'] = doc.metadata['section_title']
            if doc.metadata.get('page'):
                section_info['page'] = doc.metadata['page']
                
            sections.append(section_info)
        
        enhanced_metadata = {
            'total_sections': len(documents),
            'total_chars': total_chars,
            'total_words': total_words,
            'sections': sections,
            'source_file': documents[0].metadata.get('source_file'),
            'file_name': documents[0].metadata.get('file_name'),
            'file_type': documents[0].metadata.get('file_type'),
            'file_size': documents[0].metadata.get('file_size'),
        }
        
        return enhanced_metadata