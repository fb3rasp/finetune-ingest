"""
Enhanced text splitting using LangChain's advanced splitters.

This module provides intelligent text splitting with semantic awareness,
metadata preservation, and configurable chunking strategies.
"""

from typing import List, Dict, Optional, Any
try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter
    )
    from langchain.schema import Document
    LANGCHAIN_SPLITTERS_AVAILABLE = True
except ImportError as e:
    print(f"LangChain text splitters not available: {e}")
    print("Install with: pip install langchain")
    LANGCHAIN_SPLITTERS_AVAILABLE = False
    
    # Create dummy classes
    class RecursiveCharacterTextSplitter: pass
    class CharacterTextSplitter: pass
    class MarkdownHeaderTextSplitter: pass
    class HTMLHeaderTextSplitter: pass
    class Document: pass
from utils.helpers import log_message


class EnhancedTextSplitter:
    """Enhanced text splitter with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        
        # Default separators for recursive splitting (hierarchical)
        self.default_separators = separators or [
            "\n\n",  # Double newlines (paragraphs)
            "\n",    # Single newlines
            " ",     # Spaces
            ""       # Characters
        ]
        
        # Initialize different splitter types
        self._init_splitters()
    
    def _init_splitters(self):
        """Initialize various text splitters."""
        # Recursive character splitter (most versatile)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.default_separators
        )
        
        # Character splitter (simple)
        self.character_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separator="\n"
        )
        
        # Markdown header splitter
        self.markdown_headers = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.markdown_headers
        )
        
        # HTML header splitter
        self.html_headers = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=self.html_headers
        )
    
    def split_documents(
        self, 
        documents: List[Document], 
        strategy: str = "recursive"
    ) -> List[Dict]:
        """
        Split documents into chunks using specified strategy.
        
        Args:
            documents: List of LangChain Document objects
            strategy: Splitting strategy ('recursive', 'character', 'markdown', 'html')
            
        Returns:
            List of chunk dictionaries with enhanced metadata
        """
        if not documents:
            return []
        
        log_message(f"Splitting {len(documents)} documents using {strategy} strategy")
        
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            # Choose splitting strategy
            if strategy == "recursive":
                splits = self.recursive_splitter.split_documents([document])
            elif strategy == "character":
                splits = self.character_splitter.split_documents([document])
            elif strategy == "markdown" and document.metadata.get('file_type') == '.md':
                splits = self._split_markdown_document(document)
            elif strategy == "html" and document.metadata.get('file_type') in ['.html', '.htm']:
                splits = self._split_html_document(document)
            else:
                # Fallback to recursive splitter
                splits = self.recursive_splitter.split_documents([document])
            
            # Convert splits to enhanced chunk format
            doc_chunks = self._convert_splits_to_chunks(splits, doc_idx)
            all_chunks.extend(doc_chunks)
        
        log_message(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _split_markdown_document(self, document: Document) -> List[Document]:
        """Split markdown document preserving header structure."""
        # First split by headers
        header_splits = self.markdown_splitter.split_text(document.page_content)
        
        # Then apply recursive splitting to each header section if needed
        final_splits = []
        for split in header_splits:
            if len(split.page_content) > self.chunk_size:
                subsplits = self.recursive_splitter.split_documents([split])
                final_splits.extend(subsplits)
            else:
                final_splits.append(split)
        
        # Preserve original metadata
        for split in final_splits:
            split.metadata.update(document.metadata)
        
        return final_splits
    
    def _split_html_document(self, document: Document) -> List[Document]:
        """Split HTML document preserving header structure."""
        # First split by headers
        header_splits = self.html_splitter.split_text(document.page_content)
        
        # Then apply recursive splitting to each header section if needed
        final_splits = []
        for split in header_splits:
            if len(split.page_content) > self.chunk_size:
                subsplits = self.recursive_splitter.split_documents([split])
                final_splits.extend(subsplits)
            else:
                final_splits.append(split)
        
        # Preserve original metadata
        for split in final_splits:
            split.metadata.update(document.metadata)
        
        return final_splits
    
    def _convert_splits_to_chunks(self, splits: List[Document], doc_idx: int) -> List[Dict]:
        """Convert LangChain document splits to chunk format."""
        chunks = []
        
        for chunk_idx, split in enumerate(splits):
            # Calculate character positions (approximate)
            start_char = chunk_idx * (self.chunk_size - self.chunk_overlap)
            end_char = start_char + len(split.page_content)
            
            chunk = {
                'text': split.page_content,
                'start_char': start_char,
                'end_char': end_char,
                'chunk_id': f"{doc_idx}_{chunk_idx}",
                'chunk_index': chunk_idx,
                'document_index': doc_idx,
                'metadata': split.metadata.copy(),
                'word_count': len(split.page_content.split()),
                'char_count': len(split.page_content)
            }
            
            # Add header information if available
            if 'Header 1' in split.metadata:
                chunk['section_header'] = split.metadata['Header 1']
            if 'Header 2' in split.metadata:
                chunk['subsection_header'] = split.metadata['Header 2']
            
            chunks.append(chunk)
        
        return chunks
    
    def split_text_adaptive(self, text: str, file_type: str = '.txt') -> List[Dict]:
        """
        Adaptively split text based on file type.
        
        Args:
            text: Text content to split
            file_type: File extension to determine splitting strategy
            
        Returns:
            List of chunk dictionaries
        """
        # Create a temporary document
        document = Document(
            page_content=text,
            metadata={'file_type': file_type}
        )
        
        # Choose strategy based on file type
        if file_type == '.md':
            strategy = 'markdown'
        elif file_type in ['.html', '.htm']:
            strategy = 'html' 
        else:
            strategy = 'recursive'
        
        return self.split_documents([document], strategy=strategy)
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        chunk_sizes = [chunk['char_count'] for chunk in chunks]
        word_counts = [chunk['word_count'] for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'total_words': sum(word_counts),
            'avg_words_per_chunk': sum(word_counts) / len(word_counts)
        }
        
        return stats
    
    def update_chunk_size(self, new_chunk_size: int, new_overlap: int = None):
        """Update chunk size and reinitialize splitters."""
        self.chunk_size = new_chunk_size
        if new_overlap is not None:
            self.chunk_overlap = new_overlap
        
        log_message(f"Updated chunk size to {self.chunk_size} with overlap {self.chunk_overlap}")
        self._init_splitters()