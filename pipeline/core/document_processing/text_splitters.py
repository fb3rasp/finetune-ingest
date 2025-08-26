from typing import List, Dict, Optional
try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter,
    )
    from langchain.schema import Document
except ImportError:
    class Document: ...  # type: ignore
from pipeline.core.utils.helpers import log_message


class EnhancedTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_separators = ["\n\n", "\n", " ", ""]
        self._init_splitters()

    def _init_splitters(self):
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.default_separators,
        )
        self.character_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")
        ])
        self.html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
            ("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")
        ])

    def split_documents(self, documents: List[Document], strategy: str = "recursive") -> List[Dict]:
        if not documents:
            return []
        log_message(f"Splitting {len(documents)} documents using {strategy} strategy")
        all_chunks: List[Dict] = []
        for doc_idx, document in enumerate(documents):
            if strategy == "recursive":
                splits = self.recursive_splitter.split_documents([document])
            elif strategy == "character":
                splits = self.character_splitter.split_documents([document])
            elif strategy == "markdown" and document.metadata.get('file_type') == '.md':
                splits = self.markdown_splitter.split_text(document.page_content)
            elif strategy == "html" and document.metadata.get('file_type') in ['.html', '.htm']:
                splits = self.html_splitter.split_text(document.page_content)
            else:
                splits = self.recursive_splitter.split_documents([document])
            for chunk_idx, split in enumerate(splits):
                text = getattr(split, 'page_content', str(split))
                start_char = chunk_idx * (self.chunk_size - self.chunk_overlap)
                end_char = start_char + len(text)
                chunk = {
                    'text': text,
                    'start_char': start_char,
                    'end_char': end_char,
                    'chunk_id': f"{doc_idx}_{chunk_idx}",
                    'chunk_index': chunk_idx,
                    'document_index': doc_idx,
                    'metadata': getattr(split, 'metadata', {}).copy() if hasattr(split, 'metadata') else {},
                    'word_count': len(text.split()),
                    'char_count': len(text),
                }
                all_chunks.append(chunk)
        log_message(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
