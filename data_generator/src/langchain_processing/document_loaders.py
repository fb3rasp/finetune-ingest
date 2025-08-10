"""
LangChain-based document loaders for various file formats.
"""

from pathlib import Path
from typing import Dict, List, Tuple
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredMarkdownLoader,
        UnstructuredHTMLLoader,
        UnstructuredWordDocumentLoader,
        TextLoader,
    )
    from langchain.schema import Document
except ImportError:
    class Document: ...  # type: ignore
from common.utils.helpers import log_message


class LangChainDocumentLoader:
    def __init__(self, incoming_dir: str = "/data/incoming"):
        self.incoming_dir = Path(incoming_dir)
        self.supported_extensions = {'.pdf', '.md', '.html', '.htm', '.docx', '.txt'}
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.md': UnstructuredMarkdownLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.txt': TextLoader,
        }

    def load_document(self, file_path: Path) -> Tuple[List[Document], Dict]:
        log_message(f"Loading {file_path.name} with LangChain")
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        metadata = {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_type': file_extension,
            'file_size': file_path.stat().st_size,
        }
        loader_class = self.loader_map[file_extension]
        loader = loader_class(str(file_path)) if file_extension != '.txt' else loader_class(str(file_path), encoding='utf-8')
        documents = loader.load()
        for doc in documents:
            doc.metadata.update(metadata)
        log_message(f"Loaded {len(documents)} sections from {file_path.name}")
        return documents, metadata

    def get_documents(self) -> List[Path]:
        if not self.incoming_dir.exists():
            log_message(f"Creating incoming directory: {self.incoming_dir}")
            try:
                self.incoming_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # Fallback to a writable local data directory if /data is not writable
                fallback = Path.cwd() / 'data' / 'incoming'
                log_message(f"Failed to create {self.incoming_dir} ({e}); falling back to {fallback}")
                fallback.mkdir(parents=True, exist_ok=True)
                self.incoming_dir = fallback
            return []
        documents = []
        for file_path in self.incoming_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                documents.append(file_path)
        log_message(f"Found {len(documents)} documents to process")
        return documents


