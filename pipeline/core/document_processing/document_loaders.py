"""
LangChain-based document loaders for various file formats.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import signal
from contextlib import contextmanager

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        PyPDFium2Loader,
        UnstructuredMarkdownLoader,
        UnstructuredHTMLLoader,
        UnstructuredWordDocumentLoader,
        TextLoader,
    )
    from langchain.schema import Document
except ImportError:
    class Document: ...  # type: ignore

from pipeline.core.utils.helpers import log_message


@contextmanager
def timeout(duration):
    """Context manager for adding timeout to operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler and a alarm for the specified duration
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


class LangChainDocumentLoader:
    def __init__(self, incoming_dir: str = "/data/incoming"):
        self.incoming_dir = Path(incoming_dir)
        self.supported_extensions = {'.pdf', '.md', '.html', '.htm', '.docx', '.txt'}
        self.loader_map = {
            '.pdf': PyPDFium2Loader,
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
        
        # Special handling for PDF files with fallback options
        if file_extension == '.pdf':
            return self._load_pdf_with_fallback(file_path, metadata)
        
        # Handle non-PDF files normally
        loader_class = self.loader_map[file_extension]
        loader = loader_class(str(file_path)) if file_extension != '.txt' else loader_class(str(file_path), encoding='utf-8')
        
        try:
            with timeout(60):
                documents = loader.load()
        except TimeoutError:
            log_message(f"ERROR: Timeout loading {file_path.name} - skipping this document")
            return [], metadata
        except Exception as e:
            log_message(f"ERROR: Error loading {file_path.name}: {e} - skipping this document")
            return [], metadata
        
        for doc in documents:
            doc.metadata.update(metadata)
        log_message(f"Loaded {len(documents)} sections from {file_path.name}")
        return documents, metadata

    def _load_pdf_with_fallback(self, file_path: Path, metadata: Dict) -> Tuple[List[Document], Dict]:
        """Try multiple PDF loading methods as fallbacks."""
        
        # Method 1: Try PyPDFium2Loader (fastest)
        try:
            log_message(f"Trying PyPDFium2Loader for {file_path.name}")
            with timeout(60):
                loader = PyPDFium2Loader(str(file_path))
                documents = loader.load()
                for doc in documents:
                    doc.metadata.update(metadata)
                log_message(f"Successfully loaded {len(documents)} sections with PyPDFium2Loader")
                return documents, metadata
        except (TimeoutError, Exception) as e:
            log_message(f"PyPDFium2Loader failed for {file_path.name}: {e}")

        # Method 2: Fallback to PyPDFLoader
        try:
            log_message(f"Trying PyPDFLoader for {file_path.name}")
            with timeout(60):
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                for doc in documents:
                    doc.metadata.update(metadata)
                log_message(f"Successfully loaded {len(documents)} sections with PyPDFLoader")
                return documents, metadata
        except (TimeoutError, Exception) as e:
            log_message(f"PyPDFLoader failed for {file_path.name}: {e}")
        
        # Method 3: Try simple text extraction with pypdf
        try:
            log_message(f"Trying pypdf direct extraction for {file_path.name}")
            import pypdf
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    except Exception as e:
                        log_message(f"Error extracting page {page_num + 1}: {e}")
                        continue
                
                if text_content.strip():
                    # Create a single document with all content
                    document = Document(
                        page_content=text_content.strip(),
                        metadata=metadata.copy()
                    )
                    log_message(f"Successfully extracted text using pypdf ({len(text_content)} characters)")
                    return [document], metadata
                else:
                    log_message(f"No text content extracted from {file_path.name}")
                    
        except Exception as e:
            log_message(f"pypdf extraction failed for {file_path.name}: {e}")
        
        # Method 4: If all else fails, return empty
        log_message(f"ERROR: All PDF loading methods failed for {file_path.name}")
        return [], metadata

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
