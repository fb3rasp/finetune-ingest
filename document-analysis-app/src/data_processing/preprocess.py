import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import PyPDF2
import pdfplumber
import markdown
from bs4 import BeautifulSoup
import docx
from ..utils.helpers import log_message

class DocumentProcessor:
    """Handles extraction and preprocessing of various document types."""
    
    def __init__(self, incoming_dir: str = "./incoming"):
        self.incoming_dir = Path(incoming_dir)
        self.supported_extensions = {'.pdf', '.md', '.html', '.htm', '.docx', '.txt'}
    
    def extract_text_from_document(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text and metadata from a document."""
        log_message(f"Processing {file_path.name}")
        
        metadata = {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'sections': []
        }
        
        try:
            if file_path.suffix.lower() == '.pdf':
                text, sections = self._extract_from_pdf(file_path)
            elif file_path.suffix.lower() == '.md':
                text, sections = self._extract_from_markdown(file_path)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                text, sections = self._extract_from_html(file_path)
            elif file_path.suffix.lower() == '.docx':
                text, sections = self._extract_from_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                text, sections = self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            metadata['sections'] = sections
            return text, metadata
            
        except Exception as e:
            log_message(f"Error processing {file_path}: {str(e)}")
            return "", metadata
    
    def _extract_from_pdf(self, file_path: Path) -> Tuple[str, List[Dict]]:
        """Extract text from PDF with section tracking."""
        text_parts = []
        sections = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    sections.append({
                        'type': 'page',
                        'number': page_num,
                        'start_char': len(' '.join(text_parts[:-1])) if len(text_parts) > 1 else 0,
                        'end_char': len(' '.join(text_parts)),
                        'title': f"Page {page_num}"
                    })
        
        return ' '.join(text_parts), sections
    
    def _extract_from_markdown(self, file_path: Path) -> Tuple[str, List[Dict]]:
        """Extract text from Markdown with header-based sections."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to HTML then extract text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Track sections based on headers
        sections = []
        lines = content.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Extract header level and title
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                sections.append({
                    'type': 'header',
                    'level': level,
                    'title': title,
                    'line_number': i + 1,
                    'start_char': current_pos
                })
            current_pos += len(line) + 1  # +1 for newline
        
        return text, sections
    
    def _extract_from_html(self, file_path: Path) -> Tuple[str, List[Dict]]:
        """Extract text from HTML with element-based sections."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        
        # Track sections based on headers
        sections = []
        for i, tag in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            sections.append({
                'type': 'header',
                'level': int(tag.name[1]),
                'title': tag.get_text().strip(),
                'element_index': i
            })
        
        return text, sections
    
    def _extract_from_docx(self, file_path: Path) -> Tuple[str, List[Dict]]:
        """Extract text from DOCX with paragraph-based sections."""
        doc = docx.Document(file_path)
        text_parts = []
        sections = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
                
                # Check if this looks like a header (bold, larger font, etc.)
                if paragraph.style.name.startswith('Heading'):
                    sections.append({
                        'type': 'header',
                        'title': paragraph.text.strip(),
                        'paragraph_index': i,
                        'style': paragraph.style.name
                    })
        
        return '\n'.join(text_parts), sections
    
    def _extract_from_txt(self, file_path: Path) -> Tuple[str, List[Dict]]:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple section detection based on empty lines or patterns
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip() and (line.isupper() or line.endswith(':')):
                sections.append({
                    'type': 'section',
                    'title': line.strip(),
                    'line_number': i + 1
                })
        
        return content, sections
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks for better Q&A generation."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'chunk_id': len(chunks)
                })
            
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def normalize_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'Page \d+.*?\n', '', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text
    
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
    
    def preprocess_document(self, file_path: Path) -> Dict:
        """Main function to preprocess a single document."""
        raw_text, metadata = self.extract_text_from_document(file_path)
        
        if not raw_text.strip():
            log_message(f"No text extracted from {file_path}")
            return None
        
        normalized_text = self.normalize_text(raw_text)
        chunks = self.chunk_text(normalized_text)
        
        return {
            'metadata': metadata,
            'full_text': normalized_text,
            'chunks': chunks,
            'word_count': len(normalized_text.split()),
            'char_count': len(normalized_text)
        }


# Legacy functions for backward compatibility
def extract_text_from_document(file_path):
    """Legacy function - use DocumentProcessor class instead."""
    processor = DocumentProcessor()
    text, _ = processor.extract_text_from_document(Path(file_path))
    return text

def normalize_text(text):
    """Legacy function - use DocumentProcessor class instead."""
    processor = DocumentProcessor()
    return processor.normalize_text(text)

def preprocess_document(file_path):
    """Legacy function - use DocumentProcessor class instead."""
    processor = DocumentProcessor()
    result = processor.preprocess_document(Path(file_path))
    return result['full_text'] if result else ""