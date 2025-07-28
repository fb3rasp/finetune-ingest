"""
LangChain-based processing module for document analysis and Q&A generation.

This module provides a modern, unified approach to document processing using
LangChain's battle-tested components for document loading, text splitting,
LLM integration, and Q&A generation.
"""

from .document_loaders import LangChainDocumentLoader
from .text_splitters import EnhancedTextSplitter
from .llm_providers import UnifiedLLMProvider, LLMProvider
from .qa_chains import QAGenerationChain
from .processors import LangChainProcessor

__all__ = [
    'LangChainDocumentLoader',
    'EnhancedTextSplitter', 
    'UnifiedLLMProvider',
    'LLMProvider',
    'QAGenerationChain',
    'LangChainProcessor'
]