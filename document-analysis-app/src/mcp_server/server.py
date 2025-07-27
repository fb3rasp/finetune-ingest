#!/usr/bin/env python3

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from mcp.server import Server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource
)
from pydantic import BaseModel

# Import our existing functionality
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.preprocess import DocumentProcessor
from data_processing.qa_generator import QAGenerator, LLMProvider
from utils.helpers import log_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune-ingest-mcp")

class ServerConfig(BaseModel):
    """Configuration for the MCP server."""
    incoming_dir: str = "./incoming"
    output_dir: str = "./output"
    default_provider: str = "openai"
    default_model: Optional[str] = None
    chunk_size: int = 1000
    questions_per_chunk: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000

class FinetuneIngestServer:
    """MCP Server for document analysis and training data generation."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.server = Server("finetune-ingest")
        self.processor = DocumentProcessor(config.incoming_dir)
        self.current_qa_generator: Optional[QAGenerator] = None
        
        # Ensure directories exist
        Path(config.incoming_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Register tools and resources
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register all available MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="list_documents",
                    description="List all documents in the incoming directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_metadata": {
                                "type": "boolean", 
                                "description": "Include file metadata",
                                "default": False
                            }
                        }
                    }
                ),
                Tool(
                    name="configure_llm",
                    description="Configure the LLM provider and model for Q&A generation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "enum": ["openai", "claude", "gemini", "local"],
                                "description": "LLM provider to use"
                            },
                            "model": {
                                "type": "string",
                                "description": "Specific model name (optional)"
                            },
                            "api_key": {
                                "type": "string", 
                                "description": "API key (optional, uses env vars if not provided)"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for generation",
                                "minimum": 0,
                                "maximum": 2
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens for response"
                            }
                        },
                        "required": ["provider"]
                    }
                ),
                Tool(
                    name="process_document",
                    description="Process a single document and extract text with metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the document file in incoming directory"
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": "Size of text chunks",
                                "default": 1000
                            }
                        },
                        "required": ["filename"]
                    }
                ),
                Tool(
                    name="generate_qa_pairs",
                    description="Generate Q&A pairs from processed document chunks",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the document file"
                            },
                            "questions_per_chunk": {
                                "type": "integer",
                                "description": "Number of questions per chunk",
                                "default": 3
                            }
                        },
                        "required": ["filename"]
                    }
                ),
                Tool(
                    name="generate_training_data",
                    description="Process all documents and generate complete training dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "output_filename": {
                                "type": "string",
                                "description": "Output filename for training data",
                                "default": "training_data.json"
                            },
                            "filter_files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Only process specific files (optional)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_provider_info", 
                    description="Get information about available LLM providers and models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "enum": ["openai", "claude", "gemini", "local"],
                                "description": "Specific provider to get info for (optional)"
                            }
                        }
                    }
                ),
                Tool(
                    name="validate_setup",
                    description="Validate that all dependencies and API keys are properly configured",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "list_documents":
                    return await self._list_documents(arguments)
                elif name == "configure_llm":
                    return await self._configure_llm(arguments)
                elif name == "process_document":
                    return await self._process_document(arguments)
                elif name == "generate_qa_pairs":
                    return await self._generate_qa_pairs(arguments)
                elif name == "generate_training_data":
                    return await self._generate_training_data(arguments)
                elif name == "get_provider_info":
                    return await self._get_provider_info(arguments)
                elif name == "validate_setup":
                    return await self._validate_setup(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _register_resources(self):
        """Register MCP resources."""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            resources = []
            
            # Add processed documents as resources
            incoming_path = Path(self.config.incoming_dir)
            if incoming_path.exists():
                for file_path in incoming_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.md', '.html', '.htm', '.docx', '.txt'}:
                        resources.append(Resource(
                            uri=f"finetune://document/{file_path.name}",
                            name=f"Document: {file_path.name}",
                            description=f"Source document ({file_path.suffix})",
                            mimeType=self._get_mime_type(file_path.suffix)
                        ))
            
            # Add training data outputs as resources
            output_path = Path(self.config.output_dir)
            if output_path.exists():
                for file_path in output_path.glob("*.json"):
                    resources.append(Resource(
                        uri=f"finetune://output/{file_path.name}",
                        name=f"Training Data: {file_path.name}",
                        description="Generated training data",
                        mimeType="application/json"
                    ))
            
            return resources
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource."""
            if uri.startswith("finetune://document/"):
                filename = uri.replace("finetune://document/", "")
                file_path = Path(self.config.incoming_dir) / filename
                if file_path.exists():
                    # Return processed document info
                    processed_doc = self.processor.preprocess_document(file_path)
                    if processed_doc:
                        return json.dumps(processed_doc, indent=2)
                    else:
                        return f"Could not process document: {filename}"
                else:
                    return f"Document not found: {filename}"
            
            elif uri.startswith("finetune://output/"):
                filename = uri.replace("finetune://output/", "")
                file_path = Path(self.config.output_dir) / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        return f.read()
                else:
                    return f"Output file not found: {filename}"
            
            return f"Unknown resource: {uri}"
    
    def _get_mime_type(self, suffix: str) -> str:
        """Get MIME type for file suffix."""
        mime_types = {
            '.pdf': 'application/pdf',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain'
        }
        return mime_types.get(suffix.lower(), 'application/octet-stream')
    
    async def _list_documents(self, args: Dict[str, Any]) -> List[TextContent]:
        """List documents in incoming directory."""
        include_metadata = args.get('include_metadata', False)
        
        documents = self.processor.get_documents()
        
        if not documents:
            return [TextContent(type="text", text="No documents found in incoming directory.")]
        
        result = {
            'total_documents': len(documents),
            'documents': []
        }
        
        for doc_path in documents:
            doc_info = {
                'filename': doc_path.name,
                'file_type': doc_path.suffix,
                'size_bytes': doc_path.stat().st_size
            }
            
            if include_metadata:
                try:
                    processed = self.processor.preprocess_document(doc_path)
                    if processed:
                        doc_info.update({
                            'word_count': processed['word_count'],
                            'char_count': processed['char_count'],
                            'chunks': len(processed['chunks']),
                            'sections': len(processed['metadata']['sections'])
                        })
                except Exception as e:
                    doc_info['error'] = str(e)
            
            result['documents'].append(doc_info)
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _configure_llm(self, args: Dict[str, Any]) -> List[TextContent]:
        """Configure LLM provider."""
        provider = args['provider']
        model = args.get('model')
        api_key = args.get('api_key')
        temperature = args.get('temperature', self.config.temperature)
        max_tokens = args.get('max_tokens', self.config.max_tokens)
        
        try:
            self.current_qa_generator = QAGenerator(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = {
                'status': 'success',
                'configuration': {
                    'provider': provider,
                    'model': self.current_qa_generator.model,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error configuring LLM: {str(e)}")]
    
    async def _process_document(self, args: Dict[str, Any]) -> List[TextContent]:
        """Process a single document."""
        filename = args['filename']
        chunk_size = args.get('chunk_size', self.config.chunk_size)
        
        file_path = Path(self.config.incoming_dir) / filename
        
        if not file_path.exists():
            return [TextContent(type="text", text=f"Document not found: {filename}")]
        
        try:
            processed_doc = self.processor.preprocess_document(file_path)
            if processed_doc:
                # Store processed document for later use
                output_path = Path(self.config.output_dir) / f"{filename}_processed.json"
                with open(output_path, 'w') as f:
                    json.dump(processed_doc, f, indent=2)
                
                result = {
                    'filename': filename,
                    'status': 'success',
                    'word_count': processed_doc['word_count'],
                    'char_count': processed_doc['char_count'],
                    'chunks': len(processed_doc['chunks']),
                    'sections': len(processed_doc['metadata']['sections']),
                    'output_file': str(output_path)
                }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                return [TextContent(type="text", text=f"Failed to process document: {filename}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error processing document: {str(e)}")]
    
    async def _generate_qa_pairs(self, args: Dict[str, Any]) -> List[TextContent]:
        """Generate Q&A pairs for a document."""
        filename = args['filename']
        questions_per_chunk = args.get('questions_per_chunk', self.config.questions_per_chunk)
        
        if not self.current_qa_generator:
            return [TextContent(type="text", text="No LLM configured. Please use configure_llm first.")]
        
        # Load processed document
        processed_file = Path(self.config.output_dir) / f"{filename}_processed.json"
        
        if not processed_file.exists():
            # Process the document first
            await self._process_document({'filename': filename})
        
        try:
            with open(processed_file, 'r') as f:
                processed_doc = json.load(f)
            
            all_qa_pairs = []
            
            for chunk in processed_doc['chunks']:
                qa_pairs = self.current_qa_generator.generate_qa_pairs(
                    chunk, 
                    processed_doc['metadata'],
                    questions_per_chunk
                )
                all_qa_pairs.extend(qa_pairs)
            
            # Save Q&A pairs
            qa_output_path = Path(self.config.output_dir) / f"{filename}_qa_pairs.json"
            with open(qa_output_path, 'w') as f:
                json.dump(all_qa_pairs, f, indent=2)
            
            result = {
                'filename': filename,
                'status': 'success',
                'qa_pairs_generated': len(all_qa_pairs),
                'questions_per_chunk': questions_per_chunk,
                'chunks_processed': len(processed_doc['chunks']),
                'output_file': str(qa_output_path)
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating Q&A pairs: {str(e)}")]
    
    async def _generate_training_data(self, args: Dict[str, Any]) -> List[TextContent]:
        """Generate complete training dataset."""
        output_filename = args.get('output_filename', 'training_data.json')
        filter_files = args.get('filter_files', [])
        
        if not self.current_qa_generator:
            return [TextContent(type="text", text="No LLM configured. Please use configure_llm first.")]
        
        try:
            # Get documents to process
            all_documents = self.processor.get_documents()
            
            if filter_files:
                documents = [doc for doc in all_documents if doc.name in filter_files]
            else:
                documents = all_documents
            
            if not documents:
                return [TextContent(type="text", text="No documents to process.")]
            
            # Process all documents
            processed_documents = []
            
            for doc_path in documents:
                processed_doc = self.processor.preprocess_document(doc_path)
                if processed_doc:
                    processed_documents.append(processed_doc)
            
            if not processed_documents:
                return [TextContent(type="text", text="No documents were successfully processed.")]
            
            # Generate training data
            output_path = Path(self.config.output_dir) / output_filename
            training_data = self.current_qa_generator.generate_training_data(
                processed_documents,
                str(output_path)
            )
            
            result = {
                'status': 'success',
                'documents_processed': len(processed_documents),
                'total_qa_pairs': training_data['metadata']['total_qa_pairs'],
                'output_file': str(output_path),
                'llm_provider': training_data['metadata']['llm_provider'],
                'model_used': training_data['metadata']['model_used']
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating training data: {str(e)}")]
    
    async def _get_provider_info(self, args: Dict[str, Any]) -> List[TextContent]:
        """Get provider information."""
        specific_provider = args.get('provider')
        
        if specific_provider:
            try:
                models = QAGenerator.get_provider_models(specific_provider)
                result = {
                    'provider': specific_provider,
                    'available_models': models,
                    'default_model': QAGenerator(provider=specific_provider).model
                }
            except Exception as e:
                result = {
                    'provider': specific_provider,
                    'error': str(e)
                }
        else:
            providers = QAGenerator.get_available_providers()
            result = {
                'available_providers': providers,
                'provider_details': {}
            }
            
            for provider in providers:
                try:
                    models = QAGenerator.get_provider_models(provider)
                    result['provider_details'][provider] = {
                        'available_models': models[:5],  # Limit to first 5 models
                        'total_models': len(models)
                    }
                except Exception as e:
                    result['provider_details'][provider] = {'error': str(e)}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _validate_setup(self, args: Dict[str, Any]) -> List[TextContent]:
        """Validate setup and dependencies."""
        validation_results = {
            'directories': {},
            'dependencies': {},
            'api_keys': {},
            'overall_status': 'unknown'
        }
        
        # Check directories
        incoming_path = Path(self.config.incoming_dir)
        output_path = Path(self.config.output_dir)
        
        validation_results['directories'] = {
            'incoming_dir': {
                'path': str(incoming_path),
                'exists': incoming_path.exists(),
                'writable': incoming_path.exists() and os.access(incoming_path, os.W_OK)
            },
            'output_dir': {
                'path': str(output_path),
                'exists': output_path.exists(),
                'writable': output_path.exists() and os.access(output_path, os.W_OK)
            }
        }
        
        # Check dependencies
        required_packages = ['openai', 'anthropic', 'google.generativeai', 'ollama']
        
        for package in required_packages:
            try:
                __import__(package.replace('.', '_') if '.' in package else package)
                validation_results['dependencies'][package] = 'available'
            except ImportError:
                validation_results['dependencies'][package] = 'missing'
        
        # Check API keys
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
        }
        
        for key, value in api_keys.items():
            validation_results['api_keys'][key] = 'set' if value else 'not_set'
        
        # Determine overall status
        dirs_ok = all(d['exists'] and d['writable'] for d in validation_results['directories'].values())
        deps_ok = all(status == 'available' for status in validation_results['dependencies'].values())
        at_least_one_key = any(status == 'set' for status in validation_results['api_keys'].values())
        
        if dirs_ok and deps_ok and at_least_one_key:
            validation_results['overall_status'] = 'ready'
        elif dirs_ok and deps_ok:
            validation_results['overall_status'] = 'needs_api_keys'
        else:
            validation_results['overall_status'] = 'needs_setup'
        
        return [TextContent(type="text", text=json.dumps(validation_results, indent=2))]

async def main():
    """Main entry point for the MCP server."""
    # Load configuration
    config = ServerConfig()
    
    # Create and run server
    server_instance = FinetuneIngestServer(config)
    
    async with server_instance.server.stdio() as streams:
        await server_instance.server.run(
            streams[0], streams[1], server_instance.server.request_handlers
        )

if __name__ == "__main__":
    asyncio.run(main()) 