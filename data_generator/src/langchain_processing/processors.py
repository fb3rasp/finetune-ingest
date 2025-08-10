from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from .document_loaders import LangChainDocumentLoader
from .text_splitters import EnhancedTextSplitter
from common.llm.llm_providers import UnifiedLLMProvider, LLMProvider
from .qa_chains import QAGenerationChain
from common.utils.helpers import log_message, save_json_atomic, load_json_if_exists


class LangChainProcessor:
    def __init__(
        self,
        incoming_dir: str = "/data/incoming",
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
        process_dir: str = "/data/process",
        results_file: str = "/data/results/training_data.json",
        # Prompts
        qa_system_message: Optional[str] = None,
        qa_additional_instructions: Optional[str] = None,
        qa_custom_template: Optional[str] = None,
        **kwargs,
    ):
        self.incoming_dir = incoming_dir
        self.process_dir = process_dir
        self.results_file = results_file
        self.splitting_strategy = splitting_strategy
        self.use_batch_processing = use_batch_processing
        self.qa_system_message = qa_system_message
        self.qa_additional_instructions = qa_additional_instructions
        self.qa_custom_template = qa_custom_template

        self.document_loader = LangChainDocumentLoader(incoming_dir)
        self.text_splitter = EnhancedTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.llm_provider = UnifiedLLMProvider(provider=provider, model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.qa_chain = QAGenerationChain(
            llm_provider=self.llm_provider,
            questions_per_chunk=questions_per_chunk,
            system_message=self.qa_system_message,
            extra_instructions=self.qa_additional_instructions,
            custom_template=self.qa_custom_template,
        )

    def _cache_dir_for(self, file_path: Path) -> Path:
        base_dir = Path(self.process_dir) / "cache"
        cache_dir = base_dir / (file_path.name + ".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def process_single_document(self, file_path: Path, resume: bool = False) -> Optional[Dict]:
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
                documents = cached_docs.get("documents")
                enhanced_metadata = cached_docs.get("metadata")
                chunks = cached_chunks

        if documents is None or enhanced_metadata is None:
            documents_obj, _ = self.document_loader.load_document(file_path)
            if not documents_obj:
                return None
            enhanced_metadata = self.document_loader.extract_enhanced_metadata(documents_obj) if hasattr(self.document_loader, 'extract_enhanced_metadata') else {
                'file_name': file_path.name,
                'source_file': str(file_path),
                'file_type': file_path.suffix.lower(),
            }
            documents = [{"page_content": d.page_content, "metadata": dict(getattr(d, "metadata", {}))} for d in documents_obj]
            save_json_atomic({"documents": documents, "metadata": enhanced_metadata}, str(docs_cache_path))

        if chunks is None:
            try:
                from langchain.schema import Document as LCDocument
                reconstructed = [LCDocument(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in documents]
                chunks = self.text_splitter.split_documents(reconstructed, strategy=self.splitting_strategy)
            except Exception:
                text = "\n\n".join(d.get("page_content", "") for d in documents)
                file_type = (enhanced_metadata or {}).get('file_type', '.txt')
                chunks = self.text_splitter.split_text_adaptive(text, file_type=file_type) if hasattr(self.text_splitter, 'split_text_adaptive') else []
            save_json_atomic(chunks, str(chunks_cache_path))

        return {
            'metadata': enhanced_metadata,
            'chunks': chunks,
            'processing_info': {
                'splitting_strategy': self.splitting_strategy,
                'chunk_size': self.text_splitter.chunk_size,
                'chunk_overlap': self.text_splitter.chunk_overlap,
                'processed_at': datetime.now().isoformat(),
            },
        }

    def process_all_documents(self, resume: bool = False) -> List[Dict]:
        processed: List[Dict] = []
        for file_path in self.document_loader.get_documents():
            result = self.process_single_document(file_path, resume=resume)
            if result:
                processed.append(result)
        return processed

    def generate_qa_training_data(self, processed_documents: Optional[List[Dict]] = None, output_file: Optional[str] = None, resume: bool = False) -> Dict:
        if processed_documents is None:
            processed_documents = self.process_all_documents(resume=resume)
        if not processed_documents:
            return {}
        output_path = output_file or self.results_file
        training_data = {
            'metadata': {
                'generated_by': 'data_generator',
                'generated_at': datetime.now().isoformat(),
                'llm_provider': self.llm_provider.provider.value,
                'model_used': self.llm_provider.model,
                'processing_config': {
                    'chunk_size': self.text_splitter.chunk_size,
                    'chunk_overlap': self.text_splitter.chunk_overlap,
                    'splitting_strategy': self.splitting_strategy,
                    'questions_per_chunk': self.qa_chain.questions_per_chunk,
                    'temperature': self.llm_provider.temperature,
                    'max_tokens': self.llm_provider.max_tokens,
                },
                'num_documents': len(processed_documents),
                'total_qa_pairs': 0,
            },
            'documents': [],
            'training_pairs': [],
        }
        all_pairs: List[Dict] = []
        existing = load_json_if_exists(output_path) if resume else None
        processed_chunk_keys = set()
        if existing:
            all_pairs = existing.get('training_pairs', [])
            for qa in all_pairs:
                cid = qa.get('chunk_id'); fname = qa.get('file_name')
                if cid is not None and fname:
                    processed_chunk_keys.add(f"{fname}::{cid}")
            training_data['documents'] = existing.get('documents', [])

        for doc in processed_documents:
            file_name = doc['metadata'].get('file_name')
            doc_pairs: List[Dict] = []
            if self.use_batch_processing:
                def _on_chunk_done(pairs_for_chunk: List[Dict]):
                    nonlocal all_pairs, training_data
                    all_pairs.extend(pairs_for_chunk)
                    training_data['training_pairs'] = all_pairs
                    training_data['metadata']['total_qa_pairs'] = len(all_pairs)
                    save_json_atomic(training_data, output_path)
                chunks_to_process = [c for c in doc['chunks'] if not (resume and f"{file_name}::{c.get('chunk_id')}" in processed_chunk_keys)]
                batch = self.qa_chain.generate_batch_qa_pairs(chunks_to_process, doc['metadata'], on_chunk_done=_on_chunk_done)
                doc_pairs.extend(batch)
            else:
                for chunk in doc['chunks']:
                    if resume and f"{file_name}::{chunk.get('chunk_id')}" in processed_chunk_keys:
                        continue
                    pairs = self.qa_chain.generate_qa_pairs(chunk, doc['metadata'])
                    doc_pairs.extend(pairs)
                    if pairs:
                        all_pairs.extend(pairs)
                        training_data['training_pairs'] = all_pairs
                        training_data['metadata']['total_qa_pairs'] = len(all_pairs)
                        save_json_atomic(training_data, output_path)

            existing_docs = {d.get('file_info', {}).get('file_name') for d in training_data['documents']}
            total_pairs_for_doc = sum(1 for qa in all_pairs if qa.get('file_name') == file_name)
            if file_name not in existing_docs:
                training_data['documents'].append({
                    'file_info': doc['metadata'],
                    'chunk_count': len(doc['chunks']),
                    'qa_pairs_count': total_pairs_for_doc,
                    'processing_info': doc.get('processing_info', {}),
                })
            else:
                for d in training_data['documents']:
                    if d.get('file_info', {}).get('file_name') == file_name:
                        d['qa_pairs_count'] = total_pairs_for_doc
                        break

        training_data['training_pairs'] = all_pairs
        training_data['metadata']['total_qa_pairs'] = len(all_pairs)
        save_json_atomic(training_data, output_path)
        return training_data


