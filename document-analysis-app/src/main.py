import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Legacy imports - keeping for backward compatibility
from data_processing.preprocess import DocumentProcessor
from data_processing.qa_generator import QAGenerator, LLMProvider
from utils.helpers import log_message, save_results

# New LangChain imports - uncomment to switch to LangChain implementation
# from langchain_processing import LangChainProcessor, UnifiedLLMProvider, LLMProvider as LCLLMProvider

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Generate training data for LLM fine-tuning')
    parser.add_argument('--incoming-dir', default='./incoming', 
                       help='Directory containing source documents')
    parser.add_argument('--output-file', default='./training_data.json',
                       help='Output file for training data')
    
    # LLM Provider Configuration
    parser.add_argument('--provider', 
                       choices=['openai', 'claude', 'gemini', 'local'],
                       default='openai',
                       help='LLM provider to use')
    parser.add_argument('--model', 
                       help='Specific model name (uses provider default if not specified)')
    parser.add_argument('--api-key', 
                       help='API key for the provider (uses env vars if not specified)')
    
    # Generation Parameters
    parser.add_argument('--questions-per-chunk', type=int, default=3,
                       help='Number of questions to generate per text chunk')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Size of text chunks for processing')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for LLM generation')
    parser.add_argument('--max-tokens', type=int, default=2000,
                       help='Maximum tokens for LLM response')
    
    # Utility options
    parser.add_argument('--list-models', action='store_true',
                       help='List available models for the specified provider')
    parser.add_argument('--list-providers', action='store_true',
                       help='List all available LLM providers')
    
    args = parser.parse_args()
    
    # Handle list providers command
    if args.list_providers:
        providers = QAGenerator.get_available_providers()
        print("Available LLM providers:")
        for provider in providers:
            print(f"  - {provider}")
            models = QAGenerator.get_provider_models(provider)
            for model in models[:3]:  # Show first 3 models
                print(f"    • {model}")
            if len(models) > 3:
                print(f"    • ... and {len(models) - 3} more")
        return
    
    # Handle list models command
    if args.list_models:
        models = QAGenerator.get_provider_models(args.provider)
        print(f"Available models for {args.provider}:")
        for model in models:
            print(f"  - {model}")
        return
    
    log_message("Initializing Document Analysis App for Training Data Generation...")
    log_message(f"Using provider: {args.provider} (Legacy Implementation)")
    log_message("Note: For enhanced LangChain features, use main_langchain.py")
    
    # Initialize components
    processor = DocumentProcessor(args.incoming_dir)
    
    try:
        qa_generator = QAGenerator(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        log_message(f"Initialized {args.provider} with model: {qa_generator.model}")
    except Exception as e:
        log_message(f"Error initializing QA generator: {e}")
        print("\nTroubleshooting:")
        print(f"1. Make sure you have the API key set for {args.provider}")
        print("   Set it in .env file or use --api-key argument")
        print(f"2. Check that the model name is correct (use --list-models)")
        print(f"3. Verify the required packages are installed (pip install -r requirements.txt)")
        
        if args.provider == 'local':
            print("4. For local provider, make sure Ollama is running:")
            print("   ollama serve")
            print(f"   ollama pull {args.model or 'llama3'}")
        return
    
    # Process documents
    log_message("Scanning for documents...")
    documents = processor.get_documents()
    
    if not documents:
        log_message("No documents found in incoming directory")
        log_message(f"Place your documents (.pdf, .md, .html, .docx, .txt) in: {args.incoming_dir}")
        
        # Create sample document for demonstration
        incoming_path = Path(args.incoming_dir)
        incoming_path.mkdir(parents=True, exist_ok=True)
        
        sample_doc = incoming_path / "sample_ml_guide.md"
        if not sample_doc.exists():
            sample_content = """# Machine Learning Guide

## Introduction
Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. This technology has revolutionized various industries and continues to shape our digital world.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common algorithms include:
- Linear Regression
- Decision Trees
- Support Vector Machines
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds patterns in data without labeled examples. Key techniques include:
- Clustering (K-means, Hierarchical)
- Dimensionality Reduction (PCA, t-SNE)
- Association Rules

### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by taking actions in an environment to maximize cumulative reward.

## Applications

Machine learning has numerous real-world applications:

1. **Natural Language Processing**: Text analysis, translation, chatbots
2. **Computer Vision**: Image recognition, object detection, medical imaging
3. **Recommendation Systems**: Personalized content, product recommendations
4. **Autonomous Vehicles**: Self-driving cars, drones
5. **Finance**: Fraud detection, algorithmic trading, credit scoring
6. **Healthcare**: Drug discovery, diagnosis assistance, treatment optimization

## Getting Started

To begin with machine learning:
1. Learn Python and relevant libraries (scikit-learn, pandas, numpy)
2. Understand basic statistics and linear algebra
3. Practice with datasets from Kaggle or UCI ML Repository
4. Start with simple projects and gradually increase complexity
5. Join ML communities and continue learning

## Conclusion

Machine learning is a powerful tool that continues to evolve rapidly. With proper understanding and practice, it can solve complex problems across various domains and create significant value for businesses and society.
"""
            sample_doc.write_text(sample_content)
            log_message(f"Created sample document: {sample_doc}")
            log_message("Run the command again to process this sample document")
        
        return
    
    processed_documents = []
    
    log_message(f"Processing {len(documents)} documents...")
    for doc_path in documents:
        # Update chunk size if specified
        original_chunk_method = processor.chunk_text
        if args.chunk_size != 1000:
            def custom_chunk(text, chunk_size=args.chunk_size, overlap=200):
                return original_chunk_method(text, chunk_size, overlap)
            processor.chunk_text = custom_chunk
        
        processed_doc = processor.preprocess_document(doc_path)
        if processed_doc:
            processed_documents.append(processed_doc)
            log_message(f"Processed: {doc_path.name} ({processed_doc['word_count']} words, {len(processed_doc['chunks'])} chunks)")
        
        # Restore original method
        processor.chunk_text = original_chunk_method
    
    if not processed_documents:
        log_message("No documents were successfully processed")
        return
    
    # Generate training data
    log_message("Generating Q&A pairs...")
    log_message(f"Configuration: {args.questions_per_chunk} questions per chunk, {args.temperature} temperature")
    
    # Temporarily modify the generator to use custom questions per chunk
    original_generate = qa_generator.generate_qa_pairs
    def custom_generate(chunk, metadata, num_questions=args.questions_per_chunk):
        return original_generate(chunk, metadata, num_questions)
    qa_generator.generate_qa_pairs = custom_generate
    
    training_data = qa_generator.generate_training_data(
        processed_documents, 
        args.output_file
    )
    
    # Generate summary report
    summary = {
        'configuration': {
            'provider': args.provider,
            'model': qa_generator.model,
            'questions_per_chunk': args.questions_per_chunk,
            'chunk_size': args.chunk_size,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens
        },
        'results': {
            'total_documents': len(processed_documents),
            'total_qa_pairs': training_data['metadata']['total_qa_pairs'],
            'documents_processed': [doc['file_info']['file_name'] for doc in training_data['documents']],
            'output_file': args.output_file
        }
    }
    
    summary_file = args.output_file.replace('.json', '_summary.json')
    save_results(summary, summary_file)
    
    log_message(f"Training data generation complete!")
    log_message(f"Generated {summary['results']['total_qa_pairs']} Q&A pairs from {summary['results']['total_documents']} documents")
    log_message(f"Training data saved to: {args.output_file}")
    log_message(f"Summary saved to: {summary_file}")
    
    # Show sample Q&A pairs
    if training_data['training_pairs']:
        log_message("\nSample Q&A pairs:")
        for i, qa in enumerate(training_data['training_pairs'][:2]):
            print(f"\nSample {i+1}:")
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer'][:150]}{'...' if len(qa['answer']) > 150 else ''}")
            print(f"Source: {qa['file_name']} (chunk {qa['chunk_id']})")

if __name__ == "__main__":
    main()