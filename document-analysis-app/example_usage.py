#!/usr/bin/env python3
"""
Example usage of the training data generation system with multiple LLM providers.
This script demonstrates all major features of the finetune-ingest app.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.preprocess import DocumentProcessor
from data_processing.qa_generator import QAGenerator, LLMProvider

def setup_example_documents(incoming_dir):
    """Create example documents for testing."""
    incoming_path = Path(incoming_dir)
    incoming_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating example documents in {incoming_dir}")
    
    # Create example markdown document
    md_content = """# Machine Learning Fundamentals

## Introduction
Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. This revolutionary technology has transformed numerous industries and continues to shape our digital landscape.

## Core Concepts

### Types of Learning

#### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. The algorithm learns from example input-output pairs and makes predictions on new, unseen data.

**Common algorithms include:**
- Linear Regression: Predicts continuous values
- Decision Trees: Creates decision rules in a tree-like structure
- Support Vector Machines (SVM): Finds optimal decision boundaries
- Neural Networks: Mimics the human brain's learning process

#### Unsupervised Learning
Unsupervised learning finds patterns in data without labeled examples. It discovers hidden structures in data where you don't know the desired output.

**Key techniques include:**
- Clustering (K-means, Hierarchical): Groups similar data points
- Dimensionality Reduction (PCA, t-SNE): Reduces data complexity
- Association Rules: Finds relationships between different data points

#### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by taking actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or penalties.

## Real-World Applications

### Natural Language Processing
- **Text Analysis**: Sentiment analysis, topic modeling
- **Machine Translation**: Google Translate, DeepL
- **Chatbots**: Customer service, virtual assistants
- **Content Generation**: GPT models, text summarization

### Computer Vision
- **Image Recognition**: Photo tagging, medical imaging
- **Object Detection**: Autonomous vehicles, security systems
- **Facial Recognition**: Biometric authentication
- **Medical Imaging**: Cancer detection, radiology assistance

### Business Applications
- **Recommendation Systems**: Netflix, Amazon, Spotify
- **Fraud Detection**: Credit card transactions, insurance claims
- **Predictive Analytics**: Sales forecasting, maintenance scheduling
- **Price Optimization**: Dynamic pricing, demand forecasting

## Getting Started with ML

### Prerequisites
1. **Programming Skills**: Python is the most popular language
2. **Mathematics**: Statistics, linear algebra, calculus
3. **Data Understanding**: Data cleaning, visualization
4. **Domain Knowledge**: Understanding the problem you're solving

### Learning Path
1. **Foundation**: Learn Python and basic statistics
2. **Libraries**: Master pandas, numpy, scikit-learn
3. **Practice**: Work on Kaggle competitions
4. **Specialization**: Choose an area (NLP, computer vision, etc.)
5. **Advanced Topics**: Deep learning, MLOps, deployment

### Best Practices
- Start with simple models before moving to complex ones
- Always validate your models on unseen data
- Understand your data before applying algorithms
- Document your experiments and results
- Consider ethical implications of your models

## Future Trends

Machine learning continues to evolve rapidly with emerging trends like:
- **Large Language Models**: GPT, BERT, and their successors
- **Edge AI**: Running ML models on mobile devices
- **AutoML**: Automated machine learning pipelines
- **Explainable AI**: Making AI decisions interpretable
- **Federated Learning**: Training models without centralizing data

## Conclusion

Machine learning is a powerful tool that, when properly understood and applied, can solve complex problems across various domains. The key to success lies in understanding both the technical aspects and the practical considerations of implementing ML solutions in real-world scenarios.
"""
    
    md_file = incoming_path / "ml_fundamentals.md"
    md_file.write_text(md_content)
    
    # Create example HTML document
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Data Science Best Practices</title>
</head>
<body>
    <h1>Data Science Best Practices</h1>
    
    <h2>Data Collection and Preparation</h2>
    <p>Quality data is the foundation of any successful data science project. Poor data leads to poor results, regardless of how sophisticated your algorithms are.</p>
    
    <h3>Data Quality Checklist</h3>
    <ul>
        <li><strong>Completeness</strong>: Ensure you have sufficient data</li>
        <li><strong>Accuracy</strong>: Verify data is correct and up-to-date</li>
        <li><strong>Consistency</strong>: Check for uniform formatting and standards</li>
        <li><strong>Relevance</strong>: Confirm data aligns with your objectives</li>
    </ul>
    
    <h2>Exploratory Data Analysis (EDA)</h2>
    <p>Before building models, spend time understanding your data through visualization and statistical analysis.</p>
    
    <h3>Key EDA Steps</h3>
    <ol>
        <li>Examine data types and distributions</li>
        <li>Identify missing values and outliers</li>
        <li>Explore relationships between variables</li>
        <li>Create informative visualizations</li>
    </ol>
    
    <h2>Model Development</h2>
    <p>Follow a systematic approach to model development that includes proper validation and testing.</p>
    
    <h3>Development Process</h3>
    <ul>
        <li>Define clear success metrics</li>
        <li>Start with simple baseline models</li>
        <li>Use cross-validation for model selection</li>
        <li>Implement proper train/validation/test splits</li>
        <li>Document all experiments and results</li>
    </ul>
    
    <h2>Model Deployment and Monitoring</h2>
    <p>Deploying a model is just the beginning. Continuous monitoring ensures your model remains effective over time.</p>
    
    <h3>Monitoring Checklist</h3>
    <ul>
        <li>Track model performance metrics</li>
        <li>Monitor data drift and concept drift</li>
        <li>Set up alerts for anomalies</li>
        <li>Plan for model retraining</li>
    </ul>
</body>
</html>"""
    
    html_file = incoming_path / "data_science_practices.html"
    html_file.write_text(html_content)
    
    # Create example text document
    txt_content = """Deep Learning Neural Networks

INTRODUCTION:
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain.

NEURAL NETWORK ARCHITECTURE:
A neural network consists of interconnected nodes (neurons) organized in layers. The basic components include:
- Input Layer: Receives the raw data
- Hidden Layers: Process the data through weighted connections
- Output Layer: Produces the final prediction or classification

TRAINING PROCESS:
Neural networks learn through a process called backpropagation:
1. Forward Pass: Data flows through the network to produce output
2. Loss Calculation: Compare output with expected results
3. Backward Pass: Adjust weights to minimize error
4. Iteration: Repeat process until convergence

ACTIVATION FUNCTIONS:
Activation functions introduce non-linearity into the network:
- ReLU (Rectified Linear Unit): Most commonly used
- Sigmoid: Good for binary classification
- Tanh: Similar to sigmoid but centered at zero
- Softmax: Used in multi-class classification

COMMON ARCHITECTURES:
- Feedforward Networks: Basic fully connected layers
- Convolutional Neural Networks (CNNs): Excellent for image processing
- Recurrent Neural Networks (RNNs): Good for sequential data
- Transformer Networks: State-of-the-art for natural language processing

APPLICATIONS:
Deep learning has revolutionized many fields including:
- Computer Vision: Image classification, object detection
- Natural Language Processing: Language translation, text generation
- Speech Recognition: Voice assistants, transcription services
- Recommendation Systems: Personalized content suggestions
- Game Playing: Chess, Go, video games

CHALLENGES:
Despite its success, deep learning faces several challenges:
- Requires large amounts of data
- Computationally expensive
- Black box nature makes interpretation difficult
- Prone to overfitting without proper regularization
- Sensitive to hyperparameter selection

FUTURE DIRECTIONS:
The field continues to evolve with new architectures and techniques:
- Attention mechanisms and transformers
- Generative adversarial networks (GANs)
- Self-supervised learning
- Neural architecture search
- Federated learning approaches"""
    
    txt_file = incoming_path / "deep_learning_guide.txt"
    txt_file.write_text(txt_content)
    
    print(f"✅ Created {len(list(incoming_path.glob('*')))} example documents")
    return list(incoming_path.glob('*'))

def demonstrate_basic_usage():
    """Demonstrate basic document processing and Q&A generation."""
    print("\n🚀 Basic Usage Demonstration")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Setup example documents
    incoming_dir = "./incoming"
    documents = setup_example_documents(incoming_dir)
    
    # Initialize processor
    print("\n📋 Initializing Document Processor...")
    processor = DocumentProcessor(incoming_dir)
    
    # Process documents
    processed_docs = []
    for doc_path in documents:
        print(f"📄 Processing: {doc_path.name}")
        processed_doc = processor.preprocess_document(doc_path)
        if processed_doc:
            processed_docs.append(processed_doc)
            print(f"   ✅ Extracted {processed_doc['word_count']} words in {len(processed_doc['chunks'])} chunks")
    
    return processed_docs

def demonstrate_multi_llm_usage(processed_docs):
    """Demonstrate usage with different LLM providers."""
    print("\n🤖 Multi-LLM Provider Demonstration")
    print("=" * 50)
    
    # Provider configurations
    provider_configs = [
        {
            'name': 'OpenAI GPT-4',
            'provider': LLMProvider.OPENAI,
            'model': 'gpt-4',
            'output_file': './output/training_data_openai.json',
            'enabled': bool(os.getenv('OPENAI_API_KEY'))
        },
        {
            'name': 'Claude 3 Sonnet',
            'provider': LLMProvider.CLAUDE, 
            'model': 'claude-3-sonnet-20240229',
            'output_file': './output/training_data_claude.json',
            'enabled': bool(os.getenv('ANTHROPIC_API_KEY'))
        },
        {
            'name': 'Google Gemini Pro',
            'provider': LLMProvider.GEMINI,
            'model': 'gemini-pro', 
            'output_file': './output/training_data_gemini.json',
            'enabled': bool(os.getenv('GOOGLE_API_KEY'))
        },
        {
            'name': 'Local Llama3',
            'provider': LLMProvider.LOCAL,
            'model': 'llama3',
            'output_file': './output/training_data_local.json',
            'enabled': True  # Assume local is always available
        }
    ]
    
    # Create output directory
    Path('./output').mkdir(exist_ok=True)
    
    successful_generations = 0
    
    # Generate training data with each available provider
    for config in provider_configs:
        print(f"\n🔧 Testing {config['name']}...")
        
        if not config['enabled']:
            print(f"   ⚠️  Skipped: No API key configured")
            continue
        
        try:
            qa_generator = QAGenerator(
                provider=config['provider'],
                model=config['model'],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Generate training data (limit to first document for demo)
            demo_docs = processed_docs[:1]  # Only use first document for demo
            training_data = qa_generator.generate_training_data(
                demo_docs, 
                config['output_file']
            )
            
            print(f"   ✅ Generated {len(training_data['training_pairs'])} Q&A pairs")
            print(f"   💾 Saved to: {config['output_file']}")
            
            # Show sample Q&A
            if training_data['training_pairs']:
                sample_qa = training_data['training_pairs'][0]
                print(f"   📝 Sample Q: {sample_qa['question'][:100]}...")
                print(f"   📝 Sample A: {sample_qa['answer'][:100]}...")
            
            successful_generations += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if config['provider'] == LLMProvider.LOCAL:
                print("      💡 Tip: Make sure Ollama is running and model is pulled:")
                print(f"         ollama pull {config['model']}")
    
    if successful_generations > 0:
        print(f"\n🎉 Successfully generated training data with {successful_generations} provider(s)")
    else:
        print(f"\n⚠️  No providers were successful. Check your API keys and configurations.")

def demonstrate_advanced_features():
    """Demonstrate advanced features like custom chunking and parameters."""
    print("\n⚙️ Advanced Features Demonstration")
    print("=" * 50)
    
    # Custom chunking demonstration
    processor = DocumentProcessor("./incoming")
    documents = processor.get_documents()
    
    if documents:
        doc = documents[0]
        print(f"\n📊 Chunking Analysis for: {doc.name}")
        
        # Test different chunk sizes
        chunk_sizes = [500, 1000, 1500]
        for chunk_size in chunk_sizes:
            processed = processor.preprocess_document(doc)
            if processed:
                # Re-chunk with custom size
                custom_chunks = processor.chunk_text(processed['full_text'], chunk_size=chunk_size)
                print(f"   Chunk size {chunk_size}: {len(custom_chunks)} chunks")
    
    # Parameter comparison demonstration
    if os.getenv('OPENAI_API_KEY'):
        print(f"\n🎛️ Parameter Comparison (using OpenAI)")
        
        # Test different temperatures
        temperatures = [0.3, 0.7, 1.0]
        
        if documents and processed_docs:
            chunk = processed_docs[0]['chunks'][0]
            metadata = processed_docs[0]['metadata']
            
            for temp in temperatures:
                try:
                    qa_gen = QAGenerator(
                        provider=LLMProvider.OPENAI,
                        temperature=temp,
                        max_tokens=1000
                    )
                    
                    qa_pairs = qa_gen.generate_qa_pairs(chunk, metadata, num_questions=1)
                    if qa_pairs:
                        print(f"   Temperature {temp}: Generated {len(qa_pairs)} questions")
                        print(f"      Q: {qa_pairs[0]['question'][:80]}...")
                
                except Exception as e:
                    print(f"   Temperature {temp}: Failed - {e}")

def show_usage_examples():
    """Show command-line usage examples."""
    print("\n📚 Command-Line Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            'description': 'List available providers',
            'command': 'python src/main.py --list-providers'
        },
        {
            'description': 'List models for OpenAI',
            'command': 'python src/main.py --provider openai --list-models'
        },
        {
            'description': 'Generate with OpenAI GPT-4',
            'command': 'python src/main.py --provider openai --model gpt-4 --temperature 0.7'
        },
        {
            'description': 'Generate with Claude',
            'command': 'python src/main.py --provider claude --model claude-3-sonnet-20240229'
        },
        {
            'description': 'Generate with custom parameters',
            'command': 'python src/main.py --provider gemini --questions-per-chunk 5 --chunk-size 800'
        },
        {
            'description': 'Use local LLM via Ollama',
            'command': 'python src/main.py --provider local --model llama3'
        }
    ]
    
    for example in examples:
        print(f"\n📝 {example['description']}:")
        print(f"   {example['command']}")

def main():
    """Main demonstration function."""
    print("🎯 Finetune-Ingest Training Data Generation Demo")
    print("=" * 60)
    
    try:
        # Basic usage demonstration
        processed_docs = demonstrate_basic_usage()
        
        if processed_docs:
            # Multi-LLM demonstration
            demonstrate_multi_llm_usage(processed_docs)
            
            # Advanced features
            demonstrate_advanced_features()
        
        # Show command-line examples
        show_usage_examples()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. 📝 Edit .env file with your API keys")
        print("2. 📁 Add your documents to ./incoming directory")
        print("3. 🚀 Run: python src/main.py --provider <your_provider>")
        print("4. 🔧 Or start MCP server: ./start_mcp_server.sh")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 