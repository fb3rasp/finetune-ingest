import json
import os
from typing import Dict, List, Optional, Union
from enum import Enum
from pathlib import Path
from utils.helpers import log_message

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL = "local"

class QAGenerator:
    """Generates training Q&A pairs from document content using multiple LLM providers."""
    
    def __init__(self, 
                 provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize QA generator with specified provider.
        
        Args:
            provider: LLM provider to use (openai, claude, gemini, local)
            model: Specific model name (provider-specific defaults if None)
            api_key: API key (will use env vars if not provided)
            **kwargs: Additional provider-specific parameters
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.kwargs = kwargs
        
        # Initialize the appropriate client
        self._init_client(api_key)
    
    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for each provider."""
        defaults = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.CLAUDE: "claude-3-5-sonnet-20241022",
            LLMProvider.GEMINI: "gemini-pro",
            LLMProvider.LOCAL: "llama3"
        }
        return defaults.get(provider, "gpt-4")
    
    def _init_client(self, api_key: Optional[str]):
        """Initialize the appropriate API client."""
        try:
            if self.provider == LLMProvider.OPENAI:
                import openai
                self.client = openai.OpenAI(
                    api_key=api_key or os.getenv('OPENAI_API_KEY')
                )
                
            elif self.provider == LLMProvider.CLAUDE:
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
                )
                
            elif self.provider == LLMProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=api_key or os.getenv('GOOGLE_API_KEY'))
                self.client = genai.GenerativeModel(self.model)
                
            elif self.provider == LLMProvider.LOCAL:
                import ollama
                self.client = ollama.Client()
                
        except ImportError as e:
            raise ImportError(f"Required package for {self.provider.value} not installed: {e}")
        except Exception as e:
            raise Exception(f"Failed to initialize {self.provider.value} client: {e}")
    
    def generate_qa_pairs(self, chunk: Dict, metadata: Dict, num_questions: int = 3) -> List[Dict]:
        """Generate Q&A pairs from a text chunk."""
        
        prompt = self._create_qa_prompt(chunk['text'], num_questions)
        
        try:
            response = self._call_llm(prompt)
            qa_pairs = self._parse_qa_response(response)
            
            # Add metadata to each Q&A pair
            for qa in qa_pairs:
                qa.update({
                    'source_file': metadata['source_file'],
                    'file_name': metadata['file_name'],
                    'file_type': metadata['file_type'],
                    'chunk_id': chunk['chunk_id'],
                    'chunk_start': chunk['start_char'],
                    'chunk_end': chunk['end_char'],
                    'source_text': chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                    'generated_by': {
                        'provider': self.provider.value,
                        'model': self.model
                    }
                })
            
            return qa_pairs
            
        except Exception as e:
            log_message(f"Error generating Q&A pairs: {str(e)}")
            return []
    
    def _create_qa_prompt(self, text: str, num_questions: int) -> str:
        """Create a prompt for Q&A generation."""
        return f"""You are an expert at creating high-quality training data for language models. 
Based on the following text, generate exactly {num_questions} question-answer pairs that would help train a chatbot to understand and respond about this content.

Guidelines:
1. Questions should be diverse (factual, analytical, explanatory)
2. Answers should be comprehensive but concise
3. Include both specific details and broader concepts
4. Questions should be natural and conversational
5. Answers should be directly supported by the text

Text to analyze:
{text}

IMPORTANT: You must respond with ONLY a valid JSON array. Do not include any other text, explanations, or formatting. Start your response with [ and end with ].

Example format:
[
  {{
    "question": "Your question here",
    "answer": "Your detailed answer here"
  }},
  {{
    "question": "Another question here", 
    "answer": "Another detailed answer here"
  }}
]

JSON Response:"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider == LLMProvider.OPENAI:
            return self._call_openai(prompt)
        elif self.provider == LLMProvider.CLAUDE:
            return self._call_claude(prompt)
        elif self.provider == LLMProvider.GEMINI:
            return self._call_gemini(prompt)
        elif self.provider == LLMProvider.LOCAL:
            return self._call_local_llm(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates training data."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.kwargs.get('temperature', 0.7),
            max_tokens=self.kwargs.get('max_tokens', 2000)
        )
        return response.choices[0].message.content
    
    def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.kwargs.get('max_tokens', 2000),
            temperature=self.kwargs.get('temperature', 0.7),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        generation_config = {
            'temperature': self.kwargs.get('temperature', 0.7),
            'max_output_tokens': self.kwargs.get('max_tokens', 2000),
        }
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    
    def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM via Ollama."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': self.kwargs.get('temperature', 0.7),
                'num_predict': self.kwargs.get('max_tokens', 2000)
            }
        )
        return response['message']['content']
    
    def _parse_qa_response(self, response: str) -> List[Dict]:
        """Parse the LLM response to extract Q&A pairs."""
        try:
            # Clean the response
            response = response.strip()
            
            # Log raw response for debugging
            log_message(f"Raw response preview: {response[:200]}...")
            
            # Try to find JSON in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = response[start_idx:end_idx]
            log_message(f"Extracted JSON: {json_str[:200]}...")
            
            qa_pairs = json.loads(json_str)
            
            # Validate structure
            for qa in qa_pairs:
                if 'question' not in qa or 'answer' not in qa:
                    raise ValueError("Invalid Q&A pair structure")
            
            log_message(f"Successfully parsed {len(qa_pairs)} Q&A pairs")
            return qa_pairs
            
        except Exception as e:
            log_message(f"Error parsing Q&A response: {str(e)}")
            log_message(f"Full response: {response}")
            # Fallback: try to extract manually
            return self._manual_parse_qa(response)
    
    def _manual_parse_qa(self, response: str) -> List[Dict]:
        """Fallback manual parsing if JSON parsing fails."""
        qa_pairs = []
        lines = response.split('\n')
        
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('Question:'):
                if current_q and current_a:
                    qa_pairs.append({'question': current_q, 'answer': current_a})
                current_q = line.split(':', 1)[1].strip()
                current_a = None
            elif line.startswith('A:') or line.startswith('Answer:'):
                current_a = line.split(':', 1)[1].strip()
            elif current_a and line:
                current_a += ' ' + line
        
        if current_q and current_a:
            qa_pairs.append({'question': current_q, 'answer': current_a})
        
        return qa_pairs

    def generate_training_data(self, processed_documents: List[Dict], output_file: str) -> Dict:
        """Generate complete training dataset from processed documents."""
        training_data = {
            'metadata': {
                'generated_by': 'finetune-ingest',
                'llm_provider': self.provider.value,
                'model_used': self.model,
                'num_documents': len(processed_documents),
                'total_qa_pairs': 0
            },
            'documents': [],
            'training_pairs': []
        }
        
        all_qa_pairs = []
        
        for doc in processed_documents:
            log_message(f"Generating Q&A pairs for {doc['metadata']['file_name']} using {self.provider.value}")
            
            doc_qa_pairs = []
            for chunk in doc['chunks']:
                chunk_qa_pairs = self.generate_qa_pairs(chunk, doc['metadata'])
                doc_qa_pairs.extend(chunk_qa_pairs)
                all_qa_pairs.extend(chunk_qa_pairs)
            
            training_data['documents'].append({
                'file_info': doc['metadata'],
                'qa_pairs_count': len(doc_qa_pairs),
                'word_count': doc['word_count']
            })
        
        training_data['training_pairs'] = all_qa_pairs
        training_data['metadata']['total_qa_pairs'] = len(all_qa_pairs)
        
        # Save the training data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        log_message(f"Generated {len(all_qa_pairs)} Q&A pairs saved to {output_file}")
        return training_data

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available LLM providers."""
        return [provider.value for provider in LLMProvider]
    
    @classmethod
    def get_provider_models(cls, provider: Union[str, LLMProvider]) -> List[str]:
        """Get available models for a specific provider."""
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        models = {
            LLMProvider.OPENAI: [
                "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", 
                "gpt-4-1106-preview", "gpt-4-0125-preview"
            ],
            LLMProvider.CLAUDE: [
                "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307", "claude-2.1", "claude-instant-1.2"
            ],
            LLMProvider.GEMINI: [
                "gemini-pro", "gemini-pro-vision", "gemini-1.5-pro-latest"
            ],
            LLMProvider.LOCAL: [
                "llama3", "llama2", "mistral", "codellama", "vicuna"
            ]
        }
        
        return models.get(provider, []) 