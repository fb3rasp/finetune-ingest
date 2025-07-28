"""
Unified LLM provider interface using LangChain.

This module provides a single interface for multiple LLM providers,
replacing the custom provider implementations with LangChain's
standardized LLM wrappers.
"""

import os
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dotenv import load_dotenv

# Load environment variables when module is imported
load_dotenv()

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun

from utils.helpers import log_message


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL = "local"


class UnifiedLLMProvider:
    """Unified interface for multiple LLM providers using LangChain."""
    
    def __init__(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Initialize the LangChain LLM
        self.llm = self._init_langchain_llm(api_key)
        
        log_message(f"Initialized {self.provider.value} with model: {self.model}")
    
    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for each provider."""
        defaults = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.CLAUDE: "claude-3-5-sonnet-20241022",
            LLMProvider.GEMINI: "gemini-pro",
            LLMProvider.LOCAL: "llama3"
        }
        return defaults.get(provider, "gpt-4")
    
    def _init_langchain_llm(self, api_key: Optional[str]):
        """Initialize the appropriate LangChain LLM."""
        try:
            if self.provider == LLMProvider.OPENAI:
                if ChatOpenAI is None:
                    raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
                return ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key or os.getenv('OPENAI_API_KEY'),
                    **self.kwargs
                )
            
            elif self.provider == LLMProvider.CLAUDE:
                if ChatAnthropic is None:
                    raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
                return ChatAnthropic(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
                    **self.kwargs
                )
            
            elif self.provider == LLMProvider.GEMINI:
                if ChatGoogleGenerativeAI is None:
                    raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
                return ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    google_api_key=api_key or os.getenv('GOOGLE_API_KEY'),
                    **self.kwargs
                )
            
            elif self.provider == LLMProvider.LOCAL:
                if ChatOllama is None:
                    raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")
                # For Ollama, we need to handle the base_url if provided
                base_url = self.kwargs.get('base_url', 'http://localhost:11434')
                return ChatOllama(
                    model=self.model,
                    temperature=self.temperature,
                    base_url=base_url,
                    **{k: v for k, v in self.kwargs.items() if k != 'base_url'}
                )
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            raise Exception(f"Failed to initialize {self.provider.value} LLM: {e}")
    
    def generate_response(
        self, 
        prompt: str, 
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message for context
            
        Returns:
            Generated response string
        """
        try:
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=prompt))
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            log_message(f"Error generating response with {self.provider.value}: {str(e)}")
            raise
    
    def generate_streaming_response(
        self, 
        prompt: str, 
        system_message: Optional[str] = None
    ):
        """
        Generate a streaming response using the configured LLM.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message for context
            
        Yields:
            Response chunks as they are generated
        """
        try:
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=prompt))
            
            # Generate streaming response
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            log_message(f"Error generating streaming response with {self.provider.value}: {str(e)}")
            raise
    
    def batch_generate(
        self, 
        prompts: List[str], 
        system_message: Optional[str] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batch.
        
        Args:
            prompts: List of prompts to process
            system_message: Optional system message for context
            
        Returns:
            List of generated responses
        """
        try:
            # Prepare batch messages
            batch_messages = []
            
            for prompt in prompts:
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))
                batch_messages.append(messages)
            
            # Generate batch responses
            responses = self.llm.batch(batch_messages)
            
            # Extract content from responses
            results = []
            for response in responses:
                if hasattr(response, 'content'):
                    results.append(response.content)
                else:
                    results.append(str(response))
            
            return results
            
        except Exception as e:
            log_message(f"Error generating batch responses with {self.provider.value}: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        try:
            # Try to use the LLM's token counting if available
            if hasattr(self.llm, 'get_num_tokens'):
                return self.llm.get_num_tokens(text)
            else:
                # Fallback to approximate counting (1 token ≈ 4 characters)
                return len(text) // 4
        except:
            # Fallback approximation
            return len(text) // 4
    
    def update_parameters(self, **kwargs):
        """Update LLM parameters dynamically."""
        valid_params = ['temperature', 'max_tokens']
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
                log_message(f"Updated {param} to {value}")
        
        # Note: LangChain LLMs typically require re-initialization for parameter changes
        # For now, we'll update the instance variables and note that the LLM might need reinit
        log_message("Parameter update complete. Some changes may require LLM re-initialization.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'provider': self.provider.value,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'llm_type': type(self.llm).__name__
        }
    
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
                "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", 
                "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
            ],
            LLMProvider.GEMINI: [
                "gemini-pro", "gemini-pro-vision", "gemini-1.5-pro-latest"
            ],
            LLMProvider.LOCAL: [
                "llama3", "llama2", "mistral", "codellama", "vicuna"
            ]
        }
        
        return models.get(provider, [])