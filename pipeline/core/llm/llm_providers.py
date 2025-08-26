"""
Unified LLM provider interface using LangChain (shared by generator and checker).
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
from langchain.schema import HumanMessage, SystemMessage

from pipeline.core.utils.helpers import log_message


class LLMProvider(Enum):
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
        defaults = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.CLAUDE: "claude-3-5-sonnet-20241022",
            LLMProvider.GEMINI: "gemini-pro",
            LLMProvider.LOCAL: "llama3",
        }
        return defaults.get(provider, "gpt-4")

    def _init_langchain_llm(self, api_key: Optional[str]):
        try:
            if self.provider == LLMProvider.OPENAI:
                if ChatOpenAI is None:
                    raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
                return ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key or os.getenv('OPENAI_API_KEY'),
                    **self.kwargs,
                )
            elif self.provider == LLMProvider.CLAUDE:
                if ChatAnthropic is None:
                    raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
                return ChatAnthropic(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
                    **self.kwargs,
                )
            elif self.provider == LLMProvider.GEMINI:
                if ChatGoogleGenerativeAI is None:
                    raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
                return ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    google_api_key=api_key or os.getenv('GOOGLE_API_KEY'),
                    **self.kwargs,
                )
            elif self.provider == LLMProvider.LOCAL:
                if ChatOllama is None:
                    raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")
                base_url = self.kwargs.get('base_url', os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
                allowed = {
                    'top_k', 'top_p', 'repeat_penalty', 'num_predict',
                    'stop', 'mirostat', 'mirostat_eta', 'mirostat_tau',
                    'presence_penalty', 'frequency_penalty', 'keep_alive', 'reasoning'
                }
                ollama_kwargs = {k: v for k, v in self.kwargs.items() if k in allowed}
                return ChatOllama(
                    model=self.model,
                    temperature=self.temperature,
                    base_url=base_url,
                    **ollama_kwargs,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            raise Exception(f"Failed to initialize {self.provider.value} LLM: {e}")

    def batch_generate(self, prompts: List[str], system_message: Optional[str] = None) -> List[str]:
        try:
            batch_messages = []
            for prompt in prompts:
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))
                batch_messages.append(messages)
            responses = self.llm.batch(batch_messages)
            results = []
            for response in responses:
                results.append(getattr(response, 'content', str(response)))
            return results
        except Exception as e:
            log_message(f"Error generating batch responses with {self.provider.value}: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'provider': self.provider.value,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'llm_type': type(self.llm).__name__,
        }

    @classmethod
    def get_available_providers(cls) -> List[str]:
        return [provider.value for provider in LLMProvider]

    @classmethod
    def get_provider_models(cls, provider: Union[str, LLMProvider]) -> List[str]:
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        models = {
            LLMProvider.OPENAI: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            LLMProvider.CLAUDE: ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            LLMProvider.GEMINI: ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro-latest"],
            LLMProvider.LOCAL: ["llama3", "llama2", "mistral", "codellama", "vicuna"],
        }
        return models.get(provider, [])
