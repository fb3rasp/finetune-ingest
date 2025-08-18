import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
    from langchain.schema import BaseOutputParser, BaseLLMOutputParser
except ImportError:
    class LLMChain: ...  # type: ignore
    class PromptTemplate: ...  # type: ignore
    class ChatPromptTemplate: ...  # type: ignore
    class PydanticOutputParser: ...  # type: ignore
    class OutputFixingParser: ...  # type: ignore
    class BaseOutputParser: ...  # type: ignore
    class BaseLLMOutputParser: ...  # type: ignore
from pydantic import BaseModel, Field

from common.llm.llm_providers import UnifiedLLMProvider
from common.utils.helpers import log_message


def log_failed_response(error_type: str, prompt: str, response: str, model: str, chunk_info: Dict = None):
    """Log detailed information about failed Q&A generation responses."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"qa_failures_{timestamp}.log")
        
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "model": model,
            "chunk_info": chunk_info or {},
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
            "separator": "=" * 80
        }
        
        # Write to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{log_entry['separator']}\n")
            f.write(f"TIMESTAMP: {log_entry['timestamp']}\n")
            f.write(f"ERROR TYPE: {log_entry['error_type']}\n")
            f.write(f"MODEL: {log_entry['model']}\n")
            f.write(f"CHUNK INFO: {json.dumps(log_entry['chunk_info'], indent=2)}\n")
            f.write(f"RESPONSE LENGTH: {log_entry['response_length']} characters\n")
            f.write(f"\nPROMPT SENT:\n{log_entry['prompt']}\n")
            f.write(f"\nMODEL RESPONSE:\n{log_entry['response']}\n")
            f.write(f"{log_entry['separator']}\n")
        
        log_message(f"Failed response logged to: {log_file}")
        
    except Exception as e:
        log_message(f"Failed to log failed response: {e}")


class QAPair(BaseModel):
    question: str = Field(description="The generated question")
    answer: str = Field(description="The comprehensive answer to the question")


class QAResponse(BaseModel):
    qa_pairs: List[QAPair] = Field(description="List of question-answer pairs")


class JSONOutputParser(BaseLLMOutputParser):
    """Custom JSON output parser with enhanced error logging."""
    
    def __init__(self):
        super().__init__()
        # Store context in a global dict to avoid Pydantic validation issues
        if not hasattr(JSONOutputParser, '_context_store'):
            JSONOutputParser._context_store = {}
    
    def set_context(self, prompt: str, model: str, chunk_info: Dict = None):
        """Set context information for logging failed responses."""
        parser_id = id(self)
        JSONOutputParser._context_store[parser_id] = {
            'prompt': prompt,
            'model': model,
            'chunk_info': chunk_info or {}
        }
    
    def _get_context(self):
        """Get context information for this parser instance."""
        parser_id = id(self)
        return JSONOutputParser._context_store.get(parser_id, {
            'prompt': 'Unknown prompt',
            'model': 'Unknown model',
            'chunk_info': {}
        })
    
    def parse_result(self, result) -> List[Dict[str, str]]:
        """Required method for BaseLLMOutputParser."""
        return self.parse(result[0].text if hasattr(result[0], 'text') else str(result[0]))
    
    def parse(self, text: str) -> List[Dict[str, str]]:
        try:
            if isinstance(text, list):
                validated_pairs = []
                for qa in text:
                    if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                        validated_pairs.append({'question': str(qa['question']).strip(), 'answer': str(qa['answer']).strip()})
                return validated_pairs
            text = str(text).strip()
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
            
            qa_pairs = json.loads(text[start_idx:end_idx])
            validated_pairs = []
            for qa in qa_pairs:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    validated_pairs.append({'question': str(qa['question']).strip(), 'answer': str(qa['answer']).strip()})
            return validated_pairs
        except Exception as e:
            error_msg = str(e)
            log_message(f"JSON parsing failed: {error_msg}")
            
            # Get context for logging
            context = self._get_context()
            
            # Log detailed failure information
            log_failed_response(
                error_type=error_msg,
                prompt=context['prompt'],
                response=str(text),
                model=context['model'],
                chunk_info=context['chunk_info']
            )
            return []


class QAGenerationChain:
    def __init__(
        self,
        llm_provider: UnifiedLLMProvider,
        questions_per_chunk: int = 3,
        use_structured_output: bool = False,
        system_message: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        custom_template: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.questions_per_chunk = questions_per_chunk
        self.use_structured_output = use_structured_output
        self.system_message = system_message
        self.extra_instructions = extra_instructions or ""
        self.custom_template = custom_template
        self.human_template_str: Optional[str] = None

        if use_structured_output:
            self.pydantic_parser = PydanticOutputParser(pydantic_object=QAResponse)
            self.output_parser = OutputFixingParser.from_llm(parser=self.pydantic_parser, llm=self.llm_provider.llm)
        else:
            self.output_parser = JSONOutputParser()

        self.prompt_template = self._create_prompt_template()
        self.qa_chain = LLMChain(llm=self.llm_provider.llm, prompt=self.prompt_template, output_parser=None if use_structured_output else self.output_parser)
        log_message(f"Initialized QA chain with {questions_per_chunk} questions per chunk")

    def _create_prompt_template(self):
        template = self.custom_template or (
            """You are an expert at creating high-quality training data for language models. 
Based on the following text, generate exactly {num_questions} question-answer pairs that would help train a chatbot to understand and respond about this content.

Guidelines:
1. Questions should be diverse (factual, analytical, explanatory, inferential)
2. Answers should be comprehensive but concise
3. Include both specific details and broader concepts
4. Questions should be natural and conversational
5. Answers should be directly supported by the text
6. Vary question types: what, how, why, when, where questions
7. Include both simple recall and complex reasoning questions

Additional instructions:
{extra_instructions}

Text to analyze:
{text}

CRITICAL: Your response must be ONLY a valid JSON array. Do not include any thinking, explanation, or extra text. Do not use <think> tags. Start directly with [ and end with ]. Example format:
[{{"question": "What is...", "answer": "The answer is..."}}, {{"question": "How does...", "answer": "It works by..."}}]"""
        )
        self.human_template_str = template
        if self.system_message:
            return ChatPromptTemplate.from_messages([("system", self.system_message), ("human", template)]).partial(extra_instructions=self.extra_instructions)
        return PromptTemplate(template=template, input_variables=["text", "num_questions"], partial_variables={"extra_instructions": self.extra_instructions})

    def generate_qa_pairs(self, chunk: Dict, metadata: Dict) -> List[Dict]:
        input_data = {"text": chunk['text'], "num_questions": self.questions_per_chunk}
        
        # Generate the full prompt for logging
        if hasattr(self.prompt_template, 'format'):
            formatted_prompt = self.prompt_template.format(**input_data)
        else:
            # For ChatPromptTemplate, format differently
            formatted_prompt = self.human_template_str.format(**input_data)
        
        # Set context for the parser in case of failure
        chunk_info = {
            'chunk_id': chunk.get('chunk_id'),
            'chunk_index': chunk.get('chunk_index'),
            'file_name': metadata.get('file_name'),
            'chunk_length': len(chunk['text']),
            'chunk_start': chunk.get('start_char'),
            'chunk_end': chunk.get('end_char')
        }
        
        if hasattr(self.output_parser, 'set_context'):
            self.output_parser.set_context(
                prompt=formatted_prompt,
                model=self.llm_provider.model,
                chunk_info=chunk_info
            )
        
        raw_response = self.qa_chain.run(input_data)
        qa_pairs = self.output_parser.parse(raw_response) if not self.use_structured_output else []
        enhanced: List[Dict] = []
        for qa in qa_pairs:
            enhanced.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'source_file': metadata.get('source_file'),
                'file_name': metadata.get('file_name'),
                'file_type': metadata.get('file_type'),
                'chunk_id': chunk.get('chunk_id'),
                'chunk_index': chunk.get('chunk_index'),
                'chunk_start': chunk.get('start_char'),
                'chunk_end': chunk.get('end_char'),
                'source_text': chunk['text'],
            })
        return enhanced

    def generate_batch_qa_pairs(self, chunks: List[Dict], metadata: Dict, max_concurrent: int = 5, on_chunk_done: Optional[Callable[[List[Dict]], None]] = None) -> List[Dict]:
        all_qa: List[Dict] = []
        for i in range(0, len(chunks), max_concurrent):
            batch = chunks[i:i+max_concurrent]
            prompts: List[str] = []
            for c in batch:
                prompts.append(self.human_template_str.format(text=c['text'], num_questions=self.questions_per_chunk, extra_instructions=self.extra_instructions))
            responses = self.llm_provider.batch_generate(prompts, system_message=self.system_message)
            for c, resp, prompt in zip(batch, responses, prompts):
                # Set context for parser in case of failure
                chunk_info = {
                    'chunk_id': c.get('chunk_id'),
                    'chunk_index': c.get('chunk_index'),
                    'file_name': metadata.get('file_name'),
                    'chunk_length': len(c['text']),
                    'chunk_start': c.get('start_char'),
                    'chunk_end': c.get('end_char')
                }
                
                if hasattr(self.output_parser, 'set_context'):
                    self.output_parser.set_context(
                        prompt=prompt,
                        model=self.llm_provider.model,
                        chunk_info=chunk_info
                    )
                
                qa_pairs = self.output_parser.parse(resp) if not self.use_structured_output else []
                enhanced_for_chunk = []
                for qa in qa_pairs:
                    enhanced_for_chunk.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'source_file': metadata.get('source_file'),
                        'file_name': metadata.get('file_name'),
                        'file_type': metadata.get('file_type'),
                        'chunk_id': c.get('chunk_id'),
                        'chunk_index': c.get('chunk_index'),
                        'chunk_start': c.get('start_char'),
                        'chunk_end': c.get('end_char'),
                        'source_text': c['text'],
                    })
                all_qa.extend(enhanced_for_chunk)
                if on_chunk_done and enhanced_for_chunk:
                    on_chunk_done(enhanced_for_chunk)
        return all_qa


