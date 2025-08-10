"""
Q&A generation chains using LangChain.

This module provides sophisticated Q&A generation using LangChain's
chain architecture with prompt templates, output parsers, and
error handling capabilities.
"""

import json
from typing import Dict, List, Optional, Any
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
    from langchain.schema import BaseOutputParser
    LANGCHAIN_CHAINS_AVAILABLE = True
except ImportError as e:
    print(f"LangChain chains not available: {e}")
    print("Install with: pip install langchain")
    LANGCHAIN_CHAINS_AVAILABLE = False
    
    # Create dummy classes
    class LLMChain: pass
    class PromptTemplate: pass
    class ChatPromptTemplate: pass
    class PydanticOutputParser: pass
    class OutputFixingParser: pass
    class BaseOutputParser: pass
from pydantic import BaseModel, Field, ValidationError

from .llm_providers import UnifiedLLMProvider
from utils.helpers import log_message


class QAPair(BaseModel):
    """Structured Q&A pair model."""
    question: str = Field(description="The generated question")
    answer: str = Field(description="The comprehensive answer to the question")


class QAResponse(BaseModel):
    """Collection of Q&A pairs."""
    qa_pairs: List[QAPair] = Field(description="List of question-answer pairs")


class JSONOutputParser(BaseOutputParser):
    """Custom JSON output parser for Q&A pairs."""
    
    def parse(self, text) -> List[Dict[str, str]]:
        """Parse JSON response into Q&A pairs."""
        try:
            # Handle case where text is already a list (LangChain may pre-parse JSON)
            if isinstance(text, list):
                validated_pairs = []
                for qa in text:
                    if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                        validated_pairs.append({
                            'question': str(qa['question']).strip(),
                            'answer': str(qa['answer']).strip()
                        })
                return validated_pairs
            
            # Clean the response
            text = text.strip()
            
            # Extract JSON from response
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = text[start_idx:end_idx]
            qa_pairs = json.loads(json_str)
            
            # Validate structure
            validated_pairs = []
            for qa in qa_pairs:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    validated_pairs.append({
                        'question': str(qa['question']).strip(),
                        'answer': str(qa['answer']).strip()
                    })
            
            return validated_pairs
            
        except Exception as e:
            log_message(f"JSON parsing failed: {str(e)}")
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text) -> List[Dict[str, str]]:
        """Fallback parsing for malformed responses."""
        if isinstance(text, list):
            return text  # Already parsed
            
        qa_pairs = []
        lines = text.split('\n')
        
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['Q:', 'Question:', '**Q:', '**Question:']):
                if current_q and current_a:
                    qa_pairs.append({'question': current_q, 'answer': current_a})
                current_q = line.split(':', 1)[1].strip().strip('*').strip()
                current_a = None
            elif any(line.startswith(prefix) for prefix in ['A:', 'Answer:', '**A:', '**Answer:']):
                current_a = line.split(':', 1)[1].strip().strip('*').strip()
            elif current_a and line and not line.startswith(('Q:', 'A:', '**')):
                current_a += ' ' + line
        
        if current_q and current_a:
            qa_pairs.append({'question': current_q, 'answer': current_a})
        
        return qa_pairs


class QAGenerationChain:
    """Advanced Q&A generation chain using LangChain."""
    
    def __init__(
        self,
        llm_provider: UnifiedLLMProvider,
        questions_per_chunk: int = 3,
        use_structured_output: bool = False,
        # New: prompt customization
        system_message: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        custom_template: Optional[str] = None
    ):
        self.llm_provider = llm_provider
        self.questions_per_chunk = questions_per_chunk
        self.use_structured_output = use_structured_output
        self.system_message = system_message
        self.extra_instructions = extra_instructions or ""
        self.custom_template = custom_template
        self.human_template_str = None  # used for batch path
        
        # Initialize output parser
        if use_structured_output:
            self.pydantic_parser = PydanticOutputParser(pydantic_object=QAResponse)
            self.output_parser = OutputFixingParser.from_llm(
                parser=self.pydantic_parser,
                llm=self.llm_provider.llm
            )
        else:
            self.output_parser = JSONOutputParser()
        
        # Create prompt template(s)
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain
        if use_structured_output:
            self.qa_chain = LLMChain(
                llm=self.llm_provider.llm,
                prompt=self.prompt_template
            )
        else:
            self.qa_chain = LLMChain(
                llm=self.llm_provider.llm,
                prompt=self.prompt_template,
                output_parser=self.output_parser
            )
        
        log_message(f"Initialized QA chain with {questions_per_chunk} questions per chunk")
    
    def _create_prompt_template(self):
        """Create the Q&A generation prompt template."""
        # Base human prompt template (custom takes precedence)
        if self.custom_template:
            template = self.custom_template
        else:
            if self.use_structured_output:
                format_instructions = self.pydantic_parser.get_format_instructions()
                template = """You are an expert at creating high-quality training data for language models. 
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

{format_instructions}

Generate the Q&A pairs:"""
            else:
                template = """You are an expert at creating high-quality training data for language models. 
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

IMPORTANT: You must respond with ONLY a valid JSON array. Do not include any other text, explanations, or formatting. Start your response with [ and end with ].

Example format:
[
  {{
    "question": "What is the main topic discussed in this text?",
    "answer": "The main topic is [comprehensive answer based on the text]"
  }},
  {{
    "question": "How does [concept] relate to [another concept]?", 
    "answer": "The relationship between these concepts is [detailed explanation]"
  }}
]

JSON Response:"""

        # Save the human template string for batch formatting
        self.human_template_str = template

        # Return appropriate PromptTemplate or ChatPromptTemplate
        if self.use_structured_output:
            if self.system_message:
                return ChatPromptTemplate.from_messages([
                    ("system", self.system_message),
                    ("human", template)
                ]).partial(extra_instructions=self.extra_instructions,
                           format_instructions=self.pydantic_parser.get_format_instructions())
            else:
                return PromptTemplate(
                    template=template,
                    input_variables=["text", "num_questions"],
                    partial_variables={"extra_instructions": self.extra_instructions,
                                       "format_instructions": self.pydantic_parser.get_format_instructions()}
                )
        else:
            if self.system_message:
                return ChatPromptTemplate.from_messages([
                    ("system", self.system_message),
                    ("human", template)
                ]).partial(extra_instructions=self.extra_instructions)
            else:
                return PromptTemplate(
                    template=template,
                    input_variables=["text", "num_questions"],
                    partial_variables={"extra_instructions": self.extra_instructions}
                )
    
    def generate_qa_pairs(self, chunk: Dict, metadata: Dict) -> List[Dict]:
        """
        Generate Q&A pairs from a text chunk.
        
        Args:
            chunk: Text chunk dictionary with 'text' key
            metadata: Document metadata
            
        Returns:
            List of Q&A pair dictionaries with enhanced metadata
        """
        try:
            # Prepare input
            input_data = {
                "text": chunk['text'],
                "num_questions": self.questions_per_chunk
            }
            
            log_message(f"Generating Q&A for chunk {chunk.get('chunk_id', 'unknown')} with {len(chunk['text'])} chars")
            
            # Generate Q&A pairs
            if self.use_structured_output:
                log_message("Using structured output")
                response = self.qa_chain.run(input_data)
                if hasattr(response, 'qa_pairs'):
                    qa_pairs = [{'question': qa.question, 'answer': qa.answer} 
                              for qa in response.qa_pairs]
                else:
                    qa_pairs = []
            else:
                log_message("Using unstructured output with parsing")
                # Generate raw response and then parse it
                raw_response = self.qa_chain.run(input_data)
                # log_message(f"Raw response: {raw_response[:200]}...")
                qa_pairs = self.output_parser.parse(raw_response)
                log_message(f"Parsed {len(qa_pairs)} Q&A pairs")
            
            log_message(f"Generated {len(qa_pairs)} Q&A pairs for chunk {chunk.get('chunk_id', 'unknown')}")
            
            # Enhance with metadata
            enhanced_qa_pairs = []
            for qa in qa_pairs:
                enhanced_qa = {
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
                    'section_header': chunk.get('section_header'),
                    'subsection_header': chunk.get('subsection_header'),
                    'generated_by': {
                        'provider': self.llm_provider.provider.value,
                        'model': self.llm_provider.model,
                        'chain_type': 'langchain_qa_generation'
                    },
                    'quality_metrics': {
                        'question_length': len(qa['question']),
                        'answer_length': len(qa['answer']),
                        'question_word_count': len(qa['question'].split()),
                        'answer_word_count': len(qa['answer'].split())
                    }
                }
                enhanced_qa_pairs.append(enhanced_qa)
            
            return enhanced_qa_pairs
            
        except Exception as e:
            log_message(f"Error generating Q&A pairs: {str(e)}")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}")
            return []
    
    def generate_batch_qa_pairs(
        self, 
        chunks: List[Dict], 
        metadata: Dict,
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Generate Q&A pairs for multiple chunks in batch.
        
        Args:
            chunks: List of text chunks
            metadata: Document metadata
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of all generated Q&A pairs
        """
        all_qa_pairs = []
        
        # Process chunks in batches to avoid overwhelming the API
        for i in range(0, len(chunks), max_concurrent):
            batch_chunks = chunks[i:i + max_concurrent]
            batch_prompts = []
            
            for chunk in batch_chunks:
                # Use the raw human template string for batch formatting
                prompt_text = self.human_template_str.format(
                    text=chunk['text'],
                    num_questions=self.questions_per_chunk,
                    extra_instructions=self.extra_instructions
                )
                batch_prompts.append(prompt_text)
            
            try:
                # Generate batch responses (pass system message if present)
                batch_responses = self.llm_provider.batch_generate(
                    batch_prompts,
                    system_message=self.system_message
                )
                
                # Process responses
                for chunk, response in zip(batch_chunks, batch_responses):
                    if self.use_structured_output:
                        try:
                            parsed_response = self.output_parser.parse(response)
                            qa_pairs = [{'question': qa.question, 'answer': qa.answer} 
                                      for qa in parsed_response.qa_pairs]
                        except:
                            qa_pairs = []
                    else:
                        qa_pairs = self.output_parser.parse(response)
                    
                    # Enhance with metadata
                    for qa in qa_pairs:
                        enhanced_qa = {
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
                            'generated_by': {
                                'provider': self.llm_provider.provider.value,
                                'model': self.llm_provider.model,
                                'chain_type': 'langchain_batch_qa_generation'
                            }
                        }
                        all_qa_pairs.append(enhanced_qa)
                
                log_message(f"Processed batch {i//max_concurrent + 1}, total Q&A pairs: {len(all_qa_pairs)}")
                
            except Exception as e:
                log_message(f"Error in batch processing: {str(e)}")
                continue
        
        return all_qa_pairs
    
    def update_questions_per_chunk(self, new_count: int):
        """Update the number of questions generated per chunk."""
        self.questions_per_chunk = new_count
        log_message(f"Updated questions per chunk to {new_count}")
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the current chain configuration."""
        return {
            'llm_provider': self.llm_provider.get_model_info(),
            'questions_per_chunk': self.questions_per_chunk,
            'use_structured_output': self.use_structured_output,
            'output_parser_type': type(self.output_parser).__name__
        }