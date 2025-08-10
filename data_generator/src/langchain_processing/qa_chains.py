import json
from typing import Dict, List, Optional, Any, Callable
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
    from langchain.schema import BaseOutputParser
except ImportError:
    class LLMChain: ...  # type: ignore
    class PromptTemplate: ...  # type: ignore
    class ChatPromptTemplate: ...  # type: ignore
    class PydanticOutputParser: ...  # type: ignore
    class OutputFixingParser: ...  # type: ignore
    class BaseOutputParser: ...  # type: ignore
from pydantic import BaseModel, Field

from common.llm.llm_providers import UnifiedLLMProvider
from common.utils.helpers import log_message


class QAPair(BaseModel):
    question: str = Field(description="The generated question")
    answer: str = Field(description="The comprehensive answer to the question")


class QAResponse(BaseModel):
    qa_pairs: List[QAPair] = Field(description="List of question-answer pairs")


class JSONOutputParser(BaseOutputParser):
    def parse(self, text) -> List[Dict[str, str]]:
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
            log_message(f"JSON parsing failed: {str(e)}")
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

IMPORTANT: Respond with ONLY a valid JSON array of objects with keys \"question\" and \"answer\"."""
        )
        self.human_template_str = template
        if self.system_message:
            return ChatPromptTemplate.from_messages([("system", self.system_message), ("human", template)]).partial(extra_instructions=self.extra_instructions)
        return PromptTemplate(template=template, input_variables=["text", "num_questions"], partial_variables={"extra_instructions": self.extra_instructions})

    def generate_qa_pairs(self, chunk: Dict, metadata: Dict) -> List[Dict]:
        input_data = {"text": chunk['text'], "num_questions": self.questions_per_chunk}
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
            for c, resp in zip(batch, responses):
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


