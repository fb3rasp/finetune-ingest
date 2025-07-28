"""
QA Validation system using LangChain for factual accuracy checking.

This module provides comprehensive validation of generated Q&A pairs
against source documents to ensure factual correctness and quality.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from langchain.schema import BaseOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False
    
    # Create dummy classes
    class LLMChain: pass
    class PromptTemplate: pass
    class PydanticOutputParser: pass
    class BaseOutputParser: pass

from pydantic import BaseModel, Field, ValidationError
from .llm_providers import UnifiedLLMProvider, LLMProvider
from utils.helpers import log_message


class ValidationScore(BaseModel):
    """Validation scoring model."""
    factual_accuracy_score: int = Field(ge=0, le=10, description="Factual accuracy score (0-10)")
    completeness_score: int = Field(ge=0, le=10, description="Answer completeness score (0-10)")
    consistency_score: int = Field(ge=0, le=10, description="Internal consistency score (0-10)")
    overall_score: float = Field(ge=0, le=10, description="Overall weighted score")
    issues_found: List[str] = Field(description="List of specific issues identified")
    recommendations: List[str] = Field(description="Suggested improvements")
    validation_status: str = Field(description="PASS|NEEDS_REVIEW|FAIL")


class ValidationResult(BaseModel):
    """Complete validation result for a QA pair."""
    qa_pair_id: str
    question: str
    answer: str
    source_file: str
    chunk_id: int
    validation_score: ValidationScore
    processing_time: float


class JSONValidationParser(BaseOutputParser):
    """Custom JSON parser for validation responses."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse validation response into structured format."""
        try:
            # Clean the response
            text = text.strip()
            
            # Extract JSON from response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'factual_accuracy_score', 'completeness_score', 
                'consistency_score', 'overall_score', 'validation_status'
            ]
            
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure lists exist
            result.setdefault('issues_found', [])
            result.setdefault('recommendations', [])
            
            return result
            
        except Exception as e:
            log_message(f"Validation parsing failed: {str(e)}")
            return self._fallback_result(text)
    
    def _fallback_result(self, text: str) -> Dict[str, Any]:
        """Generate fallback validation result."""
        return {
            'factual_accuracy_score': 5,
            'completeness_score': 5,
            'consistency_score': 5,
            'overall_score': 5.0,
            'issues_found': ['Failed to parse validation response'],
            'recommendations': ['Manual review required'],
            'validation_status': 'NEEDS_REVIEW'
        }


class QAValidator:
    """
    Main QA validation class using LangChain for factual accuracy checking.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,  # Low temperature for consistent validation
        validation_threshold: float = 8.0,
        batch_size: int = 10
    ):
        """
        Initialize QA validator.
        
        Args:
            provider: LLM provider ('openai', 'claude', 'gemini', 'local')
            model: Specific model name
            api_key: API key (uses env vars if not provided)
            temperature: LLM temperature for validation
            validation_threshold: Minimum score for PASS status
            batch_size: Number of QA pairs to process in batch
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for QA validation. Install with: pip install langchain")
        
        self.provider = provider
        self.model = model
        self.validation_threshold = validation_threshold
        self.batch_size = batch_size
        
        # Initialize LLM provider
        self.llm_provider = UnifiedLLMProvider(
            provider=LLMProvider(provider.lower()),
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=1000
        )
        
        # Initialize validation chain
        self.validation_parser = JSONValidationParser()
        self.validation_chain = self._create_validation_chain()
        
        log_message(f"Initialized QA validator with {provider} model: {self.llm_provider.model}")
    
    def _create_validation_chain(self) -> LLMChain:
        """Create the LangChain validation chain."""
        template = """You are an expert fact-checker evaluating the accuracy of AI-generated training data.

TASK: Determine if the provided answer is factually accurate based on the source text.

SOURCE TEXT:
{source_text}

QUESTION: {question}

GENERATED ANSWER: {answer}

EVALUATION CRITERIA:
1. FACTUAL ACCURACY (0-10): Is the answer supported by the source text? Are all facts correct?
2. COMPLETENESS (0-10): Does the answer adequately address the question? Is important information missing?
3. CONSISTENCY (0-10): Are there any contradictions with the source or internal inconsistencies?

SCORING GUIDELINES:
- 9-10: Excellent - Fully accurate, complete, and consistent
- 7-8: Good - Mostly accurate with minor issues
- 5-6: Fair - Some accuracy issues or incomplete
- 3-4: Poor - Significant inaccuracies or contradictions  
- 0-2: Fail - Majorly incorrect or unsupported by source

Calculate overall_score as: (factual_accuracy_score * 0.5) + (completeness_score * 0.3) + (consistency_score * 0.2)

Provide your evaluation as valid JSON only, no other text:
{{
  "factual_accuracy_score": <0-10 integer>,
  "completeness_score": <0-10 integer>, 
  "consistency_score": <0-10 integer>,
  "overall_score": <calculated float>,
  "issues_found": ["specific issue 1", "specific issue 2"],
  "recommendations": ["improvement 1", "improvement 2"],
  "validation_status": "PASS or NEEDS_REVIEW or FAIL"
}}

Validation status rules:
- PASS: overall_score >= {threshold} and factual_accuracy_score >= 8
- FAIL: overall_score < 6.0 or factual_accuracy_score < 5
- NEEDS_REVIEW: everything else

JSON Response:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["source_text", "question", "answer"],
            partial_variables={"threshold": str(self.validation_threshold)}
        )
        
        return LLMChain(
            llm=self.llm_provider.llm,
            prompt=prompt,
            output_parser=self.validation_parser
        )
    
    def validate_qa_pair(
        self, 
        qa_pair: Dict[str, Any], 
        source_text: str
    ) -> ValidationResult:
        """
        Validate a single QA pair against source text.
        
        Args:
            qa_pair: QA pair dictionary with question, answer, metadata
            source_text: Source text chunk to validate against
            
        Returns:
            ValidationResult with scores and analysis
        """
        start_time = datetime.now()
        
        try:
            # Run validation chain
            validation_response = self.validation_chain.run(
                source_text=source_text,
                question=qa_pair['question'],
                answer=qa_pair['answer']
            )
            
            # Create validation score
            validation_score = ValidationScore(**validation_response)
            
            # Create result
            result = ValidationResult(
                qa_pair_id=qa_pair.get('qa_pair_id', f"{qa_pair.get('file_name', 'unknown')}_chunk{qa_pair.get('chunk_id', 0)}"),
                question=qa_pair['question'],
                answer=qa_pair['answer'],
                source_file=qa_pair.get('file_name', 'unknown'),
                chunk_id=qa_pair.get('chunk_id', 0),
                validation_score=validation_score,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            return result
            
        except Exception as e:
            log_message(f"Validation failed for QA pair: {str(e)}")
            
            # Create fallback result
            fallback_score = ValidationScore(
                factual_accuracy_score=5,
                completeness_score=5,
                consistency_score=5,
                overall_score=5.0,
                issues_found=[f"Validation error: {str(e)}"],
                recommendations=["Manual review required"],
                validation_status="NEEDS_REVIEW"
            )
            
            return ValidationResult(
                qa_pair_id=qa_pair.get('qa_pair_id', 'error'),
                question=qa_pair.get('question', ''),
                answer=qa_pair.get('answer', ''),
                source_file=qa_pair.get('file_name', 'unknown'),
                chunk_id=qa_pair.get('chunk_id', 0),
                validation_score=fallback_score,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def validate_training_data(
        self, 
        training_data_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate complete training dataset.
        
        Args:
            training_data_path: Path to training data JSON file
            output_path: Optional path to save validation report
            
        Returns:
            Comprehensive validation report
        """
        log_message(f"Starting validation of training data: {training_data_path}")
        
        # Load training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        qa_pairs = training_data.get('training_pairs', [])
        documents = training_data.get('documents', [])
        
        # Create document lookup for source text retrieval
        doc_lookup = {}
        for doc in documents:
            file_name = doc['file_info']['file_name']
            if 'chunks' in doc:
                doc_lookup[file_name] = doc['chunks']
        
        # Validate each QA pair
        validation_results = []
        total_pairs = len(qa_pairs)
        
        log_message(f"Validating {total_pairs} QA pairs...")
        
        for i, qa_pair in enumerate(qa_pairs):
            if (i + 1) % 10 == 0:
                log_message(f"Progress: {i + 1}/{total_pairs} validated")
            
            # Get source text for this QA pair
            file_name = qa_pair.get('file_name', '')
            chunk_id = qa_pair.get('chunk_id', 0)
            
            source_text = qa_pair.get('source_text', '')
            if not source_text and file_name in doc_lookup:
                chunks = doc_lookup[file_name]
                if chunk_id < len(chunks):
                    source_text = chunks[chunk_id].get('text', '')
            
            if not source_text:
                log_message(f"Warning: No source text found for QA pair {i}")
                source_text = "Source text not available"
            
            # Validate the QA pair
            result = self.validate_qa_pair(qa_pair, source_text)
            validation_results.append(result)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(validation_results)
        
        # Create validation report
        validation_report = {
            'validation_metadata': {
                'validator_model': f"{self.provider}:{self.llm_provider.model}",
                'validation_timestamp': datetime.now().isoformat(),
                'total_qa_pairs': len(qa_pairs),
                'validation_threshold': self.validation_threshold,
                'source_file': training_data_path
            },
            'summary_statistics': summary_stats,
            'validation_results': [result.dict() for result in validation_results],
            'flagged_issues': self._categorize_issues(validation_results)
        }
        
        # Save validation report if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            log_message(f"Validation report saved to: {output_path}")
        
        log_message(f"Validation complete! Pass rate: {summary_stats['pass_rate']:.1%}")
        
        return validation_report
    
    def _generate_summary_stats(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        total = len(results)
        if total == 0:
            return {}
        
        pass_count = sum(1 for r in results if r.validation_score.validation_status == 'PASS')
        needs_review_count = sum(1 for r in results if r.validation_score.validation_status == 'NEEDS_REVIEW')
        fail_count = sum(1 for r in results if r.validation_score.validation_status == 'FAIL')
        
        avg_score = sum(r.validation_score.overall_score for r in results) / total
        avg_factual = sum(r.validation_score.factual_accuracy_score for r in results) / total
        avg_completeness = sum(r.validation_score.completeness_score for r in results) / total
        avg_consistency = sum(r.validation_score.consistency_score for r in results) / total
        
        return {
            'total_qa_pairs': total,
            'pass_count': pass_count,
            'needs_review_count': needs_review_count,
            'fail_count': fail_count,
            'pass_rate': pass_count / total,
            'average_scores': {
                'overall': round(avg_score, 2),
                'factual_accuracy': round(avg_factual, 2),
                'completeness': round(avg_completeness, 2),
                'consistency': round(avg_consistency, 2)
            },
            'total_processing_time': sum(r.processing_time for r in results)
        }
    
    def _categorize_issues(self, results: List[ValidationResult]) -> Dict[str, List[str]]:
        """Categorize issues found during validation."""
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for result in results:
            issues = result.validation_score.issues_found
            
            for issue in issues:
                issue_lower = issue.lower()
                if any(keyword in issue_lower for keyword in ['contradict', 'false', 'incorrect', 'wrong']):
                    high_priority.append(issue)
                elif any(keyword in issue_lower for keyword in ['incomplete', 'missing', 'unclear']):
                    medium_priority.append(issue)
                else:
                    low_priority.append(issue)
        
        return {
            'high_priority': list(set(high_priority)),
            'medium_priority': list(set(medium_priority)),
            'low_priority': list(set(low_priority))
        }
    
    def filter_training_data(
        self, 
        training_data_path: str,
        output_path: str,
        min_score: float = 7.0
    ) -> Dict[str, Any]:
        """
        Filter training data based on validation scores.
        
        Args:
            training_data_path: Input training data path
            output_path: Output path for filtered data
            min_score: Minimum score threshold for inclusion
            
        Returns:
            Statistics about filtering process
        """
        # First validate the data
        validation_report = self.validate_training_data(training_data_path)
        
        # Load original training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        # Filter QA pairs based on scores
        original_pairs = training_data['training_pairs']
        filtered_pairs = []
        
        for i, result in enumerate(validation_report['validation_results']):
            if result['validation_score']['overall_score'] >= min_score:
                filtered_pairs.append(original_pairs[i])
        
        # Update training data
        filtered_data = training_data.copy()
        filtered_data['training_pairs'] = filtered_pairs
        filtered_data['metadata']['total_qa_pairs'] = len(filtered_pairs)
        filtered_data['metadata']['filtered_by_validation'] = True
        filtered_data['metadata']['validation_min_score'] = min_score
        
        # Save filtered data
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        filter_stats = {
            'original_count': len(original_pairs),
            'filtered_count': len(filtered_pairs),
            'removed_count': len(original_pairs) - len(filtered_pairs),
            'retention_rate': len(filtered_pairs) / len(original_pairs) if original_pairs else 0
        }
        
        log_message(f"Filtered training data: {filter_stats['retention_rate']:.1%} retention rate")
        log_message(f"Saved filtered data to: {output_path}")
        
        return filter_stats