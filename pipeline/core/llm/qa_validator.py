from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseOutputParser
except ImportError:
    class LLMChain: ...  # type: ignore
    class PromptTemplate: ...  # type: ignore
    class BaseOutputParser: ...  # type: ignore

from pydantic import BaseModel, Field
from pipeline.core.llm.llm_providers import UnifiedLLMProvider, LLMProvider
from pipeline.core.utils.helpers import log_message


class ValidationScore(BaseModel):
    factual_accuracy_score: int = Field(ge=0, le=10)
    completeness_score: int = Field(ge=0, le=10)
    consistency_score: int = Field(ge=0, le=10)
    overall_score: float = Field(ge=0, le=10)
    issues_found: List[str] = []
    recommendations: List[str] = []
    validation_status: str


class ValidationResult(BaseModel):
    qa_pair_id: str
    qa_pair_index: int  # Index in the original training data
    question: str
    answer: str
    source_file: str
    chunk_id: str  # Changed from int to str to handle string chunk IDs
    chunk_index: Optional[int] = None
    validation_score: ValidationScore
    processing_time: float
    original_qa_metadata: Optional[Dict[str, Any]] = None  # Store original QA pair metadata


class JSONValidationParser(BaseOutputParser):
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            text = text.strip()
            start_idx = text.find('{'); end_idx = text.rfind('}') + 1
            result = json.loads(text[start_idx:end_idx])
            result.setdefault('issues_found', [])
            result.setdefault('recommendations', [])
            
            # Normalize scores to 0-10 range if they're out of bounds
            for score_key in ['factual_accuracy_score', 'completeness_score', 'consistency_score', 'overall_score']:
                if score_key in result:
                    score = result[score_key]
                    if isinstance(score, (int, float)):
                        # If score is 0-100, convert to 0-10
                        if score > 10:
                            result[score_key] = min(10, max(0, score / 10))
                            log_message(f"Normalized {score_key} from {score} to {result[score_key]}")
                        # Clamp to 0-10 range
                        else:
                            result[score_key] = min(10, max(0, score))
            
            return result
        except Exception as e:
            log_message(f"Validation parsing failed: {e}")
            return {
                'factual_accuracy_score': 5,
                'completeness_score': 5,
                'consistency_score': 5,
                'overall_score': 5.0,
                'issues_found': ['Failed to parse validation response'],
                'recommendations': ['Manual review required'],
                'validation_status': 'NEEDS_REVIEW',
            }


class QAValidator:
    def __init__(self, provider: str = 'openai', model: Optional[str] = None, api_key: Optional[str] = None, temperature: float = 0.1, validation_threshold: float = 8.0, batch_size: int = 10, verbose: bool = False, reasoning: bool = False, **provider_kwargs):
        self.provider = provider
        self.model = model
        self.validation_threshold = validation_threshold
        self.batch_size = batch_size
        self.verbose = verbose
        self.llm_provider = UnifiedLLMProvider(provider=LLMProvider(provider.lower()), model=model, api_key=api_key, temperature=temperature, max_tokens=1000, reasoning=reasoning, **provider_kwargs)
        self.validation_parser = JSONValidationParser()
        self.validation_chain = self._create_validation_chain()

    def _create_validation_chain(self) -> LLMChain:
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

IMPORTANT: All scores must be integers from 0-10 only. Do not use percentages or scores above 10.

Provide your evaluation as valid JSON only:
{{
  "factual_accuracy_score": <0-10 integer>,
  "completeness_score": <0-10 integer>, 
  "consistency_score": <0-10 integer>,
  "overall_score": <calculated float 0-10>,
  "issues_found": ["specific issue 1", "specific issue 2"],
  "recommendations": ["improvement 1", "improvement 2"],
  "validation_status": "PASS or NEEDS_REVIEW or FAIL"
}}

Validation status rules:
- PASS: overall_score >= 8.0 and factual_accuracy_score >= 8
- FAIL: overall_score < 6.0 or factual_accuracy_score < 5
- NEEDS_REVIEW: everything else

JSON Response:"""
        prompt = PromptTemplate(template=template, input_variables=["source_text", "question", "answer"])
        return LLMChain(llm=self.llm_provider.llm, prompt=prompt, output_parser=self.validation_parser)

    def validate_qa_pair(self, qa_pair: Dict[str, Any], source_text: str, qa_index: int = 0) -> ValidationResult:
        start_time = datetime.now()
        resp = self.validation_chain.run(source_text=source_text, question=qa_pair['question'], answer=qa_pair['answer'])
        score = ValidationScore(**resp)
        
        # Create a comprehensive ID for cross-referencing
        qa_pair_id = qa_pair.get('qa_pair_id', f"qa_{qa_index}")
        if not qa_pair_id or qa_pair_id == 'qa_0':
            # Generate a more descriptive ID if missing
            file_name = qa_pair.get('file_name', 'unknown')
            chunk_id = qa_pair.get('chunk_id', 0)
            qa_pair_id = f"{file_name}_chunk{chunk_id}_qa{qa_index}"
        
        # Store original metadata for cross-referencing
        original_metadata = {
            'file_name': qa_pair.get('file_name'),
            'chunk_id': qa_pair.get('chunk_id'),
            'chunk_index': qa_pair.get('chunk_index'),
            'chunk_start': qa_pair.get('chunk_start'),
            'chunk_end': qa_pair.get('chunk_end'),
            'source_text_length': len(source_text) if source_text else 0,
        }
        
        return ValidationResult(
            qa_pair_id=qa_pair_id,
            qa_pair_index=qa_index,
            question=qa_pair['question'],
            answer=qa_pair['answer'],
            source_file=qa_pair.get('file_name') or qa_pair.get('source_file', 'unknown'),
            chunk_id=str(qa_pair.get('chunk_id', 'unknown')),
            chunk_index=qa_pair.get('chunk_index'),
            validation_score=score,
            processing_time=(datetime.now()-start_time).total_seconds(),
            original_qa_metadata=original_metadata,
        )

    def validate_training_data(self, training_data_path: str, output_path: Optional[str] = None, resume: bool = False) -> Dict[str, Any]:
        # Load existing validation results if resuming
        existing_results = []
        completed_indices = set()
        
        if resume and output_path and Path(output_path).exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_report = json.load(f)
                existing_results_dict = existing_report.get('validation_results', [])
                completed_indices = {r['qa_pair_index'] for r in existing_results_dict}
                log_message(f"Resuming validation: {len(existing_results_dict)} pairs already validated")
                
                # Convert dictionary results back to ValidationResult objects
                for result_dict in existing_results_dict:
                    try:
                        # Convert nested validation_score dict back to ValidationScore object
                        validation_score_dict = result_dict.get('validation_score', {})
                        validation_score = ValidationScore(**validation_score_dict)
                        
                        # Create ValidationResult object
                        validation_result = ValidationResult(
                            qa_pair_id=result_dict.get('qa_pair_id', ''),
                            qa_pair_index=result_dict.get('qa_pair_index', 0),
                            question=result_dict.get('question', ''),
                            answer=result_dict.get('answer', ''),
                            source_file=result_dict.get('source_file', ''),
                            chunk_id=result_dict.get('chunk_id', ''),
                            chunk_index=result_dict.get('chunk_index'),
                            validation_score=validation_score,
                            processing_time=result_dict.get('processing_time', 0.0),
                            original_qa_metadata=result_dict.get('original_qa_metadata')
                        )
                        existing_results.append(validation_result)
                    except Exception as e:
                        log_message(f"Warning: Could not convert existing result: {e}")
                        continue
                        
            except Exception as e:
                log_message(f"Warning: Could not load existing validation results: {e}")
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        qa_pairs = data.get('training_pairs', [])
        documents = data.get('documents', [])
        doc_lookup = {d['file_info']['file_name']: d.get('chunks', []) for d in documents if 'file_info' in d}
        results: List[ValidationResult] = existing_results
        
        total_pairs = len(qa_pairs)
        log_message(f"Starting validation of {total_pairs} QA pairs...")
        
        for i, qa in enumerate(qa_pairs, 1):
            qa_index = i - 1
            
            # Skip if already completed
            if qa_index in completed_indices:
                if self.verbose:
                    log_message(f"[{i}/{total_pairs}] Already validated, skipping")
                continue
            
            # Progress logging (verbose mode)
            if self.verbose:
                log_message(f"[{i}/{total_pairs}] Validating QA pair from {qa.get('file_name', 'unknown')} (chunk {qa.get('chunk_id', 'N/A')})")
            elif i % 50 == 0 or i == 1:
                # Basic progress for non-verbose mode
                log_message(f"[{i}/{total_pairs}] Processing...")
            
            # Get source text
            src = qa.get('source_text') or ''
            if not src and qa.get('file_name') in doc_lookup:
                chunks = doc_lookup[qa['file_name']]
                if qa.get('chunk_id', 0) < len(chunks):
                    src = chunks[qa['chunk_id']].get('text', '')
            
            # Validate the QA pair (pass the index for cross-referencing)
            result = self.validate_qa_pair(qa, src or 'Source text not available', qa_index=qa_index)
            results.append(result)
            completed_indices.add(qa_index)
            
            # Save progress incrementally if resuming
            if resume and output_path:
                log_message(f"Saving incremental progress: {len(results)} results to {output_path}")
                self._save_incremental_progress(results, output_path, training_data_path)
            
            # Log the result with scores (verbose mode)
            if self.verbose:
                score_info = result.validation_score
                log_message(f"[{i}/{total_pairs}] Result: {score_info.validation_status} | "
                           f"Overall: {score_info.overall_score:.1f}/10 | "
                           f"Accuracy: {score_info.factual_accuracy_score}/10 | "
                           f"Completeness: {score_info.completeness_score}/10 | "
                           f"Consistency: {score_info.consistency_score}/10")
                
                # Log issues if any
                if score_info.issues_found:
                    log_message(f"[{i}/{total_pairs}] Issues: {', '.join(score_info.issues_found[:2])}{'...' if len(score_info.issues_found) > 2 else ''}")
            
            # Progress summary every 10 items or at key milestones (verbose mode)
            # or every 100 items for non-verbose mode
            summary_interval = 10 if self.verbose else 100
            if (i % summary_interval == 0 or i in [1, 5, total_pairs]) and (self.verbose or i % 100 == 0 or i in [1, total_pairs]):
                current_results = results[-min(summary_interval, len(results)):]
                pass_count = sum(1 for r in current_results if r.validation_score.validation_status == 'PASS')
                avg_score = sum(r.validation_score.overall_score for r in current_results) / len(current_results)
                log_message(f"[{i}/{total_pairs}] Progress summary - Last {len(current_results)} items: "
                           f"{pass_count}/{len(current_results)} PASS ({pass_count/len(current_results)*100:.1f}%), "
                           f"Avg score: {avg_score:.1f}/10")
        total = len(results)
        pass_count = sum(1 for r in results if r.validation_score.validation_status == 'PASS')
        needs_review_count = sum(1 for r in results if r.validation_score.validation_status == 'NEEDS_REVIEW')
        fail_count = sum(1 for r in results if r.validation_score.validation_status == 'FAIL')
        avg = lambda xs: round(sum(xs)/len(xs), 2) if xs else 0.0
        
        # Calculate score distribution
        overall_scores = [r.validation_score.overall_score for r in results]
        score_distribution = {}
        for i in range(11):  # 0-10 scores
            count = sum(1 for score in overall_scores if i <= score < i + 1)
            percentage = (count / total * 100) if total > 0 else 0
            score_distribution[f"{i}-{i+1}"] = {"count": count, "percentage": round(percentage, 1)}
        
        # Group into broader ranges for summary
        score_ranges = {
            "0-1": sum(1 for score in overall_scores if 0 <= score < 1),
            "1-2": sum(1 for score in overall_scores if 1 <= score < 2),
            "2-3": sum(1 for score in overall_scores if 2 <= score < 3),
            "3-4": sum(1 for score in overall_scores if 3 <= score < 4),
            "4-5": sum(1 for score in overall_scores if 4 <= score < 5),
            "5-6": sum(1 for score in overall_scores if 5 <= score < 6),
            "6-7": sum(1 for score in overall_scores if 6 <= score < 7),
            "7-8": sum(1 for score in overall_scores if 7 <= score < 8),
            "8-9": sum(1 for score in overall_scores if 8 <= score < 9),
            "9-10": sum(1 for score in overall_scores if 9 <= score <= 10),
        }
        
        summary = {
            'total_qa_pairs': total,
            'pass_count': pass_count,
            'needs_review_count': needs_review_count,
            'fail_count': fail_count,
            'pass_rate': (pass_count/total) if total else 0.0,
            'average_scores': {
                'overall': avg([r.validation_score.overall_score for r in results]),
                'factual_accuracy': avg([r.validation_score.factual_accuracy_score for r in results]),
                'completeness': avg([r.validation_score.completeness_score for r in results]),
                'consistency': avg([r.validation_score.consistency_score for r in results]),
            },
            'score_distribution': score_distribution,
            'score_ranges': score_ranges,
            'total_processing_time': sum(r.processing_time for r in results),
        }
        # Create cross-reference index for easy lookup
        cross_reference_index = {}
        validation_results_detailed = []
        
        for r in results:
            result_dict = r.dict()
            
            # Add to cross-reference index
            cross_reference_index[r.qa_pair_index] = {
                'qa_pair_id': r.qa_pair_id,
                'validation_status': r.validation_score.validation_status,
                'overall_score': r.validation_score.overall_score,
                'source_file': r.source_file,
                'chunk_id': r.chunk_id,
            }
            
            validation_results_detailed.append(result_dict)
        
        # Organize flagged issues by priority
        flagged_issues = {
            'high_priority': [],  # FAIL status or score < 5
            'medium_priority': [],  # NEEDS_REVIEW or score 5-7
            'low_priority': []  # Minor issues in otherwise good pairs
        }
        
        for r in results:
            if r.validation_score.validation_status == 'FAIL' or r.validation_score.overall_score < 5:
                flagged_issues['high_priority'].append({
                    'qa_pair_index': r.qa_pair_index,
                    'qa_pair_id': r.qa_pair_id,
                    'status': r.validation_score.validation_status,
                    'score': r.validation_score.overall_score,
                    'issues': r.validation_score.issues_found[:3],  # Top 3 issues
                    'question_preview': r.question[:100] + '...' if len(r.question) > 100 else r.question,
                })
            elif r.validation_score.validation_status == 'NEEDS_REVIEW' or r.validation_score.overall_score < 7:
                flagged_issues['medium_priority'].append({
                    'qa_pair_index': r.qa_pair_index,
                    'qa_pair_id': r.qa_pair_id,
                    'status': r.validation_score.validation_status,
                    'score': r.validation_score.overall_score,
                    'issues': r.validation_score.issues_found[:2],  # Top 2 issues
                })
            elif r.validation_score.issues_found:
                flagged_issues['low_priority'].append({
                    'qa_pair_index': r.qa_pair_index,
                    'qa_pair_id': r.qa_pair_id,
                    'score': r.validation_score.overall_score,
                    'issues': r.validation_score.issues_found[:1],  # Top 1 issue
                })
        
        report = {
            'validation_metadata': {
                'validator_model': f"{self.provider}:{self.llm_provider.model}",
                'validation_timestamp': datetime.now().isoformat(),
                'total_qa_pairs': len(qa_pairs),
                'validation_threshold': self.validation_threshold,
                'source_file': training_data_path,
                'processing_info': {
                    'verbose_mode': self.verbose,
                    'batch_size': self.batch_size,
                    'total_processing_time_seconds': summary['total_processing_time'],
                }
            },
            'summary_statistics': summary,
            'cross_reference_index': cross_reference_index,
            'validation_results': validation_results_detailed,
            'flagged_issues': flagged_issues,
        }
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        
        # Print score distribution summary
        log_message("=" * 50)
        log_message("VALIDATION SUMMARY")
        log_message("=" * 50)
        log_message(f"Total Q&A pairs: {total}")
        log_message(f"PASS: {pass_count} ({pass_count/total*100:.1f}%)")
        log_message(f"NEEDS_REVIEW: {needs_review_count} ({needs_review_count/total*100:.1f}%)")
        log_message(f"FAIL: {fail_count} ({fail_count/total*100:.1f}%)")
        log_message("")
        log_message("SCORE DISTRIBUTION:")
        for range_name, count in score_ranges.items():
            percentage = (count / total * 100) if total > 0 else 0
            log_message(f"Overall: {range_name}: {count} ({percentage:.1f}%)")
        
        return report

    def _save_incremental_progress(self, results: List[ValidationResult], output_path: str, training_data_path: str):
        """Save validation progress incrementally."""
        try:
            # Convert ValidationResult objects to dictionaries
            validation_results_detailed = [r.dict() for r in results]
            
            # Create cross-reference index
            cross_reference_index = {}
            for r in results:
                cross_reference_index[r.qa_pair_index] = {
                    'qa_pair_id': r.qa_pair_id,
                    'validation_status': r.validation_score.validation_status,
                    'overall_score': r.validation_score.overall_score,
                    'source_file': r.source_file,
                    'chunk_id': r.chunk_id,
                }
            
            # Calculate current statistics
            total = len(results)
            pass_count = sum(1 for r in results if r.validation_score.validation_status == 'PASS')
            needs_review_count = sum(1 for r in results if r.validation_score.validation_status == 'NEEDS_REVIEW')
            fail_count = sum(1 for r in results if r.validation_score.validation_status == 'FAIL')
            avg = lambda xs: round(sum(xs)/len(xs), 2) if xs else 0.0
            
            summary = {
                'total_qa_pairs': total,
                'pass_count': pass_count,
                'needs_review_count': needs_review_count,
                'fail_count': fail_count,
                'pass_rate': (pass_count/total) if total else 0.0,
                'average_scores': {
                    'overall': avg([r.validation_score.overall_score for r in results]),
                    'factual_accuracy': avg([r.validation_score.factual_accuracy_score for r in results]),
                    'completeness': avg([r.validation_score.completeness_score for r in results]),
                    'consistency': avg([r.validation_score.consistency_score for r in results]),
                },
                'total_processing_time': sum(r.processing_time for r in results),
            }
            
            # Create progress report
            progress_report = {
                'validation_metadata': {
                    'validator_model': f"{self.provider}:{self.llm_provider.model}",
                    'validation_timestamp': datetime.now().isoformat(),
                    'total_qa_pairs': total,
                    'validation_threshold': self.validation_threshold,
                    'source_file': training_data_path,
                    'status': 'in_progress',
                    'processing_info': {
                        'verbose_mode': self.verbose,
                        'batch_size': self.batch_size,
                        'total_processing_time_seconds': summary['total_processing_time'],
                    }
                },
                'summary_statistics': summary,
                'cross_reference_index': cross_reference_index,
                'validation_results': validation_results_detailed,
            }
            
            # Save progress report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(progress_report, f, indent=2)
                
        except Exception as e:
            log_message(f"Warning: Failed to save incremental progress: {e}")
