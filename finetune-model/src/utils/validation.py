import time
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and testing framework"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 test_questions: List[str], template_format: str = "alpaca"):
        self.model = model
        self.tokenizer = tokenizer
        self.test_questions = test_questions
        self.template_format = template_format
        
        # Ensure model is in eval mode
        self.model.eval()
        
    def evaluate_model(self, save_results: bool = True, results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation
        
        Args:
            save_results: Whether to save results to file
            results_dir: Directory to save results (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("🔍 Starting comprehensive model evaluation...")
        
        evaluation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": self._get_model_info(),
            "response_quality": self._evaluate_response_quality(),
            "template_adherence": self._check_template_adherence(),
            "inference_performance": self._measure_inference_performance(),
            "consistency_check": self._check_response_consistency(),
            "memory_usage": self._measure_memory_usage()
        }
        
        # Calculate overall scores
        evaluation_results["overall_scores"] = self._calculate_overall_scores(evaluation_results)
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(evaluation_results, results_dir)
        
        # Log summary
        self._log_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
                "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else 'unknown'
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {"error": str(e)}
    
    def _evaluate_response_quality(self) -> Dict[str, Any]:
        """Evaluate response quality metrics"""
        logger.info("📊 Evaluating response quality...")
        
        responses = []
        total_time = 0
        
        for i, question in enumerate(self.test_questions):
            logger.info(f"Processing question {i+1}/{len(self.test_questions)}")
            
            start_time = time.time()
            response = self._generate_response(question)
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_time += generation_time
            
            # Analyze response
            analysis = self._analyze_response(question, response)
            
            responses.append({
                "question": question,
                "response": response,
                "generation_time": generation_time,
                "analysis": analysis
            })
        
        # Calculate aggregate metrics
        avg_length = sum(r["analysis"]["word_count"] for r in responses) / len(responses)
        avg_time = total_time / len(responses)
        
        return {
            "responses": responses,
            "aggregate_metrics": {
                "average_response_length": avg_length,
                "average_generation_time": avg_time,
                "total_evaluation_time": total_time
            }
        }
    
    def _generate_response(self, question: str, max_new_tokens: int = 256, 
                          temperature: float = 0.6, top_p: float = 0.9) -> str:
        """Generate response for a given question"""
        # Format prompt according to template
        prompt = self._format_prompt(question)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        
        # Move to model device if needed
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if self.template_format == "alpaca" and "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            # Fallback: remove the input prompt
            response = full_response[len(prompt):].strip()
        
        return response
    
    def _format_prompt(self, question: str) -> str:
        """Format question according to template format"""
        if self.template_format == "alpaca":
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:"""
        elif self.template_format == "chatml":
            return f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        elif self.template_format == "vicuna":
            return f"USER: {question}\nASSISTANT: "
        else:
            return question  # Fallback to raw question
    
    def _analyze_response(self, question: str, response: str) -> Dict[str, Any]:
        """Analyze various aspects of the response"""
        analysis = {
            "word_count": len(response.split()),
            "character_count": len(response),
            "sentence_count": len([s for s in response.split('.') if s.strip()]),
            "has_content": len(response.strip()) > 0,
            "is_complete": not response.strip().endswith("..."),
            "repetition_score": self._calculate_repetition_score(response),
            "relevance_keywords": self._count_relevant_keywords(question, response)
        }
        
        return analysis
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score (lower is better)"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Repetition score: 1 - (unique_words / total_words)
        # 0 = no repetition, 1 = maximum repetition
        return 1.0 - (unique_words / total_words)
    
    def _count_relevant_keywords(self, question: str, response: str) -> int:
        """Count relevant keywords from question that appear in response"""
        # Extract meaningful words from question (remove common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        question_words = set(word.lower().strip('.,!?;:') for word in question.split())
        question_words -= common_words
        
        response_lower = response.lower()
        relevant_count = sum(1 for word in question_words if word in response_lower)
        
        return relevant_count
    
    def _check_template_adherence(self) -> Dict[str, Any]:
        """Check if model responses adhere to expected format"""
        logger.info("📝 Checking template adherence...")
        
        adherence_results = {
            "total_questions": len(self.test_questions),
            "correct_format": 0,
            "issues": []
        }
        
        for i, question in enumerate(self.test_questions[:3]):  # Check first 3 for performance
            response = self._generate_response(question)
            
            # Check if response follows expected format
            issues = self._check_response_format(response)
            
            if not issues:
                adherence_results["correct_format"] += 1
            else:
                adherence_results["issues"].extend([f"Q{i+1}: {issue}" for issue in issues])
        
        adherence_results["adherence_rate"] = adherence_results["correct_format"] / min(3, len(self.test_questions))
        
        return adherence_results
    
    def _check_response_format(self, response: str) -> List[str]:
        """Check individual response format"""
        issues = []
        
        # Check for basic issues
        if not response.strip():
            issues.append("Empty response")
        elif len(response.strip()) < 10:
            issues.append("Response too short")
        elif response.count('\n\n') > 3:
            issues.append("Too many paragraph breaks")
        elif '###' in response:
            issues.append("Contains template markers in response")
        
        return issues
    
    def _measure_inference_performance(self) -> Dict[str, Any]:
        """Measure inference performance metrics"""
        logger.info("⚡ Measuring inference performance...")
        
        # Use a standard question for consistent measurement
        test_question = self.test_questions[0] if self.test_questions else "What is AI?"
        
        # Warm up
        for _ in range(2):
            self._generate_response(test_question, max_new_tokens=50)
        
        # Measure performance
        times = []
        tokens_per_second = []
        
        for _ in range(5):  # Run 5 times for average
            start_time = time.time()
            response = self._generate_response(test_question, max_new_tokens=100)
            end_time = time.time()
            
            generation_time = end_time - start_time
            response_tokens = len(self.tokenizer.encode(response))
            tps = response_tokens / generation_time if generation_time > 0 else 0
            
            times.append(generation_time)
            tokens_per_second.append(tps)
        
        return {
            "average_generation_time": sum(times) / len(times),
            "min_generation_time": min(times),
            "max_generation_time": max(times),
            "average_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "max_tokens_per_second": max(tokens_per_second)
        }
    
    def _check_response_consistency(self) -> Dict[str, Any]:
        """Check response consistency for same questions"""
        logger.info("🔄 Checking response consistency...")
        
        if not self.test_questions:
            return {"error": "No test questions available"}
        
        # Test with first question
        test_question = self.test_questions[0]
        responses = []
        
        # Generate multiple responses to the same question
        for _ in range(3):
            response = self._generate_response(test_question, temperature=0.7)
            responses.append(response)
        
        # Calculate similarity between responses
        similarities = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                similarity = self._calculate_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        return {
            "test_question": test_question,
            "responses": responses,
            "average_similarity": avg_similarity,
            "consistency_score": avg_similarity  # Higher = more consistent
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _measure_memory_usage(self) -> Dict[str, Any]:
        """Measure memory usage during inference"""
        logger.info("💾 Measuring memory usage...")
        
        memory_info = {}
        
        if torch.cuda.is_available():
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Measure before generation
            memory_before = torch.cuda.memory_allocated()
            
            # Generate a response
            if self.test_questions:
                self._generate_response(self.test_questions[0])
            
            # Measure after generation
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_info = {
                "memory_before_mb": memory_before / (1024**2),
                "memory_after_mb": memory_after / (1024**2),
                "peak_memory_mb": peak_memory / (1024**2),
                "memory_increase_mb": (memory_after - memory_before) / (1024**2)
            }
        else:
            memory_info = {"note": "CUDA not available, memory measurement skipped"}
        
        return memory_info
    
    def _calculate_overall_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall evaluation scores"""
        scores = {}
        
        try:
            # Response quality score (0-1)
            response_quality = results.get("response_quality", {})
            avg_metrics = response_quality.get("aggregate_metrics", {})
            avg_length = avg_metrics.get("average_response_length", 0)
            
            # Score based on reasonable response length (50-200 words is good)
            length_score = min(1.0, max(0.0, (avg_length - 10) / 190))
            scores["response_length_score"] = length_score
            
            # Template adherence score
            template_adherence = results.get("template_adherence", {})
            adherence_rate = template_adherence.get("adherence_rate", 0)
            scores["template_adherence_score"] = adherence_rate
            
            # Performance score (higher tokens/sec is better)
            performance = results.get("inference_performance", {})
            tps = performance.get("average_tokens_per_second", 0)
            # Normalize to 0-1 scale (assume 50 tokens/sec is excellent)
            performance_score = min(1.0, tps / 50)
            scores["performance_score"] = performance_score
            
            # Consistency score
            consistency = results.get("consistency_check", {})
            consistency_score = consistency.get("consistency_score", 0)
            scores["consistency_score"] = consistency_score
            
            # Overall score (weighted average)
            overall = (
                length_score * 0.3 +
                adherence_rate * 0.3 +
                performance_score * 0.2 +
                consistency_score * 0.2
            )
            scores["overall_score"] = overall
            
        except Exception as e:
            logger.warning(f"Error calculating overall scores: {e}")
            scores["error"] = str(e)
        
        return scores
    
    def _save_evaluation_results(self, results: Dict[str, Any], results_dir: Optional[str] = None):
        """Save evaluation results to file"""
        try:
            if results_dir is None:
                results_dir = "evaluation_results"
            
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
            filepath = results_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Evaluation results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def _log_evaluation_summary(self, results: Dict[str, Any]):
        """Log evaluation summary"""
        logger.info("=" * 60)
        logger.info("🎯 EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        # Overall scores
        overall_scores = results.get("overall_scores", {})
        if "overall_score" in overall_scores:
            score = overall_scores["overall_score"]
            logger.info(f"Overall Score: {score:.2f}/1.00 ({'✅ Excellent' if score > 0.8 else '⚠️ Good' if score > 0.6 else '❌ Needs Improvement'})")
        
        # Component scores
        for score_name, score_value in overall_scores.items():
            if score_name != "overall_score" and isinstance(score_value, (int, float)):
                logger.info(f"  {score_name.replace('_', ' ').title()}: {score_value:.2f}")
        
        # Performance metrics
        performance = results.get("inference_performance", {})
        if "average_tokens_per_second" in performance:
            tps = performance["average_tokens_per_second"]
            logger.info(f"Performance: {tps:.1f} tokens/second")
        
        # Memory usage
        memory = results.get("memory_usage", {})
        if "peak_memory_mb" in memory:
            peak_mb = memory["peak_memory_mb"]
            logger.info(f"Peak Memory Usage: {peak_mb:.1f} MB")
        
        logger.info("=" * 60)
