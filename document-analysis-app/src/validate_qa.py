#!/usr/bin/env python3
"""
QA Validation CLI tool for training data quality assurance.

This script validates generated Q&A pairs against source documents
to ensure factual accuracy and quality before model training.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_processing.qa_validator import QAValidator
from langchain_processing.llm_providers import UnifiedLLMProvider
from utils.helpers import log_message


def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description='Validate Q&A training data for factual accuracy'
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input training data JSON file to validate')
    parser.add_argument('--output', '-o',
                       help='Output validation report JSON file')
    parser.add_argument('--filtered-output',
                       help='Output filtered training data (removes low-scoring pairs)')
    
    # Validator configuration
    parser.add_argument('--validator-provider', 
                       choices=['openai', 'claude', 'gemini', 'local'],
                       default='openai',
                       help='LLM provider for validation (default: openai)')
    parser.add_argument('--validator-model',
                       help='Specific validator model (uses provider default if not specified)')
    parser.add_argument('--validator-api-key',
                       help='API key for validator (uses env vars if not specified)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Validation temperature (default: 0.1 for consistency)')
    
    # Validation parameters
    parser.add_argument('--threshold', type=float, default=8.0,
                       help='Minimum score threshold for PASS status (default: 8.0)')
    parser.add_argument('--filter-threshold', type=float, default=7.0,
                       help='Minimum score for filtered output (default: 7.0)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing (default: 10)')
    
    # Display options
    parser.add_argument('--show-details', action='store_true',
                       help='Show detailed validation results for each QA pair')
    parser.add_argument('--show-failed-only', action='store_true',
                       help='Only show failed and needs-review pairs')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress messages')
    
    # Utility options
    parser.add_argument('--list-models', action='store_true',
                       help='List available validator models')
    parser.add_argument('--list-providers', action='store_true',
                       help='List available validator providers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be validated without running validation')
    
    args = parser.parse_args()
    
    # Handle list providers command
    if args.list_providers:
        providers = UnifiedLLMProvider.get_available_providers()
        print("Available validator providers:")
        for provider in providers:
            print(f"  - {provider}")
            models = UnifiedLLMProvider.get_provider_models(provider)
            for model in models[:3]:
                print(f"    • {model}")
            if len(models) > 3:
                print(f"    • ... and {len(models) - 3} more")
        return
    
    # Handle list models command
    if args.list_models:
        models = UnifiedLLMProvider.get_provider_models(args.validator_provider)
        print(f"Available models for {args.validator_provider}:")
        for model in models:
            print(f"  - {model}")
        return
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Set default output path if not provided
    if not args.output:
        args.output = str(input_path.parent / f"{input_path.stem}_validation_report.json")
    
    if not args.quiet:
        log_message("Initializing QA Validation System...")
        log_message(f"Input file: {args.input}")
        log_message(f"Validator: {args.validator_provider} (threshold: {args.threshold})")
    
    # Dry run mode
    if args.dry_run:
        import json
        with open(args.input, 'r') as f:
            training_data = json.load(f)
        
        qa_count = len(training_data.get('training_pairs', []))
        doc_count = len(training_data.get('documents', []))
        
        print(f"Dry Run - Would validate:")
        print(f"  - {qa_count} Q&A pairs")
        print(f"  - From {doc_count} documents")
        print(f"  - Using {args.validator_provider} validator")
        print(f"  - Threshold: {args.threshold}")
        print(f"  - Output report: {args.output}")
        
        if args.filtered_output:
            print(f"  - Filtered output: {args.filtered_output}")
        
        return 0
    
    try:
        # Initialize validator
        validator = QAValidator(
            provider=args.validator_provider,
            model=args.validator_model,
            api_key=args.validator_api_key,
            temperature=args.temperature,
            validation_threshold=args.threshold,
            batch_size=args.batch_size
        )
        
        if not args.quiet:
            log_message(f"Validator initialized with model: {validator.llm_provider.model}")
        
        # Run validation
        validation_report = validator.validate_training_data(
            training_data_path=args.input,
            output_path=args.output
        )
        
        # Display results
        stats = validation_report['summary_statistics']
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Q&A pairs validated: {stats['total_qa_pairs']}")
        print(f"PASS: {stats['pass_count']} ({stats['pass_rate']:.1%})")
        print(f"NEEDS REVIEW: {stats['needs_review_count']}")
        print(f"FAIL: {stats['fail_count']}")
        print(f"\nAverage Scores:")
        print(f"  Overall: {stats['average_scores']['overall']}/10")
        print(f"  Factual Accuracy: {stats['average_scores']['factual_accuracy']}/10")
        print(f"  Completeness: {stats['average_scores']['completeness']}/10")
        print(f"  Consistency: {stats['average_scores']['consistency']}/10")
        print(f"\nProcessing time: {stats['total_processing_time']:.1f} seconds")
        print(f"Report saved to: {args.output}")
        
        # Show detailed results if requested
        if args.show_details or args.show_failed_only:
            print(f"\n{'='*60}")
            print("DETAILED VALIDATION RESULTS")
            print(f"{'='*60}")
            
            for result in validation_report['validation_results']:
                status = result['validation_score']['validation_status']
                
                if args.show_failed_only and status == 'PASS':
                    continue
                
                score = result['validation_score']['overall_score']
                issues = result['validation_score']['issues_found']
                
                print(f"\nQA Pair: {result['qa_pair_id']}")
                print(f"Status: {status} (Score: {score}/10)")
                print(f"Question: {result['question'][:100]}{'...' if len(result['question']) > 100 else ''}")
                
                if issues:
                    print(f"Issues: {', '.join(issues)}")
                
                if status != 'PASS':
                    recommendations = result['validation_score']['recommendations']
                    if recommendations:
                        print(f"Recommendations: {', '.join(recommendations)}")
        
        # Show flagged issues
        flagged = validation_report['flagged_issues']
        if any(flagged.values()):
            print(f"\n{'='*60}")
            print("FLAGGED ISSUES SUMMARY")
            print(f"{'='*60}")
            
            if flagged['high_priority']:
                print(f"\nHigh Priority Issues:")
                for issue in flagged['high_priority'][:5]:  # Show top 5
                    print(f"  • {issue}")
            
            if flagged['medium_priority']:
                print(f"\nMedium Priority Issues:")
                for issue in flagged['medium_priority'][:5]:  # Show top 5
                    print(f"  • {issue}")
        
        # Generate filtered output if requested
        if args.filtered_output:
            if not args.quiet:
                log_message(f"Generating filtered training data (min score: {args.filter_threshold})...")
            
            filter_stats = validator.filter_training_data(
                training_data_path=args.input,
                output_path=args.filtered_output,
                min_score=args.filter_threshold
            )
            
            print(f"\n{'='*60}")
            print("FILTERING RESULTS")
            print(f"{'='*60}")
            print(f"Original Q&A pairs: {filter_stats['original_count']}")
            print(f"Filtered Q&A pairs: {filter_stats['filtered_count']}")
            print(f"Removed Q&A pairs: {filter_stats['removed_count']}")
            print(f"Retention rate: {filter_stats['retention_rate']:.1%}")
            print(f"Filtered data saved to: {args.filtered_output}")
        
        # Return appropriate exit code
        if stats['pass_rate'] >= 0.8:
            return 0  # Success
        elif stats['pass_rate'] >= 0.6:
            return 1  # Warning
        else:
            return 2  # Failure
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return 130
    
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)