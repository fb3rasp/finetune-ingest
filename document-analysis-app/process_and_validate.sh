#!/bin/bash

# =============================================================================
# Document Processing and QA Validation Pipeline
# =============================================================================
# This script processes all documents in the incoming folder, generates 
# training data, and validates it for factual accuracy using Claude Sonnet.
#
# Usage: ./process_and_validate.sh [OPTIONS]
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCOMING_DIR="./incoming"
PROCESSING_DIR="./processing"
LOG_FILE="$PROCESSING_DIR/pipeline.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default configuration
PROVIDER="claude"
MODEL="claude-3-5-sonnet-20241022"
CHUNK_SIZE=1200
CHUNK_OVERLAP=300
QUESTIONS_PER_CHUNK=5
SPLITTING_STRATEGY="recursive"
TEMPERATURE=0.7
VALIDATION_THRESHOLD=8.0
FILTER_THRESHOLD=7.0
BATCH_PROCESSING=true
QUIET_MODE=false
DRY_RUN=false
SKIP_VALIDATION=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console with colors (unless quiet mode)
    if [[ "$QUIET_MODE" != "true" ]]; then
        case "$level" in
            "INFO")  echo -e "${BLUE}[INFO]${NC} $message" ;;
            "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
            "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
            "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
            "STEP") echo -e "${PURPLE}[STEP]${NC} $message" ;;
            *) echo "[$level] $message" ;;
        esac
    fi
}

print_banner() {
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "${CYAN}"
        echo "============================================================================="
        echo "    Document Processing & QA Validation Pipeline"
        echo "    Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================================="
        echo -e "${NC}"
    fi
}

print_config() {
    log_message "INFO" "Pipeline Configuration:"
    log_message "INFO" "  Provider: $PROVIDER"
    log_message "INFO" "  Model: $MODEL"
    log_message "INFO" "  Chunk Size: $CHUNK_SIZE"
    log_message "INFO" "  Questions per Chunk: $QUESTIONS_PER_CHUNK"
    log_message "INFO" "  Validation Threshold: $VALIDATION_THRESHOLD"
    log_message "INFO" "  Filter Threshold: $FILTER_THRESHOLD"
    log_message "INFO" "  Incoming Directory: $INCOMING_DIR"
    log_message "INFO" "  Processing Directory: $PROCESSING_DIR"
}

check_prerequisites() {
    log_message "STEP" "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "src/main.py" ]]; then
        log_message "ERROR" "Not in document-analysis-app directory. Please run from project root."
        exit 1
    fi
    
    # Check Python environment
    if ! command -v python &> /dev/null; then
        log_message "ERROR" "Python not found. Please ensure Python is installed and activated."
        exit 1
    fi
    
    # Check if required modules are available
    if ! python -c "import sys; sys.path.insert(0, './src'); from langchain_processing.qa_validator import QAValidator" 2>/dev/null; then
        log_message "ERROR" "Required Python modules not found. Please run: pip install -r requirements.txt"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        log_message "WARN" ".env file not found. Please ensure API keys are set."
    fi
    
    # Check API key for Claude
    if [[ -z "$ANTHROPIC_API_KEY" && "$PROVIDER" == "claude" ]]; then
        log_message "WARN" "ANTHROPIC_API_KEY not set. Make sure it's in your .env file."
    fi
    
    log_message "SUCCESS" "Prerequisites check completed"
}

setup_directories() {
    log_message "STEP" "Setting up directory structure..."
    
    # Create processing directory
    mkdir -p "$PROCESSING_DIR"
    mkdir -p "$PROCESSING_DIR/logs"
    mkdir -p "$PROCESSING_DIR/raw_data"
    mkdir -p "$PROCESSING_DIR/validated_data"
    mkdir -p "$PROCESSING_DIR/reports"
    
    # Create incoming directory if it doesn't exist
    mkdir -p "$INCOMING_DIR"
    
    log_message "SUCCESS" "Directory structure created"
}

count_documents() {
    local count=0
    if [[ -d "$INCOMING_DIR" ]]; then
        count=$(find "$INCOMING_DIR" -type f \( -name "*.pdf" -o -name "*.md" -o -name "*.html" -o -name "*.htm" -o -name "*.docx" -o -name "*.txt" \) | wc -l)
    fi
    echo "$count"
}

process_documents() {
    log_message "STEP" "Processing documents from $INCOMING_DIR..."
    
    local doc_count=$(count_documents)
    if [[ "$doc_count" -eq 0 ]]; then
        log_message "ERROR" "No documents found in $INCOMING_DIR"
        log_message "INFO" "Please place your documents (.pdf, .md, .html, .docx, .txt) in the incoming directory"
        exit 1
    fi
    
    log_message "INFO" "Found $doc_count documents to process"
    
    # Generate output filename with timestamp
    local output_file="$PROCESSING_DIR/raw_data/training_data_${TIMESTAMP}.json"
    local summary_file="$PROCESSING_DIR/raw_data/training_data_${TIMESTAMP}_summary.json"
    
    # Build command arguments
    local cmd_args=(
        "--provider" "$PROVIDER"
        "--model" "$MODEL"
        "--incoming-dir" "$INCOMING_DIR"
        "--output-file" "$output_file"
        "--chunk-size" "$CHUNK_SIZE"
        "--chunk-overlap" "$CHUNK_OVERLAP"
        "--questions-per-chunk" "$QUESTIONS_PER_CHUNK"
        "--splitting-strategy" "$SPLITTING_STRATEGY"
        "--temperature" "$TEMPERATURE"
    )
    
    # Provider-specific args for Ollama
    if [[ "$PROVIDER" == "local" && -n "$OLLAMA_BASE_URL" ]]; then
        cmd_args+=("--ollama-base-url" "$OLLAMA_BASE_URL")
    fi

    # Prompt customization args
    if [[ -n "$QA_SYSTEM_MESSAGE" ]]; then
        cmd_args+=("--qa-system-message" "$QA_SYSTEM_MESSAGE")
    fi
    if [[ -n "$QA_EXTRA_INSTRUCTIONS" ]]; then
        cmd_args+=("--qa-extra-instructions" "$QA_EXTRA_INSTRUCTIONS")
    fi
    if [[ -n "$QA_PROMPT_TEMPLATE_FILE" ]]; then
        cmd_args+=("--qa-prompt-template-file" "$QA_PROMPT_TEMPLATE_FILE")
    fi

    if [[ "$BATCH_PROCESSING" == "true" ]]; then
        cmd_args+=("--batch-processing")
    fi
    
    # Run document processing
    log_message "INFO" "Running document processing with enhanced LangChain implementation..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_message "INFO" "DRY RUN: Would execute: python src/main.py ${cmd_args[*]}"
        echo "$output_file" # Return the output file path for validation
        return 0
    fi
    
    if python src/main.py "${cmd_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        log_message "SUCCESS" "Document processing completed"
        log_message "INFO" "Training data saved to: $output_file"
        log_message "INFO" "Summary saved to: $summary_file"
        
        # Store output file path for validation step
        echo "$output_file"
    else
        log_message "ERROR" "Document processing failed"
        exit 1
    fi
}

validate_training_data() {
    local input_file="$1"
    
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_message "INFO" "Skipping validation as requested"
        return 0
    fi
    
    log_message "STEP" "Validating training data with Claude Sonnet..."
    
    # Generate validation output files
    local validation_report="$PROCESSING_DIR/reports/validation_report_${TIMESTAMP}.json"
    local filtered_data="$PROCESSING_DIR/validated_data/training_data_validated_${TIMESTAMP}.json"
    
    # Build validation command arguments
    local val_cmd_args=(
        "--input" "$input_file"
        "--output" "$validation_report"
        "--filtered-output" "$filtered_data"
        "--validator-provider" "$PROVIDER"
        "--validator-model" "$MODEL"
        "--threshold" "$VALIDATION_THRESHOLD"
        "--filter-threshold" "$FILTER_THRESHOLD"
        "--temperature" "0.1"  # Lower temperature for consistent validation
    )

    # Provider-specific args for Ollama validator
    if [[ "$PROVIDER" == "local" && -n "$OLLAMA_BASE_URL" ]]; then
        val_cmd_args+=("--ollama-base-url" "$OLLAMA_BASE_URL")
    fi
    
    if [[ "$QUIET_MODE" == "true" ]]; then
        val_cmd_args+=("--quiet")
    fi
    
    log_message "INFO" "Validation configuration:"
    log_message "INFO" "  Validator: $PROVIDER ($MODEL)"
    log_message "INFO" "  Pass threshold: $VALIDATION_THRESHOLD"
    log_message "INFO" "  Filter threshold: $FILTER_THRESHOLD"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_message "INFO" "DRY RUN: Would execute: python src/validate_qa.py ${val_cmd_args[*]}"
        return 0
    fi
    
    # Run validation
    local validation_exit_code
    if python src/validate_qa.py "${val_cmd_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        validation_exit_code=${PIPESTATUS[0]}
        
        if [[ "$validation_exit_code" -eq 0 ]]; then
            log_message "SUCCESS" "Validation completed successfully (high pass rate ≥80%)"
        elif [[ "$validation_exit_code" -eq 1 ]]; then
            log_message "WARN" "Validation completed with warnings (pass rate 60-80%)"
        else
            log_message "ERROR" "Validation completed with low pass rate (<60%)"
        fi
        
        log_message "INFO" "Validation report saved to: $validation_report"
        log_message "INFO" "Filtered training data saved to: $filtered_data"
        
        # Store paths for summary
        echo "$validation_report|$filtered_data"
        
    else
        log_message "ERROR" "Validation failed"
        exit 1
    fi
}

generate_pipeline_summary() {
    local training_data_file="$1"
    local validation_info="$2"  # Format: "report_path|filtered_data_path"
    
    log_message "STEP" "Generating pipeline summary..."
    
    local summary_file="$PROCESSING_DIR/pipeline_summary_${TIMESTAMP}.json"
    local validation_report=""
    local filtered_data=""
    
    if [[ -n "$validation_info" && "$validation_info" != "" ]]; then
        validation_report=$(echo "$validation_info" | cut -d'|' -f1)
        filtered_data=$(echo "$validation_info" | cut -d'|' -f2)
    fi
    
    # Create summary JSON
    cat > "$summary_file" << EOF
{
  "pipeline_summary": {
    "timestamp": "$TIMESTAMP",
    "processing_date": "$(date '+%Y-%m-%d %H:%M:%S')",
    "configuration": {
      "provider": "$PROVIDER",
      "model": "$MODEL",
      "chunk_size": $CHUNK_SIZE,
      "chunk_overlap": $CHUNK_OVERLAP,
      "questions_per_chunk": $QUESTIONS_PER_CHUNK,
      "splitting_strategy": "$SPLITTING_STRATEGY",
      "temperature": $TEMPERATURE,
      "validation_threshold": $VALIDATION_THRESHOLD,
      "filter_threshold": $FILTER_THRESHOLD
    },
    "input": {
      "incoming_directory": "$INCOMING_DIR",
      "document_count": $(count_documents)
    },
    "outputs": {
      "raw_training_data": "$training_data_file",
      "validation_report": "$validation_report",
      "filtered_training_data": "$filtered_data",
      "pipeline_log": "$LOG_FILE"
    },
    "directories": {
      "processing": "$PROCESSING_DIR",
      "raw_data": "$PROCESSING_DIR/raw_data",
      "validated_data": "$PROCESSING_DIR/validated_data",
      "reports": "$PROCESSING_DIR/reports",
      "logs": "$PROCESSING_DIR/logs"
    }
  }
}
EOF
    
    log_message "SUCCESS" "Pipeline summary saved to: $summary_file"
}

print_final_summary() {
    local training_data_file="$1"
    local validation_info="$2"
    
    if [[ "$QUIET_MODE" == "true" ]]; then
        return 0
    fi
    
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "    PIPELINE EXECUTION COMPLETED"
    echo "============================================================================="
    echo -e "${NC}"
    
    echo -e "${GREEN}✓ Documents processed successfully${NC}"
    echo -e "  Raw training data: $training_data_file"
    
    if [[ "$SKIP_VALIDATION" != "true" && -n "$validation_info" ]]; then
        local validation_report=$(echo "$validation_info" | cut -d'|' -f1)
        local filtered_data=$(echo "$validation_info" | cut -d'|' -f2)
        
        echo -e "${GREEN}✓ Validation completed${NC}"
        echo -e "  Validation report: $validation_report"
        echo -e "  Filtered data: $filtered_data"
    fi
    
    echo -e "\n${BLUE}Processing Directory Structure:${NC}"
    echo -e "  📁 $PROCESSING_DIR/"
    echo -e "     ├── 📁 raw_data/          # Original training data"
    echo -e "     ├── 📁 validated_data/    # Quality-filtered training data"  
    echo -e "     ├── 📁 reports/           # Validation reports"
    echo -e "     └── 📁 logs/              # Processing logs"
    
    echo -e "\n${PURPLE}Next Steps:${NC}"
    echo -e "  • Review validation report for quality insights"
    echo -e "  • Use filtered training data for model fine-tuning"
    echo -e "  • Check pipeline log for detailed processing information"
    
    echo -e "\n${CYAN}=============================================================================${NC}"
}

show_help() {
    cat << EOF
Document Processing and QA Validation Pipeline

USAGE:
    ./process_and_validate.sh [OPTIONS]

DESCRIPTION:
    Processes all documents in the incoming folder, generates training data,
    and validates it for factual accuracy using Claude Sonnet.

OPTIONS:
    -h, --help                  Show this help message
    -q, --quiet                 Run in quiet mode (minimal output)
    -d, --dry-run              Show what would be executed without running
    --skip-validation          Skip the validation step
    
    --provider PROVIDER        LLM provider (default: claude)
    --model MODEL              Specific model (default: claude-3-5-sonnet-20241022)
    --chunk-size SIZE          Text chunk size (default: 1200)
    --chunk-overlap SIZE       Chunk overlap (default: 300)
    --questions-per-chunk N    Questions per chunk (default: 5)
    --splitting-strategy STRAT Splitting strategy (default: recursive)
    --temperature TEMP         Generation temperature (default: 0.7)
    
    --validation-threshold T   Validation pass threshold (default: 8.0)
    --filter-threshold T       Filtering threshold (default: 7.0)
    
    --incoming-dir DIR         Input directory (default: ./incoming)
    --processing-dir DIR       Processing directory (default: ./processing)

EXAMPLES:
    # Basic usage with defaults
    ./process_and_validate.sh
    
    # Dry run to see what would be executed
    ./process_and_validate.sh --dry-run
    
    # Custom configuration
    ./process_and_validate.sh --questions-per-chunk 7 --validation-threshold 7.5
    
    # Skip validation (only process documents)
    ./process_and_validate.sh --skip-validation
    
    # Quiet mode
    ./process_and_validate.sh --quiet

DIRECTORY STRUCTURE:
    incoming/                   # Place your documents here
    processing/
    ├── raw_data/              # Generated training data
    ├── validated_data/        # Quality-filtered training data
    ├── reports/               # Validation reports
    └── logs/                  # Processing logs

EOF
}

# =============================================================================
# Command Line Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quiet)
            QUIET_MODE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-overlap)
            CHUNK_OVERLAP="$2"
            shift 2
            ;;
        --questions-per-chunk)
            QUESTIONS_PER_CHUNK="$2"
            shift 2
            ;;
        --splitting-strategy)
            SPLITTING_STRATEGY="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --validation-threshold)
            VALIDATION_THRESHOLD="$2"
            shift 2
            ;;
        --filter-threshold)
            FILTER_THRESHOLD="$2"
            shift 2
            ;;
        --incoming-dir)
            INCOMING_DIR="$2"
            shift 2
            ;;
        --processing-dir)
            PROCESSING_DIR="$2"
            shift 2
            ;;
        *)
            log_message "ERROR" "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Pipeline Execution
# =============================================================================

main() {
    # Initialize directories first (before any logging)
    mkdir -p "$PROCESSING_DIR/logs"
    
    # Initialize
    print_banner
    setup_directories
    print_config
    check_prerequisites
    
    # Document Processing
    log_message "INFO" "Starting document processing pipeline..."
    training_data_file=$(process_documents)
    
    # Validation
    validation_info=""
    if [[ "$SKIP_VALIDATION" != "true" ]]; then
        validation_info=$(validate_training_data "$training_data_file")
    fi
    
    # Generate summary
    generate_pipeline_summary "$training_data_file" "$validation_info"
    
    # Final summary
    print_final_summary "$training_data_file" "$validation_info"
    
    log_message "SUCCESS" "Pipeline execution completed successfully!"
}

# Error handling
set -e
trap 'log_message "ERROR" "Pipeline execution failed at line $LINENO"' ERR

# Execute main function
main "$@"