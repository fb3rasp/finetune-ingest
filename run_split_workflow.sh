#!/bin/bash

# Split Workflow Runner for Q&A Data Generation
# This script runs the three-step split workflow with proper error handling
# Updated for new folder structure: document_chunker/ and training_data_generator/

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default directories (can be overridden with environment variables)
INCOMING_DIR=${GENERATOR_INCOMING_DIR:-"../data/documents"}
CHUNKS_DIR=${GENERATOR_PROCESS_DIR:-"../data/document_chunks"}
QA_DIR=${GENERATOR_OUTPUT_DIR:-"../data/document_training_data"}
OUTPUT_FILE=${GENERATOR_OUTPUT_FILE:-"../data/document_training_data/training_data.json"}

# Parse command line arguments
RESUME=""
STEP=""
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="--resume"
            shift
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "Split Workflow Runner for Q&A Data Generation"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --resume          Resume processing from where it was interrupted"
    echo "  --step STEP       Run only specific step (1=chunk, 2=qa, 3=combine, all=default)"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GENERATOR_INCOMING_DIR    Source documents directory (default: /data/incoming)"
    echo "  GENERATOR_PROCESS_DIR     Chunks directory (default: /data/chunks)"
    echo "  GENERATOR_OUTPUT_DIR      Q&A results directory (default: /data/qa_results)"
    echo "  GENERATOR_OUTPUT_FILE     Final training data file (default: /data/results/training_data.json)"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run complete workflow"
    echo "  $0 --resume               # Resume interrupted workflow"
    echo "  $0 --step 2               # Run only Q&A generation step"
    echo "  $0 --step 2 --resume      # Resume Q&A generation step"
    echo ""
    echo "New Folder Structure:"
    echo "  document_chunker/         # Document chunking module"
    echo "  training_data_generator/  # Q&A generation and combining module"
    exit 0
fi

echo -e "${GREEN}Starting Split Workflow for Q&A Data Generation${NC}"
echo "Directories:"
echo "  Incoming: $INCOMING_DIR"
echo "  Chunks:   $CHUNKS_DIR"
echo "  Q&A:      $QA_DIR"
echo "  Output:   $OUTPUT_FILE"
echo ""

# Function to run a step with error handling
run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3
    
    echo -e "${YELLOW}Step $step_num: $step_name${NC}"
    echo "Command: $command"
    echo ""
    
    if eval "$command"; then
        echo -e "${GREEN}✓ Step $step_num completed successfully${NC}"
        echo ""
    else
        echo -e "${RED}✗ Step $step_num failed${NC}"
        exit 1
    fi
}

# Check if Python environment is activated
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please activate your conda environment:${NC}"
    echo "  source /Users/rainer/.zshrc"
    echo "  conda activate finetune"
    exit 1
fi

# Check if we're in the right directory (project root)
if [ ! -d "document_chunker" ] || [ ! -d "training_data_generator" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    echo "Expected folder structure:"
    echo "  document_chunker/"
    echo "  training_data_generator/"
    exit 1
fi

# Step 1: Document Chunking
if [ -z "$STEP" ] || [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
    if [ ! -f "document_chunker/src/chunk_documents.py" ]; then
        echo -e "${RED}Error: document_chunker/src/chunk_documents.py not found${NC}"
        exit 1
    fi
    run_step 1 "Document Chunking" "cd document_chunker && python src/chunk_documents.py --incoming-dir '$INCOMING_DIR' --chunks-dir '$CHUNKS_DIR' $RESUME && cd .."
fi

# Step 2: Q&A Generation  
if [ -z "$STEP" ] || [ "$STEP" = "2" ] || [ "$STEP" = "all" ]; then
    if [ ! -f "training_data_generator/src/generate_qa.py" ]; then
        echo -e "${RED}Error: training_data_generator/src/generate_qa.py not found${NC}"
        exit 1
    fi
    run_step 2 "Q&A Generation" "cd training_data_generator && python src/generate_qa.py --chunks-dir '$CHUNKS_DIR' --qa-dir '$QA_DIR' $RESUME && cd .."
fi

# Step 3: Combine Results
if [ -z "$STEP" ] || [ "$STEP" = "3" ] || [ "$STEP" = "all" ]; then
    if [ ! -f "training_data_generator/src/combine_qa_results.py" ]; then
        echo -e "${RED}Error: training_data_generator/src/combine_qa_results.py not found${NC}"
        exit 1
    fi
    run_step 3 "Combine Results" "cd training_data_generator && python src/combine_qa_results.py --qa-dir '$QA_DIR' --output-file '$OUTPUT_FILE' && cd .."
fi

echo -e "${GREEN}🎉 Split workflow completed successfully!${NC}"
echo "Final training data saved to: $OUTPUT_FILE"

# Show summary if final file exists
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Training Data Summary:"
    python -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    print(f\"  Documents: {data['metadata']['num_documents']}\")
    print(f\"  Q&A Pairs: {data['metadata']['total_qa_pairs']}\")
    print(f\"  Provider: {data['metadata'].get('processing_config', {}).get('llm_provider', 'N/A')}\")
    print(f\"  Model: {data['metadata'].get('processing_config', {}).get('model_used', 'N/A')}\")
except Exception as e:
    print(f\"  Could not read summary: {e}\")
"
fi
