#!/bin/bash

# Start script for the Finetune Ingest MCP Server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Finetune Ingest MCP Server...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp config_example.env .env
    echo -e "${RED}Please edit .env file with your API keys before running the server${NC}"
    echo -e "${YELLOW}Example: OPENAI_API_KEY=your_actual_key_here${NC}"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | xargs)
fi

# Create directories if they don't exist
mkdir -p incoming output

# Validate setup
echo -e "${YELLOW}Validating setup...${NC}"
python3 -c "
import os
from pathlib import Path

# Check directories
incoming = Path('incoming')
output = Path('output')
if not incoming.exists():
    incoming.mkdir(parents=True)
    print('Created incoming directory')
if not output.exists():
    output.mkdir(parents=True)
    print('Created output directory')

# Check at least one API key is set
api_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
if not any(os.getenv(key) for key in api_keys):
    print('WARNING: No API keys found in environment')
    print('Set at least one API key in .env file:')
    for key in api_keys:
        print(f'  {key}=your_key_here')
else:
    print('API keys configured')

print('Setup validation complete')
"

# Start the MCP server
echo -e "${GREEN}Starting MCP server...${NC}"
echo -e "${YELLOW}Server will listen for MCP connections on stdio${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

python3 src/mcp_server/server.py

echo -e "${GREEN}MCP server stopped${NC}" 