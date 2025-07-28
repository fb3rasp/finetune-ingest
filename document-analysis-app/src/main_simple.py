#!/usr/bin/env python3
"""
Simple main entry point that automatically falls back to legacy implementation
if LangChain packages are not available.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point with automatic fallback."""
    
    # Try to use LangChain implementation first
    try:
        from langchain_processing import LangChainProcessor
        print("[INFO] Using enhanced LangChain implementation")
        
        # Import and run the LangChain main
        from main_langchain import main as langchain_main
        langchain_main()
        
    except ImportError as e:
        print(f"[INFO] LangChain packages not available: {e}")
        print("[INFO] Falling back to legacy implementation")
        
        # Import and run the legacy main
        from main import main as legacy_main
        legacy_main()

if __name__ == "__main__":
    main()