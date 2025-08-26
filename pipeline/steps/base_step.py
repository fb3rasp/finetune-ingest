"""
Base class for pipeline steps.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.utils.helpers import log_message
from pipeline.config import PipelineConfig


class BaseStep(ABC):
    """Base class for all pipeline steps."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled."""
        if self.config.verbose or level == "error":
            log_message(message)
    
    @abstractmethod
    def run(self, **kwargs) -> bool:
        """Run the pipeline step. Returns True if successful."""
        pass
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites for this step are met."""
        return True
