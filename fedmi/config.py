"""Re-export: Configuration."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import ExperimentConfig

__all__ = ["ExperimentConfig"]
