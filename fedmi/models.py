"""Re-export: Models."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import SimpleCNN, get_model

__all__ = ["SimpleCNN", "get_model"]
