import unittest
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import individual test modules
from tests.test_visualizers import test_visualizers

class TestVisualizers(unittest.TestCase):
    def test_visualizer_pipeline(self):
        """Runs the visualizer integration test with dummy data."""
        # Using the existing functional test logic wrapped in unittest
        try:
            test_visualizers()
        except Exception as e:
            self.fail(f"Visualizer pipeline failed: {e}")

if __name__ == "__main__":
    unittest.main()
