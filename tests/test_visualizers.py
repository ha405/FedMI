import os
import shutil
import sys
# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.generate_dummy_data import generate_dummy_data
from analysis.visualizer.heatmap import HeatmapVisualizer
from analysis.visualizer.metrics import MetricsVisualizer
from analysis.visualizer.consistency import ConsistencyVisualizer
from analysis.visualizer.graph import GraphVisualizer

def test_visualizers():
    # Use absolute path relative to script location to put dummy output inside tests/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dummy_output_classes")
    circuits_dir = os.path.join(output_dir, "circuits")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(circuits_dir, exist_ok=True)

    # 1. Generate Dummy Data
    json_path = os.path.join(circuits_dir, "all_circuits.json")
    print(f"Generating dummy data at {json_path}...")
    generate_dummy_data(json_path)

    # 2. Run Visualizers
    print("\nStarting Visualizer Class Tests...")
    
    try:
        print("Testing CrossEvalVisualizer...")
        CrossEvalVisualizer(output_dir).run()
        
        print("Testing HeatmapVisualizer...")
        HeatmapVisualizer(output_dir).run()
        
        print("Testing MetricsVisualizer...")
        MetricsVisualizer(output_dir).run()
        
        print("Testing ConsistencyVisualizer...")
        ConsistencyVisualizer(output_dir).run()
        
        print("Testing GraphVisualizer...")
        GraphVisualizer(output_dir).run()
        
        print("\nAll visualizers ran successfully!")
        
        # Check files
        figs_dir = os.path.join(output_dir, "figures")
        files = os.listdir(figs_dir)
        print(f"Generated figures in {figs_dir}:")
        for f in files:
            print(f" - {f}")

        if len(files) > 0:
            print("\nVerification PASSED.")
        else:
            print("\nVerification FAILED: No figures generated.")

    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualizers()
