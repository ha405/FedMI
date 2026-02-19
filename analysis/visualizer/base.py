import os
import json
import matplotlib.pyplot as plt

class BaseVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.circuits_file = os.path.join(output_dir, "circuits", "all_circuits.json")
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.data = None

    def load_data(self):
        if not os.path.exists(self.circuits_file):
            print(f"[{self.__class__.__name__}] Warning: Circuits file not found at {self.circuits_file}")
            return False
        
        try:
            with open(self.circuits_file, 'r') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error loading data: {e}")
            return False

    def save_plot(self, fig, filename):
        save_path = os.path.join(self.figures_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[{self.__class__.__name__}] Saved plot: {save_path}")
        plt.close(fig)

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")
