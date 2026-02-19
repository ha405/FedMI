import os
import json
import subprocess
import shutil
import sys

# =======================================================
#  CONFIGURATION FOR THE AUTOMATED EXPERIMENT SUITE
# =======================================================

# --- Experiment 1: IID Baseline ---
IID_CONFIG = {
    "name": "iid_baseline",
    "num_rounds": 10,
    "partition": "iid",
    "num_clients": 3
}

# --- Experiment 2: Non-IID (Label Skew) ---
NON_IID_CONFIG = {
    "name": "non_iid_skew",
    "num_rounds": 10,
    "partition": "manual",
    "num_clients": 3,
    # Extreme skew: Client 0 -> {0,1,2}, Client 1 -> {3,4,5}, Client 2 -> {6,7,8}
    "client_map": {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
}

# =======================================================

def run_command(command):
    """Helper to run a shell command and print its output."""
    print(f"\nExecuting: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with exit code {process.returncode}")

def archive_results(exp_name, checkpoint_dir):
    """Moves results to exps/ directory."""
    print(f"\n--- Archiving Results for {exp_name} ---")
    archive_dir = f"exps/{exp_name}"
    os.makedirs(archive_dir, exist_ok=True)
    
    if os.path.exists(checkpoint_dir):
        target_path = f"{archive_dir}/data"
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        
        # Rename is atomic and fast, but requires same filesystem. 
        # Fallback to move if rename fails (though usually fine on local)
        try:
            os.rename(checkpoint_dir, target_path)
            print(f"Moved {checkpoint_dir} -> {target_path}")
        except OSError:
            shutil.move(checkpoint_dir, target_path)
            print(f"Moved {checkpoint_dir} -> {target_path}")

def main():
    print("=============================================")
    print("  STARTING FEDMI AUTOMATED EXPERIMENT SUITE  ")
    print("=============================================")

    # --- 1. RUN IID EXPERIMENT ---
    print("\n\n--- [1/2] RUNNING EXPERIMENT: IID BASELINE ---")
    config = IID_CONFIG
    checkpoint_dir = f"./checkpoints/{config['name']}"
    
    # Ensure clean start
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    # Use sys.executable to ensure the current python environment is used
    run_command(
        f"\"{sys.executable}\" main.py --partition {config['partition']} "
        f"--num_rounds {config['num_rounds']} "
        f"--num_clients {config['num_clients']} "
        f"--output_dir \"{checkpoint_dir}\" "
        f"--dataset MNIST" 
    )
    
    archive_results(config['name'], checkpoint_dir)


    # --- 2. RUN NON-IID EXPERIMENT ---
    print("\n\n--- [2/2] RUNNING EXPERIMENT: NON-IID (LABEL SKEW) ---")
    config = NON_IID_CONFIG
    checkpoint_dir = f"./checkpoints/{config['name']}"
    client_map_json = json.dumps(config['client_map'])

    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    run_command(
        f"\"{sys.executable}\" main.py --partition {config['partition']} "
        f"--num_rounds {config['num_rounds']} "
        f"--output_dir \"{checkpoint_dir}\" "
        f"--manual_allocation '{client_map_json}' "
        f"--dataset MNIST"
    )

    archive_results(config['name'], checkpoint_dir)


    print("\n\n=============================================")
    print("      ALL EXPERIMENTS COMPLETE! ✅")
    print("=============================================")

if __name__ == "__main__":
    main()