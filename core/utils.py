import torch
import torch.nn.functional as F
import os
import glob
import json

def save_circuits_to_json(all_circuits, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(all_circuits, f, indent=2)
    print(f"Circuits saved to {path}")

def load_latest_checkpoint(global_model, config, load_circuits=True):
    if not os.path.exists(config.output_dir):
        return 0, {}
    # Look in checkpoints subdirectory
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return 0, {}
        
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_round_*.pt"))
    if not files:
        return 0, {}
    
    latest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(f"  [Resume] Loading checkpoint: {latest_file}")
    
    checkpoint = torch.load(latest_file, map_location=config.device)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    start_round = checkpoint['round']
    
    all_circuits = {}
    if load_circuits:
        all_circuits = checkpoint.get('all_circuits', {})
        
    print(f"  [Resume] Resuming from Round {start_round + 1}")
    return start_round, all_circuits

def save_checkpoint(global_model, round_num, all_circuits, config, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    path = os.path.join(checkpoint_dir, f"checkpoint_round_{round_num}.pt")
    state = {
        'round': round_num,
        'model_state_dict': global_model.state_dict(),
        'all_circuits': all_circuits,
    }
    torch.save(state, path)
    print(f"  [Checkpoint] Saved global state to {path}")
    
def save_local_model(model, round_num, client_idx, config):
    # Save in checkpoints/round_X/
    round_dir = os.path.join(config.output_dir, "checkpoints", f"round_{round_num + 1}")
    os.makedirs(round_dir, exist_ok=True)
    path = os.path.join(round_dir, f"client_{client_idx}_model.pt")
    torch.save(model.state_dict(), path)
