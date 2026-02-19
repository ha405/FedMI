import json
import os
import random

def generate_dummy_data(output_path):
    rounds = ["round_1", "round_2", "round_3"]
    clients = ["client_0", "client_1"]
    classes = ["0 - zero", "1 - one"]
    
    layer_names = ["conv1", "conv2", "conv3"]
    layer_sizes = [64, 128, 256]
    
    all_data = {}
    
    for r_idx, r in enumerate(rounds):
        round_data = {
            "clients_local_model": {},
            "clients_global_model": {}
        }
        
        for c_idx, client in enumerate(clients):
            # Local Model Data
            local_client_data = {}
            for cls in classes:
                # Generate random active nodes
                active_nodes = {}
                for l_name, l_size in zip(layer_names, layer_sizes):
                     # Randomly select 10 nodes
                     active_nodes[l_name] = sorted(random.sample(range(l_size), 10))
                
                # Connectivity (mock)
                connectivity = {}
                for i in range(len(layer_names)-1):
                    src = layer_names[i]
                    dst = layer_names[i+1]
                    # Random edges
                    edges = {}
                    for node in range(10): # Mock destination nodes
                         edges[str(node)] = [1, 2, 3] # Mock source nodes
                    connectivity[dst] = edges

                local_client_data[cls] = {
                    "active_nodes": active_nodes,
                    "connectivity": connectivity,
                    "metrics": {
                        "accuracy": 85.0 + r_idx,
                        "necessity": 80.0
                    }
                }
            round_data["clients_local_model"][client] = local_client_data
            
            # Global Model Data
            global_client_data = {}
            for cls in classes:
                active_nodes = {}
                for l_name, l_size in zip(layer_names, layer_sizes):
                     active_nodes[l_name] = sorted(random.sample(range(l_size), 10))
                     
                connectivity = {}
                for i in range(len(layer_names)-1):
                    dst = layer_names[i+1]
                    edges = {}
                    for node in range(10): 
                         edges[str(node)] = [1, 2, 3]
                    connectivity[dst] = edges

                global_client_data[cls] = {
                    "active_nodes": active_nodes,
                    "connectivity": connectivity,
                    "metrics": {
                        "accuracy": 80.0 + r_idx,
                        "necessity": 75.0,
                        "local_mask_on_global_weights_acc": 70.0 + r_idx * 0.5
                    }
                }
            round_data["clients_global_model"][client] = global_client_data
            
        all_data[r] = round_data
        
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Generated dummy data at {output_path}")

if __name__ == "__main__":
    os.makedirs("tests/dummy_exp/circuits", exist_ok=True)
    generate_dummy_data("tests/dummy_exp/circuits/all_circuits.json")
