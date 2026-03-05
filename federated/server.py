import torch
import copy
import os
from .client import FederatedClient
from core.utils import save_local_model
from circuits.evaluation import extract_sparse_connectivity, filter_connectivity_by_circuit, evaluate_circuit, evaluate_circuit_necessity
from circuits.discovery import discover_client_circuit, compute_layer_means

class FederatedServer:
    def __init__(self, global_model, testloader, config, class_names):
        self.global_model = global_model
        self.testloader = testloader
        self.config = config
        self.class_names = class_names
        self.device = config.device
        
    def aggregate(self, client_models):
        """
        Federated Averaging (FedAvg).
        """
        weights = [1.0 / len(client_models)] * len(client_models)
        global_state = self.global_model.state_dict()
        target_device = next(self.global_model.parameters()).device
        
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32).to(target_device)
            
            for i, client_model in enumerate(client_models):
                global_state[key] += weights[i] * client_model.state_dict()[key].to(target_device).float()
                
        self.global_model.load_state_dict(global_state)

    def orchestrate_round(self, round_num, clients: list, log_file=None):
        print(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
        
        client_models = []
        round_circuits = {
            "clients_local_model": {}, 
            "clients_global_model": {}
        }
        for i, client in enumerate(clients):
            print(f"  Client {i}: Local Training...")
            
            # Send copy of global model to client
            c_model_copy = copy.deepcopy(self.global_model)
            
            # Train
            trained_model = client.train(c_model_copy)
            save_local_model(trained_model, round_num, i, self.config)
            client_models.append(trained_model)
            
            # Discover circuits (Local Model)
            print(f"  Client {i}: Discovering Circuits...")
            c_circuits = client.discover_circuits(trained_model, self.testloader)
            round_circuits["clients_local_model"][f"client_{i}"] = c_circuits

        self.aggregate(client_models)

        print(f"  Global Model: Running per-client circuit discovery & evaluation...")
        
        global_phys_conn = extract_sparse_connectivity(self.global_model)
        
        for i, client in enumerate(clients):
            gm_copy = copy.deepcopy(self.global_model)
            global_means = None
            if self.config.use_mean_ablation:
                global_means = compute_layer_means(gm_copy, client.dataloader, self.config)
                
            cg_circs = {}
            if self.config.classes_to_discover_per_client and i in self.config.classes_to_discover_per_client:
                classes_for_this_client = self.config.classes_to_discover_per_client[i]
            elif self.config.classes_to_analyze is not None:
                classes_for_this_client = self.config.classes_to_analyze
            else:
                classes_for_this_client = list(range(self.config.num_classes))


            for tc in classes_for_this_client:
                if self.class_names and 0 <= tc < len(self.class_names):
                    name = self.class_names[tc]
                else:
                    name = str(tc)
                circ_global = discover_client_circuit(gm_copy, client.dataloader, tc, self.config, layer_means=global_means)
                func_conn = filter_connectivity_by_circuit(global_phys_conn, circ_global)
                acc_global = evaluate_circuit(gm_copy, self.testloader, circ_global, tc, self.config, layer_means=global_means)
                inv_acc = evaluate_circuit_necessity(gm_copy, self.testloader, circ_global, tc, self.config)
                acc_cross = 0.0
                try:
                    local_circ_nodes = round_circuits["clients_local_model"][f"client_{i}"][name]["active_nodes"]
                    if log_file:
                        log_file.write(f"\n[Round {round_num + 1} | Client {i} | Class {name}] Cross-Eval:\n")
                    
                    acc_cross = evaluate_circuit(
                        gm_copy, self.testloader, local_circ_nodes, tc, self.config, 
                        layer_means=global_means, 
                        log_file=log_file, 
                        class_names=self.class_names
                    )
                except KeyError:
                    acc_cross = 0.0
                    
                cg_circs[name] = {
                    "active_nodes": circ_global,
                    "connectivity": func_conn,
                    "metrics": {
                        "accuracy": acc_global,
                        "necessity": inv_acc,
                        "local_mask_on_global_weights_acc": acc_cross
                    }
                }
                
                counts = {layer: len(idx) for layer, idx in circ_global.items()}
                print(f"    Global + Client {i} Data - {name}: {counts}")
                print(f"      > Global Mask Acc: {acc_global:.2f}% | Local Mask on Global Weights Acc: {acc_cross:.2f}%")
                
            round_circuits["clients_global_model"][f"client_{i}"] = cg_circs
            
        return round_circuits
