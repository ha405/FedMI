import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from core.dataset import get_client_class_counts
from circuits.pruning import get_current_sparsity, apply_weight_sparsity, apply_weight_sparsity
from circuits.discovery import discover_client_circuit, compute_layer_means
from circuits.evaluation import evaluate_circuit, evaluate_circuit_necessity, extract_sparse_connectivity, filter_connectivity_by_circuit

class FederatedClient:
    def __init__(self, client_id, dataloader, config, class_names):
        self.client_id = client_id
        self.dataloader = dataloader
        self.config = config
        self.class_names = class_names
        self.device = config.device
        
    def train(self, model):
        """
        Performs local training on the client's data.
        Returns the trained model.
        """
        model.train()
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # FedRS Logic
        cdist = None
        use_rs = self.config.use_fedrs
        if use_rs:
            cnts = get_client_class_counts(self.dataloader, self.config.num_classes)
            if cnts.sum() > 0:
                dist = cnts / cnts.sum()
            else:
                dist = torch.ones(self.config.num_classes) / self.config.num_classes
            
            if dist.max() > 0:
                cdist = dist / dist.max()
            else:
                cdist = torch.ones(self.config.num_classes)
                
            alpha = self.config.fedrs_alpha
            cdist = cdist * (1.0 - alpha) + alpha
            cdist = cdist.to(self.device).view(1, -1)

        total_steps = len(self.dataloader) * self.config.local_epochs
        current_step = 0
        
        for epoch in range(self.config.local_epochs):
            model.train()
            # Using leave=False to avoid cluttering stdout
            # progress_bar = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}", leave=False)
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if use_rs and cdist is not None:
                    outputs = outputs * cdist
                    
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Dynamic Sparsity
                if self.config.train_mode == 'sparse':
                    sparsity_to_apply = get_current_sparsity(
                        current_step=current_step, total_steps=total_steps,
                        final_sparsity=self.config.target_sparsity
                    )
                    apply_weight_sparsity(model, sparsity_to_apply)
                
                current_step += 1
                
        # Final sparsity application to ensure target is met
        if self.config.train_mode == 'sparse':
            apply_weight_sparsity(model, self.config.target_sparsity) 
        return model

    def discover_circuits(self, model, testloader, test_layer_means=None):
        """
        Discovers circuits for specific classes based on client's data.
        Returns a dictionary of discovered circuits and metrics.
        """
        client_circuits = {}
        if self.config.classes_to_discover_per_client and self.client_id in self.config.classes_to_discover_per_client:
            classes_to_analyze = self.config.classes_to_discover_per_client[self.client_id]
        elif self.config.classes_to_analyze is not None:
            classes_to_analyze = self.config.classes_to_analyze
        else:
            classes_to_analyze = list(range(self.config.num_classes))

            
        # Compute layer means if needed (for mean ablation)
        layer_means = None
        if self.config.use_mean_ablation:
            layer_means = compute_layer_means(model, self.dataloader, self.config)
            
        physical_connectivity = extract_sparse_connectivity(model)

        for tc in classes_to_analyze:
            if self.class_names and 0 <= tc < len(self.class_names):
                name = self.class_names[tc]
            else:
                name = str(tc)
            
            # 1. Discover Circuit
            circ = discover_client_circuit(model, self.dataloader, tc, self.config, layer_means=layer_means)
            
            # 2. Filter Connectivity
            func_conn = filter_connectivity_by_circuit(physical_connectivity, circ)
            
            # 3. Evaluate on Test Set
            acc = evaluate_circuit(model, testloader, circ, tc, self.config, layer_means=layer_means)
            inv_acc = evaluate_circuit_necessity(model, testloader, circ, tc, self.config)
            
            client_circuits[name] = {
                "active_nodes": circ,
                "connectivity": func_conn,
                "metrics": {
                    "accuracy": acc,
                    "necessity": inv_acc
                }
            }
            
        return client_circuits
