import torch
import torch.nn as nn
import torch.optim as optim

from circuits.pruning import get_current_sparsity, apply_weight_sparsity
from circuits.discovery import discover_all_classes_cached, precollect_all_class_samples, is_valid_layer
from circuits.evaluation import (
    evaluate_circuit_cached, evaluate_circuit_necessity_cached,
    extract_sparse_connectivity, filter_connectivity_by_circuit
)
from core.data_cache import EvaluationCache


class FederatedClient:
    def __init__(self, client_id, train_dataloader, discovery_dataloader, config, class_names):
        self.client_id = client_id
        self.config = config
        self.class_names = class_names
        self.device = config.device
        
        # Preload train dataloader to GPU memory to avoid PCIe transfers
        # and simultaneously collect discovery samples to avoid a separate loop
        self.dataloader = []
        classes = list(range(config.num_classes))
        max_per_class = 1024
        
        class_inputs = {c: [] for c in classes}
        class_labels = {c: [] for c in classes}
        counts = {c: 0 for c in classes}

        for inputs, labels in train_dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.dataloader.append((inputs, labels))
            
            for c in classes:
                if counts[c] < max_per_class:
                    mask = (labels == c)
                    if mask.any():
                        class_inputs[c].append(inputs[mask])
                        class_labels[c].append(labels[mask])
                        counts[c] += mask.sum().item()

        # To complete num_samples, refer to train set in general 
        if discovery_dataloader is not None and any(counts[c] < max_per_class for c in classes):
            for b_inputs, b_labels in discovery_dataloader:
                b_inputs = b_inputs.to(self.device, non_blocking=True)
                b_labels = b_labels.to(self.device, non_blocking=True)
                for c in classes:
                    if counts[c] < max_per_class:
                        mask = (b_labels == c)
                        if mask.any():
                            class_inputs[c].append(b_inputs[mask])
                            class_labels[c].append(b_labels[mask])
                            counts[c] += mask.sum().item()
                if all(counts[c] >= max_per_class for c in classes):
                    break

        # Ensure discovery data is preloaded and isn't loaded again and again
        self.class_samples = {}
        for c in classes:
            if class_inputs[c]:
                self.class_samples[c] = (
                    torch.cat(class_inputs[c])[:max_per_class],
                    torch.cat(class_labels[c])[:max_per_class]
                )

        # Persistent model to avoid repeated torch.compile overhead
        from core.models import get_model
        self.model = get_model(self.config).to(self.device)
        # self.model = torch.compile(self.model, dynamic=True)
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        #     self.model(dummy_input)

    def train(self, global_model):
        # Update compiled model with the global weights
        target_model = getattr(self.model, "_orig_mod", self.model)
        target_model.load_state_dict(global_model.state_dict())
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_steps = len(self.dataloader) * self.config.local_epochs
        current_step = 0

        running_loss = torch.tensor(0.0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total = 0

        for _ in range(self.config.local_epochs):
            for inputs, labels in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.detach()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum()

                if self.config.train_mode == 'sparse':
                    apply_weight_sparsity(self.model, get_current_sparsity(
                        current_step=current_step,
                        total_steps=total_steps,
                        final_sparsity=self.config.target_sparsity
                    ))
                current_step += 1

        if self.config.train_mode == 'sparse':
            apply_weight_sparsity(self.model, self.config.target_sparsity)

        avg_loss = running_loss.item() / current_step if current_step > 0 else 0.0
        train_acc = 100.0 * correct.item() / total if total > 0 else 0.0
        
        # Return the uncompiled model to avoid '_orig_mod.' prefixes in state_dict
        return getattr(self.model, "_orig_mod", self.model), {"loss": avg_loss, "accuracy": train_acc}

    def evaluate_on_test(self, model, cache: EvaluationCache) -> dict:
        """Evaluate local model on global test set for this client's available classes only.
        Returns {class_id: accuracy_float}.
        """
        classes = list(range(self.config.num_classes))

        # Use our persistent compiled model for evaluation
        if model is not self.model:
            target_model = getattr(self.model, "_orig_mod", self.model)
            target_model.load_state_dict(model.state_dict())
            
        self.model.eval()
        class_acc = {}
        with torch.no_grad():
            for c in classes:
                inputs, labels = cache.get_class_data(c)
                if inputs.shape[0] == 0:
                    continue
                correct = torch.tensor(0, device=self.device)
                total = 0
                for i in range(0, inputs.shape[0], self.config.batch_size):
                    out = self.model(inputs[i:i + self.config.batch_size])
                    _, pred = torch.max(out, 1)
                    lbl = labels[i:i + self.config.batch_size]
                    total += lbl.size(0)
                    correct += (pred == lbl).sum()
                class_acc[c] = round(100.0 * correct.item() / total, 4) if total > 0 else 0.0
        return class_acc

    def discover_circuits(self, model, evaluation_cache: EvaluationCache):
        classes_to_analyze = list(range(self.config.num_classes))

        if model is not self.model:
            target_model = getattr(self.model, "_orig_mod", self.model)
            target_model.load_state_dict(model.state_dict())
            
        physical_connectivity = extract_sparse_connectivity(self.model)

        client_circuits = {}
        
        # Vectorized discovery: 1 call instead of looping
        all_circs = discover_all_classes_cached(self.model, self.class_samples, self.config)
        
        for tc in classes_to_analyze:
            name = self.class_names[tc] if self.class_names and 0 <= tc < len(self.class_names) else str(tc)

            circ = all_circs[tc]

            client_circuits[name] = {
                "active_nodes": circ,
                "connectivity": filter_connectivity_by_circuit(physical_connectivity, circ),
                "metrics": {
                    "accuracy": evaluate_circuit_cached(model, evaluation_cache, circ, tc, self.config),
                    "necessity": evaluate_circuit_necessity_cached(model, evaluation_cache, circ, tc, self.config),
                }
            }

        return client_circuits
