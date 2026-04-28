import torch
import torch.nn as nn
import torch.optim as optim

from circuits.pruning import get_current_sparsity, apply_weight_sparsity
from circuits.discovery import discover_client_circuit_cached, compute_layer_means, precollect_all_class_samples, is_valid_layer
from circuits.evaluation import (
    evaluate_circuit_cached, evaluate_circuit_necessity_cached,
    extract_sparse_connectivity, filter_connectivity_by_circuit
)
from core.data_cache import EvaluationCache


class FederatedClient:
    def __init__(self, client_id, train_dataloader, discovery_dataloader, config, class_names):
        self.client_id = client_id
        self.dataloader = train_dataloader
        self.discovery_dataloader = discovery_dataloader if discovery_dataloader is not None else train_dataloader
        self.config = config
        self.class_names = class_names
        self.device = config.device

    def train(self, model):
        model.train()
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_steps = len(self.dataloader) * self.config.local_epochs
        current_step = 0
        running_loss = 0.0
        correct = 0
        total = 0

        for _ in range(self.config.local_epochs):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if self.config.train_mode == 'sparse':
                    apply_weight_sparsity(model, get_current_sparsity(
                        current_step=current_step,
                        total_steps=total_steps,
                        final_sparsity=self.config.target_sparsity
                    ))
                current_step += 1

        if self.config.train_mode == 'sparse':
            apply_weight_sparsity(model, self.config.target_sparsity)

        avg_loss = running_loss / current_step if current_step > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        return model, {"loss": avg_loss, "accuracy": train_acc}

    def evaluate_on_test(self, model, cache: EvaluationCache) -> dict:
        """Evaluate local model on global test set for this client's available classes only.
        Returns {class_id: accuracy_float}.
        """
        classes = None
        if self.config.classes_to_discover_per_client:
            # Robust to int or string keys (from JSON resume)
            classes = self.config.classes_to_discover_per_client.get(self.client_id) or \
                      self.config.classes_to_discover_per_client.get(str(self.client_id))
        
        if classes is None:
            classes = self.config.classes_to_analyze or list(range(self.config.num_classes))

        model.eval()
        class_acc = {}
        with torch.no_grad():
            for c in classes:
                inputs, labels = cache.get_class_data(c)
                if inputs.shape[0] == 0:
                    continue
                correct, total = 0, 0
                for i in range(0, inputs.shape[0], self.config.batch_size):
                    out = model(inputs[i:i + self.config.batch_size])
                    _, pred = torch.max(out, 1)
                    lbl = labels[i:i + self.config.batch_size]
                    total += lbl.size(0)
                    correct += (pred == lbl).sum().item()
                class_acc[c] = round(100.0 * correct / total, 4) if total > 0 else 0.0
        return class_acc

    def discover_circuits(self, model, evaluation_cache: EvaluationCache):
        if self.config.classes_to_discover_per_client and self.client_id in self.config.classes_to_discover_per_client:
            classes_to_analyze = self.config.classes_to_discover_per_client[self.client_id]
        elif self.config.classes_to_analyze is not None:
            classes_to_analyze = self.config.classes_to_analyze
        else:
            classes_to_analyze = list(range(self.config.num_classes))

        layer_means = compute_layer_means(model, self.discovery_dataloader, self.config) if self.config.use_mean_ablation else None
        physical_connectivity = extract_sparse_connectivity(model)

        class_samples = precollect_all_class_samples(
            self.discovery_dataloader, classes_to_analyze,
            max_per_class=1024, device=self.device
        )

        client_circuits = {}
        for tc in classes_to_analyze:
            name = self.class_names[tc] if self.class_names and 0 <= tc < len(self.class_names) else str(tc)

            if tc in class_samples:
                c_inputs, c_labels = class_samples[tc]
                circ = discover_client_circuit_cached(model, c_inputs, c_labels, tc, self.config, layer_means=layer_means)
            else:
                circ = {n: [] for n, m in model.named_modules() if is_valid_layer(n, m)}

            client_circuits[name] = {
                "active_nodes": circ,
                "connectivity": filter_connectivity_by_circuit(physical_connectivity, circ),
                "metrics": {
                    "accuracy": evaluate_circuit_cached(model, evaluation_cache, circ, tc, self.config, layer_means=layer_means),
                    "necessity": evaluate_circuit_necessity_cached(model, evaluation_cache, circ, tc, self.config),
                }
            }

        return client_circuits
