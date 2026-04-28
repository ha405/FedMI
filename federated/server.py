import torch
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import FederatedClient
from core.utils import save_local_model
from core.data_cache import EvaluationCache
from circuits.evaluation import (
    extract_sparse_connectivity, filter_connectivity_by_circuit,
    evaluate_circuit_cached, evaluate_circuit_necessity_cached,
)
from circuits.discovery import (
    discover_client_circuit_cached, precollect_all_class_samples, is_valid_layer,
)

class FederatedServer:
    def __init__(self, global_model, config, class_names, evaluation_cache: EvaluationCache):
        self.global_model = global_model
        self.config = config
        self.class_names = class_names
        self.device = config.device
        self.evaluation_cache = evaluation_cache

    def aggregate(self, client_models):
        weights = 1.0 / len(client_models)
        global_state = self.global_model.state_dict()
        target_device = next(self.global_model.parameters()).device

        for key in global_state:
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32, device=target_device)
            for model in client_models:
                global_state[key] += weights * model.state_dict()[key].to(target_device).float()

        self.global_model.load_state_dict(global_state)

    def _discover_global(self, client, gm_copy, local_circuits, log_file=None):
        classes = list(range(self.config.num_classes))

        physical_conn = extract_sparse_connectivity(gm_copy)
        class_samples = precollect_all_class_samples(client.discovery_dataloader, classes, max_per_class=1024, device=self.device)

        cg_circs = {}
        for tc in classes:
            name = self.class_names[tc] if self.class_names and 0 <= tc < len(self.class_names) else str(tc)

            if tc in class_samples:
                c_inputs, c_labels = class_samples[tc]
                circ = discover_client_circuit_cached(gm_copy, c_inputs, c_labels, tc, self.config)
            else:
                circ = {n: [] for n, m in gm_copy.named_modules() if is_valid_layer(n, m)}

            acc_global = evaluate_circuit_cached(gm_copy, self.evaluation_cache, circ, tc, self.config)
            inv_acc = evaluate_circuit_necessity_cached(gm_copy, self.evaluation_cache, circ, tc, self.config)

            cg_circs[name] = {
                "active_nodes": circ,
                "connectivity": filter_connectivity_by_circuit(physical_conn, circ),
                "metrics": {
                    "accuracy": acc_global,
                    "necessity": inv_acc,
                }
            }
            print(f"    [C{client.client_id}|{name}] Global Acc: {acc_global:.2f}% | Nec: {inv_acc:.2f}%")

        return client.client_id, cg_circs

    def orchestrate_round(self, round_num, clients: list, log_file=None):
        print(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
        client_models = [None] * len(clients)
        client_train_metrics = {}
        client_test_metrics = {}
        round_circuits = {"clients_local_model": {}, "clients_global_model": {}}

        def _train(i, client):
            model_copy = copy.deepcopy(self.global_model)
            trained_model, metrics = client.train(model_copy)
            save_local_model(trained_model, round_num, i, self.config)
            return i, trained_model, metrics

        with ThreadPoolExecutor(max_workers=min(len(clients), 5)) as ex:
            for i, trained, metrics in [f.result() for f in as_completed(
                {ex.submit(_train, i, c): i for i, c in enumerate(clients)}
            )]:
                client_models[i] = trained
                client_train_metrics[i] = metrics
                client_test_metrics[i] = clients[i].evaluate_on_test(trained, self.evaluation_cache)

        def _discover_local(i, client, model):
            return i, client.discover_circuits(model, self.evaluation_cache)

        with ThreadPoolExecutor(max_workers=min(len(clients), 5)) as ex:
            for i, circs in [f.result() for f in as_completed(
                {ex.submit(_discover_local, i, c, client_models[i]): i for i, c in enumerate(clients)}
            )]:
                round_circuits["clients_local_model"][f"client_{i}"] = circs

        self.aggregate(client_models)

        with ThreadPoolExecutor(max_workers=min(len(clients), 5)) as ex:
            futures = {
                ex.submit(
                    self._discover_global,
                    client,
                    copy.deepcopy(self.global_model),
                    round_circuits["clients_local_model"].get(f"client_{i}", {}),
                    log_file
                ): i
                for i, client in enumerate(clients)
            }
            for cid, cg_circs in [f.result() for f in as_completed(futures)]:
                round_circuits["clients_global_model"][f"client_{cid}"] = cg_circs

        return round_circuits, client_train_metrics, client_test_metrics
