import torch
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

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
        self.client_streams = (
            [torch.cuda.Stream() for _ in range(config.num_clients)]
            if 'cuda' in str(config.device) else []
        )

    def _clone_global_model(self):
        from core.models import get_model
        model_copy = get_model(self.config)
        model_copy.load_state_dict(self.global_model.state_dict())
        model_copy.to(self.config.device)
        return model_copy

    def aggregate(self, client_models):
        weights = 1.0 / len(client_models)
        target_device = self.config.device
        
        client_state_dicts = [m.state_dict() for m in client_models]
        new_state = {}
        for key in client_state_dicts[0]:
            # Aggregate purely on GPU to eliminate PCIe transfers
            acc = client_state_dicts[0][key].clone().float().mul_(weights)
            for sd in client_state_dicts[1:]:
                acc.add_(sd[key].float(), alpha=weights)
            new_state[key] = acc.to(target_device)
        self.global_model.load_state_dict(new_state)

    def _discover_global(self, client, gm_copy, local_circuits, stream=None, log_file=None):
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            classes = list(range(self.config.num_classes))

            physical_conn = extract_sparse_connectivity(gm_copy)
            class_samples = client.class_samples

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

            if stream is not None:
                stream.synchronize()

        return client.client_id, cg_circs

    def orchestrate_round(self, round_num, clients: list, log_file=None):
        print(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
        n = len(clients)
        client_models = [None] * n
        client_train_metrics = {}
        client_test_metrics = {}
        round_circuits = {"clients_local_model": {}, "clients_global_model": {}}

        # Phase 1: train + evaluate all clients in parallel, each on its own CUDA stream.
        def _train(i, client):
            stream = self.client_streams[i] if self.client_streams else None
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                model_copy = self._clone_global_model()
                trained_model, metrics = client.train(model_copy)
                test_metrics = client.evaluate_on_test(trained_model, self.evaluation_cache)
                if stream is not None:
                    stream.synchronize()
            save_local_model(trained_model, round_num, i, self.config)
            return i, trained_model, metrics, test_metrics

        with ThreadPoolExecutor(max_workers=n) as ex:
            for i, trained, metrics, test_metrics in [f.result() for f in as_completed(
                {ex.submit(_train, i, c): i for i, c in enumerate(clients)}
            )]:
                client_models[i] = trained
                client_train_metrics[i] = metrics
                client_test_metrics[i] = test_metrics

        # Phase 2: local discovery + aggregation in parallel.
        # Snapshot state dicts upfront so aggregation reads stable tensors
        # while discovery concurrently touches model metadata (requires_grad, hooks).
        client_state_dicts = [m.state_dict() for m in client_models]

        def _aggregate():
            weights = 1.0 / len(client_state_dicts)
            target_device = self.config.device
            new_state = {}
            for key in client_state_dicts[0]:
                acc = client_state_dicts[0][key].clone().float().mul_(weights)
                for sd in client_state_dicts[1:]:
                    acc.add_(sd[key].float(), alpha=weights)
                new_state[key] = acc.to(target_device)
            self.global_model.load_state_dict(new_state)

        def _discover_local(i, client, model):
            stream = self.client_streams[i] if self.client_streams else None
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                result = client.discover_circuits(model, self.evaluation_cache)
                if stream is not None:
                    stream.synchronize()
            return i, result

        with ThreadPoolExecutor(max_workers=n + 1) as ex:
            agg_future = ex.submit(_aggregate)
            disc_futures = {ex.submit(_discover_local, i, c, client_models[i]): i for i, c in enumerate(clients)}
            for i, circs in [f.result() for f in as_completed(disc_futures)]:
                round_circuits["clients_local_model"][f"client_{i}"] = circs
            agg_future.result()

        # Phase 3: global discovery in parallel, each client on its own stream.
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {
                ex.submit(
                    self._discover_global,
                    client,
                    self._clone_global_model(),
                    round_circuits["clients_local_model"].get(f"client_{i}", {}),
                    self.client_streams[i] if self.client_streams else None,
                    log_file,
                ): i
                for i, client in enumerate(clients)
            }
            for cid, cg_circs in [f.result() for f in as_completed(futures)]:
                round_circuits["clients_global_model"][f"client_{cid}"] = cg_circs

        return round_circuits, client_train_metrics, client_test_metrics
