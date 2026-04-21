from typing import Dict, List


def stitch_circuits(client_circuits: Dict[str, dict], strategy: str = "union") -> dict:
    all_layers = set()
    for circuit in client_circuits.values():
        all_layers.update(circuit.keys())

    stitched = {}
    for layer in all_layers:
        layer_index_sets = [
            set(circuit.get(layer, []))
            for circuit in client_circuits.values()
        ]
        if strategy == "union":
            merged = set()
            for s in layer_index_sets:
                merged |= s
            stitched[layer] = sorted(merged)

        elif strategy == "intersection":
            merged = layer_index_sets[0].copy() if layer_index_sets else set()
            for s in layer_index_sets[1:]:
                merged &= s
            stitched[layer] = sorted(merged)

        elif strategy == "majority":
            from collections import Counter
            counts = Counter()
            for s in layer_index_sets:
                for idx in s:
                    counts[idx] += 1
            threshold = len(layer_index_sets) / 2
            stitched[layer] = sorted(idx for idx, cnt in counts.items() if cnt > threshold)

        else:
            raise ValueError(f"Unknown stitch strategy: '{strategy}'. Choose: union, intersection, majority")

    return stitched


def available_strategies() -> List[str]:
    return ["union", "intersection", "majority"]
