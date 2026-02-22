import sys
from .base import BaseExperiment
from playground.core import load_config, load_model, load_dataset, load_circuits, stitch_circuits
from playground.core.evaluator import eval_full, eval_sufficiency, eval_necessity
from playground.core.stitcher import available_strategies


class StitchCircuitExperiment(BaseExperiment):
    def run(self):
        args = self.args
        cfg = load_config(args.exp_dir)
        testloader = load_dataset(cfg)
        model = load_model(args.exp_dir, cfg, round_num=args.round)
        circuits = load_circuits(args.exp_dir, source="local", round_key=args.round_key or "last")

        class_names = [str(i) for i in range(cfg.num_classes)]
        target_classes = args.classes or list(range(cfg.num_classes))

        for tc in target_classes:
            class_name = str(tc)
            client_circuits_for_class = {}
            for client_key, client_data in circuits.items():
                if class_name in client_data and client_data[class_name].get("active_nodes"):
                    client_circuits_for_class[client_key] = client_data[class_name]["active_nodes"]

            if not client_circuits_for_class:
                print(f"\n[stitch] Class '{class_name}': no client circuits found — skipping")
                continue

            self.print_header(f"Stitch — class '{class_name}' | strategy: {args.strategy}")
            print(f"  Clients contributing: {sorted(client_circuits_for_class.keys())}")

            stitched = stitch_circuits(client_circuits_for_class, strategy=args.strategy)
            node_counts = {l: len(v) for l, v in stitched.items()}
            self.print_result("Stitched nodes per layer:", str(node_counts))

            suff = eval_sufficiency(model, testloader, stitched, tc, cfg)
            self.print_result("Sufficiency (stitched circuit):", f"{suff:.2f}%")

            nec = eval_necessity(model, testloader, stitched, tc, cfg)
            self.print_result("Necessity (stitched removed):", f"{nec:.2f}%")

            if args.compare_clients:
                print()
                for client_key, individual_circuit in client_circuits_for_class.items():
                    s = eval_sufficiency(model, testloader, individual_circuit, tc, cfg)
                    saved = circuits[client_key][class_name].get("metrics", {}).get("accuracy", "?")
                    self.print_result(f"  {client_key} individual suff:", f"{s:.2f}%  (saved={saved})")


def add_args(subparsers):
    p = subparsers.add_parser("stitch", help="Merge client circuits and evaluate the stitched result.")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--strategy", choices=available_strategies(), default="union",
                   help="How to merge circuits across clients")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Classes to stitch (default: all)")
    p.add_argument("--round", type=int, default=None, help="Checkpoint round (default: last)")
    p.add_argument("--round_key", default=None, help="Circuit round key, e.g. 'round_5' (default: last)")
    p.add_argument("--compare_clients", action="store_true",
                   help="Also evaluate each client's individual circuit alongside the stitched one")
    return p
