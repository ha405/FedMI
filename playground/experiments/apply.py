import sys
from .base import BaseExperiment
from playground.core import load_config, load_model, load_dataset, load_circuits
from playground.core.evaluator import eval_full, eval_sufficiency, eval_necessity


class ApplyCircuitExperiment(BaseExperiment):
    def run(self):
        args = self.args
        cfg = load_config(args.exp_dir)
        testloader = load_dataset(cfg)
        model = load_model(args.exp_dir, cfg, round_num=args.round)
        circuits = load_circuits(args.exp_dir, source=args.source, round_key=args.round_key or "last")

        client_key = f"client_{args.client_id}"
        if client_key not in circuits:
            sys.exit(f"[ERROR] {client_key} not found. Available: {sorted(circuits.keys())}")

        client_circuits = circuits[client_key]
        class_names = [str(i) for i in range(cfg.num_classes)]

        self.print_header(f"Full Model Baseline — {args.exp_dir}")
        result = eval_full(model, testloader, cfg)
        self.print_result("Overall accuracy:", f"{result['overall']:.2f}%")
        for cls, acc in result["per_class"].items():
            self.print_result(f"  Class {cls}:", f"{acc}%" if acc is not None else "N/A")

        for class_name, class_data in sorted(client_circuits.items()):
            circuit = class_data.get("active_nodes", {})
            if not circuit:
                continue

            tc = int(class_name) if class_name.isdigit() else None
            if tc is None or tc >= cfg.num_classes:
                continue

            self.print_header(f"Circuit: {client_key} / class '{class_name}' ({args.source})")

            if "sufficient" in args.eval_modes or "all" in args.eval_modes:
                suff = eval_sufficiency(model, testloader, circuit, tc, cfg)
                self.print_result("Sufficiency (circuit only):", f"{suff:.2f}%")

            if "necessary" in args.eval_modes or "all" in args.eval_modes:
                nec = eval_necessity(model, testloader, circuit, tc, cfg)
                self.print_result("Necessity (circuit removed):", f"{nec:.2f}%")

            saved_acc = class_data.get("metrics", {}).get("accuracy")
            if saved_acc is not None:
                self.print_result("Saved accuracy (from run):", f"{saved_acc:.2f}%")

            layer_counts = {l: len(v) for l, v in circuit.items()}
            self.print_result("Active nodes per layer:", str(layer_counts))


def add_args(subparsers):
    p = subparsers.add_parser("apply", help="Apply circuits to a model and evaluate on the test set.")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--client_id", type=int, required=True)
    p.add_argument("--source", choices=["local", "global"], default="local",
                   help="Which model's circuits to load (local or global)")
    p.add_argument("--round", type=int, default=None, help="Checkpoint round (default: last)")
    p.add_argument("--round_key", default=None, help="Circuit round key, e.g. 'round_5' (default: last)")
    p.add_argument("--eval_modes", nargs="+", default=["all"],
                   choices=["all", "sufficient", "necessary"],
                   help="Which evaluations to run")
    return p
