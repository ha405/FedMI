"""
analysis/circuit_consistency.py

Modes
-----
1.  Single-path:
        python analysis/circuit_consistency.py <exp_dir> [round_num]

2.  Inter-client comparison across all model families:
        python analysis/circuit_consistency.py --compare [results_dir]

3.  Local-vs-global comparison across all model families:
        python analysis/circuit_consistency.py --local-global [results_dir]

4.  Intra-client stability across all model families:
        python analysis/circuit_consistency.py --intra-client [results_dir]

Modes 2, 3 & 4 save figures to results/figures_<family>/
"""

import os, sys, json, itertools

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ───────────────── IoU helpers ─────────────────

def _nodes(circuit_data, layer):
    try:
        return [int(x) for x in circuit_data.get("active_nodes", {}).get(layer, [])]
    except Exception:
        return []


def _iou(a, b):
    s1, s2 = set(a), set(b)
    u = len(s1 | s2)
    return (len(s1 & s2) / u) if u else 1.0


def avg_layer_iou(d1, d2, layers):
    vals = [_iou(_nodes(d1, l), _nodes(d2, l)) for l in layers]
    return float(np.mean(vals)) if vals else np.nan


def layer_names_from_round(round_data):
    names = set()
    for client_data in round_data.values():
        for class_data in client_data.values():
            names.update(class_data.get("active_nodes", {}).keys())
    return sorted(names)


# ───────────────── Data loading ─────────────────

def load_all_circuits(exp_dir):
    path = os.path.join(exp_dir, "circuits", "all_circuits.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"all_circuits.json not found in {exp_dir}/circuits/")
    with open(path) as f:
        data = json.load(f)
    round_keys = sorted(data.keys(), key=lambda r: int(r.split("_")[1]))
    return data, round_keys


def load_single_round(exp_dir, round_num):
    path = os.path.join(exp_dir, "circuits", f"circuits_round_{round_num}.json")
    if not os.path.exists(path):
        # fall back to all_circuits
        data, rkeys = load_all_circuits(exp_dir)
        key = f"round_{round_num}"
        if key not in data:
            raise KeyError(f"round_{round_num} not found")
        return data[key]
    with open(path) as f:
        return json.load(f)


def last_round_number(exp_dir):
    circuits_dir = os.path.join(exp_dir, "circuits")
    files = [f for f in os.listdir(circuits_dir) if f.startswith("circuits_round_")]
    nums = [int(f.replace("circuits_round_", "").replace(".json", "")) for f in files]
    return max(nums) if nums else None


# ───────────────── Per-round inter-client IoU ─────────────────

def inter_client_iou_all_classes(round_data, model_key="clients_local_model"):
    """
    Returns {class_id: mean_iou_over_client_pairs} for one round.
    """
    clients_data = round_data[model_key]
    client_keys = sorted(clients_data.keys())
    layers = layer_names_from_round(clients_data)

    # collect all class ids
    all_classes = sorted(clients_data[client_keys[0]].keys(), key=lambda x: int(x))

    result = {}
    for cls in all_classes:
        pair_ious = [
            avg_layer_iou(clients_data[c1][cls], clients_data[c2][cls], layers)
            for c1, c2 in itertools.combinations(client_keys, 2)
        ]
        result[cls] = float(np.nanmean(pair_ious)) if pair_ious else np.nan
    return result


def mean_iou_over_rounds(exp_dir, model_key="clients_local_model"):
    """
    Returns (round_numbers, per_class_series, mean_series)
      per_class_series: {cls: [iou_r1, iou_r2, ...]}
      mean_series: [mean_iou_r1, mean_iou_r2, ...]
    """
    data, round_keys = load_all_circuits(exp_dir)
    round_nums = [int(rk.split("_")[1]) for rk in round_keys]

    per_class = None
    mean_series = []

    for rk in round_keys:
        cls_iou = inter_client_iou_all_classes(data[rk], model_key)
        if per_class is None:
            per_class = {cls: [] for cls in cls_iou}
        for cls, v in cls_iou.items():
            per_class[cls].append(v)
        mean_series.append(float(np.nanmean(list(cls_iou.values()))))

    return round_nums, per_class, mean_series


# ───────────────── Local vs Global IoU ─────────────────

def local_vs_global_iou_all_classes(round_data):
    """
    For each class, compute mean IoU between each client's local circuit and
    that same client's global circuit, averaged over all clients.
    Returns {class_id: mean_iou}.
    """
    local_data  = round_data["clients_local_model"]
    global_data = round_data["clients_global_model"]
    client_keys = sorted(local_data.keys())
    layers      = layer_names_from_round(local_data)
    all_classes = sorted(local_data[client_keys[0]].keys(), key=lambda x: int(x))

    result = {}
    for cls in all_classes:
        per_client = [
            avg_layer_iou(local_data[c][cls], global_data[c][cls], layers)
            for c in client_keys
            if cls in local_data[c] and cls in global_data.get(c, {})
        ]
        result[cls] = float(np.nanmean(per_client)) if per_client else np.nan
    return result


def local_vs_global_over_rounds(exp_dir):
    """
    Returns (round_numbers, per_class_series, mean_series) for local vs global.
    """
    data, round_keys = load_all_circuits(exp_dir)
    round_nums = [int(rk.split("_")[1]) for rk in round_keys]

    per_class   = None
    mean_series = []

    for rk in round_keys:
        cls_iou = local_vs_global_iou_all_classes(data[rk])
        if per_class is None:
            per_class = {cls: [] for cls in cls_iou}
        for cls, v in cls_iou.items():
            per_class[cls].append(v)
        mean_series.append(float(np.nanmean(list(cls_iou.values()))))

    return round_nums, per_class, mean_series


# ───────────────── Plot helpers ─────────────────

def style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


# ───────────────── Mode 1: single checkpoint ─────────────────

def run_single(exp_dir, round_num=None):
    if round_num is None:
        round_num = last_round_number(exp_dir)
        if round_num is None:
            sys.exit("No circuit files found.")

    print(f"Loading round {round_num} from {exp_dir}")
    rd = load_single_round(exp_dir, round_num)
    cls_iou = inter_client_iou_all_classes(rd)

    classes = sorted(cls_iou.keys(), key=lambda x: int(x))
    values = [cls_iou[c] for c in classes]
    mean_val = float(np.nanmean(values))

    figures_dir = os.path.join(exp_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(classes)), values, color=plt.cm.tab10(np.linspace(0, 1, len(classes))))
    ax.axhline(mean_val, linestyle="--", color="black", linewidth=1.2, label=f"Mean = {mean_val:.3f}")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"class {c}" for c in classes], rotation=45, ha="right")
    ax.set_title(f"Inter-client Circuit Consistency — Round {round_num}\n{os.path.basename(exp_dir)}")
    ax.set_ylabel("Mean IoU (avg over client pairs)")
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(figures_dir, f"inter_client_all_classes_round{round_num}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")

    # also save as CSV for reference
    csv_path = os.path.join(figures_dir, f"inter_client_all_classes_round{round_num}.csv")
    with open(csv_path, "w") as f:
        f.write("class,mean_iou\n")
        for c in classes:
            f.write(f"{c},{cls_iou[c]:.6f}\n")
    print(f"Saved: {csv_path}")


# ───────────────── Model families ─────────────────
#
# Each family:  (slug, display_title, [(dir, legend_label), ...])
# slug         → used for output subfolder  figures_<slug>/
# display_title → shown in plot titles

MODEL_FAMILIES = [
    (
        "cifar_cnn",
        "CIFAR-10 · CNN",
        [
            ("CIFAR_CNN",     "IID"),
            ("CIFAR_CNN_05",  "α = 0.5"),
            ("CIFAR_CNN_02",  "α = 0.2"),
            ("CIFAR_CNN_005", "α = 0.05"),
        ],
    ),
    (
        "cifar_resnet",
        "CIFAR-10 · ResNet",
        [
            ("CIFAR_ResNet",     "IID"),
            ("CIFAR_ResNet_05",  "α = 0.5"),
            ("CIFAR_ResNet_02",  "α = 0.2"),
            ("CIFAR_ResNet_005", "α = 0.05"),
        ],
    ),
    (
        "fmnist_cnn",
        "Fashion-MNIST · CNN",
        [
            ("FMNIST_CNN",     "IID"),
            ("FMNIST_CNN_05",  "α = 0.5"),
            ("FMNIST_CNN_02",  "α = 0.2"),
            ("FMNIST_CNN_005", "α = 0.05"),
        ],
    ),
    (
        "fmnist_resnet",
        "Fashion-MNIST · ResNet",
        [
            ("fmnist_resnet",     "IID"),
            ("fmnist_resnet_05",  "α = 0.5"),
            ("fmnist_resnet_02",  "α = 0.2"),
            ("fmnist_resnet_005", "α = 0.05"),
        ],
    ),
    (
        "cifar_cnn_dense",
        "CIFAR-10 · CNN (Dense)",
        [
            ("CIFAR_CNN_dense",     "IID"),
            ("CIFAR_CNN_005_dense", "α = 0.05"),
        ],
    ),
]

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

# keep for single-path mode (backward compat)
CONFIGS = MODEL_FAMILIES[0][2]


def _collect_inter_client(results_dir, configs):
    all_rounds_data = {}
    final_class_iou = {}
    for dirname, label in configs:
        exp_dir = os.path.join(results_dir, dirname)
        if not os.path.isdir(exp_dir):
            print(f"  WARNING: {exp_dir} not found, skipping.")
            continue
        print(f"  {dirname} ...")
        rnums, per_class, mean_series = mean_iou_over_rounds(exp_dir)
        all_rounds_data[dirname] = (rnums, per_class, mean_series, label)
        data, _ = load_all_circuits(exp_dir)
        last_key = sorted(data.keys(), key=lambda r: int(r.split("_")[1]))[-1]
        final_class_iou[dirname] = inter_client_iou_all_classes(data[last_key])
    return all_rounds_data, final_class_iou


def _plot_line(all_rounds_data, configs, figures_dir, title, ylabel, fname):
    SAMPLE_POINTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (dirname, _) in enumerate(configs):
        if dirname not in all_rounds_data:
            continue
        rnums, _, mean_series, lbl = all_rounds_data[dirname]
        arr_r = np.array(rnums)
        arr_m = np.array(mean_series)
        mask  = np.isin(arr_r, SAMPLE_POINTS)
        ax.plot(arr_r[mask], arr_m[mask], label=lbl, color=COLORS[i],
                linewidth=2.2, marker="o", markersize=6,
                markerfacecolor="white", markeredgewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_xticks(SAMPLE_POINTS)
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    ax.legend(title="Data distribution")
    fig.tight_layout()
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_per_class_bars(final_class_iou, configs, figures_dir, title, ylabel, fname):
    all_classes = sorted(next(iter(final_class_iou.values())).keys(), key=lambda x: int(x))
    n_classes = len(all_classes)
    n_configs = sum(1 for d, _ in configs if d in final_class_iou)
    bar_w = 0.8 / n_configs
    x = np.arange(n_classes)
    fig, ax = plt.subplots(figsize=(14, 6))
    offset = 0
    for i, (dirname, label) in enumerate(configs):
        if dirname not in final_class_iou:
            continue
        vals = [final_class_iou[dirname].get(c, np.nan) for c in all_classes]
        pos  = x + (offset - (n_configs - 1) / 2) * bar_w
        ax.bar(pos, vals, bar_w * 0.9, label=label, color=COLORS[i])
        offset += 1
    ax.set_xticks(x)
    ax.set_xticklabels([f"class {c}" for c in all_classes])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    ax.legend(title="Data distribution")
    fig.tight_layout()
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_summary_bar(final_class_iou, configs, figures_dir, title, ylabel, fname):
    labels, means, colors = [], [], []
    for i, (dirname, label) in enumerate(configs):
        if dirname not in final_class_iou:
            continue
        labels.append(label)
        means.append(float(np.nanmean(list(final_class_iou[dirname].values()))))
        colors.append(COLORS[i])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, color=colors, width=0.5)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.1)
    style_ax(ax)
    fig.tight_layout()
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def run_comparison(results_dir):
    for slug, display, configs in MODEL_FAMILIES:
        print(f"\n[inter-client] {display}")
        figures_dir = os.path.join(results_dir, f"figures_{slug}")
        os.makedirs(figures_dir, exist_ok=True)

        all_rounds_data, final_class_iou = _collect_inter_client(results_dir, configs)
        if not all_rounds_data:
            continue

        _plot_line(
            all_rounds_data, configs, figures_dir,
            title=f"Inter-client Circuit Consistency — {display}\nIID → non-IID (mean over all classes & client pairs)",
            ylabel="Mean IoU (avg over all classes & client pairs)",
            fname=f"{slug}_inter_client_mean_iou_over_rounds.png",
        )
        _plot_per_class_bars(
            final_class_iou, configs, figures_dir,
            title=f"Inter-client Consistency per Class (final round) — {display}",
            ylabel="Mean IoU (avg over client pairs)",
            fname=f"{slug}_inter_client_per_class_final_round.png",
        )
        _plot_summary_bar(
            final_class_iou, configs, figures_dir,
            title=f"Inter-client Consistency (final round) — {display}",
            ylabel="Mean IoU (all classes & client pairs)",
            fname=f"{slug}_inter_client_summary_bar.png",
        )


# ───────────────── Mode 3: Local vs Global comparison ─────────────────

# ───────────────── Intra-client stability ─────────────────

def intra_client_iou_over_rounds(exp_dir):
    """
    For each consecutive pair of rounds (N, N+1), compute IoU between each
    client's circuit at round N and round N+1, averaged over clients & classes.
    Returns (round_numbers, per_class_series, mean_series) where round_numbers
    are the N+1 values (i.e. the later round in each pair).
    """
    data, round_keys = load_all_circuits(exp_dir)

    first_local = data[round_keys[0]]["clients_local_model"]
    client_keys = sorted(first_local.keys())
    layers      = layer_names_from_round(first_local)
    all_classes = sorted(first_local[client_keys[0]].keys(), key=lambda x: int(x))

    round_nums  = []
    per_class   = {cls: [] for cls in all_classes}
    mean_series = []

    for i in range(len(round_keys) - 1):
        rk_a = round_keys[i]
        rk_b = round_keys[i + 1]
        data_a = data[rk_a]["clients_local_model"]
        data_b = data[rk_b]["clients_local_model"]
        round_nums.append(int(rk_b.split("_")[1]))

        cls_means = {}
        for cls in all_classes:
            per_client = [
                avg_layer_iou(data_a[c][cls], data_b[c][cls], layers)
                for c in client_keys
                if cls in data_a.get(c, {}) and cls in data_b.get(c, {})
            ]
            cls_means[cls] = float(np.nanmean(per_client)) if per_client else np.nan
        for cls, v in cls_means.items():
            per_class[cls].append(v)
        mean_series.append(float(np.nanmean(list(cls_means.values()))))

    return round_nums, per_class, mean_series


def _collect_intra_client(results_dir, configs):
    all_rounds_data = {}
    for dirname, label in configs:
        exp_dir = os.path.join(results_dir, dirname)
        if not os.path.isdir(exp_dir):
            print(f"  WARNING: {exp_dir} not found, skipping.")
            continue
        print(f"  {dirname} ...")
        rnums, per_class, mean_series = intra_client_iou_over_rounds(exp_dir)
        all_rounds_data[dirname] = (rnums, per_class, mean_series, label)
    return all_rounds_data


def run_intra_client_comparison(results_dir):
    for slug, display, configs in MODEL_FAMILIES:
        print(f"\n[intra-client] {display}")
        figures_dir = os.path.join(results_dir, f"figures_{slug}")
        os.makedirs(figures_dir, exist_ok=True)

        all_rounds_data = _collect_intra_client(results_dir, configs)
        if not all_rounds_data:
            continue

        _plot_line(
            all_rounds_data, configs, figures_dir,
            title=f"Intra-client Circuit Stability — {display}\n"
                  "IoU between consecutive rounds (mean over clients & classes)",
            ylabel="Mean IoU (round N vs. round N+1, avg over clients & classes)",
            fname=f"{slug}_intra_client_mean_iou_over_rounds.png",
        )

def _collect_local_global(results_dir, configs):
    all_rounds_data = {}
    final_class_iou = {}
    for dirname, label in configs:
        exp_dir = os.path.join(results_dir, dirname)
        if not os.path.isdir(exp_dir):
            print(f"  WARNING: {exp_dir} not found, skipping.")
            continue
        print(f"  {dirname} ...")
        rnums, per_class, mean_series = local_vs_global_over_rounds(exp_dir)
        all_rounds_data[dirname] = (rnums, per_class, mean_series, label)
        data, _ = load_all_circuits(exp_dir)
        last_key = sorted(data.keys(), key=lambda r: int(r.split("_")[1]))[-1]
        final_class_iou[dirname] = local_vs_global_iou_all_classes(data[last_key])
    return all_rounds_data, final_class_iou


def _plot_inter_vs_local_global(results_dir, configs, figures_dir, display, slug):
    """Side-by-side bar comparing inter-client vs local-global at final round."""
    inter_iou = {}
    local_iou = {}
    for dirname, label in configs:
        exp_dir = os.path.join(results_dir, dirname)
        if not os.path.isdir(exp_dir):
            continue
        data, _ = load_all_circuits(exp_dir)
        last_key = sorted(data.keys(), key=lambda r: int(r.split("_")[1]))[-1]
        inter_iou[dirname] = float(np.nanmean(
            list(inter_client_iou_all_classes(data[last_key]).values())
        ))
        local_iou[dirname] = float(np.nanmean(
            list(local_vs_global_iou_all_classes(data[last_key]).values())
        ))

    present  = [(d, l) for d, l in configs if d in inter_iou]
    labels_p = [l for _, l in present]
    inter_v  = [inter_iou[d] for d, _ in present]
    local_v  = [local_iou[d]  for d, _ in present]
    clrs     = [COLORS[i] for i, (d, _) in enumerate(configs) if d in inter_iou]

    x2, w2 = np.arange(len(present)), 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x2 - w2/2, inter_v, w2, label="Inter-client",
                color=clrs, edgecolor="white")
    b2 = ax.bar(x2 + w2/2, local_v, w2, label="Local vs. Global",
                color=clrs, alpha=0.55, edgecolor="white", hatch="//")
    for bar, v in zip(list(b1) + list(b2), inter_v + local_v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels_p, fontsize=10)
    ax.set_ylabel("Mean IoU (final round, all classes)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Circuit Consistency: Inter-client vs. Local–Global (final round)\n{display}")
    style_ax(ax)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{slug}_inter_vs_local_global_summary.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def run_local_global_comparison(results_dir):
    for slug, display, configs in MODEL_FAMILIES:
        print(f"\n[local-vs-global] {display}")
        figures_dir = os.path.join(results_dir, f"figures_{slug}")
        os.makedirs(figures_dir, exist_ok=True)

        all_rounds_data, final_class_iou = _collect_local_global(results_dir, configs)
        if not all_rounds_data:
            continue

        _plot_line(
            all_rounds_data, configs, figures_dir,
            title=f"Local vs. Global Circuit Consistency — {display}\nIID → non-IID (mean over all classes & clients)",
            ylabel="Mean IoU (local circuit vs. global circuit, per client)",
            fname=f"{slug}_local_vs_global_mean_iou_over_rounds.png",
        )
        _plot_per_class_bars(
            final_class_iou, configs, figures_dir,
            title=f"Local vs. Global Consistency per Class (final round) — {display}",
            ylabel="Mean IoU (local vs. global, avg over clients)",
            fname=f"{slug}_local_vs_global_per_class_final_round.png",
        )
        _plot_inter_vs_local_global(results_dir, configs, figures_dir, display, slug)


# ───────────────── Single-experiment analysis (called from main.py) ─────────────────

def _plot_single_line(round_nums, mean_series, figures_dir, title, ylabel, fname):
    SAMPLE_POINTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, ax = plt.subplots(figsize=(9, 5))
    arr_r = np.array(round_nums)
    arr_m = np.array(mean_series)
    mask  = np.isin(arr_r, SAMPLE_POINTS)
    if not mask.any():
        mask = np.ones(len(arr_r), dtype=bool)
    ax.plot(arr_r[mask], arr_m[mask], color=COLORS[0], linewidth=2.2,
            marker="o", markersize=6, markerfacecolor="white", markeredgewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    fig.tight_layout()
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def run_single_experiment_analysis(exp_dir, figures_dir=None):
    """Generate inter, intra, and local-vs-global mean IoU figures for one experiment."""
    if not os.path.exists(os.path.join(exp_dir, "circuits", "all_circuits.json")):
        print(f"  [analysis] all_circuits.json not found in {exp_dir}, skipping.")
        return

    if figures_dir is None:
        figures_dir = os.path.join(exp_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    name = os.path.basename(exp_dir)

    try:
        rnums, _, mean_series = mean_iou_over_rounds(exp_dir)
        _plot_single_line(
            rnums, mean_series, figures_dir,
            title=f"Inter-client Circuit Consistency — {name}",
            ylabel="Mean IoU (avg over client pairs & classes)",
            fname="inter_client_mean_iou_over_rounds.png",
        )
    except Exception as e:
        print(f"  [analysis] inter-client failed: {e}")

    try:
        rnums, _, mean_series = local_vs_global_over_rounds(exp_dir)
        _plot_single_line(
            rnums, mean_series, figures_dir,
            title=f"Local vs. Global Circuit Consistency — {name}",
            ylabel="Mean IoU (local vs. global, avg over clients & classes)",
            fname="local_vs_global_mean_iou_over_rounds.png",
        )
    except Exception as e:
        print(f"  [analysis] local-vs-global failed: {e}")

    try:
        rnums, _, mean_series = intra_client_iou_over_rounds(exp_dir)
        _plot_single_line(
            rnums, mean_series, figures_dir,
            title=f"Intra-client Circuit Stability — {name}",
            ylabel="Mean IoU (round N vs. N+1, avg over clients & classes)",
            fname="intra_client_mean_iou_over_rounds.png",
        )
    except Exception as e:
        print(f"  [analysis] intra-client failed: {e}")


# ───────────────── Main ─────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "--compare":
        results_dir = args[1] if len(args) > 1 else "results"
        if not os.path.isdir(results_dir):
            sys.exit(f"Results dir not found: {results_dir}")
        run_comparison(results_dir)
    elif args and args[0] == "--local-global":
        results_dir = args[1] if len(args) > 1 else "results"
        if not os.path.isdir(results_dir):
            sys.exit(f"Results dir not found: {results_dir}")
        run_local_global_comparison(results_dir)
    elif args and args[0] == "--intra-client":
        results_dir = args[1] if len(args) > 1 else "results"
        if not os.path.isdir(results_dir):
            sys.exit(f"Results dir not found: {results_dir}")
        run_intra_client_comparison(results_dir)
    else:
        if not args:
            exp_dir = input("Experiment path: ").strip()
            round_num = None
        else:
            exp_dir = args[0]
            round_num = int(args[1]) if len(args) > 1 else None

        if not os.path.isdir(exp_dir):
            sys.exit(f"Not a directory: {exp_dir}")
        run_single(exp_dir, round_num)
