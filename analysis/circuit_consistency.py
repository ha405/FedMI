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

5.  Sparse vs dense CIFAR comparison (for appendix):
        python analysis/circuit_consistency.py --sparse-vs-dense [results_dir]
"""

import os, sys, json, itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":              13,
    "axes.labelsize":         14,
    "xtick.labelsize":        12,
    "ytick.labelsize":        12,
    "legend.fontsize":        12,
    "legend.title_fontsize":  12,
    "lines.linewidth":        3.0,
    "lines.markersize":       9,
})

LINE_KW = dict(linewidth=3.0, marker="o", markersize=9,
               markerfacecolor="white", markeredgewidth=2.5)

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]


# ── IoU helpers ───────────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

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
        data, _ = load_all_circuits(exp_dir)
        key = f"round_{round_num}"
        if key not in data:
            raise KeyError(f"round_{round_num} not found")
        return data[key]
    with open(path) as f:
        return json.load(f)


def last_round_number(exp_dir):
    circuits_dir = os.path.join(exp_dir, "circuits")
    files = [f for f in os.listdir(circuits_dir) if f.startswith("circuits_round_")]
    nums  = [int(f.replace("circuits_round_", "").replace(".json", "")) for f in files]
    return max(nums) if nums else None


# ── Per-round IoU computations ────────────────────────────────────────────────

def inter_client_iou_all_classes(round_data, model_key="clients_local_model"):
    clients_data = round_data[model_key]
    client_keys  = sorted(clients_data.keys())
    layers       = layer_names_from_round(clients_data)
    all_classes  = sorted(clients_data[client_keys[0]].keys(), key=lambda x: int(x))
    result = {}
    for cls in all_classes:
        pair_ious = [
            avg_layer_iou(clients_data[c1][cls], clients_data[c2][cls], layers)
            for c1, c2 in itertools.combinations(client_keys, 2)
        ]
        result[cls] = float(np.nanmean(pair_ious)) if pair_ious else np.nan
    return result


def mean_iou_over_rounds(exp_dir, model_key="clients_local_model"):
    data, round_keys = load_all_circuits(exp_dir)
    round_nums, per_class, mean_series = [], None, []
    for rk in round_keys:
        cls_iou = inter_client_iou_all_classes(data[rk], model_key)
        if per_class is None:
            per_class = {cls: [] for cls in cls_iou}
        for cls, v in cls_iou.items():
            per_class[cls].append(v)
        mean_series.append(float(np.nanmean(list(cls_iou.values()))))
        round_nums.append(int(rk.split("_")[1]))
    return round_nums, per_class, mean_series


def local_vs_global_iou_all_classes(round_data):
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
    data, round_keys = load_all_circuits(exp_dir)
    round_nums, per_class, mean_series = [], None, []
    for rk in round_keys:
        cls_iou = local_vs_global_iou_all_classes(data[rk])
        if per_class is None:
            per_class = {cls: [] for cls in cls_iou}
        for cls, v in cls_iou.items():
            per_class[cls].append(v)
        mean_series.append(float(np.nanmean(list(cls_iou.values()))))
        round_nums.append(int(rk.split("_")[1]))
    return round_nums, per_class, mean_series


def intra_client_iou_over_rounds(exp_dir):
    data, round_keys = load_all_circuits(exp_dir)
    first_local = data[round_keys[0]]["clients_local_model"]
    client_keys = sorted(first_local.keys())
    layers      = layer_names_from_round(first_local)
    all_classes = sorted(first_local[client_keys[0]].keys(), key=lambda x: int(x))
    round_nums, per_class, mean_series = [], {cls: [] for cls in all_classes}, []
    for i in range(len(round_keys) - 1):
        rk_a, rk_b = round_keys[i], round_keys[i + 1]
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


# ── Axes styling ──────────────────────────────────────────────────────────────

def style_ax(ax, max_round=100):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    # uniform x-ticks every 10 rounds
    step = 10 if max_round >= 50 else 5
    ax.set_xticks(range(0, max_round + 1, step))
    ax.set_xlim(0, max_round + 1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


# ── Smoothing ─────────────────────────────────────────────────────────────────

def _smooth(values, window=7):
    """Simple symmetric moving average; edge values padded with edge values."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


# ── Core line plot ────────────────────────────────────────────────────────────

def _plot_line(all_rounds_data, configs, figures_dir, fname,
               colors=None, legend_title="Data distribution",
               marker_every=20, smooth_window=7):
    if colors is None:
        colors = COLORS

    fig, ax = plt.subplots(figsize=(10, 5))
    max_round = 0

    for i, (dirname, _) in enumerate(configs):
        if dirname not in all_rounds_data:
            continue
        rnums, _, mean_series, lbl = all_rounds_data[dirname]
        r = np.array(rnums)
        m = _smooth(mean_series, smooth_window)
        max_round = max(max_round, int(r.max()) if len(r) else max_round)

        c = colors[i % len(colors)]
        ax.plot(r, m, color=c, linewidth=3.0, label=lbl)

        # markers spaced out
        mask = (r % marker_every == 0)
        if mask.any():
            ax.plot(r[mask], m[mask], color=c, linestyle="none",
                    marker="o", markersize=9,
                    markerfacecolor="white", markeredgewidth=2.5)

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.05)
    style_ax(ax, max_round or 100)
    ax.legend(title=legend_title, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Model families ────────────────────────────────────────────────────────────

MODEL_FAMILIES = [
    (
        "cifar_cnn",
        [
            ("CIFAR_CNN",     "IID"),
            ("CIFAR_CNN_05",  "α=0.5"),
            ("CIFAR_CNN_02",  "α=0.2"),
            ("CIFAR_CNN_005", "α=0.05"),
        ],
    ),
    (
        "cifar_resnet",
        [
            ("CIFAR_ResNet",     "IID"),
            ("CIFAR_ResNet_05",  "α=0.5"),
            ("CIFAR_ResNet_02",  "α=0.2"),
            ("CIFAR_ResNet_005", "α=0.05"),
        ],
    ),
    (
        "fmnist_cnn",
        [
            ("FMNIST_CNN",     "IID"),
            ("FMNIST_CNN_05",  "α=0.5"),
            ("FMNIST_CNN_02",  "α=0.2"),
            ("FMNIST_CNN_005", "α=0.05"),
        ],
    ),
    (
        "fmnist_resnet",
        [
            ("fmnist_resnet",     "IID"),
            ("fmnist_resnet_05",  "α=0.5"),
            ("fmnist_resnet_02",  "α=0.2"),
            ("fmnist_resnet_005", "α=0.05"),
        ],
    ),
    (
        "cifar_cnn_dense",
        [
            ("CIFAR_CNN_dense",     "IID"),
            ("CIFAR_CNN_005_dense", "α=0.05"),
        ],
    ),
]


# ── Data collectors ───────────────────────────────────────────────────────────

def _collect(results_dir, configs, fn):
    out = {}
    for dirname, label in configs:
        exp_dir = os.path.join(results_dir, dirname)
        if not os.path.isdir(exp_dir):
            print(f"  WARNING: {dirname} not found, skipping.")
            continue
        print(f"  {dirname} ...")
        try:
            rnums, per_class, mean_series = fn(exp_dir)
            out[dirname] = (rnums, per_class, mean_series, label)
        except Exception as e:
            print(f"  WARNING: {dirname} failed: {e}")
    return out


# ── Mode runners ──────────────────────────────────────────────────────────────

def run_comparison(results_dir):
    for slug, configs in MODEL_FAMILIES:
        print(f"\n[inter-client] {slug}")
        figures_dir = os.path.join(results_dir, f"figures_{slug}")
        os.makedirs(figures_dir, exist_ok=True)
        data = _collect(results_dir, configs, mean_iou_over_rounds)
        if not data:
            continue
        _plot_line(data, configs, figures_dir,
                   fname=f"inter_{slug}.png")


def run_local_global_comparison(results_dir):
    for slug, configs in MODEL_FAMILIES:
        print(f"\n[local-vs-global] {slug}")
        figures_dir = os.path.join(results_dir, f"figures_{slug}")
        os.makedirs(figures_dir, exist_ok=True)
        data = _collect(results_dir, configs, local_vs_global_over_rounds)
        if not data:
            continue
        _plot_line(data, configs, figures_dir,
                   fname=f"local_global_{slug}.png")


def run_intra_client_comparison(results_dir):
    for slug, configs in MODEL_FAMILIES:
        print(f"\n[intra-client] {slug}")
        figures_dir = os.path.join(results_dir, f"figures_{slug}")
        os.makedirs(figures_dir, exist_ok=True)
        data = _collect(results_dir, configs, intra_client_iou_over_rounds)
        if not data:
            continue
        _plot_line(data, configs, figures_dir,
                   fname=f"intra_{slug}.png")


# ── Sparse vs Dense (appendix) ────────────────────────────────────────────────

def run_sparse_vs_dense(results_dir):
    print("\n[sparse-vs-dense] CIFAR-10 CNN")
    figures_dir = os.path.join(results_dir, "figures_cifar_cnn_sparse_vs_dense")
    os.makedirs(figures_dir, exist_ok=True)

    configs = [
        ("CIFAR_CNN",           "Sparse IID"),
        ("CIFAR_CNN_dense",     "Dense IID"),
        ("CIFAR_CNN_005",       "Sparse α=0.05"),
        ("CIFAR_CNN_005_dense", "Dense α=0.05"),
    ]
    colors = ["#1565C0", "#90CAF9", "#B71C1C", "#EF9A9A"]

    for fn, fname in [
        (mean_iou_over_rounds,         "inter_cifar_cnn_sparse_vs_dense.png"),
        (local_vs_global_over_rounds,  "local_global_cifar_cnn_sparse_vs_dense.png"),
        (intra_client_iou_over_rounds, "intra_cifar_cnn_sparse_vs_dense.png"),
    ]:
        data = _collect(results_dir, configs, fn)
        if data:
            _plot_line(data, configs, figures_dir, fname=fname,
                       colors=colors, legend_title="Model / Setting")


# ── Single-experiment (called from main.py) ───────────────────────────────────

def _plot_single_line(round_nums, mean_series, figures_dir, fname):
    r = np.array(round_nums)
    m = _smooth(mean_series, window=7)
    max_round = int(r.max()) if len(r) else 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, m, color=COLORS[0], linewidth=3.0)
    mask = (r % 20 == 0)
    if mask.any():
        ax.plot(r[mask], m[mask], color=COLORS[0], linestyle="none",
                marker="o", markersize=9,
                markerfacecolor="white", markeredgewidth=2.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.05)
    style_ax(ax, max_round)
    fig.tight_layout()
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def run_single_experiment_analysis(exp_dir, figures_dir=None):
    if not os.path.exists(os.path.join(exp_dir, "circuits", "all_circuits.json")):
        return
    if figures_dir is None:
        figures_dir = os.path.join(exp_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    for fn, fname in [
        (mean_iou_over_rounds,         "inter_client.png"),
        (local_vs_global_over_rounds,  "local_global.png"),
        (intra_client_iou_over_rounds, "intra_client.png"),
    ]:
        try:
            rnums, _, mean_series = fn(exp_dir)
            _plot_single_line(rnums, mean_series, figures_dir, fname)
        except Exception as e:
            print(f"  [analysis] {fname} failed: {e}")


# ── Mode 1: single checkpoint ─────────────────────────────────────────────────

def run_single(exp_dir, round_num=None):
    if round_num is None:
        round_num = last_round_number(exp_dir)
        if round_num is None:
            sys.exit("No circuit files found.")
    print(f"Loading round {round_num} from {exp_dir}")
    rd       = load_single_round(exp_dir, round_num)
    cls_iou  = inter_client_iou_all_classes(rd)
    classes  = sorted(cls_iou.keys(), key=lambda x: int(x))
    values   = [cls_iou[c] for c in classes]
    mean_val = float(np.nanmean(values))
    figures_dir = os.path.join(exp_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(classes)), values,
           color=plt.cm.tab10(np.linspace(0, 1, len(classes))))
    ax.axhline(mean_val, linestyle="--", color="black", linewidth=2,
               label=f"Mean = {mean_val:.3f}")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"class {c}" for c in classes], rotation=45, ha="right")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.05)
    style_ax(ax, len(classes))
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figures_dir, f"inter_round{round_num}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

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
    elif args and args[0] == "--sparse-vs-dense":
        results_dir = args[1] if len(args) > 1 else "results"
        if not os.path.isdir(results_dir):
            sys.exit(f"Results dir not found: {results_dir}")
        run_sparse_vs_dense(results_dir)
    else:
        if not args:
            exp_dir   = input("Experiment path: ").strip()
            round_num = None
        else:
            exp_dir   = args[0]
            round_num = int(args[1]) if len(args) > 1 else None
        if not os.path.isdir(exp_dir):
            sys.exit(f"Not a directory: {exp_dir}")
        run_single(exp_dir, round_num)
