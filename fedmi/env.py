"""
FedMI Environment Bootstrap
============================
Auto-detects Colab / Kaggle / local, resolves paths,
patches configs, and provides disk-cleanup utilities.

Usage (in a notebook):
    from fedmi.env import setup, patch_config, print_info
    setup()          # clone repo, install deps, configure sys.path
    print_info()     # GPU / env summary
"""

import os
import sys
import glob
import shutil
import subprocess

# ──────────────────────────────────────────────
#  Environment Detection
# ──────────────────────────────────────────────

def detect_env() -> str:
    """Return 'colab', 'kaggle', or 'local'."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


ENV = detect_env()

# ──────────────────────────────────────────────
#  Path Resolution
# ──────────────────────────────────────────────

def _find_project_root() -> str:
    """Walk up from this file to find the repo root (contains main.py)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "main.py")):
            return d
        d = os.path.dirname(d)
    # Fallback: assume cwd is the repo root
    return os.getcwd()


def _default_dirs():
    root = _find_project_root()
    if ENV == "colab":
        data = os.path.join(root, "data")
        ckpt = os.path.join(root, "checkpoints")
    elif ENV == "kaggle":
        data = os.path.join(root, "data")
        ckpt = "/kaggle/working/checkpoints"
    else:
        data = os.path.join(root, "data")
        ckpt = os.path.join(root, "checkpoints")
    return root, data, ckpt


PROJECT_ROOT, DATA_DIR, CHECKPOINT_DIR = _default_dirs()

# ──────────────────────────────────────────────
#  Setup
# ──────────────────────────────────────────────

def setup(repo_url: str | None = None, branch: str = "cvpr", install_deps: bool = True):
    """
    One-call bootstrap for Colab / Kaggle.

    1.  Clone the repo if PROJECT_ROOT doesn't exist yet.
    2.  pip-install core dependencies.
    3.  Add PROJECT_ROOT to sys.path.
    """
    global PROJECT_ROOT, DATA_DIR, CHECKPOINT_DIR

    if ENV in ("colab", "kaggle") and repo_url and not os.path.isfile(os.path.join(PROJECT_ROOT, "main.py")):
        dest = "/content/FedMI" if ENV == "colab" else "/kaggle/working/FedMI"
        if not os.path.exists(dest):
            print(f"[fedmi.env] Cloning {repo_url} → {dest}")
            subprocess.check_call(["git", "clone", "-b", branch, repo_url, dest])
        PROJECT_ROOT, DATA_DIR, CHECKPOINT_DIR = _default_dirs()

    # Ensure project root is on sys.path
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    if install_deps:
        _install_deps()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _install_deps():
    """Install pip packages that are NOT already in Colab/Kaggle.
    torch, torchvision, tqdm, matplotlib, numpy are pre-installed.
    Only install truly missing packages here.
    """
    # Packages that might not be pre-installed on cloud platforms
    optional_deps = ["einops"]  # add any extra deps here as needed
    for pkg in optional_deps:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[fedmi.env] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ──────────────────────────────────────────────
#  Config Patching
# ──────────────────────────────────────────────

def patch_config(cfg, output_subdir: str = "experiment_run"):
    """
    Patch an ExperimentConfig in-place for Colab/Kaggle.
    
    Sets:
      - data_root   → DATA_DIR
      - output_dir  → CHECKPOINT_DIR/<output_subdir>
      - device      → 'cuda' if available, else 'cpu'
      - num_workers → 0 (multi-process dataloading is unreliable in notebooks)
    """
    import torch
    cfg.data_root = DATA_DIR
    cfg.output_dir = os.path.join(CHECKPOINT_DIR, output_subdir)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.num_workers = 0
    return cfg

# ──────────────────────────────────────────────
#  Disk Management
# ──────────────────────────────────────────────

def cleanup_checkpoints(exp_dir: str, keep_latest: int = 2):
    """
    Delete old round checkpoints to free disk space.
    Keeps the *keep_latest* most recent ``checkpoint_round_*.pt`` files.
    """
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"[fedmi.env] No checkpoints dir at {ckpt_dir}")
        return

    files = sorted(
        glob.glob(os.path.join(ckpt_dir, "checkpoint_round_*.pt")),
        key=lambda f: int(os.path.basename(f).split("_")[-1].replace(".pt", "")),
    )
    to_delete = files[:-keep_latest] if len(files) > keep_latest else []
    for f in to_delete:
        os.remove(f)
        print(f"  Deleted: {os.path.basename(f)}")

    # Also clean round subdirectories (client model files)
    round_dirs = sorted(
        glob.glob(os.path.join(ckpt_dir, "round_*")),
        key=lambda d: int(os.path.basename(d).split("_")[-1]),
    )
    to_delete_dirs = round_dirs[:-keep_latest] if len(round_dirs) > keep_latest else []
    for d in to_delete_dirs:
        shutil.rmtree(d, ignore_errors=True)
        print(f"  Deleted dir: {os.path.basename(d)}")

    if to_delete or to_delete_dirs:
        print(f"[fedmi.env] Kept latest {keep_latest} checkpoints in {ckpt_dir}")
    else:
        print(f"[fedmi.env] Nothing to clean (≤{keep_latest} checkpoints)")

# ──────────────────────────────────────────────
#  Info Banner
# ──────────────────────────────────────────────

def print_info():
    """Print a summary banner showing environment, paths, and GPU."""
    import torch
    
    sep = "=" * 55
    print(sep)
    print("  FedMI — Environment Info")
    print(sep)
    print(f"  Runtime      : {ENV}")
    print(f"  Project Root : {PROJECT_ROOT}")
    print(f"  Data Dir     : {DATA_DIR}")
    print(f"  Checkpoint   : {CHECKPOINT_DIR}")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  PyTorch      : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Memory   : {mem:.1f} GB")
    else:
        print("  GPU          : ⚠ None (CPU only)")
    print(sep)

# ──────────────────────────────────────────────
#  Log / File Viewing Utilities
# ──────────────────────────────────────────────

def show_log(log_path: str, tail: int = 50):
    """
    Print the last *tail* lines of a log file.
    Useful for viewing training_log.txt in notebooks.
    """
    if not os.path.exists(log_path):
        print(f"⚠ Log file not found: {log_path}")
        return
    with open(log_path, "r") as f:
        lines = f.readlines()
    total = len(lines)
    start = max(0, total - tail)
    print(f"── {os.path.basename(log_path)} ({total} lines, showing last {min(tail, total)}) ──")
    for line in lines[start:]:
        print(line, end="")
    print(f"── end of log ──")


def show_file(file_path: str):
    """Print the entire contents of a text file."""
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return
    with open(file_path, "r") as f:
        print(f.read())


def list_experiments(base_dir: str = None):
    """List all experiment directories under checkpoints."""
    if base_dir is None:
        base_dir = CHECKPOINT_DIR
    if not os.path.isdir(base_dir):
        print(f"⚠ Directory not found: {base_dir}")
        return []
    experiments = []
    for name in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, name)
        if os.path.isdir(exp_path):
            config_exists = os.path.isfile(os.path.join(exp_path, "config.json"))
            log_exists = os.path.isfile(os.path.join(exp_path, "logs", "training_log.txt"))
            print(f"  📁 {name}  [config: {'✅' if config_exists else '❌'}]  [logs: {'✅' if log_exists else '❌'}]")
            experiments.append(exp_path)
    if not experiments:
        print("  (no experiments found)")
    return experiments


def show_images(exp_dir: str):
    """
    Find and display all images (.png, .jpg, .jpeg, .svg) generated
    by the codebase inside an experiment directory.
    Uses IPython.display so images render inline in Colab/Kaggle.
    """
    from IPython.display import display, Image as IPImage, HTML

    img_extensions = {".png", ".jpg", ".jpeg", ".svg"}
    found = []
    for root, dirs, files in os.walk(exp_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in img_extensions:
                found.append(os.path.join(root, f))

    if not found:
        print(f"⚠ No images found in {exp_dir}")
        return

    print(f"Found {len(found)} image(s) in {exp_dir}:\n")
    for img_path in found:
        rel = os.path.relpath(img_path, exp_dir)
        display(HTML(f"<h4>📊 {rel}</h4>"))
        display(IPImage(filename=img_path))
