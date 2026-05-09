import os
import sys
import json
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader
from core.models import get_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FAMILIES = [
    ("CIFAR-10 · CNN",         ["CIFAR_CNN",    "CIFAR_CNN_05",    "CIFAR_CNN_02",    "CIFAR_CNN_005"]),
    ("CIFAR-10 · ResNet",      ["CIFAR_ResNet", "CIFAR_ResNet_05", "CIFAR_ResNet_02", "CIFAR_ResNet_005"]),
    ("Fashion-MNIST · CNN",    ["FMNIST_CNN",   "FMNIST_CNN_05",   "FMNIST_CNN_02",   "FMNIST_CNN_005"]),
    ("Fashion-MNIST · ResNet", ["fmnist_resnet","fmnist_resnet_05","fmnist_resnet_02","fmnist_resnet_005"]),
]


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def _get_stages(model):
    if hasattr(model, 'block1'):
        stem = lambda x: model.maxpool(model.relu(model.bn1(model.conv1(x))))
        return [stem, model.block1, model.block2, model.block3, model.block4], model.avgpool
    stem = lambda x: model.pool(torch.relu(model.conv1(x)))
    return [
        stem,
        lambda x: model.pool(torch.relu(model.conv2(x))),
        lambda x: model.pool(torch.relu(model.conv3(x))),
    ], nn.Identity()


def get_model_features(model, x, block_idx):
    stages, pool = _get_stages(model)
    with torch.no_grad():
        for stage in stages[:block_idx]:
            x = stage(x)
        x = torch.flatten(pool(x), 1)
    return x


def extract_features(model, loader, device, block_idx):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = get_model_features(model, images, block_idx)
            all_features.append(features.cpu())
            all_labels.append(labels)
    return TensorDataset(torch.cat(all_features), torch.cat(all_labels))


def train_probe(feature_loader, input_dim, num_classes, device, epochs=50, lr=1e-4):
    probe = LinearProbe(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        probe.train()
        total_loss, correct, total = 0, 0, 0
        for features, labels in feature_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}: Loss={total_loss/len(feature_loader):.4f} Acc={correct/total:.4f}")
    return probe


def evaluate_probe(probe, feature_loader, device):
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in feature_loader:
            features, labels = features.to(device), labels.to(device)
            logits = probe(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def find_latest_checkpoint(checkpoints_dir):
    files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_round_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_round_*.pt files in {checkpoints_dir}")
    return max(files, key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))


def run_one(results_dir, device, epochs=50, batch_size=128, lr=1e-4, final_only=True):
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        print(f"  WARNING: directory not found, skipping: {results_dir}")
        return
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"  WARNING: config.json not found, skipping: {results_dir}")
        return

    config = ExperimentConfig.load(config_path)
    config.device = device
    if not os.path.isabs(config.data_root):
        config.data_root = os.path.join(REPO_ROOT, config.data_root)

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    try:
        ckpt_path = find_latest_checkpoint(ckpt_dir)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        return
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")

    model = get_model(config)
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(sd)
    model.eval().to(device)

    train_dataset, test_dataset = get_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    stages, _ = _get_stages(model)
    num_blocks = len(stages)
    blocks_to_probe = [num_blocks] if final_only else list(range(1, num_blocks + 1))

    out_dir = os.path.join(results_dir, "probes")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for block_idx in blocks_to_probe:
        print(f"\n  [Block {block_idx}/{num_blocks}] Extracting features...")
        feat_train = extract_features(model, train_loader, device, block_idx)
        feat_test = extract_features(model, test_loader, device, block_idx)
        input_dim = feat_train.tensors[0].shape[1]

        train_feat_loader = DataLoader(feat_train, batch_size=batch_size, shuffle=True)
        test_feat_loader = DataLoader(feat_test, batch_size=batch_size, shuffle=False)

        probe = train_probe(train_feat_loader, input_dim, config.num_classes, device, epochs=epochs, lr=lr)
        accuracy = evaluate_probe(probe, test_feat_loader, device)
        print(f"  Block {block_idx} test accuracy: {accuracy:.4f}")
        summary[block_idx] = accuracy

        torch.save({
            'probe_state_dict': probe.state_dict(),
            'input_dim': input_dim,
            'num_classes': config.num_classes,
            'block': block_idx,
            'accuracy': accuracy,
        }, os.path.join(out_dir, f"probe_block{block_idx}.pt"))

        with open(os.path.join(out_dir, f"probe_block{block_idx}_accuracy.json"), "w") as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_checkpoint': ckpt_path,
                'config_file': config_path,
                'block': block_idx,
                'epochs': epochs,
                'learning_rate': lr,
                'batch_size': batch_size,
                'test_accuracy': float(accuracy),
                'accuracy_percentage': f"{accuracy*100:.2f}%",
                'input_dim': input_dim,
                'num_classes': config.num_classes,
            }, f, indent=4)

    with open(os.path.join(out_dir, "probe_summary.json"), "w") as f:
        json.dump({k: float(v) for k, v in summary.items()}, f, indent=4)
    print(f"  Probes saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on federated model checkpoints")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Single results directory (e.g. results/CIFAR_CNN)")
    parser.add_argument("--all", action="store_true",
                        help="Run probes for all model/dataset families")
    parser.add_argument("--results_base", type=str, default="results",
                        help="Base results directory used with --all (default: results)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default=None, help="(legacy) Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="(legacy) Path to config.json")
    parser.add_argument("--block", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--all-blocks", action="store_true",
                        help="Probe all blocks (default: final block only)")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}\n")

    final_only = not args.all_blocks

    if args.all:
        for display, dirnames in MODEL_FAMILIES:
            print(f"\n[{display}]")
            for dirname in dirnames:
                print(f"\n  -- {dirname}")
                run_one(os.path.join(args.results_base, dirname), device,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        final_only=final_only)
        return

    if args.results_dir:
        run_one(args.results_dir, device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                final_only=final_only)
        return

    # Legacy single-checkpoint mode
    if args.model is None or args.config is None:
        parser.error("Provide --results_dir, --all, or both --model and --config")

    checkpoint_dir = os.path.dirname(args.model)
    out_dir = os.path.abspath(os.path.join(checkpoint_dir, "probes"))
    os.makedirs(out_dir, exist_ok=True)

    config = ExperimentConfig.load(args.config)
    config.device = device

    model = get_model(config)
    sd = torch.load(args.model, map_location=device)
    if 'model_state_dict' in sd:
        model.load_state_dict(sd['model_state_dict'])
    else:
        model.load_state_dict(sd)
    model.eval().to(device)

    train_dataset, test_dataset = get_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Extracting features from Block {args.block}...")
    feat_train = extract_features(model, train_loader, device, args.block)
    feat_test = extract_features(model, test_loader, device, args.block)
    input_dim = feat_train.tensors[0].shape[1]

    train_feat_loader = DataLoader(feat_train, batch_size=args.batch_size, shuffle=True)
    probe = train_probe(train_feat_loader, input_dim, config.num_classes, device,
                        epochs=args.epochs, lr=args.lr)

    test_feat_loader = DataLoader(feat_test, batch_size=args.batch_size, shuffle=False)
    accuracy = evaluate_probe(probe, test_feat_loader, device)
    print(f"\nFinal Test Accuracy (Block {args.block}): {accuracy:.4f}")

    torch.save({
        'probe_state_dict': probe.state_dict(),
        'input_dim': input_dim,
        'num_classes': config.num_classes,
        'block': args.block,
        'accuracy': accuracy,
    }, os.path.join(out_dir, f"probe_block{args.block}.pt"))

    with open(os.path.join(out_dir, f"probe_block{args.block}_accuracy.json"), "w") as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': os.path.abspath(args.model),
            'config_file': os.path.abspath(args.config),
            'block': args.block,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'test_accuracy': float(accuracy),
            'accuracy_percentage': f"{accuracy*100:.2f}%",
            'input_dim': input_dim,
            'num_classes': config.num_classes,
        }, f, indent=4)
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
