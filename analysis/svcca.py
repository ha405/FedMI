import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.resnet import ResNet
from core.models import SimpleCNN, get_model
from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader

def load_backbone(ckpt_path, num_classes, device, arch):
    config_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "config.json")
    
    if os.path.exists(config_path):
        config = ExperimentConfig.load(config_path)
        config.device = device
        model = get_model(config)
    else:
        if arch == "resnet":
            model = ResNet(num_classes=num_classes)
        else:
            model = SimpleCNN(num_classes=num_classes, input_channels=3, conv_channels=[64, 128, 256])
    
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

@torch.no_grad()
def _forward_to_block(model, x, arch):
    if arch == "resnet":
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.block1(x)
        x = model.block2(x)
        x = model.block3(x)
        x = model.block4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)
    else:
        x = torch.relu(model.conv1(x))
        x = model.pool(x)
        x = torch.relu(model.conv2(x))
        x = model.pool(x)
        x = torch.relu(model.conv3(x))
        x = model.pool(x)
        return torch.flatten(x, 1)

def extract_features(model, loader, device, arch):
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, ys in loader:
            images = images.to(device)
            feats.append(_forward_to_block(model, images, arch).cpu())
            labels.append(ys)
    return TensorDataset(torch.cat(feats), torch.cat(labels))

def save_features(ds, path):
    torch.save(ds, path)

def load_features(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def cca_intervention_original_head(iid_train_ds, niid_train_ds, niid_test_ds, original_model, k_keep_list, device):
    X_train = iid_train_ds.tensors[0].to(device).float()
    Y_train = niid_train_ds.tensors[0].to(device).float()
    Y_test = niid_test_ds.tensors[0].to(device).float()
    Y_test_labels = niid_test_ds.tensors[1].to(device)
    mu_x = X_train.mean(dim=0)
    mu_y = Y_train.mean(dim=0)
    X_c = X_train - mu_x
    Y_c = Y_train - mu_y
    U_x, S_x, Vh_x = torch.linalg.svd(X_c, full_matrices=False)
    U_y, S_y, Vh_y = torch.linalg.svd(Y_c, full_matrices=False)
    M = U_x.T @ U_y
    A, Sigma, Bh = torch.linalg.svd(M)
    B = Bh.T
    S_y_inv = torch.diag(1.0 / torch.clamp(S_y, min=1e-5))
    S_x_mat = torch.diag(S_x)
    print(f"{'k':<10} | {'Accuracy':<15}")
    print("-" * 30)
    original_model.eval()
    results = {}
    for k in k_keep_list:
        Y_test_c = Y_test - mu_y
        U_y_test = Y_test_c @ Vh_y.T @ S_y_inv
        C_y_test = U_y_test @ B
        C_y_intervened = C_y_test.clone()
        if k < C_y_intervened.shape[1]:
            C_y_intervened[:, k:] = 0.0
        X_test_reconstructed_c = C_y_intervened @ A.T @ S_x_mat @ Vh_x
        X_test_reconstructed = X_test_reconstructed_c + mu_x
        with torch.no_grad():
            logits = original_model.fc(X_test_reconstructed)
            _, preds = torch.max(logits, 1)
            acc = (preds == Y_test_labels).sum().item() / len(Y_test_labels)
        results[int(k)] = float(acc)
        print(f"{k:<10} | {acc:.4f}")
    return results

def _build_dataloaders(config):
    _, testset = get_dataset(config)
    test_loader = get_test_dataloader(testset, config)
    train_loader = get_test_dataloader(testset, config)
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, choices=["cnn", "resnet"])
    parser.add_argument("--iid-ckpt", type=str, required=True)
    parser.add_argument("--niid-ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Path to config.json (auto-detected from checkpoint if omitted)")
    parser.add_argument("--save-features", type=str, default=None)
    parser.add_argument("--features-dir", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--out", type=str, default="cca_results.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config_path = args.config or os.path.join(os.path.dirname(os.path.dirname(args.iid_ckpt)), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    config = ExperimentConfig.load(config_path)
    config.device = device
    config.num_classes = args.num_classes
    
    layer = "final"
    iid_train, niid_train, niid_test = {}, {}, {}
    model_iid = load_backbone(args.iid_ckpt, args.num_classes, device, args.arch)
    if args.features_dir:
        iid_train[layer] = load_features(os.path.join(args.features_dir, f"iid_train_{layer}.pt"))
        niid_train[layer] = load_features(os.path.join(args.features_dir, f"niid_train_{layer}.pt"))
        niid_test[layer] = load_features(os.path.join(args.features_dir, f"niid_test_{layer}.pt"))
    else:
        model_niid = load_backbone(args.niid_ckpt, args.num_classes, device, args.arch)
        train_loader, test_loader = _build_dataloaders(config)
        iid_train[layer] = extract_features(model_iid, train_loader, device, args.arch)
        niid_train[layer] = extract_features(model_niid, train_loader, device, args.arch)
        niid_test[layer] = extract_features(model_niid, test_loader, device, args.arch)
        if args.save_features:
            os.makedirs(args.save_features, exist_ok=True)
            for t, d in [("iid_train", iid_train[layer]), ("niid_train", niid_train[layer]), ("niid_test", niid_test[layer])]:
                save_features(d, os.path.join(args.save_features, f"{t}_{layer}.pt"))
    niid_test_loader = DataLoader(niid_test[layer], batch_size=256)
    with torch.no_grad():
        c = t = 0
        for X_b, y_b in niid_test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            p = model_iid.fc(X_b).argmax(dim=1)
            c += (p == y_b).sum().item()
            t += y_b.size(0)
        baseline = c / t
    print(f"Baseline (NIID raw on IID head): {baseline:.4f}")
    feat_dim = iid_train[layer].tensors[0].shape[1]
    k_list = [1, 5, 10, 20, 40, 64, 128, 256]
    if args.arch == "cnn":
        k_list += [512, 1024, 2048, 4096]
    k_list = [k for k in k_list if k <= feat_dim]
    if feat_dim not in k_list:
        k_list.append(feat_dim)
    res = cca_intervention_original_head(iid_train[layer], niid_train[layer], niid_test[layer], model_iid, k_list, device)
    output = {"layer": layer, "arch": args.arch, "baseline": baseline, "results": res}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.out}")

if __name__ == "__main__":
    main()