import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader
from core.models import get_model

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=5):
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
    all_features = []
    all_labels = []
    
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
            print(f"Epoch {epoch+1:02d}: Loss = {total_loss/len(feature_loader):.4f} | Acc = {correct/total:.4f}")
            
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

def main():
    parser = argparse.ArgumentParser(description="Train a Linear Probe")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config.json")
    parser.add_argument("--block", type=int, choices=[1, 2, 3, 4], default=4, help="Block/Layer index to probe")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    config = ExperimentConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_model(config)
    sd = torch.load(args.model, map_location=device)
    if 'model_state_dict' in sd: 
        model.load_state_dict(sd['model_state_dict'])
    else: 
        model.load_state_dict(sd)
    model.eval()
    
    print("Loading data...")
    train_dataset, test_dataset = get_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Extracting features from Block {args.block}...")
    feat_train = extract_features(model, train_loader, device, args.block)
    feat_test = extract_features(model, test_loader, device, args.block)
    
    input_dim = feat_train.tensors[0].shape[1]
    
    print(f"Training probe...")
    train_feat_loader = DataLoader(feat_train, batch_size=args.batch_size, shuffle=True)
    probe = train_probe(train_feat_loader, input_dim, config.num_classes, device, epochs=args.epochs, lr=args.lr)
    
    print(f"Evaluating probe...")
    test_feat_loader = DataLoader(feat_test, batch_size=args.batch_size, shuffle=False)
    accuracy = evaluate_probe(probe, test_feat_loader, device)
    print(f"\nFinal Test Accuracy (Block {args.block}): {accuracy:.4f}")
    
    output_path = os.path.join(os.path.dirname(args.model), f"probe_block{args.block}.pt")
    torch.save({
        'probe_state_dict': probe.state_dict(), 
        'input_dim': input_dim, 
        'num_classes': config.num_classes, 
        'block': args.block, 
        'accuracy': accuracy
    }, output_path)
    print(f"Probe saved to: {output_path}")

if __name__ == "__main__":
    main()