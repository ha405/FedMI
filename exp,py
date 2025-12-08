import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Model Architecture (Global Scope) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, 10)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x

# --- 2. Helper Functions (Global Scope) ---
def binary_gate(x):
    return (x > 0).float() - torch.sigmoid(x).detach() + torch.sigmoid(x)

def get_gate_hook(gate_param):
    def hook(module, input, output):
        return output * binary_gate(gate_param)
    return hook

def get_hard_mask_hook(indices, device):
    def hook(module, input, output):
        mask = torch.zeros(1, output.shape[1], 1, 1).to(device)
        mask[:, indices, :, :] = 1.0
        return output * mask
    return hook

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    class_names = trainset.classes

    # Initialize and Train Dense Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Dense Model (5 Epochs)...")
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(trainloader):.3f}")

    print("Dense Training Complete.\n")

    # Freeze Weights
    for param in model.parameters():
        param.requires_grad = False

    # Classes to analyze (e.g., Plane, Car, Bird)
    classes_to_analyze = [0, 1, 2] 

    for target_class_idx in classes_to_analyze:
        target_name = class_names[target_class_idx]
        print(f"========================================")
        print(f"DISCOVERING CIRCUIT FOR CLASS: {target_name.upper()}")
        print(f"========================================")

        # Initialize Gates
        gate_params = {}
        hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Initialize gates to be open (value > 0)
                gate_params[name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
                hooks.append(module.register_forward_hook(get_gate_hook(gate_params[name])))

        # Optimizer for gates only
        gate_optimizer = optim.Adam(gate_params.values(), lr=0.1)
        
        data_iter = iter(trainloader)
        
        # Optimization Loop
        for step in range(150):
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                inputs, labels = next(data_iter)
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Filter batch for target class
            target_mask = (labels == target_class_idx)
            if target_mask.sum() == 0:
                continue
                
            inputs = inputs[target_mask]
            labels = labels[target_mask]
            
            gate_optimizer.zero_grad()
            outputs = model(inputs)
            
            # Task Loss (Classification)
            task_loss = criterion(outputs, labels)
            
            # L0 Regularization (Sparsity)
            l0_loss = 0
            for name in gate_params:
                prob = torch.sigmoid(gate_params[name])
                l0_loss += prob.sum()
                
            # Lambda determines how aggressively we prune. 
            # 0.05 is usually a good starting point for L0 surrogate.
            total_loss = task_loss + (0.05 * l0_loss)
            
            total_loss.backward()
            gate_optimizer.step()

        # Remove Training Hooks
        for h in hooks:
            h.remove()

        print(f"Circuit Definition for {target_name}:")
        final_active_indices = {}
        
        # Identify Active Channels
        for name in gate_params:
            mask = (gate_params[name] > 0).float().detach().cpu().numpy().flatten()
            indices = np.where(mask == 1)[0]
            final_active_indices[name] = indices
            print(f"  {name}: Kept {len(indices)}/{len(mask)} channels")
            print(f"    Indices: {indices}")

        # Attach Hard Mask Hooks for Inference
        inference_hooks = []
        for name, module in model.named_modules():
            if name in final_active_indices:
                inference_hooks.append(module.register_forward_hook(
                    get_hard_mask_hook(final_active_indices[name], device)
                ))

        # Evaluate Accuracy using ONLY the circuit
        correct_target = 0
        total_target = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                target_mask = (labels == target_class_idx)
                if target_mask.sum() == 0:
                    continue
                    
                outputs = model(images[target_mask])
                _, predicted = torch.max(outputs.data, 1)
                
                total_target += target_mask.sum().item()
                correct_target += (predicted == labels[target_mask]).sum().item()

        print(f"Accuracy on {target_name} using ONLY the circuit: {100 * correct_target / total_target:.2f}%")

        # Visualize one example
        img_idx = -1
        for i in range(len(testset)):
            if testset[i][1] == target_class_idx:
                img_idx = i
                break
                
        img, label = testset[img_idx]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
            prob = F.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)

        # Clean up hooks for next class
        for h in inference_hooks:
            h.remove()

        plt.figure(figsize=(2, 2))
        plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
        plt.title(f"{target_name}\nPred: {class_names[pred.item()]}\nConf: {conf.item():.2f}")
        plt.axis('off')
        plt.show()
        print("\n")