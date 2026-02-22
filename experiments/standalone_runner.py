import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import shutil
from tqdm import tqdm

from core.dataset import get_dataset, get_test_dataloader, get_dataloader
from core.models import get_model
from circuits.pruning import get_current_sparsity, apply_weight_sparsity

class StandaloneRunner:
    """
    Runner for single-client/centralized experiments (No Federation).
    """
    def __init__(self, config):
        self.config = config
        
    def set_seed(self):
        seed = self.config.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def setup(self):
        self.set_seed()
        
        # 1. Output Dir
        if not self.config.resume:
            if os.path.exists(self.config.output_dir):
                print(f"Cleaning existing directory: {self.config.output_dir}")
                shutil.rmtree(self.config.output_dir, ignore_errors=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.dirs = {
            "checkpoints": os.path.join(self.config.output_dir, "checkpoints"),
            "logs": os.path.join(self.config.output_dir, "logs"),
            "figures": os.path.join(self.config.output_dir, "figures")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        with open(os.path.join(self.config.output_dir, "config.json"), 'w') as f:
             json.dump(self.config.__dict__, f, indent=4, default=str)
        
        # 2. Data
        print(f"Loading dataset: {self.config.dataset_name}")
        trainset, testset = get_dataset(self.config)
        self.testloader = get_test_dataloader(testset, self.config)
        
        # Standalone: Use full trainset
        # (Could add logic here to use subset if desired, but default is full)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers
        )
        self.class_names = trainset.classes
            
        # 3. Model
        self.model = get_model(self.config)
        torch.save(self.model.state_dict(), os.path.join(self.dirs["checkpoints"], "initialization.pt"))
        
    def run(self):
        print(f"\n==================================================")
        print(f"STARTING STANDALONE EXPERIMENT: {self.config.output_dir}")
        print(f"Mode: {self.config.experiment_mode}")
        print(f"==================================================")
        
        self.model.to(self.config.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        log_path = os.path.join(self.dirs["logs"], "training_log.txt")
        mode = "a" if self.config.resume else "w"
        
        # Total epochs = num_rounds * local_epochs (interpret rounds as super-epochs or just use 1 round)
        # Simplified: Use num_rounds as total epochs for standalone
        total_epochs = self.config.num_rounds
        
        with open(log_path, mode) as log_f:
            if not self.config.resume:
                log_f.write("=== Standalone Training Log ===\n")
            
            total_steps = len(self.trainloader) * total_epochs
            current_step = 0
            
            for epoch in range(total_epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Training Loop
                pbar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Sparsity
                    if self.config.train_mode == 'sparse':
                        sparsity = get_current_sparsity(current_step, total_steps, self.config.target_sparsity)
                        apply_weight_sparsity(self.model, sparsity)
                    
                    current_step += 1
                
                # Final Sparsity Fix
                if self.config.train_mode == 'sparse':
                    apply_weight_sparsity(self.model, self.config.target_sparsity)
                
                train_acc = 100. * correct / total
                train_loss = running_loss / len(self.trainloader)
                
                # Evaluation
                test_acc = self.evaluate()
                
                log_msg = f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%"
                print(log_msg)
                log_f.write(log_msg + "\n")
                
                # Periodic Save
                if (epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs:
                    save_path = os.path.join(self.dirs["checkpoints"], f"checkpoint_epoch_{epoch+1}.pt")
                    torch.save(self.model.state_dict(), save_path)
                    
        print("\n--- Standalone Experiment Complete ---")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total
