import torch
import time
from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader

def test_caching():
    print("=== Testing Dataloader Caching ===")
    config = ExperimentConfig(dataset_name='MNIST', batch_size=256)
    device = config.device
    print(f"Device: {device}")
    
    # 1. Load Dataset
    t0 = time.time()
    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    print(f"Dataset loaded in {time.time() - t0:.2f}s")
    
    # 2. Re-implement logging-heavy caching to see where it gets stuck
    cache = {i: {"inputs": [], "labels": []} for i in range(config.num_classes)}
    
    t1 = time.time()
    batch_count = 0
    total_samples = 0
    for inputs, labels in testloader:
        batch_count += 1
        total_samples += len(labels)
        
        # Original logic inside cache_dataloader_by_class
        for i in range(len(labels)):
            c = labels[i].item()
            if 0 <= c < config.num_classes:
                cache[c]["inputs"].append(inputs[i].unsqueeze(0))
                cache[c]["labels"].append(labels[i].unsqueeze(0))
                
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches ({total_samples} samples)...")
            
    print(f"Batch iteration finished in {time.time() - t1:.2f}s")
    
    # 3. Test Tensor Concatenation (Often the actual bottleneck)
    print("Starting Tensor Concatenation and Device Transfer...")
    t2 = time.time()
    for c in range(config.num_classes):
        if cache[c]["inputs"]:
            # Profiling the cat operation
            ct0 = time.time()
            cache[c]["inputs"] = torch.cat(cache[c]["inputs"]).to(device)
            cache[c]["labels"] = torch.cat(cache[c]["labels"]).to(device)
            print(f"  Class {c} concat and transfer took {time.time() - ct0:.4f}s")
        else:
            cache[c]["inputs"] = torch.empty(0).to(device)
            cache[c]["labels"] = torch.empty(0, dtype=torch.long).to(device)
            
    print(f"Concatenation finished in {time.time() - t2:.2f}s")
    print("=== Test Complete ===")
    
if __name__ == "__main__":
    test_caching()
