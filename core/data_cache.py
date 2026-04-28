import torch
from torch.utils.data import DataLoader


class EvaluationCache:
    """
    Caches the evaluation dataset in GPU/CPU memory and pre-splits by class.
    """

    def __init__(self, testloader: DataLoader, device: str, num_classes: int):
        all_inputs, all_labels = [], []
        for inputs, labels in testloader:
            all_inputs.append(inputs)
            all_labels.append(labels)

        self.inputs = torch.cat(all_inputs).to(device)
        self.labels = torch.cat(all_labels).to(device)
        self.device = device
        self.num_classes = num_classes

        # Pre-build per-class index tensors for fast filtering
        self.class_indices = {}
        for c in range(num_classes):
            self.class_indices[c] = (self.labels == c).nonzero(as_tuple=True)[0]

        # Pre-build per-class data tensors
        self._class_inputs = {}
        self._class_labels = {}
        for c in range(num_classes):
            idx = self.class_indices[c]
            if len(idx) > 0:
                self._class_inputs[c] = self.inputs[idx]
                self._class_labels[c] = self.labels[idx]
            else:
                self._class_inputs[c] = torch.empty(0, *self.inputs.shape[1:], device=device)
                self._class_labels[c] = torch.empty(0, dtype=torch.long, device=device)

    def get_class_data(self, target_class: int):
        """Return (inputs, labels) for a single class — already on device."""
        return self._class_inputs[target_class], self._class_labels[target_class]

    def iterate_batches(self, batch_size: int = 256):
        """Yield (inputs, labels) batches over the full test set. Replaces DataLoader iteration."""
        n = self.inputs.shape[0]
        for i in range(0, n, batch_size):
            yield self.inputs[i:i + batch_size], self.labels[i:i + batch_size]

    def iterate_class_batches(self, target_class: int, batch_size: int = 256):
        """Yield batches for a single class."""
        inp, lab = self._class_inputs[target_class], self._class_labels[target_class]
        n = inp.shape[0]
        for i in range(0, n, batch_size):
            yield inp[i:i + batch_size], lab[i:i + batch_size]

    def __len__(self):
        return self.inputs.shape[0]

    def class_count(self, target_class: int) -> int:
        return len(self.class_indices[target_class])
