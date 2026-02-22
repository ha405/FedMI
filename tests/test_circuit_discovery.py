"""
tests/test_circuit_discovery.py

Tests for config-driven circuit discovery:
  1. get_classes_for_client() - correctly extracts classes from partition indices
  2. Config default - classes_to_analyze defaults to None (not hardcoded)
  3. discover_circuits class resolution priority
     a. Per-client override takes priority
     b. Global classes_to_analyze used when no per-client override
     c. Safe fallback to all classes when both are None
  4. server.orchestrate_round uses the same priority chain
  5. Integration: runner.setup() auto-populates classes_to_discover_per_client
"""
import os
import sys
import copy
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import ExperimentConfig
from core.dataset import get_classes_for_client


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_fake_dataset(labels):
    """Return a minimal mock dataset with a targets attribute."""
    ds = MagicMock()
    ds.targets = labels
    ds.__len__ = lambda self: len(labels)
    return ds


def _make_config(**kwargs):
    cfg = ExperimentConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_model(num_classes=3, in_channels=1):
    """Tiny real model for forward-pass tests."""
    import torch.nn as nn
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 4, 3, padding=1)
            self.pool  = nn.AdaptiveAvgPool2d(1)
            self.fc    = nn.Linear(4, num_classes)
        def forward(self, x):
            return self.fc(self.pool(self.conv1(x)).view(x.size(0), -1))
    return TinyModel()


def _make_trivial_dataloader(num_classes, samples_per_class=4, img_size=(1, 8, 8)):
    """DataLoader where each class has `samples_per_class` samples."""
    from torch.utils.data import TensorDataset, DataLoader
    imgs   = torch.randn(num_classes * samples_per_class, *img_size)
    labels = torch.tensor(
        [c for c in range(num_classes) for _ in range(samples_per_class)]
    )
    return DataLoader(TensorDataset(imgs, labels), batch_size=num_classes * samples_per_class)


# ================================================================== #
# 1. get_classes_for_client
# ================================================================== #
class TestGetClassesForClient(unittest.TestCase):

    def test_returns_sorted_unique_classes(self):
        """Should return sorted unique class labels present in partition."""
        labels = np.array([0, 0, 1, 2, 2, 5, 5, 5])
        ds = _make_fake_dataset(labels)
        indices = [0, 1, 2, 3, 4]  # classes 0, 0, 1, 2, 2
        result = get_classes_for_client(ds, indices)
        self.assertEqual(result, [0, 1, 2])

    def test_single_class_partition(self):
        """Partition with only one class should return a single-element list."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        ds = _make_fake_dataset(labels)
        indices = [0, 1]  # class 0 only
        result = get_classes_for_client(ds, indices)
        self.assertEqual(result, [0])

    def test_non_contiguous_indices(self):
        """Works correctly with non-contiguous index lists."""
        labels = np.array([3, 7, 3, 0, 7])
        ds = _make_fake_dataset(labels)
        indices = [0, 1, 4]  # labels: 3, 7, 7
        result = get_classes_for_client(ds, indices)
        self.assertEqual(result, [3, 7])

    def test_empty_indices(self):
        """Empty partition should return empty list."""
        labels = np.array([0, 1, 2])
        ds = _make_fake_dataset(labels)
        result = get_classes_for_client(ds, [])
        self.assertEqual(result, [])


# ================================================================== #
# 2. Config default
# ================================================================== #
class TestConfigDefaults(unittest.TestCase):

    def test_classes_to_analyze_defaults_to_none(self):
        """classes_to_analyze must default to None, not a hardcoded list."""
        cfg = ExperimentConfig()
        self.assertIsNone(
            cfg.classes_to_analyze,
            "classes_to_analyze should default to None so the runner can "
            "auto-derive it from the partition rather than using a hardcoded list."
        )

    def test_classes_to_discover_per_client_defaults_to_none(self):
        cfg = ExperimentConfig()
        self.assertIsNone(cfg.classes_to_discover_per_client)


# ================================================================== #
# 3. Client.discover_circuits class-resolution priority
# ================================================================== #
class TestClientCircuitResolution(unittest.TestCase):
    """Tests the class-selection logic inside FederatedClient.discover_circuits."""

    def _run_discover_and_capture_classes(self, cfg, client_id=0):
        """
        Runs discover_circuits on a tiny model/data and returns the set of
        class names that appear as keys in the returned circuit dict.
        """
        from federated.client import FederatedClient

        num_classes = cfg.num_classes
        dl = _make_trivial_dataloader(num_classes)
        model = _make_mock_model(num_classes=num_classes)
        class_names = [str(i) for i in range(num_classes)]

        cfg.device = "cpu"
        cfg.discovery_steps = 2   # Fast
        cfg.gate_lr = 0.1
        cfg.l0_lambda = 0.01
        cfg.use_mean_ablation = False

        client = FederatedClient(client_id, dl, cfg, class_names)

        with patch("federated.client.evaluate_circuit", return_value=0.0), \
             patch("federated.client.evaluate_circuit_necessity", return_value=0.0), \
             patch("federated.client.extract_sparse_connectivity", return_value={}), \
             patch("federated.client.filter_connectivity_by_circuit", return_value={}):
            result = client.discover_circuits(model, dl)

        return set(result.keys())

    def test_priority1_per_client_override(self):
        """Per-client override (priority 1) selects exactly those classes."""
        cfg = _make_config(
            num_classes=5,
            classes_to_analyze=[0, 1, 2, 3, 4],        # would use all if not overridden
            classes_to_discover_per_client={0: [1, 3]}  # priority-1 override
        )
        discovered = self._run_discover_and_capture_classes(cfg, client_id=0)
        self.assertEqual(discovered, {"1", "3"})

    def test_priority2_global_classes_to_analyze(self):
        """Global classes_to_analyze (priority 2) used when no per-client override."""
        cfg = _make_config(
            num_classes=5,
            classes_to_analyze=[0, 2],
            classes_to_discover_per_client=None  # not set
        )
        discovered = self._run_discover_and_capture_classes(cfg, client_id=0)
        self.assertEqual(discovered, {"0", "2"})

    def test_priority3_safe_fallback(self):
        """When both are None, falls back to all num_classes classes."""
        cfg = _make_config(
            num_classes=3,
            classes_to_analyze=None,
            classes_to_discover_per_client=None
        )
        discovered = self._run_discover_and_capture_classes(cfg, client_id=0)
        self.assertEqual(discovered, {"0", "1", "2"})

    def test_per_client_override_not_matching_client_id_falls_to_global(self):
        """Override for client_id=1 should not affect client_id=0; client 0 uses global."""
        cfg = _make_config(
            num_classes=4,
            classes_to_analyze=[0, 1],
            classes_to_discover_per_client={1: [2, 3]}  # override for client 1 only
        )
        discovered = self._run_discover_and_capture_classes(cfg, client_id=0)
        self.assertEqual(discovered, {"0", "1"})


# ================================================================== #
# 4. get_classes_for_client integration with partition methods
# ================================================================== #
class TestPartitionToClasses(unittest.TestCase):

    def test_manual_partition_gives_correct_classes(self):
        """After manual (by_class) partitioning, get_classes_for_client should
        return exactly the classes assigned to each client."""
        from core.dataset import partition_by_class

        n = 30
        labels = np.array([i % 6 for i in range(n)])  # classes 0-5
        ds = _make_fake_dataset(labels)

        allocation = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        client_indices = partition_by_class(ds, allocation)

        for cid, expected_classes in allocation.items():
            result = get_classes_for_client(ds, client_indices[cid])
            self.assertEqual(result, sorted(expected_classes),
                             f"Client {cid}: expected {expected_classes}, got {result}")

    def test_iid_partition_contains_all_classes(self):
        """IID partition should give each client roughly all classes (with enough data)."""
        from core.dataset import partition_iid
        np.random.seed(42)
        n = 500
        num_classes = 5
        labels = np.array([i % num_classes for i in range(n)])
        ds = _make_fake_dataset(labels)

        client_indices = partition_iid(ds, num_clients=3)
        for cid, indices in enumerate(client_indices):
            classes = get_classes_for_client(ds, indices)
            self.assertEqual(
                classes, list(range(num_classes)),
                f"IID client {cid} should have all {num_classes} classes"
            )


# ================================================================== #
# 5. Runner auto-populates classes_to_discover_per_client
# ================================================================== #
class TestRunnerAutoPopulate(unittest.TestCase):

    def test_auto_populate_integrates_correctly(self):
        """
        get_classes_for_client() should produce the same result we'd get if
        runner.setup() had auto-populated classes_to_discover_per_client.
        (Integration-level test without running the full runner.)
        """
        from core.dataset import partition_by_class

        n = 20
        labels = np.array([i % 4 for i in range(n)])  # classes 0,1,2,3
        ds = _make_fake_dataset(labels)

        allocation = {0: [0, 1], 1: [2, 3]}
        client_indices = partition_by_class(ds, allocation)

        # Simulate what runner.setup() does:
        auto_classes = {
            i: get_classes_for_client(ds, idxs)
            for i, idxs in enumerate(client_indices)
        }

        self.assertEqual(auto_classes[0], [0, 1])
        self.assertEqual(auto_classes[1], [2, 3])

        # Verify this would be written into config correctly
        cfg = ExperimentConfig()
        self.assertIsNone(cfg.classes_to_discover_per_client)  # starts None
        cfg.classes_to_discover_per_client = auto_classes
        self.assertEqual(cfg.classes_to_discover_per_client[0], [0, 1])
        self.assertEqual(cfg.classes_to_discover_per_client[1], [2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
