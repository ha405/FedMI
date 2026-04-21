import torch
import unittest
from circuits.cka import linear_cka

class TestCKA(unittest.TestCase):
    def test_linear_cka_identical(self):
        # CKA of a matrix with itself should be close to 1.0
        X = torch.randn(100, 50)
        score = linear_cka(X, X)
        self.assertAlmostEqual(score, 1.0, places=5)
        
    def test_linear_cka_orthogonal(self):
        # Two orthogonal matrices (or completely random/uncorrelated) should have low CKA
        X = torch.randn(1000, 50)
        Y = torch.randn(1000, 50)
        score = linear_cka(X, Y)
        # Expected to be small but maybe not exactly 0 due to random chance, usually < 0.1
        self.assertTrue(score < 0.1)

    def test_linear_cka_different_dims(self):
        # CKA works with different feature dimensions
        X = torch.randn(100, 50)
        # Create Y that is derived from X linearly, but embedded in higher dim
        proj = torch.randn(50, 200)
        Y = X @ proj
        
        score = linear_cka(X, Y)
        # Should be relatively high since one is a random linear projection of the other,
        # but projection into higher dim with random matrix preserves structure well.
        # Definitely > 0.5 usually > 0.8
        self.assertTrue(score > 0.5)
        
    def test_linear_cka_mismatched_samples(self):
        X = torch.randn(100, 50)
        Y = torch.randn(50, 50)
        with self.assertRaises(ValueError):
            linear_cka(X, Y)

if __name__ == "__main__":
    unittest.main()
