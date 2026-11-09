"""
Unit tests for Comparative Evaluation Suite & Probing Metrics.

Enforces:
- Section 6.3 of agents.md: Test output isolation in tests/outputs/
"""

import torch

from wm_control.metrics import RidgeRegressionProbe


def test_ridge_regression_probe_synthetic():
    """Verify RidgeRegressionProbe recovers ground truth linear relationships."""
    torch.manual_seed(42)
    n, d, targets = 200, 5, 2
    X = torch.randn(n, d)
    true_w = torch.randn(d, targets)
    true_b = torch.tensor([1.5, -2.0])

    Y = X @ true_w + true_b + torch.randn(n, targets) * 0.05

    # Train / test split
    X_train, X_test = X[:150], X[150:]
    Y_train, Y_test = Y[:150], Y[150:]

    probe = RidgeRegressionProbe(alpha=0.1)
    probe.fit(X_train, Y_train)
    scores = probe.score_r2(X_test, Y_test)

    assert "mean_r2" in scores
    # With low noise, R^2 should easily exceed 0.90
    assert scores["mean_r2"] > 0.90
