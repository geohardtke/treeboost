"""
TreeBoost - High-performance Gradient Boosted Decision Tree engine

A pure Rust GBDT implementation designed for large-scale tabular data
with robust handling of dirty/noisy data.

Key Features:
- Histogram-based training: u8 bins for memory efficiency
- Shannon Entropy regularized splits: Drift-resilient objective
- Pseudo-Huber loss: Robust to outliers
- Split Conformal Prediction: Distribution-free prediction intervals
- Zero-copy serialization: Fast model loading via rkyv

Usage:
    import numpy as np
    from treeboost import GBDTConfig, GBDTModel

    # Prepare data
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(1000) * 0.1).astype(np.float32)

    # Configure model
    config = GBDTConfig()
    config.num_rounds = 100
    config.max_depth = 6
    config.learning_rate = 0.1

    # Train
    model = GBDTModel.train(X, y, config)

    # Predict
    predictions = model.predict(X)

    # Feature importances
    importances = model.feature_importances()

    # Save/load
    model.save("model.rkyv")
    loaded = GBDTModel.load("model.rkyv")

Advanced Features:
    # Pseudo-Huber loss (robust to outliers)
    config.use_pseudo_huber_loss(delta=1.0)

    # Conformal prediction intervals
    config.calibration_ratio = 0.2
    config.conformal_quantile = 0.9
    preds, lower, upper = model.predict_with_intervals(X)

    # Early stopping
    config.early_stopping_rounds = 10
    config.validation_ratio = 0.2

    # Subsampling (stochastic gradient boosting)
    config.subsample = 0.8   # Row subsampling
    config.colsample = 0.8   # Column subsampling

    # Monotonic constraints
    # 1 = increasing, -1 = decreasing, 0 = none
    config.set_monotonic_constraints([1, -1, 0, 0, 0])

    # Feature interaction constraints
    # Features in same group can interact together
    config.set_interaction_groups([[0, 1, 2], [3, 4]])
"""

from ._core import GBDTConfig, GBDTModel

__all__ = ["GBDTConfig", "GBDTModel"]
__version__ = "0.1.0"
