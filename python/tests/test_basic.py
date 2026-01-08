"""Basic tests for TreeBoost Python bindings."""

import numpy as np
import pytest


def test_import():
    """Test that the module can be imported."""
    from treeboost import GBDTConfig, GBDTModel
    assert GBDTConfig is not None
    assert GBDTModel is not None


def test_config_defaults():
    """Test default configuration values."""
    from treeboost import GBDTConfig

    config = GBDTConfig()
    assert config.num_rounds == 100
    assert abs(config.learning_rate - 0.1) < 1e-6
    assert config.max_depth == 6
    assert config.max_leaves == 31


def test_config_setters():
    """Test configuration setters."""
    from treeboost import GBDTConfig

    config = GBDTConfig()
    config.num_rounds = 50
    config.learning_rate = 0.05
    config.max_depth = 4

    assert config.num_rounds == 50
    assert abs(config.learning_rate - 0.05) < 1e-6
    assert config.max_depth == 4


def test_train_basic():
    """Test basic training and prediction."""
    from treeboost import GBDTConfig, GBDTModel

    # Generate simple regression data
    np.random.seed(42)
    n_samples = 500
    n_features = 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1).astype(np.float32)

    # Configure and train
    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4
    config.learning_rate = 0.1

    model = GBDTModel.train(X, y, config)

    # Check model properties
    assert model.num_trees == 20
    assert model.num_features == 5

    # Predict
    predictions = model.predict(X)
    assert predictions.shape == (n_samples,)

    # Check predictions are reasonable (R^2 > 0.5)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.5


def test_feature_importance():
    """Test feature importance computation."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 500
    n_features = 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Target only depends on first two features
    y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4

    model = GBDTModel.train(X, y, config)
    importances = model.feature_importance()

    assert importances.shape == (n_features,)
    # First two features should have higher importance
    assert importances[0] > importances[3]
    assert importances[1] > importances[4]
    # Importances should sum to 1
    assert abs(np.sum(importances) - 1.0) < 0.01


def test_pseudo_huber_loss():
    """Test Pseudo-Huber loss for robustness."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    # Add outliers
    y[0:10] = 100.0  # Outliers

    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4
    config.use_pseudo_huber_loss(delta=1.0)

    model = GBDTModel.train(X, y, config)
    predictions = model.predict(X)

    # Model should not be overly influenced by outliers
    # Check predictions on non-outlier data
    non_outlier_preds = predictions[10:]
    non_outlier_y = y[10:]
    mse = np.mean((non_outlier_preds - non_outlier_y) ** 2)
    assert mse < 1.0


def test_conformal_prediction():
    """Test conformal prediction intervals."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 3).astype(np.float32)
    y = (X[:, 0] + np.random.randn(n_samples) * 0.5).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4
    config.calibration_ratio = 0.2
    config.conformal_quantile = 0.9

    model = GBDTModel.train(X, y, config)

    # Should have conformal quantile
    assert model.conformal_quantile is not None

    # Get prediction intervals
    preds, lower, upper = model.predict_with_intervals(X)

    assert preds.shape == (n_samples,)
    assert lower.shape == (n_samples,)
    assert upper.shape == (n_samples,)

    # Lower should be less than upper
    assert np.all(lower < upper)


def test_monotonic_constraints():
    """Test monotonic constraints."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 3).astype(np.float32)
    y = X[:, 0].astype(np.float32)  # Simple linear relationship

    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4
    config.set_monotonic_constraints([1, 0, 0])  # Feature 0: increasing

    model = GBDTModel.train(X, y, config)
    assert model.num_trees > 0


def test_interaction_constraints():
    """Test feature interaction constraints."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4
    config.set_interaction_groups([[0, 1], [2, 3, 4]])

    model = GBDTModel.train(X, y, config)
    assert model.num_trees > 0


def test_early_stopping():
    """Test early stopping configuration."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 100
    config.max_depth = 4
    config.early_stopping_rounds = 5
    config.validation_ratio = 0.2

    model = GBDTModel.train(X, y, config)

    # Model should train successfully (early stopping may or may not trigger
    # depending on data - for simple linear data it often doesn't)
    assert model.num_trees > 0
    assert model.num_trees <= 100


def test_subsampling():
    """Test row and column subsampling."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 20
    config.max_depth = 4
    config.subsample = 0.8
    config.colsample = 0.8

    model = GBDTModel.train(X, y, config)
    predictions = model.predict(X)

    assert predictions.shape == (n_samples,)


def test_save_load(tmp_path):
    """Test model serialization."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 200

    X = np.random.randn(n_samples, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 10
    config.max_depth = 3

    model = GBDTModel.train(X, y, config)
    original_preds = model.predict(X)

    # Save and load
    model_path = str(tmp_path / "model.rkyv")
    model.save(model_path)

    loaded_model = GBDTModel.load(model_path)
    loaded_preds = loaded_model.predict(X)

    # Predictions should match
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)


def test_feature_names():
    """Test custom feature names."""
    from treeboost import GBDTConfig, GBDTModel

    np.random.seed(42)
    n_samples = 200

    X = np.random.randn(n_samples, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    config = GBDTConfig()
    config.num_rounds = 10
    config.max_depth = 3

    feature_names = ["age", "income", "score"]
    model = GBDTModel.train(X, y, config, feature_names=feature_names)

    assert model.feature_names == feature_names


def test_preset_configs():
    """Test preset-based configuration helpers."""
    from treeboost import GBDTConfig

    accuracy_cfg = GBDTConfig.preset("accuracy")
    assert accuracy_cfg.max_depth >= 8

    conformal_cfg = GBDTConfig.preset("conformal")
    assert conformal_cfg.calibration_ratio > 0.0
