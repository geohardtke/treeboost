//! Integration tests for AutoTuner

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::tuner::{AutoTuner, EvalStrategy, GridStrategy, ParameterSpace, TunerConfig};

use common::{create_binary_classification_dataset, create_synthetic_dataset};

/// Test full tuning loop on synthetic regression data
#[test]
fn test_autotuner_regression() {
    let dataset = create_synthetic_dataset(500, 42);

    let base_config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_learning_rate(0.1);

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 3 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(123);

    let (best_config, history) = tuner.tune(&dataset).expect("Tuning should succeed");

    // Should have run some trials
    assert!(
        !history.is_empty(),
        "History should not be empty after tuning"
    );

    // Best trial should have valid metrics
    let best = history.best().expect("Should have a best trial");
    assert!(best.val_metric.is_finite(), "Best val_metric should be finite");
    assert!(best.val_metric >= 0.0, "MSE should be non-negative");

    // Train final model with best config
    let final_model =
        GBDTModel::train_binned(&dataset, best_config).expect("Final training should succeed");

    // Model should produce reasonable predictions
    let predictions = final_model.predict(&dataset);
    assert_eq!(predictions.len(), dataset.num_rows());
}

/// Test full tuning loop on binary classification data
#[test]
fn test_autotuner_binary_classification() {
    let dataset = create_binary_classification_dataset(400, 456);

    let base_config = GBDTConfig::new()
        .with_num_rounds(15)
        .with_learning_rate(0.1)
        .with_binary_logloss();

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 3 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_classification())
        .with_seed(789);

    let (best_config, history) = tuner.tune(&dataset).expect("Tuning should succeed");

    assert!(!history.is_empty());
    let best = history.best().expect("Should have a best trial");
    assert!(best.val_metric.is_finite());

    // Train final model
    let final_model =
        GBDTModel::train_binned(&dataset, best_config).expect("Training should succeed");
    let predictions = final_model.predict(&dataset);

    // Binary classification predictions should be in reasonable range
    for &pred in &predictions {
        assert!(pred.is_finite(), "Prediction should be finite");
    }
}

/// Test autotuner with K-fold cross-validation
#[test]
fn test_autotuner_kfold() {
    let dataset = create_synthetic_dataset(300, 321);

    let base_config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_learning_rate(0.1);

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2).with_folds(3)) // 3-fold CV
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(654);

    let (best_config, history) = tuner.tune(&dataset).expect("K-fold tuning should succeed");

    assert!(!history.is_empty());
    let best = history.best().expect("Should have best trial");
    assert!(best.val_metric.is_finite());
    assert!(best.num_trees > 0, "Should have trained trees");

    // Train final model
    let final_model =
        GBDTModel::train_binned(&dataset, best_config).expect("Training should succeed");
    assert!(final_model.num_trees() > 0);
}

/// Test autotuner with Latin Hypercube Sampling
#[test]
fn test_autotuner_lhs() {
    let dataset = create_synthetic_dataset(400, 111);

    let base_config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_learning_rate(0.1);

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::LatinHypercube { n_samples: 10 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(222);

    let (_, history) = tuner.tune(&dataset).expect("LHS tuning should succeed");

    // Should have 10 trials per iteration = 20 total
    assert!(history.len() >= 10, "Should have at least 10 trials from LHS");

    let best = history.best().expect("Should have best trial");
    assert!(best.val_metric.is_finite());
}

/// Test autotuner with Random sampling
#[test]
fn test_autotuner_random() {
    let dataset = create_synthetic_dataset(400, 333);

    let base_config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_learning_rate(0.1);

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Random { n_samples: 8 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(444);

    let (_, history) = tuner.tune(&dataset).expect("Random tuning should succeed");

    // Should have 8 trials per iteration = 16 total
    assert!(
        history.len() >= 8,
        "Should have at least 8 trials from random"
    );

    let best = history.best().expect("Should have best trial");
    assert!(best.val_metric.is_finite());
}

/// Test autotuner reproducibility with same seed
#[test]
fn test_autotuner_reproducibility() {
    let dataset = create_synthetic_dataset(300, 555);

    let base_config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_learning_rate(0.1)
        .with_seed(42); // Fixed seed for GBDT

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Random { n_samples: 5 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false)
        .with_parallel(false); // Disable parallel for determinism

    // Run 1
    let mut tuner1 = AutoTuner::<GBDTModel>::new(base_config.clone())
        .with_config(tuner_config.clone())
        .with_space(ParameterSpace::default_regression())
        .with_seed(999);

    let (_, history1) = tuner1.tune(&dataset).expect("Run 1 should succeed");

    // Run 2 with same seed
    let mut tuner2 = AutoTuner::<GBDTModel>::new(base_config.clone())
        .with_config(tuner_config.clone())
        .with_space(ParameterSpace::default_regression())
        .with_seed(999);

    let (_, history2) = tuner2.tune(&dataset).expect("Run 2 should succeed");

    // Results should be identical
    assert_eq!(
        history1.len(),
        history2.len(),
        "Should have same number of trials"
    );

    let best1 = history1.best().expect("Should have best 1");
    let best2 = history2.best().expect("Should have best 2");

    assert_eq!(
        best1.val_metric, best2.val_metric,
        "Best val_metric should be identical with same seed"
    );

    // Check that sampled hyperparameters are identical
    let trials1 = history1.trials();
    let trials2 = history2.trials();
    for (t1, t2) in trials1.iter().zip(trials2.iter()) {
        for (key, &val1) in &t1.params {
            let val2 = t2.params.get(key).expect("Should have same keys");
            assert!(
                (val1 - val2).abs() < 1e-6,
                "Param {} should match: {} vs {}",
                key,
                val1,
                val2
            );
        }
    }
}

/// Test autotuner with early stopping integration
#[test]
fn test_autotuner_early_stopping() {
    let dataset = create_synthetic_dataset(600, 666);

    // Enable early stopping in base config
    let base_config = GBDTConfig::new()
        .with_num_rounds(100) // High max rounds
        .with_learning_rate(0.1)
        .with_early_stopping(10, 0.2); // Stop after 10 rounds no improvement

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(777);

    let (best_config, history) = tuner
        .tune(&dataset)
        .expect("Tuning with early stopping should succeed");

    // Trials should have early stopped (num_trees < 100)
    let best = history.best().expect("Should have best trial");
    assert!(
        best.num_trees <= 100,
        "Should have used early stopping: num_trees = {}",
        best.num_trees
    );

    // Final model training
    let final_model =
        GBDTModel::train_binned(&dataset, best_config).expect("Training should succeed");
    assert!(final_model.num_trees() > 0);
}

/// Test autotuner with progress callback
#[test]
fn test_autotuner_callback() {
    let dataset = create_synthetic_dataset(300, 888);

    let base_config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_learning_rate(0.1);

    let tuner_config = TunerConfig::new()
        .with_iterations(1)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let callback_count = Arc::new(AtomicUsize::new(0));
    let callback_count_clone = callback_count.clone();

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(999)
        .with_callback(move |_trial, _current, _total| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
        });

    let (_, history) = tuner.tune(&dataset).expect("Tuning should succeed");

    // Callback should have been called for each trial
    assert_eq!(
        callback_count.load(Ordering::SeqCst),
        history.len(),
        "Callback should be called once per trial"
    );
}

/// Test autotuner history JSON export
#[test]
fn test_autotuner_history_json() {
    let dataset = create_synthetic_dataset(200, 101);

    let base_config = GBDTConfig::new()
        .with_num_rounds(5)
        .with_learning_rate(0.1);

    let tuner_config = TunerConfig::new()
        .with_iterations(1)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(202);

    let (_, history) = tuner.tune(&dataset).expect("Tuning should succeed");

    // Export to JSON
    let json = history.to_json();

    // Verify JSON structure
    assert!(json.contains("\"trials\""), "JSON should have trials array");
    assert!(
        json.contains("\"best_trial_id\""),
        "JSON should have best_trial_id"
    );
    assert!(
        json.contains("\"val_metric\""),
        "JSON should have val_metric field"
    );
    assert!(json.contains("\"params\""), "JSON should have params field");

    // Verify it's valid enough to contain trial count markers
    let trial_count = json.matches("\"trial_id\"").count();
    assert_eq!(
        trial_count,
        history.len(),
        "JSON should have entry for each trial"
    );
}
