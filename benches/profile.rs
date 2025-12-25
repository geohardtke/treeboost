//! Profiling benchmarks to identify training and prediction bottlenecks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use std::time::Duration;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

/// Generate synthetic regression dataset
fn generate_data(num_rows: usize, num_features: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::with_capacity(num_rows * num_features);
    let mut targets = Vec::with_capacity(num_rows);

    for _ in 0..num_rows {
        let mut row_sum = 0.0;
        for f in 0..num_features {
            let val: f64 = rng.gen_range(0.0..10.0);
            features.push(val);
            row_sum += val * (f as f64 + 1.0) * 0.1;
        }
        targets.push(row_sum + rng.gen_range(-1.0..1.0));
    }
    (features, targets)
}

/// Convert to TreeBoost format
fn to_treeboost_dataset(
    features: &[f64],
    targets: &[f64],
    num_rows: usize,
    num_features: usize,
) -> BinnedDataset {
    let mut all_binned = Vec::with_capacity(num_rows * num_features);
    let mut all_info = Vec::with_capacity(num_features);

    for f in 0..num_features {
        let mut col_values: Vec<f64> = (0..num_rows)
            .map(|r| features[r * num_features + f])
            .collect();
        col_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let num_bins = 255usize;
        let mut boundaries = Vec::with_capacity(num_bins - 1);
        for i in 1..num_bins {
            let idx = (i * col_values.len()) / num_bins;
            let val = col_values[idx.min(col_values.len() - 1)];
            if boundaries.is_empty() || val > *boundaries.last().unwrap() {
                boundaries.push(val);
            }
        }

        for r in 0..num_rows {
            let val = features[r * num_features + f];
            let bin = boundaries
                .binary_search_by(|b| b.partial_cmp(&val).unwrap())
                .unwrap_or_else(|i| i) as u8;
            all_binned.push(bin);
        }

        all_info.push(FeatureInfo {
            name: format!("f{}", f),
            feature_type: FeatureType::Numeric,
            num_bins: (boundaries.len() + 1).min(255) as u8,
            bin_boundaries: boundaries,
        });
    }

    let targets_f32: Vec<f32> = targets.iter().map(|&t| t as f32).collect();
    BinnedDataset::new(num_rows, all_binned, targets_f32, all_info)
}

fn benchmark_prediction_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("PredictionProfile");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);

    let num_features = 10;
    let num_rounds = 50;
    let train_rows = 10_000;
    let test_rows = 10_000;

    // Train model
    let (train_features, train_targets) = generate_data(train_rows, num_features, 42);
    let train_data = to_treeboost_dataset(&train_features, &train_targets, train_rows, num_features);
    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(6)
        .with_learning_rate(0.1)
        .with_min_samples_leaf(5);
    let model = GBDTModel::train(&train_data, config).unwrap();

    // Test data
    let (test_features, test_targets) = generate_data(test_rows, num_features, 123);
    let test_data = to_treeboost_dataset(&test_features, &test_targets, test_rows, num_features);

    // Benchmark parallel prediction (default)
    group.bench_function("predict_parallel", |b| {
        b.iter(|| black_box(model.predict(&test_data)));
    });

    // Benchmark sequential prediction
    group.bench_function("predict_sequential", |b| {
        b.iter(|| black_box(model.predict_sequential(&test_data)));
    });

    // Benchmark just bin access pattern
    group.bench_function("bin_access_only", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for row in 0..test_rows {
                for f in 0..num_features {
                    sum += test_data.get_bin(row, f) as u64;
                }
            }
            black_box(sum)
        });
    });

    // Benchmark bin access with column-first pattern (cache friendly)
    group.bench_function("bin_access_col_first", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for f in 0..num_features {
                let col = test_data.feature_column(f);
                for &bin in col {
                    sum += bin as u64;
                }
            }
            black_box(sum)
        });
    });

    // Benchmark tree traversal with precomputed bins per row
    let trees = model.trees();
    group.bench_function("tree_traverse_precomputed", |b| {
        b.iter(|| {
            let mut preds = Vec::with_capacity(test_rows);
            for row in 0..test_rows {
                // Precompute bins for this row
                let bins: Vec<u8> = (0..num_features)
                    .map(|f| test_data.get_bin(row, f))
                    .collect();

                let mut pred = model.base_prediction();
                for tree in trees {
                    pred += tree.predict(|f| bins[f]);
                }
                preds.push(pred);
            }
            black_box(preds)
        });
    });

    group.finish();
}

fn benchmark_training_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("TrainingProfile");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    let num_features = 10;

    // Small dataset training
    group.bench_function("train_1k_rows_10_rounds", |b| {
        let (features, targets) = generate_data(1_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 1_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // Medium dataset training
    group.bench_function("train_10k_rows_10_rounds", |b| {
        let (features, targets) = generate_data(10_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 10_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // More boosting rounds
    group.bench_function("train_10k_rows_50_rounds", |b| {
        let (features, targets) = generate_data(10_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 10_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(50)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // Deeper trees
    group.bench_function("train_10k_rows_depth8", |b| {
        let (features, targets) = generate_data(10_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 10_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(8)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // More features
    let num_features_wide = 50;
    group.bench_function("train_10k_rows_50_features", |b| {
        let (features, targets) = generate_data(10_000, num_features_wide, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 10_000, num_features_wide);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // With entropy regularization
    group.bench_function("train_10k_rows_entropy_reg", |b| {
        let (features, targets) = generate_data(10_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 10_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1)
            .with_entropy_weight(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // With Pseudo-Huber loss
    group.bench_function("train_10k_rows_huber", |b| {
        let (features, targets) = generate_data(10_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 10_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1)
            .with_pseudo_huber_loss(1.0);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // Large dataset
    group.bench_function("train_100k_rows_10_rounds", |b| {
        let (features, targets) = generate_data(100_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 100_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    group.finish();
}

criterion_group!(benches, benchmark_prediction_components, benchmark_training_components);
criterion_main!(benches);
