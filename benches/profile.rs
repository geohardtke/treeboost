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

fn benchmark_histogram_building(c: &mut Criterion) {
    use treeboost::histogram::{Histogram, HistogramBuilder};

    let mut group = c.benchmark_group("HistogramProfile");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let num_rows = 100_000;
    let num_features = 10;

    let (features, targets) = generate_data(num_rows, num_features, 42);
    let dataset = to_treeboost_dataset(&features, &targets, num_rows, num_features);

    // Create gradients/hessians
    let gradients: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.001).collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];

    // Contiguous row indices (root node case)
    let contiguous_rows: Vec<usize> = (0..num_rows).collect();

    // Sparse row indices (child node case - every other row)
    let sparse_rows: Vec<usize> = (0..num_rows).step_by(2).collect();

    let builder = HistogramBuilder::new();

    group.bench_function("histogram_contiguous_100k", |b| {
        b.iter(|| {
            black_box(builder.build(&dataset, &contiguous_rows, &gradients, &hessians))
        });
    });

    group.bench_function("histogram_sparse_50k", |b| {
        b.iter(|| {
            black_box(builder.build(&dataset, &sparse_rows, &gradients, &hessians))
        });
    });

    // Single feature histogram - the innermost hot loop
    let feature_col = dataset.feature_column(0);
    group.bench_function("single_feature_accumulate_100k", |b| {
        b.iter(|| {
            let mut hist = Histogram::new();
            for i in 0..num_rows {
                let bin = feature_col[i];
                hist.accumulate(bin, gradients[i], hessians[i]);
            }
            black_box(hist)
        });
    });

    // Zip-based single feature (what contiguous path uses)
    group.bench_function("single_feature_zip_100k", |b| {
        b.iter(|| {
            let mut hist = Histogram::new();
            for ((&bin, &grad), &hess) in feature_col.iter().zip(&gradients).zip(&hessians) {
                hist.accumulate(bin, grad, hess);
            }
            black_box(hist)
        });
    });

    // Batch accumulate (new optimized path)
    group.bench_function("single_feature_batch_100k", |b| {
        b.iter(|| {
            let mut hist = Histogram::new();
            hist.accumulate_batch(feature_col, &gradients, &hessians);
            black_box(hist)
        });
    });

    // Indexed access (what child nodes use) - simulates sparse row_indices
    group.bench_function("single_feature_indexed_100k", |b| {
        b.iter(|| {
            let mut hist = Histogram::new();
            let bins = hist.bins_mut();
            let len = contiguous_rows.len();
            let chunks = len / 8;
            let remainder = len % 8;

            unsafe {
                for i in 0..chunks {
                    let base = i * 8;
                    let idx0 = *contiguous_rows.get_unchecked(base);
                    let idx1 = *contiguous_rows.get_unchecked(base + 1);
                    let idx2 = *contiguous_rows.get_unchecked(base + 2);
                    let idx3 = *contiguous_rows.get_unchecked(base + 3);
                    let idx4 = *contiguous_rows.get_unchecked(base + 4);
                    let idx5 = *contiguous_rows.get_unchecked(base + 5);
                    let idx6 = *contiguous_rows.get_unchecked(base + 6);
                    let idx7 = *contiguous_rows.get_unchecked(base + 7);

                    let bin0 = *feature_col.get_unchecked(idx0) as usize;
                    let bin1 = *feature_col.get_unchecked(idx1) as usize;
                    let bin2 = *feature_col.get_unchecked(idx2) as usize;
                    let bin3 = *feature_col.get_unchecked(idx3) as usize;
                    let bin4 = *feature_col.get_unchecked(idx4) as usize;
                    let bin5 = *feature_col.get_unchecked(idx5) as usize;
                    let bin6 = *feature_col.get_unchecked(idx6) as usize;
                    let bin7 = *feature_col.get_unchecked(idx7) as usize;

                    let grad0 = *gradients.get_unchecked(idx0);
                    let grad1 = *gradients.get_unchecked(idx1);
                    let grad2 = *gradients.get_unchecked(idx2);
                    let grad3 = *gradients.get_unchecked(idx3);
                    let grad4 = *gradients.get_unchecked(idx4);
                    let grad5 = *gradients.get_unchecked(idx5);
                    let grad6 = *gradients.get_unchecked(idx6);
                    let grad7 = *gradients.get_unchecked(idx7);

                    let hess0 = *hessians.get_unchecked(idx0);
                    let hess1 = *hessians.get_unchecked(idx1);
                    let hess2 = *hessians.get_unchecked(idx2);
                    let hess3 = *hessians.get_unchecked(idx3);
                    let hess4 = *hessians.get_unchecked(idx4);
                    let hess5 = *hessians.get_unchecked(idx5);
                    let hess6 = *hessians.get_unchecked(idx6);
                    let hess7 = *hessians.get_unchecked(idx7);

                    bins.get_unchecked_mut(bin0).accumulate(grad0, hess0);
                    bins.get_unchecked_mut(bin1).accumulate(grad1, hess1);
                    bins.get_unchecked_mut(bin2).accumulate(grad2, hess2);
                    bins.get_unchecked_mut(bin3).accumulate(grad3, hess3);
                    bins.get_unchecked_mut(bin4).accumulate(grad4, hess4);
                    bins.get_unchecked_mut(bin5).accumulate(grad5, hess5);
                    bins.get_unchecked_mut(bin6).accumulate(grad6, hess6);
                    bins.get_unchecked_mut(bin7).accumulate(grad7, hess7);
                }
                let base = chunks * 8;
                for i in 0..remainder {
                    let idx = *contiguous_rows.get_unchecked(base + i);
                    let bin = *feature_col.get_unchecked(idx) as usize;
                    bins.get_unchecked_mut(bin).accumulate(
                        *gradients.get_unchecked(idx),
                        *hessians.get_unchecked(idx),
                    );
                }
            }
            black_box(hist)
        });
    });

    // Benchmark a single tree grow (the main per-round cost)
    group.bench_function("single_tree_grow_100k", |b| {
        use treeboost::tree::TreeGrower;
        use treeboost::loss::{LossFunction, MseLoss};

        let loss = MseLoss::new();
        let predictions = vec![0.0f32; num_rows];
        let targets_f32: Vec<f32> = targets.iter().map(|&t| t as f32).collect();

        // Pre-compute gradients/hessians
        let (grads, hess): (Vec<f32>, Vec<f32>) = targets_f32.iter()
            .zip(&predictions)
            .map(|(&t, &p)| loss.gradient_hessian(t, p))
            .unzip();

        let grower = TreeGrower::new()
            .with_max_depth(6)
            .with_max_leaves(31)
            .with_learning_rate(0.1)
            .with_lambda(1.0);

        let row_indices: Vec<usize> = (0..num_rows).collect();

        b.iter(|| {
            black_box(grower.grow_with_indices(&dataset, &grads, &hess, &row_indices))
        });
    });

    group.finish();
}

fn benchmark_training_components(c: &mut Criterion) {
    use treeboost::dataset::QuantileBinner;

    let mut group = c.benchmark_group("TrainingProfile");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    let num_features = 10;

    // Benchmark binning separately (to match Python benchmark behavior)
    group.bench_function("binning_8k_20features", |b| {
        let num_rows = 8000;
        let num_features = 20;
        let (features, _targets) = generate_data(num_rows, num_features, 42);
        let binner = QuantileBinner::new(255);

        b.iter(|| {
            let mut binned_data = Vec::with_capacity(num_rows * num_features);
            for f in 0..num_features {
                let col: Vec<f64> = (0..num_rows)
                    .map(|r| features[r * num_features + f])
                    .collect();
                let boundaries = binner.compute_boundaries(&col);
                let binned = binner.bin_column(&col, &boundaries);
                binned_data.extend(binned);
            }
            black_box(binned_data)
        });
    });

    // Training with pre-binned data (matches Python: binning done once before training loop)
    group.bench_function("train_8k_20features_100rounds_prebinned", |b| {
        let num_rows = 8000;
        let num_features = 20;
        let (features, targets) = generate_data(num_rows, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, num_rows, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(100)
            .with_max_depth(6)
            .with_max_leaves(31)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    // Training with random noise targets (matches Python test_timing.py exactly)
    group.bench_function("train_8k_20features_100rounds_random", |b| {
        let num_rows = 8000;
        let num_features = 20;
        // Use pure random targets like Python benchmark
        let mut rng = StdRng::seed_from_u64(42);
        let features: Vec<f64> = (0..num_rows * num_features)
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0) // Standard normal-ish
            .collect();
        let targets: Vec<f64> = (0..num_rows)
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0) // Random targets
            .collect();
        let dataset = to_treeboost_dataset(&features, &targets, num_rows, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(100)
            .with_max_depth(6)
            .with_max_leaves(31)
            .with_learning_rate(0.1);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

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

    // Large dataset with GOSS
    group.bench_function("train_100k_rows_10_rounds_goss", |b| {
        let (features, targets) = generate_data(100_000, num_features, 42);
        let dataset = to_treeboost_dataset(&features, &targets, 100_000, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(6)
            .with_learning_rate(0.1)
            .with_goss(true);

        b.iter(|| black_box(GBDTModel::train(&dataset, config.clone()).unwrap()));
    });

    group.finish();
}

criterion_group!(benches, benchmark_prediction_components, benchmark_histogram_building, benchmark_training_components);
criterion_main!(benches);
