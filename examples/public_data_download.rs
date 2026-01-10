//! Public Dataset Example
//!
//! Demonstrates training TreeBoost on publicly available datasets:
//! - California Housing (real-world regression)
//! - Data preparation and exploration
//! - Train/test split
//! - Model evaluation
//!
//! This example generates synthetic data that mimics the structure of
//! the California Housing dataset for self-contained demonstration.
//! In practice, you can replace this with your own data files.
//!
//! To use real data:
//! 1. Download: https://www.kaggle.com/datasets/codenameneha/housing-prices-data
//! 2. Place in: datasets/california_housing.csv
//! 3. Uncomment the DatasetLoader section below
//!
//! Run with: cargo run --release --example public_data_download

#[path = "common/mod.rs"]
mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

/// Generate synthetic California Housing-like dataset
/// Dimensions: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
fn generate_synthetic_housing_data(n_samples: usize, seed: u64) -> BinnedDataset {
    let mut rng = common::SimpleRng::new(seed);

    let n_features = 8;

    // Generate features (column-major layout)
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    // First, generate row-major data for targets computation
    let mut row_major_data: Vec<Vec<f32>> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let med_inc = rng.next_range(0.5, 15.0);
        let house_age = rng.next_range(1.0, 52.0);
        let ave_rooms = rng.next_range(2.0, 10.0);
        let ave_bedrms = rng.next_range(0.5, 6.0);
        let population = rng.next_range(10.0, 35000.0);
        let ave_occup = rng.next_range(0.5, 10.0);
        let latitude = rng.next_range(32.0, 42.0);
        let longitude = rng.next_range(-125.0, -114.0);

        // Simulate house prices as function of features
        let target = 0.5
            + med_inc * 0.3
            + (house_age / 10.0) * 0.1
            + (10.0 / ave_rooms) * 0.2
            + (latitude - 32.0) / 10.0 * 0.1
            + rng.next_f32() * 0.5;

        targets.push(target.max(0.1));
        row_major_data.push(vec![
            med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude,
        ]);
    }

    // Convert to column-major u8 features
    for f in 0..n_features {
        for r in 0..n_samples {
            let val = row_major_data[r][f];
            let normalized = match f {
                0 => val / 15.0,           // MedInc
                1 => val / 52.0,           // HouseAge
                2 => val / 10.0,           // AveRooms
                3 => val / 6.0,            // AveBedrms
                4 => val / 35000.0,        // Population
                5 => val / 10.0,           // AveOccup
                6 => (val - 32.0) / 10.0,  // Latitude
                7 => (val + 125.0) / 11.0, // Longitude
                _ => 0.0,
            };
            features.push((normalized * 255.0).min(255.0) as u8);
        }
    }

    let feature_names = vec![
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];

    let feature_info: Vec<FeatureInfo> = feature_names
        .iter()
        .map(|&name| FeatureInfo {
            name: name.to_string(),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        })
        .collect();

    BinnedDataset::new(n_samples, features, targets, feature_info)
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TreeBoost: Public Dataset Example");
    println!("{}", "=".repeat(70));
    println!();

    // Dataset configuration
    let total_samples = 10000;
    let train_ratio = 0.7;
    let train_samples = (total_samples as f32 * train_ratio) as usize;

    println!("1. Data Preparation");
    println!("   Using synthetic California Housing-like data");
    println!("   Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude");
    println!("   Target: Median house value (in $100k)");
    println!();

    // Generate synthetic data
    println!("2. Generating synthetic dataset...");
    let full_dataset = generate_synthetic_housing_data(total_samples, 42);
    println!("   Total samples: {}", total_samples);
    println!("   Training samples: {}", train_samples);
    println!("   Test samples: {}", total_samples - train_samples);
    println!();

    // Split data
    println!("3. Splitting into train/test sets...");
    let train_dataset = common::extract_subset(&full_dataset, 0, train_samples);
    let test_dataset = common::extract_subset(&full_dataset, train_samples, total_samples);

    println!("   Training set: {} samples", train_dataset.num_rows());
    println!("   Test set: {} samples", test_dataset.num_rows());
    println!();

    // Explore training set statistics
    println!("4. Training Data Statistics");
    let train_targets = train_dataset.targets();
    let min_target = train_targets.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_target = train_targets
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mean_target = train_targets.iter().sum::<f32>() / train_targets.len() as f32;

    println!("   Target (Median House Value in $100k):");
    println!("     Min: {:.4}", min_target);
    println!("     Max: {:.4}", max_target);
    println!("     Mean: {:.4}", mean_target);
    println!();

    // Configure and train model
    println!("5. Training GBDT model...");
    let config = GBDTConfig::new()
        .with_num_rounds(100)
        .with_max_depth(6)
        .with_learning_rate(0.1)
        .with_subsample(0.8)?
        .with_colsample(0.8)?
        .with_seed(42);

    let start = std::time::Instant::now();
    let model = GBDTModel::train_binned(&train_dataset, config).expect("Training failed");
    let train_time = start.elapsed();

    println!("   Time: {:.2?}", train_time);
    println!("   Trees: {}", model.num_trees());
    println!();

    // Make predictions
    println!("6. Making predictions on test set...");
    let predictions = model.predict(&test_dataset);
    let test_targets = test_dataset.targets();
    println!("   Generated {} predictions", predictions.len());
    println!();

    // Evaluate model
    println!("7. Model Evaluation");

    let mae: f32 = predictions
        .iter()
        .zip(test_targets.iter())
        .map(|(pred, &target)| (target - pred).abs())
        .sum::<f32>()
        / predictions.len() as f32;

    let rmse: f32 = (predictions
        .iter()
        .zip(test_targets.iter())
        .map(|(pred, &target)| (target - pred).powi(2))
        .sum::<f32>()
        / predictions.len() as f32)
        .sqrt();

    let mean_pred = predictions.iter().sum::<f32>() / predictions.len() as f32;
    let ss_res: f32 = predictions
        .iter()
        .zip(test_targets.iter())
        .map(|(pred, &target)| (target - pred).powi(2))
        .sum();
    let ss_tot: f32 = test_targets
        .iter()
        .map(|&target| (target - mean_pred).powi(2))
        .sum();
    let r_squared = 1.0 - (ss_res / ss_tot);

    println!("   Mean Absolute Error: ${:.2}k", mae * 100.0);
    println!("   Root Mean Squared Error: ${:.2}k", rmse * 100.0);
    println!("   R² Score: {:.4}", r_squared);
    println!();

    // Feature importance
    println!("8. Feature Importance");
    let feature_names = vec![
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];
    let importances = model.feature_importance();
    let mut indexed: Vec<(usize, f32, &str)> = importances
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp, feature_names[i]))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (_, importance, name) in indexed.iter() {
        let bar = "*".repeat(((importance * 50.0) as usize).min(50));
        println!("   {:12} {:.4} {}", name, importance, bar);
    }
    println!();

    // Sample predictions
    println!("9. Sample Predictions vs Actual (in $100k):");
    println!(
        "   {:>6} {:>10} {:>10} {:>10}",
        "ID", "Predicted", "Actual", "Error"
    );
    println!("   {}", "-".repeat(40));

    for i in (0..predictions.len()).step_by((predictions.len() / 5).max(1)) {
        let error = (predictions[i] - test_targets[i]).abs();
        println!(
            "   {:>6} {:>10.2} {:>10.2} {:>10.2}",
            i,
            predictions[i] * 100.0,
            test_targets[i] * 100.0,
            error * 100.0
        );
    }
    println!();

    // Save model
    println!("10. Saving model...");
    let model_path = "/tmp/housing_model.rkyv";
    treeboost::serialize::save_model(&model, model_path).expect("Failed to save model");
    println!("    Saved to: {}", model_path);
    println!();

    println!("{}", "=".repeat(70));
    println!("Example completed successfully!");
    println!();
    println!("To use real California Housing data:");
    println!("1. Download from: https://www.kaggle.com/datasets/codenameneha/housing-prices-data");
    println!("2. Save as: datasets/california_housing.csv");
    println!("3. Use DatasetLoader::load_csv() to load the data");
    println!("{}", "=".repeat(70));
}
