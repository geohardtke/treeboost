//! Test progress callback functionality

use polars::prelude::*;
use std::sync::{Arc, Mutex};
use treeboost::{AutoBuilder, ConsoleProgress, ProgressCallback, ProgressUpdate, TrainingPhase};

/// Create a simple test dataset
fn create_test_dataset(n_rows: usize) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(42);

    let x1: Vec<f64> = (0..n_rows).map(|i| i as f64 / 10.0).collect();
    let x2: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 100.0).collect();

    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&x1, &x2)| 2.0 * x1 + 0.5 * x2 + rng.f64() * 5.0)
        .collect();

    df!(
        "x1" => x1,
        "x2" => x2,
        "target" => y
    )
    .unwrap()
}

/// Custom progress tracker that records all updates
struct TestProgress {
    updates: Arc<Mutex<Vec<ProgressUpdate>>>,
}

impl TestProgress {
    fn new() -> Self {
        Self {
            updates: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_updates(&self) -> Vec<ProgressUpdate> {
        self.updates.lock().unwrap().clone()
    }
}

impl ProgressCallback for TestProgress {
    fn on_progress(&self, update: &ProgressUpdate) {
        self.updates.lock().unwrap().push(update.clone());
    }
}

#[test]
fn test_progress_callback_receives_all_phases() {
    let df = create_test_dataset(200);

    let progress = Arc::new(TestProgress::new());
    let progress_clone = Arc::clone(&progress);

    let builder = AutoBuilder::new().with_progress_callback(progress_clone);

    let _result = builder.fit(&df, "target").expect("Training should succeed");

    // Get all progress updates
    let updates = progress.get_updates();

    // Should have received updates for all phases
    assert!(!updates.is_empty(), "Should receive progress updates");

    // Verify we got the key phases
    let phases: Vec<TrainingPhase> = updates.iter().map(|u| u.phase).collect();

    // Should at least have profiling, analysis, tuning, training, complete
    assert!(
        phases.contains(&TrainingPhase::Profiling),
        "Should have profiling phase"
    );
    assert!(
        phases.contains(&TrainingPhase::Training),
        "Should have training phase"
    );
    assert!(
        phases.contains(&TrainingPhase::Complete),
        "Should have complete phase"
    );

    // Progress should generally increase
    let last_update = updates.last().unwrap();
    assert_eq!(
        last_update.phase,
        TrainingPhase::Complete,
        "Last phase should be Complete"
    );
    assert_eq!(
        last_update.progress_pct, 100,
        "Final progress should be 100%"
    );

    println!("Received {} progress updates", updates.len());
    for update in &updates {
        println!(
            "  [{:3}%] {:?} - {:?}",
            update.progress_pct, update.phase, update.elapsed
        );
    }
}

#[test]
fn test_console_progress_callback() {
    let df = create_test_dataset(150);

    // Use the built-in console progress
    let builder = AutoBuilder::new()
        .with_progress_callback(Arc::new(ConsoleProgress::detailed()));

    let _result = builder.fit(&df, "target").expect("Training should succeed");

    // Visual inspection - should print progress bars to console
    println!("Console progress test completed");
}

#[test]
fn test_progress_with_time_budget() {
    use std::time::Duration;

    let df = create_test_dataset(200);

    let progress = Arc::new(TestProgress::new());
    let progress_clone = Arc::clone(&progress);

    let builder = AutoBuilder::new()
        .with_time_budget(Duration::from_secs(10))
        .with_progress_callback(progress_clone);

    let _result = builder.fit(&df, "target").expect("Training should succeed");

    let updates = progress.get_updates();

    // With time budget, some phases might be skipped
    // But we should still get progress updates
    assert!(!updates.is_empty(), "Should receive progress updates");
    assert_eq!(
        updates.last().unwrap().phase,
        TrainingPhase::Complete,
        "Should still complete"
    );

    // Check that messages reflect skipped phases
    let messages: Vec<String> = updates
        .iter()
        .filter_map(|u| u.message.clone())
        .collect();

    println!("Progress messages with time budget:");
    for msg in &messages {
        println!("  - {}", msg);
    }
}

#[test]
fn test_progress_percentages_increase() {
    let df = create_test_dataset(180);

    let progress = Arc::new(TestProgress::new());
    let progress_clone = Arc::clone(&progress);

    let builder = AutoBuilder::new().with_progress_callback(progress_clone);

    let _result = builder.fit(&df, "target").expect("Training should succeed");

    let updates = progress.get_updates();

    // Verify progress percentages generally increase
    let mut prev_pct = 0;
    for update in &updates {
        assert!(
            update.progress_pct >= prev_pct,
            "Progress should not decrease: {} -> {}",
            prev_pct,
            update.progress_pct
        );
        prev_pct = update.progress_pct;
    }

    assert_eq!(prev_pct, 100, "Final progress should be 100%");
}
