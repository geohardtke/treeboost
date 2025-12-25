//! Find crossover point where parallel gradient computation beats sequential

use rayon::prelude::*;
use std::time::Instant;

/// Returns (stop_size, recommended_size)
/// - stop_size: where speedup > 2.0x
/// - recommended_size: earliest size where moving mean of last 3 speedups >= 1.2x
fn find_crossover() -> (usize, usize) {
    let mut size = 20000;
    let step = 5000;
    let iterations = 100;

    let mut speedups: Vec<(usize, f64)> = Vec::new();
    let mut recommended: Option<usize> = None;

    loop {
        let targets: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let predictions: Vec<f32> = (0..size).map(|i| i as f32 * 0.01 + 0.5).collect();
        let train_indices: Vec<usize> = (0..size).collect();
        let mut gradients = vec![0.0f32; size];
        let mut hessians = vec![0.0f32; size];

        // Warmup
        for _ in 0..10 {
            for &idx in &train_indices {
                let residual = predictions[idx] - targets[idx];
                gradients[idx] = residual;
                hessians[idx] = 1.0;
            }
        }

        // Sequential timing
        let start = Instant::now();
        for _ in 0..iterations {
            for &idx in &train_indices {
                let residual = predictions[idx] - targets[idx];
                gradients[idx] = residual;
                hessians[idx] = 1.0;
            }
        }
        let seq_time = start.elapsed().as_nanos() as f64 / iterations as f64;

        // Warmup parallel
        for _ in 0..10 {
            train_indices.par_iter().for_each(|&idx| {
                let residual = predictions[idx] - targets[idx];
                unsafe {
                    let grad_ptr = gradients.as_ptr() as *mut f32;
                    let hess_ptr = hessians.as_ptr() as *mut f32;
                    *grad_ptr.add(idx) = residual;
                    *hess_ptr.add(idx) = 1.0;
                }
            });
        }

        // Parallel timing
        let start = Instant::now();
        for _ in 0..iterations {
            train_indices.par_iter().for_each(|&idx| {
                let residual = predictions[idx] - targets[idx];
                unsafe {
                    let grad_ptr = gradients.as_ptr() as *mut f32;
                    let hess_ptr = hessians.as_ptr() as *mut f32;
                    *grad_ptr.add(idx) = residual;
                    *hess_ptr.add(idx) = 1.0;
                }
            });
        }
        let par_time = start.elapsed().as_nanos() as f64 / iterations as f64;

        let speedup = seq_time / par_time;
        speedups.push((size, speedup));

        // Check moving mean of last 3
        if recommended.is_none() && speedups.len() >= 3 {
            let last_3: Vec<f64> = speedups.iter().rev().take(3).map(|(_, s)| *s).collect();
            let mean = last_3.iter().sum::<f64>() / 3.0;
            if mean >= 1.2 {
                // Find the earliest size in this window
                recommended = Some(speedups[speedups.len() - 3].0);
            }
        }

        println!(
            "{}\t\t{:.1} µs\t\t{:.1} µs\t\t{:.2}x",
            size,
            seq_time / 1000.0,
            par_time / 1000.0,
            speedup
        );

        if speedup > 2.0 {
            return (size, recommended.unwrap_or(size));
        }

        size += step;

        if size > 1_000_000 {
            return (1_000_000, recommended.unwrap_or(1_000_000));
        }
    }
}

fn main() {
    println!("Finding crossover point for parallel gradient computation...");
    println!("Stop when speedup > 2.0x, recommend earliest with moving mean >= 1.2x\n");

    let mut stop_sizes = Vec::new();
    let mut recommended_sizes = Vec::new();

    for run in 1..=5 {
        println!("=== Run {} ===", run);
        println!("Size\t\tSequential\tParallel\tSpeedup");
        println!("----\t\t----------\t--------\t-------");

        let (stop, recommended) = find_crossover();
        println!("Stop at: {}, Recommended: {}\n", stop, recommended);
        stop_sizes.push(stop);
        recommended_sizes.push(recommended);
    }

    let avg_stop = stop_sizes.iter().sum::<usize>() as f64 / stop_sizes.len() as f64;
    let avg_rec = recommended_sizes.iter().sum::<usize>() as f64 / recommended_sizes.len() as f64;

    println!("=== Results ===");
    println!("Stop sizes (2.0x): {:?}", stop_sizes);
    println!("Recommended sizes (1.2x moving mean): {:?}", recommended_sizes);
    println!("\nAverage stop: {:.0} rows", avg_stop);
    println!("Average recommended: {:.0} rows", avg_rec);
    println!("\nUse threshold: {} rows", (avg_rec / 5000.0).ceil() as usize * 5000);
}
