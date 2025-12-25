# TreeBoost Benchmark Results

Comparison of TreeBoost against other pure-Rust GBDT implementations.

## Competitors

| Implementation | Crate               | Description                                                                       |
| -------------- | ------------------- | --------------------------------------------------------------------------------- |
| **TreeBoost**  | `treeboost`         | Histogram-based, Rayon parallel, T-Digest binning, Shannon Entropy regularization |
| **gbdt-rs**    | `gbdt = "0.1"`      | Baidu MesaTEE, pure safe Rust, XGBoost model compatibility                        |
| **forust**     | `forust-ml = "0.4"` | Histogram-based, Rayon parallel, leaf-wise growth                                 |

## Test Configuration

- **Features**: 10 numeric features
- **Trees**: 50 boosting rounds
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Hardware**: Single machine benchmark (results may vary by hardware)

## Training Performance

| Dataset Size | TreeBoost | forust | gbdt-rs | TreeBoost vs forust | TreeBoost vs gbdt-rs               |
| ------------ | --------- | ------ | ------- | ------------------- | ---------------------------------- |
| 1K rows      | 84 ms     | 108 ms | 32 ms   | **1.29x faster**    | 0.38x (gbdt-rs wins on small data) |
| 10K rows     | 140 ms    | 179 ms | 276 ms  | **1.28x faster**    | **1.97x faster**                   |
| 100K rows    | 456 ms    | 579 ms | 3.35 s  | **1.27x faster**    | **7.3x faster**                    |

### Throughput (elements/second)

| Dataset Size | TreeBoost | forust | gbdt-rs |
| ------------ | --------- | ------ | ------- |
| 1K rows      | 11.9K     | 9.2K   | 31.4K   |
| 10K rows     | 71.6K     | 55.8K  | 36.2K   |
| 100K rows    | **219K**  | 173K   | 29.8K   |

### Key Observations - Training

- TreeBoost scales efficiently with data size due to histogram-based split finding
- gbdt-rs is faster on small datasets (1K) due to lower overhead, but O(N²) complexity causes it to fall behind at scale
- TreeBoost consistently outperforms forust by ~25-30% across all dataset sizes

## Prediction Performance

| Dataset Size | TreeBoost | forust  | gbdt-rs | TreeBoost vs forust | TreeBoost vs gbdt-rs |
| ------------ | --------- | ------- | ------- | ------------------- | -------------------- |
| 100 rows     | 32 µs     | 96 µs   | 142 µs  | **3.0x faster**     | **4.4x faster**      |
| 1K rows      | 135 µs    | 896 µs  | 1.26 ms | **6.6x faster**     | **9.3x faster**      |
| 10K rows     | 856 µs    | 8.99 ms | 11.6 ms | **10.5x faster**    | **13.6x faster**     |

### Throughput (elements/second)

| Dataset Size | TreeBoost | forust | gbdt-rs |
| ------------ | --------- | ------ | ------- |
| 100 rows     | 3.1M      | 1.0M   | 704K    |
| 1K rows      | 7.4M      | 1.1M   | 792K    |
| 10K rows     | **11.7M** | 1.1M   | 860K    |

### Key Observations - Prediction

- TreeBoost uses Rayon parallel prediction with row-wise bin caching
- Prediction throughput improves with batch size due to parallelism efficiency
- At 10K rows: TreeBoost is **10.5x faster** than forust and **13.6x faster** than gbdt-rs

## Parallel Training (100K rows, 100 rounds)

| Implementation  | Time       | Throughput        |
| --------------- | ---------- | ----------------- |
| **TreeBoost**   | **833 ms** | **120K elem/s**   |
| forust-parallel | 2.17 s     | 46.1K elem/s      |
| gbdt-rs         | 6.47 s     | 15.5K elem/s      |

TreeBoost is **2.6x faster** than forust in parallel mode and **7.8x faster** than gbdt-rs.

## Optimization Details

### Prediction Optimizations

1. **Row-wise Bin Caching**: Pre-compute all feature bins for each row before tree traversal

   - Avoids repeated column-major memory access patterns
   - Reduces cache misses during tree traversal

2. **Rayon Parallelization**: Parallel iteration over rows

   - Each row's prediction is independent
   - Near-linear scaling with CPU cores

3. **Inline Hints**: Critical path functions marked `#[inline]`

### Training Optimizations

1. **Histogram-based Split Finding**: O(bins) instead of O(N) per split
2. **T-Digest Quantile Binning**: Streaming quantile estimation for bin boundaries
3. **Histogram Subtraction Trick**: Compute sibling histograms from parent - smaller child
4. **Feature-parallel Histogram Construction**: Rayon work-stealing across features

## Running Benchmarks

```bash
# Full competitor benchmark
cargo bench --bench competitors

# Profile benchmarks (training + prediction)
cargo bench --bench profile

# Training benchmark only
cargo bench --bench competitors -- "Training"

# Prediction benchmark only
cargo bench --bench competitors -- "Prediction"

# Parallel training only
cargo bench --bench competitors -- "ParallelTraining"

# TreeBoost only (skip competitors)
cargo bench --bench competitors -- "TreeBoost"
```

## Methodology

- Benchmarks use Criterion with 30-50 samples per measurement
- Warm-up period of 3 seconds before each benchmark
- Results show mean time with confidence intervals
- All competitors use equivalent hyperparameters where possible
