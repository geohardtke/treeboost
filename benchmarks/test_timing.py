#!/usr/bin/env python3
"""Simple timing test for TreeBoost training."""

import time
import numpy as np
from treeboost import GBDTConfig, GBDTModel

# Same parameters as Rust benchmark
num_rows = 100000
num_features = 20
num_rounds = 100
max_depth = 6
learning_rate = 0.1

# Generate random data
np.random.seed(42)
X = np.random.randn(num_rows, num_features).astype(np.float32)
y = np.random.randn(num_rows).astype(np.float32)

# Create config
config = GBDTConfig()
config.num_rounds = num_rounds
config.max_depth = max_depth
config.learning_rate = learning_rate
config.max_leaves = 31

# Warmup
print("Warming up...")
model = GBDTModel.train(X, y, config)

# Timed runs
n_iterations = 5
times = []
for i in range(n_iterations):
    start = time.perf_counter()
    model = GBDTModel.train(X, y, config)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Iteration {i+1}: {elapsed:.2f} ms")

mean_ms = np.mean(times)
std_ms = np.std(times)
print(f"\nMean: {mean_ms:.2f} ms (±{std_ms:.2f})")
print(f"Config: {num_rows} rows, {num_features} features, {num_rounds} rounds, depth={max_depth}")
