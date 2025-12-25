#!/usr/bin/env python3
"""
Generate synthetic parquet sample files for TreeBoost testing.

Usage:
    python scripts/generate_samples.py [--output-dir samples/synthetic]

This script generates large parquet files for testing:
- large_regression.parquet: 100K rows, numeric only
- large_mixed.parquet: 100K rows, mixed numeric + categorical
- large_dirty.parquet: 100K rows with missing values, outliers, rare categories
- large_high_cardinality.parquet: 100K rows with 10K+ unique categories
- stress_test.parquet: 1M rows for stress testing
"""

import argparse
import random
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required. Install with: pip install pyarrow")
    exit(1)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)


def generate_large_regression(n_rows: int = 100_000) -> pa.Table:
    """Generate large numeric-only regression dataset."""
    print(f"Generating large_regression.parquet ({n_rows:,} rows)...")

    # 10 numeric features
    features = {}
    for i in range(10):
        features[f"feature_{i}"] = [random.gauss(0, 1) for _ in range(n_rows)]

    # Target: linear combination + noise
    target = []
    for row in range(n_rows):
        y = sum(features[f"feature_{i}"][row] * (i + 1) * 0.5 for i in range(5))
        y += random.gauss(0, 0.1)
        target.append(y)

    features["target"] = target
    return pa.table(features)


def generate_large_mixed(n_rows: int = 100_000) -> pa.Table:
    """Generate large mixed numeric + categorical dataset."""
    print(f"Generating large_mixed.parquet ({n_rows:,} rows)...")

    neighborhoods = ["downtown", "suburbs", "rural", "industrial", "waterfront"]
    property_types = ["apartment", "house", "condo", "townhouse", "duplex"]
    conditions = ["excellent", "good", "fair", "poor"]

    data = {
        "price": [random.uniform(50000, 1000000) for _ in range(n_rows)],
        "sqft": [random.uniform(500, 5000) for _ in range(n_rows)],
        "bedrooms": [random.randint(1, 6) for _ in range(n_rows)],
        "bathrooms": [random.randint(1, 4) for _ in range(n_rows)],
        "year_built": [random.randint(1950, 2024) for _ in range(n_rows)],
        "lot_size": [random.uniform(1000, 20000) for _ in range(n_rows)],
        "neighborhood": [random.choice(neighborhoods) for _ in range(n_rows)],
        "property_type": [random.choice(property_types) for _ in range(n_rows)],
        "condition": [random.choice(conditions) for _ in range(n_rows)],
        "has_pool": [random.choice(["true", "false"]) for _ in range(n_rows)],
        "has_garage": [random.choice(["true", "false"]) for _ in range(n_rows)],
    }

    # Target based on features
    target = []
    for i in range(n_rows):
        base = data["sqft"][i] * 150 + data["bedrooms"][i] * 10000
        if data["neighborhood"][i] == "waterfront":
            base *= 1.5
        elif data["neighborhood"][i] == "downtown":
            base *= 1.2
        base += random.gauss(0, 10000)
        target.append(base)

    data["target"] = target
    return pa.table(data)


def generate_large_dirty(n_rows: int = 100_000) -> pa.Table:
    """Generate large dataset with dirty data issues."""
    print(f"Generating large_dirty.parquet ({n_rows:,} rows)...")

    # Frequent categories (80% of data)
    frequent_cats = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]
    # Rare categories (will appear < 10 times each)
    rare_cats = [f"rare_{i}" for i in range(500)]

    data = {
        "value": [],
        "category": [],
        "group": [],
        "score": [],
        "target": [],
    }

    for i in range(n_rows):
        # Value: 5% missing, 2% outliers
        if random.random() < 0.05:
            data["value"].append(None)
        elif random.random() < 0.02:
            data["value"].append(random.choice([-9999, 99999, 1e9]))
        else:
            data["value"].append(random.gauss(100, 20))

        # Category: 80% frequent, 20% rare (spread across 500 rare categories)
        if random.random() < 0.80:
            data["category"].append(random.choice(frequent_cats))
        else:
            data["category"].append(random.choice(rare_cats))

        # Group: 3% missing
        if random.random() < 0.03:
            data["group"].append(None)
        else:
            data["group"].append(f"group_{random.randint(1, 10)}")

        # Score: 2% missing
        if random.random() < 0.02:
            data["score"].append(None)
        else:
            data["score"].append(random.uniform(0, 1))

        # Target: 1% missing
        if random.random() < 0.01:
            data["target"].append(None)
        else:
            data["target"].append(random.gauss(200, 50))

    return pa.table(data)


def generate_large_high_cardinality(n_rows: int = 100_000) -> pa.Table:
    """Generate large dataset with high-cardinality categoricals."""
    print(f"Generating large_high_cardinality.parquet ({n_rows:,} rows)...")

    # Generate unique IDs
    n_users = 10000
    n_products = 5000
    n_regions = 200
    n_merchants = 1000

    data = {
        "user_id": [f"user_{random.randint(1, n_users):05d}" for _ in range(n_rows)],
        "product_id": [f"prod_{random.randint(1, n_products):04d}" for _ in range(n_rows)],
        "region": [f"region_{random.randint(1, n_regions):03d}" for _ in range(n_rows)],
        "merchant_id": [f"merchant_{random.randint(1, n_merchants):04d}" for _ in range(n_rows)],
        "amount": [random.uniform(10, 1000) for _ in range(n_rows)],
        "quantity": [random.randint(1, 20) for _ in range(n_rows)],
        "discount": [random.uniform(0, 0.3) for _ in range(n_rows)],
    }

    # Target based on amount and quantity
    target = []
    for i in range(n_rows):
        y = data["amount"][i] * data["quantity"][i] * (1 - data["discount"][i])
        y += random.gauss(0, 10)
        target.append(y)

    data["target"] = target
    return pa.table(data)


def generate_stress_test(n_rows: int = 1_000_000) -> pa.Table:
    """Generate large stress test dataset (1M rows)."""
    print(f"Generating stress_test.parquet ({n_rows:,} rows)...")

    categories = ["cat_a", "cat_b", "cat_c"]

    # Use list comprehensions for speed
    data = {
        "f0": [random.gauss(0, 1) for _ in range(n_rows)],
        "f1": [random.gauss(10, 5) for _ in range(n_rows)],
        "f2": [random.gauss(-5, 2) for _ in range(n_rows)],
        "f3": [random.uniform(0, 100) for _ in range(n_rows)],
        "f4": [random.uniform(-50, 50) for _ in range(n_rows)],
        "cat": [random.choice(categories) for _ in range(n_rows)],
    }

    # Simple target
    target = [
        data["f0"][i] * 2 + data["f1"][i] * 0.5 + random.gauss(0, 0.5)
        for i in range(n_rows)
    ]

    data["target"] = target
    return pa.table(data)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic parquet samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples/synthetic"),
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Generate smaller files for quick testing (10K rows instead of 100K)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # Adjust row counts
    base_rows = 10_000 if args.small else 100_000
    stress_rows = 100_000 if args.small else 1_000_000

    # Generate and save each dataset
    datasets = [
        ("large_regression.parquet", generate_large_regression(base_rows)),
        ("large_mixed.parquet", generate_large_mixed(base_rows)),
        ("large_dirty.parquet", generate_large_dirty(base_rows)),
        ("large_high_cardinality.parquet", generate_large_high_cardinality(base_rows)),
        ("stress_test.parquet", generate_stress_test(stress_rows)),
    ]

    for filename, table in datasets:
        output_path = args.output_dir / filename
        pq.write_table(table, output_path, compression="snappy")
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  -> Saved {output_path} ({file_size:.2f} MB)")

    print("\nDone! Generated files:")
    for filename, _ in datasets:
        print(f"  - {args.output_dir / filename}")


if __name__ == "__main__":
    main()
