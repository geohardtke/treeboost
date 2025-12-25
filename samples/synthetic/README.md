# Synthetic Test Samples

These files contain synthetic data designed for testing TreeBoost edge cases.

## Generating Large Parquet Files

The CSV files are tracked in git. For large parquet test files, run:

```bash
# Install pyarrow if needed
pip install pyarrow

# Generate all parquet files (100K-1M rows)
python scripts/generate_samples.py

# Or generate smaller files for quick testing (10K-100K rows)
python scripts/generate_samples.py --small
```

This creates:
- `large_regression.parquet` (100K rows, numeric only)
- `large_mixed.parquet` (100K rows, mixed types)
- `large_dirty.parquet` (100K rows, missing/outliers/rare categories)
- `large_high_cardinality.parquet` (100K rows, 10K+ unique categories)
- `stress_test.parquet` (1M rows, for performance testing)

## CSV Files (tracked in git)

### basic_regression.csv
- 50 rows, 5 numeric features + target
- Clean numeric-only data
- Tests: basic training, prediction, serialization

### mixed_types.csv
- 40 rows, numeric + categorical columns
- Columns: price, sqft, bedrooms, bathrooms, neighborhood, property_type, has_pool, year_built
- Categories: downtown/suburbs/rural, apartment/house/condo/townhouse, true/false
- Tests: mixed type handling, categorical encoding

### dirty_data.csv
- 50 rows with intentional data quality issues
- Missing values (empty cells)
- Outliers: negative values, extreme values (50000, 99999, 88888)
- Rare categories: rare_typo_1, rare_typo_2, rare_typo_3, rare_once, rare_only_two
- Frequent categories: frequent_a, frequent_b
- Tests: CMS filtering, outlier handling, missing value handling

### high_cardinality.csv
- 110 rows with high-cardinality categorical columns
- user_id: 110 unique values (user_001 to user_110)
- product_id: 100 unique values (prod_001 to prod_100)
- region: 50 unique values (region_01 to region_50)
- Tests: CMS filter with many rare categories, target encoding with high cardinality

### edge_cases.csv
- 30 rows testing edge conditions
- constant_col: single value column (always_same)
- near_constant: 27 "mostly_a", 2 "rare_b", 1 "rare_c"
- zero_variance_num: all values are 5.0
- extreme_small: values around 0.000001-0.000009
- extreme_large: values in billions (111111111 to 999999999)
- duplicate_rows: groups of identical feature patterns
- sparse_values: many missing values
- binary_feature: 0/1 only
- Tests: constant column handling, near-zero variance, extreme value binning, duplicate detection
