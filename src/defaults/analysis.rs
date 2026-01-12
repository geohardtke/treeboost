// Standard
pub const DEFAULT_MAX_SAMPLE_ROWS: usize = 20_000;
pub const DEFAULT_LINEAR_MAX_ITER: usize = 200;
pub const DEFAULT_TREE_MAX_DEPTH: usize = 6;
pub const DEFAULT_TOP_FEATURES: usize = 50;
pub const DEFAULT_ANALYSIS_SEED: u64 = 42;

/// Minimum cardinality threshold for categorical columns to be considered date/time columns.
///
/// Categorical columns with cardinality > this threshold are likely to be datetime columns
/// stored as strings (e.g., "2024-01-15", "2024-01-16", ...) rather than categorical
/// time periods (e.g., "morning", "afternoon", "evening" with cardinality = 3).
///
/// Used by panel data detection to distinguish between:
/// - High cardinality (>10): Likely datetime column (timestamps as strings)
/// - Low cardinality (≤10): Likely categorical period (time_of_day, season, etc.)
pub const MIN_DATE_CARDINALITY: usize = 10;

// Fast/Thorough presets
pub const FAST_MAX_SAMPLE_ROWS: usize = 5_000;
pub const FAST_LINEAR_MAX_ITER: usize = 20;
pub const FAST_TREE_MAX_DEPTH: usize = 3;
pub const FAST_TOP_FEATURES: usize = 20;
pub const THOROUGH_MAX_SAMPLE_ROWS: usize = 50_000;
pub const THOROUGH_LINEAR_MAX_ITER: usize = 100;
pub const THOROUGH_TREE_MAX_DEPTH: usize = 5;
pub const THOROUGH_TOP_FEATURES: usize = 100;
