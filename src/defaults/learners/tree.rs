// Structure
pub const DEFAULT_MAX_DEPTH: usize = 6;
pub const DEFAULT_MAX_LEAVES: usize = 31;
pub const SHALLOW_MAX_DEPTH: usize = 4;
pub const DEEP_MAX_DEPTH: usize = 10;

// Regularization
pub const DEFAULT_TREE_LAMBDA: f32 = 1.0;
pub const REGULARIZED_TREE_LAMBDA: f32 = 2.0;
pub const EXPRESSIVE_TREE_LAMBDA: f32 = 0.0;
pub const DEFAULT_ENTROPY_WEIGHT: f32 = 0.0;
pub const REGULARIZED_ENTROPY_WEIGHT: f32 = 0.3;
pub const DEFAULT_MIN_GAIN: f32 = 0.0;

// Samples
pub const DEFAULT_MIN_SAMPLES_LEAF: usize = 1;
pub const DEFAULT_MIN_HESSIAN_LEAF: f32 = 1.0;
pub const DEFAULT_COLSAMPLE: f32 = 1.0;
pub const ROBUST_COLSAMPLE: f32 = 0.8; // For noise-robust training (feature bagging)

// Step size
pub const DEFAULT_LEARNING_RATE: f32 = 0.1;
