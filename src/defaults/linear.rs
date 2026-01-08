// Regularization
pub const DEFAULT_LAMBDA: f32 = 1.0;
pub const MIN_LAMBDA: f32 = 1e-6;
pub const DEFAULT_L1_RATIO: f32 = 0.0;
pub const LASSO_L1_RATIO: f32 = 1.0;
pub const ELASTIC_NET_L1_RATIO: f32 = 0.5;

// Boosting
pub const DEFAULT_SHRINKAGE_FACTOR: f32 = 0.3;
pub const AGGRESSIVE_SHRINKAGE: f32 = 0.7;
pub const CONSERVATIVE_SHRINKAGE: f32 = 0.1;

// Optimization
pub const DEFAULT_MAX_ITER: usize = 100;
pub const DEFAULT_TOL: f32 = 1e-6;
pub const DEFAULT_MAX_WEIGHT: f32 = 100.0;
pub const DEFAULT_EXTRAPOLATION_DAMPING: f32 = 0.0;
pub const SAFE_EXTRAPOLATION_DAMPING: f32 = 0.1;
