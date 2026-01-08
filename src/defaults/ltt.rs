// R2 thresholds
pub const STRONG_LINEAR_R2: f32 = 0.6;
pub const WEAK_LINEAR_R2: f32 = 0.3;

// Validation
pub const DEFAULT_LTT_VAL_RATIO: f32 = 0.2;

// Shrinkage
pub const HIGH_SHRINKAGE_MIN: f32 = 0.5;
pub const LOW_SHRINKAGE_MAX: f32 = 0.7;
pub const DEFAULT_LTT_SHRINKAGE: f32 = 0.7;

// Variance scoring
pub const HIGH_VARIANCE_THRESHOLD: f32 = 1.0;
pub const DEPTH_WEIGHT_HIGH_VAR: f32 = 0.15;
pub const ROUNDS_WEIGHT_HIGH_VAR: f32 = 0.001;
pub const LR_PENALTY_HIGH_VAR: f32 = 0.5;
pub const DEPTH_WEIGHT_LOW_VAR: f32 = 0.1;
pub const LR_WEIGHT_LOW_VAR: f32 = 1.0;
pub const ROUNDS_PENALTY_LOW_VAR: f32 = 0.0005;
pub const MAX_DEPTH_THRESHOLD: u32 = 8;
pub const MIN_LR_THRESHOLD: f32 = 0.02;
pub const EXTREME_CONFIG_PENALTY: f32 = 0.1;

// Shrinkage probe (tiny tree fit for selection)
pub const SHRINKAGE_PROBE_BINS: usize = 64;
pub const SHRINKAGE_PROBE_ROUNDS: usize = 200;
pub const SHRINKAGE_PROBE_DEPTH: usize = 8;
pub const SHRINKAGE_PROBE_LR: f32 = 0.05;
pub const SHRINKAGE_PROBE_MIN_SAMPLES_LEAF: usize = 5;

// Grid search
pub const DEFAULT_LAMBDA_GRID: &[f32] = &[0.01, 0.1, 1.0, 10.0];
pub const DEFAULT_L1_RATIO_GRID: &[f32] = &[0.0, 0.5, 1.0];
pub const DEFAULT_MAX_DEPTH_GRID: &[u32] = &[4, 6, 8];
pub const DEFAULT_LR_GRID: &[f32] = &[0.05, 0.1, 0.15];
pub const DEFAULT_ROUNDS_GRID: &[u32] = &[300, 500, 800];
pub const DEFAULT_SHRINKAGE_GRID: &[f32] = &[
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 1.0,
];
pub const DEFAULT_EXTRAPOLATION_DAMPING_GRID: &[f32] = &[0.0, 0.1, 0.2];
pub const QUICK_LAMBDA_GRID: &[f32] = &[0.1, 1.0];
pub const THOROUGH_LAMBDA_GRID: &[f32] = &[0.01, 0.1, 0.5, 1.0, 10.0];
pub const QUICK_L1_RATIO_GRID: &[f32] = &[0.0, 1.0];
pub const THOROUGH_L1_RATIO_GRID: &[f32] = &[0.0, 0.25, 0.5, 0.75, 1.0];
pub const QUICK_MAX_DEPTH_GRID: &[u32] = &[4, 6];
pub const THOROUGH_MAX_DEPTH_GRID: &[u32] = &[3, 4, 5, 6, 7, 8];
pub const QUICK_LR_GRID: &[f32] = &[0.05, 0.1];
pub const THOROUGH_LR_GRID: &[f32] = &[0.01, 0.03, 0.05, 0.1, 0.15, 0.2];
pub const QUICK_ROUNDS_GRID: &[u32] = &[300, 500];
pub const THOROUGH_ROUNDS_GRID: &[u32] = &[200, 400, 600, 800, 1000];
pub const QUICK_SHRINKAGE_GRID: &[f32] = DEFAULT_SHRINKAGE_GRID;
pub const THOROUGH_SHRINKAGE_GRID: &[f32] = DEFAULT_SHRINKAGE_GRID;
pub const QUICK_EXTRAPOLATION_DAMPING_GRID: &[f32] = &[0.0];
pub const THOROUGH_EXTRAPOLATION_DAMPING_GRID: &[f32] = &[0.0, 0.1, 0.2, 0.3];
