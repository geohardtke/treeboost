// Multi-seed ensemble
pub const DEFAULT_N_SEEDS: usize = 5;
pub const DEFAULT_N_FOLDS: usize = 5;

// Hill climbing selection
pub const DEFAULT_MAX_MODELS: usize = 0;
pub const DEFAULT_MIN_IMPROVEMENT: f32 = 1e-6;
pub const DEFAULT_PATIENCE: usize = 5;

// Ridge stacking
pub const DEFAULT_STACKING_ALPHA: f32 = 10.0;
pub const DEFAULT_RANK_TRANSFORM: bool = false;
pub const DEFAULT_FIT_INTERCEPT: bool = true;
pub const DEFAULT_MIN_WEIGHT: f32 = 0.0;
