pub const DEFAULT_VALIDATION_RATIO: f32 = 0.2;
pub const LINEAR_SIGNAL_THRESHOLD: f32 = 0.3;
pub const AUTO_FEATURES_DEFAULT_COUNT: usize = 50;

// Tree tuner ranges
pub const QUICK_DEPTH_RANGE: (usize, usize) = (3, 6);
pub const STANDARD_DEPTH_RANGE: (usize, usize) = (3, 8);
pub const THOROUGH_DEPTH_RANGE: (usize, usize) = (3, 10);
pub const QUICK_LR_RANGE: (f32, f32) = (0.05, 0.15);
pub const STANDARD_LR_RANGE: (f32, f32) = (0.01, 0.15);
