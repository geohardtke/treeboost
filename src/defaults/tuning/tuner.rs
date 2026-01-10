// Iterations
pub const DEFAULT_N_ITERATIONS: usize = 5;
pub const QUICK_N_ITERATIONS: usize = 2;
pub const THOROUGH_N_ITERATIONS: usize = 7;
pub const SMOKE_TEST_N_ITERATIONS: usize = 1;

// Rounds
pub const DEFAULT_TUNER_ROUNDS: usize = 100;
pub const QUICK_TUNER_ROUNDS: usize = 50;
pub const THOROUGH_TUNER_ROUNDS: usize = 200;

// Search
pub const DEFAULT_INITIAL_SPREAD: f32 = 1.0;
pub const DEFAULT_ZOOM_FACTOR: f32 = 0.8;

// Early Stopping
pub const DEFAULT_TUNER_EARLY_STOP: usize = 10;
pub const QUICK_TUNER_EARLY_STOP: usize = 5;
pub const THOROUGH_TUNER_EARLY_STOP: usize = 20;

// Thresholds
pub const DEFAULT_IMPROVEMENT_THRESHOLD: f32 = 0.001;
pub const QUICK_IMPROVEMENT_THRESHOLD: f32 = 0.01;
pub const THOROUGH_IMPROVEMENT_THRESHOLD: f32 = 0.0001;
pub const DEFAULT_MIN_F1_SCORE: f32 = 0.8;

// Validation
pub const DEFAULT_TUNER_VAL_RATIO: f32 = 0.2;
