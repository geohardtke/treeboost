//! Distribution shift detection and monitoring
//!
//! Provides tools for detecting data drift between training and inference,
//! and monitoring CV vs holdout gaps during training.
//!
//! # Overview
//!
//! This module helps identify two types of distribution shift:
//!
//! 1. **Training-time**: Monitoring the gap between cross-validation and holdout
//!    metrics. Large or increasing gaps signal potential data structure issues.
//!
//! 2. **Inference-time**: Comparing feature distributions between training data
//!    and new inference data using metrics like PSI, KL divergence, and KS test.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::monitoring::{ShiftDetector, PSI};
//!
//! // Create detector from training data
//! let detector = ShiftDetector::from_dataset(&train_data)
//!     .with_metric(PSI::new(10))
//!     .with_thresholds(0.1, 0.25);
//!
//! // Check inference data for drift
//! let result = detector.check(&inference_data);
//! if result.alert == AlertLevel::Critical {
//!     println!("Critical drift: {:?}", result.drifted_features);
//! }
//! ```

mod detector;
mod metrics;
mod tracker;

pub use detector::{AlertLevel, ShiftDetector, ShiftResult};
pub use metrics::{DistributionMetric, JensenShannon, KolmogorovSmirnov, PSI};
pub use tracker::{CVHoldoutTracker, GapRecord, Trend};
