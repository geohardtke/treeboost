//! Ensemble learning module
//!
//! Provides multi-seed training, hill climbing selection, and Ridge stacking
//! for building robust ensemble models from multiple GBDT base learners.
//!
//! # Overview
//!
//! This module implements a two-stage ensemble approach:
//!
//! 1. **Multi-seed training**: Train multiple models with different random seeds to reduce variance
//! 2. **Hill climbing selection**: Greedily select models that improve CV score
//! 3. **Ridge stacking**: Combine selected models using regularized linear regression
//!
//! # Example
//!
//! ```ignore
//! use treeboost::ensemble::{EnsembleBuilder, MultiSeedConfig, StackingConfig};
//! use treeboost::{GBDTConfig, BinnedDataset};
//!
//! let config = GBDTConfig::new().with_num_rounds(100);
//!
//! let ensemble = EnsembleBuilder::new(config)
//!     .with_n_seeds(5)
//!     .with_ridge_alpha(10.0)
//!     .with_rank_transform(true)
//!     .build(&dataset)?;
//!
//! let predictions = ensemble.predict(&test_data);
//! ```

mod ltt;
mod model;
mod multi_seed;
mod selection;
mod stacking;
mod traits;

pub use ltt::{LttEnsemble, LttEnsembleStats};
pub use model::{EnsembleBuilder, EnsembleStats, StackedEnsemble};
pub use multi_seed::TrainedMember;
pub use multi_seed::{MultiSeedConfig, MultiSeedTrainer};
pub use selection::{HillClimbingSelector, SelectionConfig};
pub use stacking::{RidgeStacker, SimpleAverageStacker, StackingConfig};
pub use traits::{EnsembleMember, Stacker};
