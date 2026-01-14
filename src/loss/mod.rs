//! Loss functions for GBDT training
//!
//! Provides objective functions with gradient and hessian computation:
//! - `MseLoss`: Mean Squared Error (standard, but sensitive to outliers)
//! - `PseudoHuberLoss`: Robust loss that transitions smoothly from L2 to L1
//! - `BinaryLogLoss`: Binary cross-entropy for binary classification
//! - `MultiClassLogLoss`: Softmax cross-entropy for multi-class classification
//! - `MultiLabelLogLoss`: Independent binary cross-entropy for multi-label classification
//! - `MultiLabelFocalLoss`: Focal loss for imbalanced multi-label classification
//! - `BetaLoss`: Beta distribution loss for bounded continuous regression in (0, 1)
//!
//! Also provides activation functions:
//! - `sigmoid`: Numerically stable sigmoid for binary classification
//! - `softmax`: Numerically stable softmax for multi-class classification

mod activation;
mod beta;
mod focal;
mod huber;
mod logloss;
mod mse;
mod multilabel;
mod softmax;
mod traits;

pub use activation::sigmoid;
pub use beta::BetaLoss;
pub use focal::MultiLabelFocalLoss;
pub use huber::PseudoHuberLoss;
pub use logloss::BinaryLogLoss;
pub use mse::MseLoss;
pub use multilabel::MultiLabelLogLoss;
pub use softmax::{softmax, MultiClassLogLoss};
pub use traits::LossFunction;
