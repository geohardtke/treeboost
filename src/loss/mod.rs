//! Loss functions for GBDT training
//!
//! Provides objective functions with gradient and hessian computation:
//! - `MseLoss`: Mean Squared Error (standard, but sensitive to outliers)
//! - `PseudoHuberLoss`: Robust loss that transitions smoothly from L2 to L1

mod huber;
mod mse;
mod traits;

pub use huber::PseudoHuberLoss;
pub use mse::MseLoss;
pub use traits::LossFunction;
