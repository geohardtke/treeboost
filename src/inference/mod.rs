//! Inference module
//!
//! Provides prediction with Split Conformal Prediction intervals

mod conformal;
mod predict;

pub use conformal::ConformalPredictor;
pub use predict::Prediction;
