//! Re-export of histogram builder for scalar backend.
//!
//! The actual histogram building logic is in `crate::histogram`.
//! This module re-exports it for use within the scalar backend.

pub use crate::histogram::HistogramBuilder;
