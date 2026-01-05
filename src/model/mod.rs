//! High-level model abstractions
//!
//! This module provides:
//! - `UniversalModel`: Unified boosting framework supporting multiple modes

mod universal;

pub use universal::{BoostingMode, UniversalConfig, UniversalModel};
