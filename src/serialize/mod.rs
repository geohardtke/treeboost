//! Serialization module
//!
//! Provides model serialization in multiple formats:
//! - rkyv: Zero-copy deserialization, fastest loading (recommended for production)
//! - bincode: Compact binary format, serde-based

mod bincode_io;
mod rkyv_io;

pub use bincode_io::{load_model_bincode, save_model_bincode};
pub use rkyv_io::{load_model, save_model};
