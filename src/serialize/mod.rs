//! Serialization module
//!
//! Provides zero-copy model serialization via rkyv

mod rkyv_io;

pub use rkyv_io::{load_model, save_model};
