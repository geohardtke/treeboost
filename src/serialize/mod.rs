//! Serialization module
//!
//! Provides model serialization in multiple formats:
//! - rkyv: Zero-copy deserialization, fastest loading (recommended for production)
//! - bincode: Compact binary format, serde-based
//! - trb: Incremental format supporting model updates and crash recovery

mod bincode_io;
pub mod rkyv_io;
mod trb;

pub use bincode_io::{load_model_bincode, save_model_bincode};
pub use rkyv_io::{
    deserialize_universal_model, load_model, load_universal_model, save_model,
    save_universal_model, serialize_universal_model,
};
pub use trb::{
    open_for_append, TrbHeader, TrbReader, TrbSegment, TrbUpdateHeader, TrbWriter, UpdateType,
    FORMAT_VERSION, TRB_MAGIC,
};
