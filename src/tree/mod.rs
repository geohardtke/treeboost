//! Decision tree structures and algorithms
//!
//! Provides tree building with:
//! - Shannon Entropy regularized split finding
//! - Best-First (Leaf-wise) growth strategy
//! - Histogram Subtraction Trick integration

mod grow;
mod node;
mod split;
mod tree;

pub use grow::TreeGrower;
pub use node::{Node, NodeType};
pub use split::{InteractionConstraints, MonotonicConstraint, SplitFinder, SplitInfo};
pub use tree::Tree;
