//! Decision tree structures and algorithms
//!
//! Provides tree building with:
//! - Shannon Entropy regularized split finding
//! - Best-First (Leaf-wise) growth strategy
//! - Histogram Subtraction Trick integration
//! - Vector split finding for multi-output learning
//! - Vector trees with multi-output leaf values

mod grow;
mod node;
mod split;
#[allow(clippy::module_inception)]
mod tree;
mod vector_split;
mod vector_tree;

pub use grow::TreeGrower;
pub use node::{Node, NodeType};
pub use split::{InteractionConstraints, MonotonicConstraint, SplitFinder, SplitInfo};
pub use tree::Tree;
pub use vector_split::{VectorSplitFinder, VectorSplitInfo};
pub use vector_tree::{VectorNode, VectorNodeType, VectorTree};
