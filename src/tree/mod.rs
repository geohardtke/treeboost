//! Decision tree structures and algorithms
//!
//! Provides tree building with:
//! - Shannon Entropy regularized split finding
//! - Best-First (Leaf-wise) growth strategy
//! - Histogram Subtraction Trick integration
//! - Vector split finding for multi-output learning
//! - Vector trees with multi-output leaf values
//! - Unified EnsembleTree for scalar and vector trees

mod ensemble;
mod grow;
mod node;
mod split;
#[allow(clippy::module_inception)]
mod tree;
mod vector_grow;
mod vector_split;
mod vector_tree;

pub use ensemble::EnsembleTree;
pub use grow::TreeGrower;
pub use node::{Node, NodeType};
pub use split::{InteractionConstraints, MonotonicConstraint, SplitFinder, SplitInfo};
pub use tree::Tree;
pub use vector_grow::VectorTreeGrower;
pub use vector_split::{VectorSplitFinder, VectorSplitInfo};
pub use vector_tree::{VectorNode, VectorNodeType, VectorTree};

/// Generate a column subsample mask for feature bagging.
///
/// Returns `None` if `colsample >= 1.0` (use all features).
/// Otherwise returns `Some(indices)` with a randomly selected subset.
///
/// `rng_seed` is the fully derived seed passed directly to the RNG.
/// Callers are responsible for combining tree seed + row count as needed.
pub fn generate_feature_mask(
    num_features: usize,
    colsample: f32,
    rng_seed: u64,
) -> Option<Vec<usize>> {
    if colsample >= 1.0 {
        return None;
    }
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let n_features = ((num_features as f32) * colsample).ceil().max(1.0) as usize;
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let mut all_features: Vec<usize> = (0..num_features).collect();
    all_features.shuffle(&mut rng);
    all_features.truncate(n_features);
    Some(all_features)
}
