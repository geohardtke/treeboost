//! Vector tree structure for multi-output learning
//!
//! Provides tree types that support multiple output dimensions (labels/targets).
//! Each leaf node stores a vector of values instead of a single scalar.
//!
//! # Memory Layout
//!
//! Leaf values are stored as flat vectors for efficient access:
//! ```text
//! leaf.values = [value_0, value_1, ..., value_K]
//! ```

use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// Node type for vector trees: internal (split) or leaf (vector values)
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub enum VectorNodeType {
    /// Internal node with split
    Internal {
        /// Feature index to split on
        feature_idx: usize,
        /// Bin threshold (samples with bin <= threshold go left)
        bin_threshold: u8,
        /// Actual split value (for raw prediction without binning)
        /// Samples with value <= split_value go left
        split_value: f64,
        /// Left child node index
        left_child: usize,
        /// Right child node index
        right_child: usize,
    },
    /// Leaf node with vector of prediction values
    Leaf {
        /// Prediction values (one per output)
        values: Vec<f32>,
    },
}

/// Tree node for vector outputs
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct VectorNode {
    /// Node type (internal or leaf)
    pub node_type: VectorNodeType,
    /// Depth in tree (root = 0)
    pub depth: usize,
    /// Number of samples in this node
    pub num_samples: usize,
    /// Sum of gradients per output
    sum_gradients: Vec<f32>,
    /// Sum of hessians per output
    sum_hessians: Vec<f32>,
}

impl VectorNode {
    /// Create a new leaf node with vector values
    pub fn leaf(
        values: Vec<f32>,
        depth: usize,
        num_samples: usize,
        sum_gradients: Vec<f32>,
        sum_hessians: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(values.len(), sum_gradients.len());
        debug_assert_eq!(values.len(), sum_hessians.len());

        Self {
            node_type: VectorNodeType::Leaf { values },
            depth,
            num_samples,
            sum_gradients,
            sum_hessians,
        }
    }

    /// Create a new internal node
    ///
    /// All parameters represent distinct, fundamental node properties.
    #[allow(clippy::too_many_arguments)]
    pub fn internal(
        feature_idx: usize,
        bin_threshold: u8,
        split_value: f64,
        left_child: usize,
        right_child: usize,
        depth: usize,
        num_samples: usize,
        sum_gradients: Vec<f32>,
        sum_hessians: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(sum_gradients.len(), sum_hessians.len());

        Self {
            node_type: VectorNodeType::Internal {
                feature_idx,
                bin_threshold,
                split_value,
                left_child,
                right_child,
            },
            depth,
            num_samples,
            sum_gradients,
            sum_hessians,
        }
    }

    /// Number of outputs
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.sum_gradients.len()
    }

    /// Check if this is a leaf node
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self.node_type, VectorNodeType::Leaf { .. })
    }

    /// Get leaf values, returning None if this is not a leaf node
    #[inline]
    pub fn leaf_values(&self) -> Option<&[f32]> {
        match &self.node_type {
            VectorNodeType::Leaf { values } => Some(values),
            VectorNodeType::Internal { .. } => None,
        }
    }

    /// Get split info, returning None if this is not an internal node
    /// Returns (feature_idx, bin_threshold, split_value, left_child, right_child)
    #[inline]
    pub fn split_info(&self) -> Option<(usize, u8, f64, usize, usize)> {
        match &self.node_type {
            VectorNodeType::Internal {
                feature_idx,
                bin_threshold,
                split_value,
                left_child,
                right_child,
            } => Some((
                *feature_idx,
                *bin_threshold,
                *split_value,
                *left_child,
                *right_child,
            )),
            VectorNodeType::Leaf { .. } => None,
        }
    }

    /// Get sum of gradients
    #[inline]
    pub fn sum_gradients(&self) -> &[f32] {
        &self.sum_gradients
    }

    /// Get sum of hessians
    #[inline]
    pub fn sum_hessians(&self) -> &[f32] {
        &self.sum_hessians
    }

    /// Compute optimal leaf weights using Newton step for all outputs
    ///
    /// weight_k = -sum_gradients_k / (sum_hessians_k + lambda)
    pub fn compute_leaf_weights(sum_grads: &[f32], sum_hess: &[f32], lambda: f32) -> Vec<f32> {
        debug_assert_eq!(sum_grads.len(), sum_hess.len());

        sum_grads
            .iter()
            .zip(sum_hess.iter())
            .map(|(g, h)| -g / (h + lambda))
            .collect()
    }
}

/// Decision tree with vector outputs
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct VectorTree {
    /// Nodes in the tree (index 0 is root)
    nodes: Vec<VectorNode>,
    /// Number of outputs (cached for efficiency)
    num_outputs: usize,
}

impl VectorTree {
    /// Create a new tree with a single root leaf
    pub fn new(
        root_values: Vec<f32>,
        num_samples: usize,
        sum_gradients: Vec<f32>,
        sum_hessians: Vec<f32>,
    ) -> Self {
        let num_outputs = root_values.len();
        Self {
            nodes: vec![VectorNode::leaf(root_values, 0, num_samples, sum_gradients, sum_hessians)],
            num_outputs,
        }
    }

    /// Create from a vector of nodes
    pub fn from_nodes(nodes: Vec<VectorNode>, num_outputs: usize) -> Self {
        debug_assert!(!nodes.is_empty());
        debug_assert!(nodes.iter().all(|n| n.num_outputs() == num_outputs));

        Self { nodes, num_outputs }
    }

    /// Number of outputs
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Get the root node
    #[inline]
    pub fn root(&self) -> &VectorNode {
        &self.nodes[0]
    }

    /// Get a node by index
    #[inline]
    pub fn get_node(&self, idx: usize) -> &VectorNode {
        &self.nodes[idx]
    }

    /// Get mutable node by index
    #[inline]
    pub fn get_node_mut(&mut self, idx: usize) -> &mut VectorNode {
        &mut self.nodes[idx]
    }

    /// Add a node and return its index
    pub fn add_node(&mut self, node: VectorNode) -> usize {
        debug_assert_eq!(node.num_outputs(), self.num_outputs);
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Number of nodes
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaves
    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Maximum depth of the tree
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Predict for a single sample using binned features
    ///
    /// # Arguments
    /// * `get_bin` - Function to get bin value for a feature: `|feature_idx| -> u8`
    ///
    /// # Returns
    /// Vector of predictions (one per output)
    pub fn predict<F>(&self, get_bin: F) -> Vec<f32>
    where
        F: Fn(usize) -> u8,
    {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];
            match &node.node_type {
                VectorNodeType::Leaf { values } => return values.clone(),
                VectorNodeType::Internal {
                    feature_idx,
                    bin_threshold,
                    left_child,
                    right_child,
                    ..
                } => {
                    let bin = get_bin(*feature_idx);
                    node_idx = if bin <= *bin_threshold {
                        *left_child
                    } else {
                        *right_child
                    };
                }
            }
        }
    }

    /// Predict for a single sample using raw feature values
    ///
    /// # Arguments
    /// * `get_value` - Function to get feature value: `|feature_idx| -> f64`
    ///
    /// # Returns
    /// Vector of predictions (one per output)
    #[inline]
    pub fn predict_raw<F>(&self, get_value: F) -> Vec<f32>
    where
        F: Fn(usize) -> f64,
    {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];
            match &node.node_type {
                VectorNodeType::Leaf { values } => return values.clone(),
                VectorNodeType::Internal {
                    feature_idx,
                    split_value,
                    left_child,
                    right_child,
                    ..
                } => {
                    let value = get_value(*feature_idx);
                    node_idx = if value <= *split_value {
                        *left_child
                    } else {
                        *right_child
                    };
                }
            }
        }
    }

    /// Batch predict: add this tree's contribution to all predictions
    ///
    /// # Arguments
    /// * `get_bin` - Function to get bin for a sample and feature: `|sample_idx, feature_idx| -> u8`
    /// * `num_samples` - Number of samples to predict
    /// * `predictions` - Mutable slice of predictions, layout: `[s0_o0, s0_o1, ..., s1_o0, s1_o1, ...]`
    pub fn predict_batch_add<F>(
        &self,
        get_bin: F,
        num_samples: usize,
        predictions: &mut [f32],
    ) where
        F: Fn(usize, usize) -> u8 + Sync,
    {
        debug_assert_eq!(predictions.len(), num_samples * self.num_outputs);

        // Parallel threshold
        const PARALLEL_THRESHOLD: usize = 10_000;

        if num_samples >= PARALLEL_THRESHOLD {
            // Parallel: process chunks of predictions
            predictions
                .par_chunks_mut(self.num_outputs)
                .enumerate()
                .for_each(|(sample_idx, pred_slice)| {
                    let leaf_values = self.predict_to_leaf(|f| get_bin(sample_idx, f));
                    for (pred, leaf_val) in pred_slice.iter_mut().zip(leaf_values.iter()) {
                        *pred += *leaf_val;
                    }
                });
        } else {
            // Sequential
            for sample_idx in 0..num_samples {
                let offset = sample_idx * self.num_outputs;
                let leaf_values = self.predict_to_leaf(|f| get_bin(sample_idx, f));
                for (k, leaf_val) in leaf_values.iter().enumerate() {
                    predictions[offset + k] += *leaf_val;
                }
            }
        }
    }

    /// Batch predict using raw feature values
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    /// * `num_features` - Number of features per row
    /// * `predictions` - Mutable slice of predictions, layout: `[s0_o0, s0_o1, ..., s1_o0, s1_o1, ...]`
    #[inline]
    pub fn predict_batch_add_raw(
        &self,
        features: &[f64],
        num_features: usize,
        predictions: &mut [f32],
    ) {
        let num_samples = predictions.len() / self.num_outputs;
        debug_assert_eq!(features.len(), num_samples * num_features);
        debug_assert_eq!(predictions.len(), num_samples * self.num_outputs);

        const PARALLEL_THRESHOLD: usize = 10_000;

        if num_samples >= PARALLEL_THRESHOLD {
            predictions
                .par_chunks_mut(self.num_outputs)
                .enumerate()
                .for_each(|(sample_idx, pred_slice)| {
                    let row_offset = sample_idx * num_features;
                    let leaf_values = self.predict_to_leaf_raw(|f| features[row_offset + f]);
                    for (pred, leaf_val) in pred_slice.iter_mut().zip(leaf_values.iter()) {
                        *pred += *leaf_val;
                    }
                });
        } else {
            for sample_idx in 0..num_samples {
                let row_offset = sample_idx * num_features;
                let offset = sample_idx * self.num_outputs;
                let leaf_values = self.predict_to_leaf_raw(|f| features[row_offset + f]);
                for (k, leaf_val) in leaf_values.iter().enumerate() {
                    predictions[offset + k] += *leaf_val;
                }
            }
        }
    }

    /// Inner prediction logic - returns reference to leaf values
    #[inline]
    fn predict_to_leaf<F>(&self, get_bin: F) -> &[f32]
    where
        F: Fn(usize) -> u8,
    {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];
            match &node.node_type {
                VectorNodeType::Leaf { values } => return values,
                VectorNodeType::Internal {
                    feature_idx,
                    bin_threshold,
                    left_child,
                    right_child,
                    ..
                } => {
                    let bin = get_bin(*feature_idx);
                    node_idx = if bin <= *bin_threshold {
                        *left_child
                    } else {
                        *right_child
                    };
                }
            }
        }
    }

    /// Inner raw prediction logic - returns reference to leaf values
    #[inline]
    fn predict_to_leaf_raw<F>(&self, get_value: F) -> &[f32]
    where
        F: Fn(usize) -> f64,
    {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];
            match &node.node_type {
                VectorNodeType::Leaf { values } => return values,
                VectorNodeType::Internal {
                    feature_idx,
                    split_value,
                    left_child,
                    right_child,
                    ..
                } => {
                    let value = get_value(*feature_idx);
                    node_idx = if value <= *split_value {
                        *left_child
                    } else {
                        *right_child
                    };
                }
            }
        }
    }

    /// Get all leaf nodes
    pub fn leaves(&self) -> impl Iterator<Item = (usize, &VectorNode)> {
        self.nodes.iter().enumerate().filter(|(_, n)| n.is_leaf())
    }

    /// Get all internal nodes
    pub fn internal_nodes(&self) -> impl Iterator<Item = (usize, &VectorNode)> {
        self.nodes.iter().enumerate().filter(|(_, n)| !n.is_leaf())
    }

    /// Get nodes as slice
    pub fn nodes(&self) -> &[VectorNode] {
        &self.nodes
    }

    /// Get mutable nodes
    pub fn nodes_mut(&mut self) -> &mut Vec<VectorNode> {
        &mut self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_node_leaf() {
        let node = VectorNode::leaf(
            vec![1.0, 2.0, 3.0],
            2,
            100,
            vec![10.0, 20.0, 30.0],
            vec![5.0, 10.0, 15.0],
        );

        assert!(node.is_leaf());
        assert_eq!(node.num_outputs(), 3);
        assert_eq!(node.leaf_values(), Some(&[1.0f32, 2.0, 3.0][..]));
    }

    #[test]
    fn test_vector_node_internal() {
        let node = VectorNode::internal(
            5,
            128,
            5.5,
            1,
            2,
            1,
            200,
            vec![15.0, 25.0],
            vec![8.0, 12.0],
        );

        assert!(!node.is_leaf());
        assert_eq!(node.num_outputs(), 2);
        let info = node.split_info();
        assert!(info.is_some());
    }

    #[test]
    fn test_vector_tree_new() {
        let tree = VectorTree::new(
            vec![0.5, 1.0],
            100,
            vec![0.0, 0.0],
            vec![100.0, 100.0],
        );

        assert_eq!(tree.num_nodes(), 1);
        assert_eq!(tree.num_outputs(), 2);
        assert!(tree.root().is_leaf());
    }

    #[test]
    fn test_leaf_weight_computation() {
        let weights = VectorNode::compute_leaf_weights(
            &[-10.0, -20.0],
            &[20.0, 40.0],
            0.0, // No regularization
        );

        assert!((weights[0] - 0.5).abs() < 1e-6);
        assert!((weights[1] - 0.5).abs() < 1e-6);
    }
}
