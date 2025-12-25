//! Decision tree structure

use crate::dataset::BinnedDataset;
use crate::tree::{Node, NodeType};
use rkyv::{Archive, Deserialize, Serialize};

/// Decision tree
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Tree {
    /// Nodes in the tree (index 0 is root)
    nodes: Vec<Node>,
}

impl Tree {
    /// Create a new tree with a single root leaf
    pub fn new(root_value: f32, num_samples: usize, sum_g: f32, sum_h: f32) -> Self {
        Self {
            nodes: vec![Node::leaf(root_value, 0, num_samples, sum_g, sum_h)],
        }
    }

    /// Create from a vector of nodes
    pub fn from_nodes(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }

    /// Get the root node
    #[inline]
    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    /// Get a node by index
    #[inline]
    pub fn get_node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }

    /// Get mutable node by index
    #[inline]
    pub fn get_node_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    /// Add a node and return its index
    pub fn add_node(&mut self, node: Node) -> usize {
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
    pub fn predict<F>(&self, get_bin: F) -> f32
    where
        F: Fn(usize) -> u8,
    {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];
            match node.node_type {
                NodeType::Leaf { value } => return value,
                NodeType::Internal {
                    feature_idx,
                    bin_threshold,
                    left_child,
                    right_child,
                } => {
                    let bin = get_bin(feature_idx);
                    node_idx = if bin <= bin_threshold {
                        left_child
                    } else {
                        right_child
                    };
                }
            }
        }
    }

    /// Predict for a single row in a dataset
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        self.predict(|feature_idx| dataset.get_bin(row_idx, feature_idx))
    }

    /// Predict for all rows in a dataset
    pub fn predict_all(&self, dataset: &BinnedDataset) -> Vec<f32> {
        (0..dataset.num_rows())
            .map(|row_idx| self.predict_row(dataset, row_idx))
            .collect()
    }

    /// Get all leaf nodes
    pub fn leaves(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf())
    }

    /// Get all internal nodes
    pub fn internal_nodes(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.is_leaf())
    }

    /// Get nodes as slice
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let tree = Tree::new(0.5, 100, 10.0, 20.0);

        assert_eq!(tree.num_nodes(), 1);
        assert_eq!(tree.num_leaves(), 1);
        assert!(tree.root().is_leaf());
        assert_eq!(tree.root().leaf_value(), Some(0.5));
    }

    #[test]
    fn test_tree_prediction() {
        // Create a simple tree:
        //        [f0 <= 5]
        //        /        \
        //    leaf=1.0   [f1 <= 10]
        //               /        \
        //           leaf=2.0   leaf=3.0

        let tree = Tree::from_nodes(vec![
            Node::internal(0, 5, 1, 2, 0, 100, 0.0, 100.0),
            Node::leaf(1.0, 1, 50, 0.0, 50.0),
            Node::internal(1, 10, 3, 4, 1, 50, 0.0, 50.0),
            Node::leaf(2.0, 2, 25, 0.0, 25.0),
            Node::leaf(3.0, 2, 25, 0.0, 25.0),
        ]);

        // Test predictions
        // f0=3 (<=5): go left -> leaf=1.0
        assert_eq!(tree.predict(|f| if f == 0 { 3 } else { 0 }), 1.0);

        // f0=7 (>5): go right, f1=5 (<=10): go left -> leaf=2.0
        assert_eq!(tree.predict(|f| if f == 0 { 7 } else { 5 }), 2.0);

        // f0=7 (>5): go right, f1=15 (>10): go right -> leaf=3.0
        assert_eq!(tree.predict(|f| if f == 0 { 7 } else { 15 }), 3.0);
    }

    #[test]
    fn test_tree_stats() {
        let tree = Tree::from_nodes(vec![
            Node::internal(0, 5, 1, 2, 0, 100, 0.0, 100.0),
            Node::leaf(1.0, 1, 50, 0.0, 50.0),
            Node::internal(1, 10, 3, 4, 1, 50, 0.0, 50.0),
            Node::leaf(2.0, 2, 25, 0.0, 25.0),
            Node::leaf(3.0, 2, 25, 0.0, 25.0),
        ]);

        assert_eq!(tree.num_nodes(), 5);
        assert_eq!(tree.num_leaves(), 3);
        assert_eq!(tree.max_depth(), 2);
    }
}
