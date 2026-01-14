//! Count-Min Sketch based category filter
//!
//! Filters rare categories to a single "unknown" bucket using
//! probabilistic counting with fixed memory.
//!
//! The Count-Min Sketch is a probabilistic data structure that provides
//! approximate frequency counts using sub-linear space. It never underestimates
//! counts but may overestimate due to hash collisions.

use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxHashSet;
use rustc_hash::FxHasher;
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};
use std::hash::{Hash, Hasher};

// ============================================================================
// Count-Min Sketch Implementation
// ============================================================================

/// Reduce two 64-bit hashes into one using Google's CityHash mixing.
#[inline(always)]
fn combine_hashes(upper: u64, lower: u64) -> u64 {
    const MUL: u64 = 0x9ddfea08eb382d69;

    let mut a = (lower ^ upper).wrapping_mul(MUL);
    a ^= a >> 47;
    let mut b = (upper ^ a).wrapping_mul(MUL);
    b ^= b >> 47;
    b.wrapping_mul(MUL)
}

/// Integer mixing function for generating independent hash functions.
#[inline(always)]
fn twang_mix64(val: u64) -> u64 {
    let mut val = (!val).wrapping_add(val << 21);
    val ^= val >> 24;
    val = val.wrapping_add(val << 3).wrapping_add(val << 8);
    val ^= val >> 14;
    val = val.wrapping_add(val << 2).wrapping_add(val << 4);
    val ^= val >> 28;
    val.wrapping_add(val << 31)
}

/// Count-Min Sketch with u64 counters.
///
/// A probabilistic data structure for approximate frequency counting.
/// Uses multiple hash functions (rows) to reduce collision probability.
///
/// # Properties
/// - Never underestimates counts
/// - May overestimate due to hash collisions
/// - Fixed memory regardless of stream size
/// - Error bounded by `eps * total_count` with probability `confidence`
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    table: Vec<u64>,
}

impl CountMinSketch {
    /// Creates a new sketch sized by target error and confidence.
    ///
    /// # Arguments
    /// * `eps` - Error tolerance (e.g., 0.01 for 1% error). Width = 2/eps.
    /// * `confidence` - Confidence level (e.g., 0.99). Depth = ceil(-log2(1-confidence)).
    ///
    /// # Panics
    /// Panics if `eps <= 0.0` or `confidence <= 0.0`.
    pub fn new(eps: f64, confidence: f64) -> Self {
        assert!(eps > 0.0, "eps must be positive");
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "confidence must be in (0, 1)"
        );

        let width = (2.0 / eps).ceil() as usize;
        let depth = (-(1.0 - confidence).log2()).ceil() as usize;

        debug_assert!(width > 0);
        debug_assert!(depth > 0);

        let table = vec![0u64; width * depth];

        Self {
            width,
            depth,
            table,
        }
    }

    /// Increment the count for a hash by 1.
    #[inline]
    pub fn inc(&mut self, hash: u64) {
        self.inc_by(hash, 1);
    }

    /// Increment the count for a hash by a specified amount.
    #[inline]
    pub fn inc_by(&mut self, hash: u64, count: u64) {
        for depth in 0..self.depth {
            let index = self.index(depth, hash);
            self.table[index] = self.table[index].saturating_add(count);
        }
    }

    /// Estimate the count for a hash (returns minimum across all rows).
    #[inline]
    pub fn estimate(&self, hash: u64) -> u64 {
        (0..self.depth)
            .map(|depth| self.table[self.index(depth, hash)])
            .min()
            .unwrap_or(0)
    }

    /// Reset all counters to zero.
    pub fn clear(&mut self) {
        self.table.fill(0);
    }

    /// Divide all counters by 2 (useful for time decay).
    pub fn halve(&mut self) {
        for c in &mut self.table {
            *c >>= 1;
        }
    }

    /// Get the table width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the table depth (number of hash functions).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.table.len() * std::mem::size_of::<u64>()
    }

    #[inline(always)]
    fn index(&self, depth: usize, hash: u64) -> usize {
        depth * self.width + (combine_hashes(twang_mix64(depth as u64), hash) as usize % self.width)
    }
}

// ============================================================================
// Category Filter (uses Count-Min Sketch)
// ============================================================================

/// Hash a string to u64 for CMS lookup.
#[inline]
fn hash_str(s: &str) -> u64 {
    let mut hasher = FxHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Category filter using Count-Min Sketch.
///
/// Uses probabilistic counting to identify rare categories
/// and map them to a single "unknown" value. This is essential
/// for handling high-cardinality categorical features with typos
/// and rare values.
///
/// # Usage
/// 1. Count all categories in a first pass using `count()` or `count_batch()`
/// 2. Call `finalize()` with all unique categories to identify frequent ones
/// 3. Use `filter()` to map rare categories to "unknown"
pub struct CategoryFilter {
    /// Count-Min Sketch for approximate frequency counting
    sketch: CountMinSketch,
    /// Minimum count threshold for a category to be kept
    min_count: u64,
    /// Known frequent categories (for exact lookup after filtering)
    frequent: FxHashSet<String>,
}

impl CategoryFilter {
    /// Create a new category filter.
    ///
    /// # Arguments
    /// * `eps` - Error tolerance (e.g., 0.001 for 0.1% error)
    /// * `confidence` - Confidence level (e.g., 0.99 for 99%)
    /// * `min_count` - Minimum frequency to keep a category
    pub fn new(eps: f64, confidence: f64, min_count: u64) -> Self {
        Self {
            sketch: CountMinSketch::new(eps, confidence),
            min_count,
            frequent: FxHashSet::default(),
        }
    }

    /// Create with default parameters suitable for high-cardinality data.
    ///
    /// Uses eps=0.001 (0.1% error), confidence=0.99, min_count=5.
    pub fn default_for_high_cardinality() -> Self {
        Self::new(0.001, 0.99, 5)
    }

    /// First pass: count a single category.
    #[inline]
    pub fn count(&mut self, category: &str) {
        self.sketch.inc(hash_str(category));
    }

    /// Count a batch of categories.
    pub fn count_batch<'a>(&mut self, categories: impl IntoIterator<Item = &'a str>) {
        for cat in categories {
            self.sketch.inc(hash_str(cat));
        }
    }

    /// Finalize the filter by identifying frequent categories.
    ///
    /// Must be called after counting and before filtering.
    /// Pass all unique categories to identify which are frequent.
    pub fn finalize(&mut self, unique_categories: impl IntoIterator<Item = String>) {
        self.frequent.clear();
        for cat in unique_categories {
            if self.sketch.estimate(hash_str(&cat)) >= self.min_count {
                self.frequent.insert(cat);
            }
        }
    }

    /// Check if a category is frequent enough to keep.
    #[inline]
    pub fn is_frequent(&self, category: &str) -> bool {
        self.frequent.contains(category)
    }

    /// Get the estimated count for a category.
    #[inline]
    pub fn estimate_count(&self, category: &str) -> u64 {
        self.sketch.estimate(hash_str(category))
    }

    /// Filter a category: returns the category if frequent, or "unknown" otherwise.
    #[inline]
    pub fn filter<'a>(&self, category: &'a str) -> &'a str {
        if self.is_frequent(category) {
            category
        } else {
            "unknown"
        }
    }

    /// Filter a batch of categories.
    pub fn filter_batch<'a>(&self, categories: &'a [String]) -> Vec<&'a str> {
        categories.iter().map(|c| self.filter(c)).collect()
    }

    /// Number of frequent categories identified.
    pub fn num_frequent(&self) -> usize {
        self.frequent.len()
    }

    /// Get all frequent categories.
    pub fn frequent_categories(&self) -> &FxHashSet<String> {
        &self.frequent
    }

    /// Get memory usage of the sketch in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.sketch.memory_bytes()
    }
}

/// Mapping from original categories to filtered indices.
///
/// Used for serialization and consistent encoding during inference.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
pub struct CategoryMapping {
    /// Map from category string to index (sorted for binary search)
    pub category_to_idx: Vec<(String, u32)>,
    /// Index for unknown categories
    pub unknown_idx: u32,
}

impl CategoryMapping {
    /// Create a mapping from a category filter.
    pub fn from_filter(filter: &CategoryFilter) -> Self {
        let mut category_to_idx: Vec<(String, u32)> = filter
            .frequent
            .iter()
            .enumerate()
            .map(|(i, cat)| (cat.clone(), i as u32))
            .collect();

        // Sort for deterministic ordering and binary search
        category_to_idx.sort_by(|a, b| a.0.cmp(&b.0));

        // Re-assign indices after sorting
        for (i, (_, idx)) in category_to_idx.iter_mut().enumerate() {
            *idx = i as u32;
        }

        let unknown_idx = category_to_idx.len() as u32;

        Self {
            category_to_idx,
            unknown_idx,
        }
    }

    /// Get index for a category (uses binary search).
    #[inline]
    pub fn get_index(&self, category: &str) -> u32 {
        match self
            .category_to_idx
            .binary_search_by(|(cat, _)| cat.as_str().cmp(category))
        {
            Ok(pos) => self.category_to_idx[pos].1,
            Err(_) => self.unknown_idx,
        }
    }

    /// Total number of categories including unknown.
    pub fn num_categories(&self) -> usize {
        self.category_to_idx.len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_min_sketch_basic() {
        let mut cms = CountMinSketch::new(0.01, 0.99);

        // Insert some values
        for _ in 0..100 {
            cms.inc(42);
        }
        for _ in 0..50 {
            cms.inc(123);
        }

        // Estimates should be at least the true count
        assert!(cms.estimate(42) >= 100);
        assert!(cms.estimate(123) >= 50);
        assert_eq!(cms.estimate(999), 0); // Never inserted
    }

    #[test]
    fn test_count_min_sketch_halve() {
        let mut cms = CountMinSketch::new(0.01, 0.99);

        for _ in 0..100 {
            cms.inc(42);
        }

        cms.halve();

        // Should be approximately halved
        assert!(cms.estimate(42) >= 50);
        assert!(cms.estimate(42) <= 55); // Some rounding tolerance
    }

    #[test]
    fn test_count_min_sketch_clear() {
        let mut cms = CountMinSketch::new(0.01, 0.99);

        for _ in 0..100 {
            cms.inc(42);
        }

        cms.clear();
        assert_eq!(cms.estimate(42), 0);
    }

    #[test]
    fn test_category_filter() {
        let mut filter = CategoryFilter::new(0.01, 0.99, 3);

        // Count categories
        for _ in 0..10 {
            filter.count("frequent");
        }
        for _ in 0..2 {
            filter.count("rare");
        }
        filter.count("very_rare");

        // Finalize
        filter.finalize(vec![
            "frequent".to_string(),
            "rare".to_string(),
            "very_rare".to_string(),
        ]);

        // Check filtering
        assert!(filter.is_frequent("frequent"));
        assert!(!filter.is_frequent("rare"));
        assert!(!filter.is_frequent("very_rare"));

        assert_eq!(filter.filter("frequent"), "frequent");
        assert_eq!(filter.filter("rare"), "unknown");
        assert_eq!(filter.filter("unseen"), "unknown");
    }

    #[test]
    fn test_category_mapping() {
        let mut filter = CategoryFilter::new(0.01, 0.99, 2);

        for _ in 0..5 {
            filter.count("cat_a");
            filter.count("cat_b");
            filter.count("cat_c");
        }
        filter.count("rare");

        filter.finalize(vec![
            "cat_a".to_string(),
            "cat_b".to_string(),
            "cat_c".to_string(),
            "rare".to_string(),
        ]);

        let mapping = CategoryMapping::from_filter(&filter);

        assert_eq!(mapping.num_categories(), 4); // 3 frequent + unknown

        // Check consistent indexing
        let idx_a = mapping.get_index("cat_a");
        let idx_b = mapping.get_index("cat_b");
        let idx_c = mapping.get_index("cat_c");
        let idx_unknown = mapping.get_index("rare");

        assert!(idx_a < 3);
        assert!(idx_b < 3);
        assert!(idx_c < 3);
        assert_eq!(idx_unknown, mapping.unknown_idx);
    }

    #[test]
    fn test_high_cardinality() {
        let mut filter = CategoryFilter::default_for_high_cardinality();

        // Simulate high cardinality with many unique strings
        for i in 0..1000 {
            let cat = format!("category_{}", i);
            for _ in 0..(i % 20) {
                filter.count(&cat);
            }
        }

        // Collect unique categories
        let unique: Vec<String> = (0..1000).map(|i| format!("category_{}", i)).collect();
        filter.finalize(unique);

        // Categories with count >= 5 should be frequent
        // i % 20 >= 5 means i in {5,6,...,19}, {25,26,...,39}, etc.
        assert!(filter.is_frequent("category_5"));
        assert!(filter.is_frequent("category_19"));
        assert!(!filter.is_frequent("category_0")); // count = 0
        assert!(!filter.is_frequent("category_4")); // count = 4
    }
}
