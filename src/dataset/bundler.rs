//! Exclusive Feature Bundling (EFB) for memory-efficient histogram construction
//!
//! EFB bundles mutually exclusive features (features that rarely have non-zero values
//! simultaneously) into single synthetic features. This reduces:
//! - Memory usage (fewer columns to store)
//! - Training time (fewer histograms to build)
//!
//! # Algorithm
//!
//! 1. **Conflict Detection**: Count how many samples have non-zero values for both
//!    features in a pair. Features with low conflict (<0.01% of samples) are
//!    considered mutually exclusive.
//!
//! 2. **Greedy Bundling**: Pack features into bundles greedily, ensuring:
//!    - Total bins in bundle ≤ 256 (to fit in u8)
//!    - Conflict threshold not exceeded
//!
//! 3. **Bin Offset Mapping**: Each feature in a bundle gets an offset in the
//!    combined bin space. Feature A bins [0-10], Feature B bins [11-20], etc.
//!
//! # Example
//!
//! One-hot encoded features are perfectly mutually exclusive:
//! - Category "Color" with one-hot: is_red, is_green, is_blue
//! - Only one can be 1 at a time → 0 conflicts → perfect bundle
//! - 3 features (3 bins each) → 1 bundled feature (9 bins total)
//!
//! # Reference
//!
//! Based on LightGBM's EFB algorithm from "LightGBM: A Highly Efficient Gradient
//! Boosting Decision Tree" (NeurIPS 2017).

use crate::defaults::bundler as bundler_defaults;
use rkyv::{Archive, Deserialize, Serialize};

use super::{BinnedDataset, FeatureInfo, DEFAULT_BIN};

/// Maximum allowed conflict ratio (fraction of samples with conflicting non-zero values)
/// LightGBM uses 0.01% (1/10000). We use a slightly higher threshold for robustness.
const MAX_CONFLICT_RATIO: f32 = bundler_defaults::DEFAULT_MAX_CONFLICT_RATIO;

/// Minimum sparsity for a feature to be considered for bundling
/// Dense features don't benefit much from bundling
const MIN_SPARSITY_FOR_BUNDLING: f32 = bundler_defaults::DEFAULT_MIN_SPARSITY;

/// Maximum number of bundle candidates to search when adding a feature
/// Limits O(n²) complexity; LightGBM uses 100
const MAX_SEARCH_BUNDLES: usize = bundler_defaults::DEFAULT_MAX_SEARCH_BUNDLES;

/// A bundle of mutually exclusive features
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct FeatureBundle {
    /// Original feature indices in this bundle
    pub feature_indices: Vec<usize>,
    /// Bin offset for each feature (cumulative bins)
    /// feature i's bins start at bin_offsets[i] and end at bin_offsets[i+1]-1
    pub bin_offsets: Vec<u8>,
    /// Total number of bins in this bundle
    pub total_bins: u8,
    /// Name for the bundled feature (for debugging)
    pub name: String,
}

impl FeatureBundle {
    /// Create a new feature bundle
    pub fn new(feature_indices: Vec<usize>, bin_counts: &[u8], names: &[String]) -> Self {
        let mut bin_offsets = Vec::with_capacity(feature_indices.len() + 1);
        bin_offsets.push(0);

        let mut total = 0u16;
        for &count in bin_counts {
            total += count as u16;
            // Saturate at 255 to prevent overflow
            bin_offsets.push(total.min(255) as u8);
        }

        let name = if names.len() == 1 {
            names[0].clone()
        } else {
            format!("bundle[{}]", names.join("+"))
        };

        Self {
            feature_indices,
            bin_offsets,
            total_bins: total.min(255) as u8,
            name,
        }
    }

    /// Get the bin offset for a feature within this bundle
    #[inline]
    pub fn bin_offset(&self, local_idx: usize) -> u8 {
        self.bin_offsets[local_idx]
    }

    /// Get the bin range for a feature within this bundle (start, end exclusive)
    #[inline]
    pub fn bin_range(&self, local_idx: usize) -> (u8, u8) {
        (self.bin_offsets[local_idx], self.bin_offsets[local_idx + 1])
    }

    /// Number of features in this bundle
    #[inline]
    pub fn len(&self) -> usize {
        self.feature_indices.len()
    }

    /// Check if bundle is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.feature_indices.is_empty()
    }

    /// Check if this is a single-feature bundle (no actual bundling)
    #[inline]
    pub fn is_single(&self) -> bool {
        self.feature_indices.len() == 1
    }
}

/// Result of the bundling process
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct BundlingResult {
    /// The feature bundles
    pub bundles: Vec<FeatureBundle>,
    /// Mapping from original feature index to (bundle_idx, local_idx)
    pub feature_to_bundle: Vec<(usize, usize)>,
    /// Number of original features
    pub num_original_features: usize,
    /// Number of bundles (effective features after bundling)
    pub num_bundles: usize,
}

impl BundlingResult {
    /// Get the bundle and local index for an original feature
    #[inline]
    pub fn get_bundle(&self, feature_idx: usize) -> (usize, usize) {
        self.feature_to_bundle[feature_idx]
    }

    /// Compression ratio (original features / bundles)
    pub fn compression_ratio(&self) -> f32 {
        if self.num_bundles == 0 {
            1.0
        } else {
            self.num_original_features as f32 / self.num_bundles as f32
        }
    }

    /// Number of features that were actually bundled (in multi-feature bundles)
    pub fn num_bundled_features(&self) -> usize {
        self.bundles
            .iter()
            .filter(|b| !b.is_single())
            .map(|b| b.len())
            .sum()
    }
}

/// Configuration for the bundling algorithm
#[derive(Debug, Clone)]
pub struct BundlerConfig {
    /// Maximum conflict ratio allowed for bundling
    pub max_conflict_ratio: f32,
    /// Minimum sparsity for considering bundling
    pub min_sparsity: f32,
    /// Maximum bins per bundle (must be ≤ 256)
    pub max_bins_per_bundle: u16,
    /// Whether bundling is enabled
    pub enabled: bool,
}

impl Default for BundlerConfig {
    fn default() -> Self {
        Self {
            max_conflict_ratio: MAX_CONFLICT_RATIO,
            min_sparsity: MIN_SPARSITY_FOR_BUNDLING,
            max_bins_per_bundle: bundler_defaults::DEFAULT_MAX_BINS_PER_BUNDLE,
            enabled: bundler_defaults::DEFAULT_BUNDLING_ENABLED,
        }
    }
}

/// Feature bundler using greedy algorithm
pub struct FeatureBundler {
    config: BundlerConfig,
}

impl Default for FeatureBundler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureBundler {
    /// Create a new bundler with default configuration
    pub fn new() -> Self {
        Self {
            config: BundlerConfig::default(),
        }
    }

    /// Create a bundler with custom configuration
    pub fn with_config(config: BundlerConfig) -> Self {
        Self { config }
    }

    /// Analyze dataset and find optimal feature bundles
    pub fn find_bundles(&self, dataset: &BinnedDataset) -> BundlingResult {
        let num_features = dataset.num_features();
        let num_rows = dataset.num_rows();

        if !self.config.enabled || num_features == 0 {
            return self.create_trivial_bundles(dataset);
        }

        // Step 1: Compute sparsity and non-zero indices for each feature
        let feature_stats: Vec<FeatureStats> = (0..num_features)
            .map(|f| self.compute_feature_stats(dataset, f))
            .collect();

        // Step 2: Filter features that are sparse enough for bundling
        let sparse_features: Vec<usize> = feature_stats
            .iter()
            .enumerate()
            .filter(|(_, stats)| stats.sparsity >= self.config.min_sparsity)
            .map(|(idx, _)| idx)
            .collect();

        // Step 3: Order features by non-zero count (ascending) for greedy bundling
        let mut ordered_sparse: Vec<usize> = sparse_features.clone();
        ordered_sparse.sort_by_key(|&f| feature_stats[f].non_zero_count);

        // Step 4: Greedy bundling
        let max_conflicts = ((num_rows as f32) * self.config.max_conflict_ratio).ceil() as usize;
        let bundles = self.greedy_bundle(dataset, &feature_stats, &ordered_sparse, max_conflicts);

        // Step 5: Add remaining dense features as single-feature bundles
        let bundled_features: std::collections::HashSet<usize> = bundles
            .iter()
            .flat_map(|b| b.feature_indices.iter().copied())
            .collect();

        let mut all_bundles = bundles;
        for f in 0..num_features {
            if !bundled_features.contains(&f) {
                // Create single-feature bundle
                let info = dataset.feature_info(f);
                all_bundles.push(FeatureBundle::new(
                    vec![f],
                    std::slice::from_ref(&info.num_bins),
                    std::slice::from_ref(&info.name),
                ));
            }
        }

        // Sort bundles by first feature index for deterministic ordering
        all_bundles.sort_by_key(|b| b.feature_indices[0]);

        // Build feature-to-bundle mapping
        let mut feature_to_bundle = vec![(0usize, 0usize); num_features];
        for (bundle_idx, bundle) in all_bundles.iter().enumerate() {
            for (local_idx, &feature_idx) in bundle.feature_indices.iter().enumerate() {
                feature_to_bundle[feature_idx] = (bundle_idx, local_idx);
            }
        }

        BundlingResult {
            num_bundles: all_bundles.len(),
            bundles: all_bundles,
            feature_to_bundle,
            num_original_features: num_features,
        }
    }

    /// Greedy bundling algorithm
    ///
    /// Uses limited search (MAX_SEARCH_BUNDLES) to avoid O(n²) complexity.
    /// Prefers checking most recently created bundles first (more likely to fit).
    fn greedy_bundle(
        &self,
        dataset: &BinnedDataset,
        feature_stats: &[FeatureStats],
        sparse_features: &[usize],
        max_conflicts: usize,
    ) -> Vec<FeatureBundle> {
        if sparse_features.is_empty() {
            return Vec::new();
        }

        // Track which features are already bundled
        let mut bundled = vec![false; dataset.num_features()];
        let mut bundles: Vec<BundleBuilder> = Vec::new();

        for &feature_idx in sparse_features {
            if bundled[feature_idx] {
                continue;
            }

            let feature_info = dataset.feature_info(feature_idx);
            let stats = &feature_stats[feature_idx];

            // Try to add to an existing bundle
            // Search from the end (most recent bundles are often best candidates)
            let mut added_to_bundle = false;
            let num_to_search = bundles.len().min(MAX_SEARCH_BUNDLES);

            // Search most recent bundles first (reverse order)
            for i in (bundles.len().saturating_sub(num_to_search)..bundles.len()).rev() {
                let bundle = &bundles[i];

                // Check bin count constraint (fast check first)
                let new_bins = bundle.total_bins as u16 + feature_info.num_bins as u16;
                if new_bins > self.config.max_bins_per_bundle {
                    continue;
                }

                // Check conflict constraint (expensive)
                let conflicts =
                    self.count_conflicts(&stats.non_zero_indices, &bundle.non_zero_union);
                if conflicts <= max_conflicts {
                    // Add to this bundle
                    let bundle = &mut bundles[i];
                    bundle.add_feature(
                        feature_idx,
                        feature_info.num_bins,
                        &feature_info.name,
                        &stats.non_zero_indices,
                    );
                    bundled[feature_idx] = true;
                    added_to_bundle = true;
                    break;
                }
            }

            if !added_to_bundle {
                // Create new bundle
                let mut new_bundle = BundleBuilder::new();
                new_bundle.add_feature(
                    feature_idx,
                    feature_info.num_bins,
                    &feature_info.name,
                    &stats.non_zero_indices,
                );
                bundles.push(new_bundle);
                bundled[feature_idx] = true;
            }
        }

        // Convert BundleBuilder to FeatureBundle
        bundles.into_iter().map(|b| b.build()).collect()
    }

    /// Count conflicts between a feature's non-zero indices and a bundle's union
    /// Returns the count, or a value > max_conflicts for early termination
    fn count_conflicts(&self, feature_indices: &[u32], bundle_union: &[u32]) -> usize {
        if feature_indices.is_empty() || bundle_union.is_empty() {
            return 0;
        }

        // Both arrays are sorted, use merge-style counting
        let mut conflicts = 0;
        let mut i = 0;
        let mut j = 0;

        // Get max conflicts from config for early termination
        let max_conflicts =
            ((feature_indices.len() as f32) * self.config.max_conflict_ratio * 10.0) as usize + 1;

        while i < feature_indices.len() && j < bundle_union.len() {
            match feature_indices[i].cmp(&bundle_union[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    conflicts += 1;
                    // Early termination if we've exceeded threshold
                    if conflicts > max_conflicts {
                        return conflicts;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        conflicts
    }

    /// Compute statistics for a feature
    fn compute_feature_stats(&self, dataset: &BinnedDataset, feature_idx: usize) -> FeatureStats {
        let column = dataset.feature_column(feature_idx);
        let mut non_zero_indices: Vec<u32> = Vec::new();

        for (row_idx, &bin) in column.iter().enumerate() {
            if bin != DEFAULT_BIN {
                non_zero_indices.push(row_idx as u32);
            }
        }

        let non_zero_count = non_zero_indices.len();
        let sparsity = if column.is_empty() {
            1.0
        } else {
            1.0 - (non_zero_count as f32 / column.len() as f32)
        };

        FeatureStats {
            non_zero_count,
            sparsity,
            non_zero_indices,
        }
    }

    /// Create trivial bundles (one feature per bundle, no actual bundling)
    fn create_trivial_bundles(&self, dataset: &BinnedDataset) -> BundlingResult {
        let num_features = dataset.num_features();

        let bundles: Vec<FeatureBundle> = (0..num_features)
            .map(|f| {
                let info = dataset.feature_info(f);
                FeatureBundle::new(
                    vec![f],
                    std::slice::from_ref(&info.num_bins),
                    std::slice::from_ref(&info.name),
                )
            })
            .collect();

        let feature_to_bundle: Vec<(usize, usize)> = (0..num_features).map(|f| (f, 0)).collect();

        BundlingResult {
            bundles,
            feature_to_bundle,
            num_original_features: num_features,
            num_bundles: num_features,
        }
    }
}

/// Statistics for a single feature
struct FeatureStats {
    /// Number of non-zero (non-default) values
    non_zero_count: usize,
    /// Sparsity ratio (fraction of zero values)
    sparsity: f32,
    /// Sorted indices of non-zero values
    non_zero_indices: Vec<u32>,
}

/// Builder for constructing a bundle
struct BundleBuilder {
    feature_indices: Vec<usize>,
    bin_counts: Vec<u8>,
    names: Vec<String>,
    total_bins: u8,
    /// Union of all non-zero indices in this bundle (sorted)
    non_zero_union: Vec<u32>,
}

impl BundleBuilder {
    fn new() -> Self {
        Self {
            feature_indices: Vec::new(),
            bin_counts: Vec::new(),
            names: Vec::new(),
            total_bins: 0,
            non_zero_union: Vec::new(),
        }
    }

    fn add_feature(
        &mut self,
        feature_idx: usize,
        num_bins: u8,
        name: &str,
        non_zero_indices: &[u32],
    ) {
        self.feature_indices.push(feature_idx);
        self.bin_counts.push(num_bins);
        self.names.push(name.to_string());
        self.total_bins = (self.total_bins as u16 + num_bins as u16).min(255) as u8;

        // Merge non_zero_indices into union (both are sorted)
        self.non_zero_union = Self::merge_sorted(&self.non_zero_union, non_zero_indices);
    }

    fn merge_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(a.len() + b.len());
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    result.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }

        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);
        result
    }

    fn build(self) -> FeatureBundle {
        FeatureBundle::new(self.feature_indices, &self.bin_counts, &self.names)
    }
}

/// Dataset with bundled features for efficient histogram construction
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct BundledDataset {
    /// Number of rows
    num_rows: usize,
    /// Bundled feature data (column-major): bundles[bundle_idx][row_idx]
    bundle_data: Vec<Vec<u8>>,
    /// Target values
    targets: Vec<f32>,
    /// Bundle metadata
    bundles: Vec<FeatureBundle>,
    /// Mapping from original feature to (bundle_idx, local_idx)
    feature_to_bundle: Vec<(usize, usize)>,
    /// Original feature info (preserved for prediction)
    original_feature_info: Vec<FeatureInfo>,
}

impl BundledDataset {
    /// Create a bundled dataset from a regular dataset and bundling result
    pub fn from_dataset(dataset: &BinnedDataset, bundling: &BundlingResult) -> Self {
        let num_rows = dataset.num_rows();

        // Create bundled columns
        let bundle_data: Vec<Vec<u8>> = bundling
            .bundles
            .iter()
            .map(|bundle| {
                let mut column = vec![0u8; num_rows];

                for (local_idx, &feature_idx) in bundle.feature_indices.iter().enumerate() {
                    let offset = bundle.bin_offset(local_idx);
                    let feature_column = dataset.feature_column(feature_idx);

                    for (row_idx, &bin) in feature_column.iter().enumerate() {
                        if bin != DEFAULT_BIN {
                            // Apply offset and store
                            // Note: This assumes mutually exclusive features
                            // If there's a conflict, later feature overwrites
                            column[row_idx] = offset.saturating_add(bin);
                        }
                    }
                }

                column
            })
            .collect();

        Self {
            num_rows,
            bundle_data,
            targets: dataset.targets().to_vec(),
            bundles: bundling.bundles.clone(),
            feature_to_bundle: bundling.feature_to_bundle.clone(),
            original_feature_info: dataset.all_feature_info().to_vec(),
        }
    }

    /// Number of rows
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of bundles (effective features)
    #[inline]
    pub fn num_bundles(&self) -> usize {
        self.bundles.len()
    }

    /// Number of original features
    #[inline]
    pub fn num_original_features(&self) -> usize {
        self.original_feature_info.len()
    }

    /// Get bundled bin value for a row and bundle
    #[inline]
    pub fn get_bundle_bin(&self, row_idx: usize, bundle_idx: usize) -> u8 {
        self.bundle_data[bundle_idx][row_idx]
    }

    /// Get bundle column (for histogram building)
    #[inline]
    pub fn bundle_column(&self, bundle_idx: usize) -> &[u8] {
        &self.bundle_data[bundle_idx]
    }

    /// Get bundle metadata
    #[inline]
    pub fn bundle(&self, bundle_idx: usize) -> &FeatureBundle {
        &self.bundles[bundle_idx]
    }

    /// Get all bundles
    #[inline]
    pub fn bundles(&self) -> &[FeatureBundle] {
        &self.bundles
    }

    /// Get target values
    #[inline]
    pub fn targets(&self) -> &[f32] {
        &self.targets
    }

    /// Get original feature info
    #[inline]
    pub fn original_feature_info(&self) -> &[FeatureInfo] {
        &self.original_feature_info
    }

    /// Get the bundle and local index for an original feature
    #[inline]
    pub fn get_feature_bundle(&self, feature_idx: usize) -> (usize, usize) {
        self.feature_to_bundle[feature_idx]
    }

    /// Compression ratio achieved
    pub fn compression_ratio(&self) -> f32 {
        if self.bundles.is_empty() {
            1.0
        } else {
            self.num_original_features() as f32 / self.num_bundles() as f32
        }
    }

    /// Memory savings from bundling (bytes saved)
    pub fn memory_savings(&self) -> usize {
        let original_size = self.num_rows * self.num_original_features();
        let bundled_size = self.num_rows * self.num_bundles();
        original_size.saturating_sub(bundled_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FeatureType;

    fn create_exclusive_dataset() -> BinnedDataset {
        // Create dataset with mutually exclusive one-hot features
        // 10 rows, 4 features that are one-hot encoded (only one is 1 at a time)
        let num_rows = 10;
        let num_features = 4;

        // One-hot pattern: each row has exactly one feature = 1, rest = 0
        let mut features = vec![0u8; num_rows * num_features];
        for row in 0..num_rows {
            let active_feature = row % num_features;
            features[active_feature * num_rows + row] = 1;
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("onehot_{}", i),
                feature_type: FeatureType::Categorical,
                num_bins: 2, // 0 or 1
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    fn create_dense_dataset() -> BinnedDataset {
        // Create dataset with dense (non-sparse) features
        let num_rows = 100;
        let num_features = 3;

        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                // All non-zero values (dense)
                features.push(((r + f * 7) % 15 + 1) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("dense_{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 16,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_feature_bundle_creation() {
        let bundle = FeatureBundle::new(
            vec![0, 1, 2],
            &[10, 5, 8],
            &["f0".to_string(), "f1".to_string(), "f2".to_string()],
        );

        assert_eq!(bundle.len(), 3);
        assert_eq!(bundle.total_bins, 23);
        assert_eq!(bundle.bin_offset(0), 0);
        assert_eq!(bundle.bin_offset(1), 10);
        assert_eq!(bundle.bin_offset(2), 15);
        assert_eq!(bundle.bin_range(0), (0, 10));
        assert_eq!(bundle.bin_range(1), (10, 15));
        assert_eq!(bundle.bin_range(2), (15, 23));
    }

    #[test]
    fn test_exclusive_features_bundled() {
        let dataset = create_exclusive_dataset();
        let bundler = FeatureBundler::new();
        let result = bundler.find_bundles(&dataset);

        // All 4 one-hot features should be bundled together
        // They have 75% sparsity (3/4 zeros per row) and 0 conflicts
        assert!(
            result.num_bundles < result.num_original_features,
            "Expected bundling to reduce features: {} bundles vs {} features",
            result.num_bundles,
            result.num_original_features
        );

        // Check compression
        assert!(result.compression_ratio() > 1.0);
    }

    #[test]
    fn test_dense_features_not_bundled() {
        let dataset = create_dense_dataset();
        let bundler = FeatureBundler::new();
        let result = bundler.find_bundles(&dataset);

        // Dense features (0% sparsity) should not be bundled
        assert_eq!(
            result.num_bundles, result.num_original_features,
            "Dense features should not be bundled"
        );
    }

    #[test]
    fn test_bundled_dataset_creation() {
        let dataset = create_exclusive_dataset();
        let bundler = FeatureBundler::new();
        let bundling = bundler.find_bundles(&dataset);
        let bundled = BundledDataset::from_dataset(&dataset, &bundling);

        assert_eq!(bundled.num_rows(), dataset.num_rows());
        assert_eq!(bundled.num_bundles(), bundling.num_bundles);

        // Verify data integrity - each row should have a non-zero value
        for row in 0..bundled.num_rows() {
            let mut found_nonzero = false;
            for bundle_idx in 0..bundled.num_bundles() {
                if bundled.get_bundle_bin(row, bundle_idx) != 0 {
                    found_nonzero = true;
                    break;
                }
            }
            // For one-hot data, each row should have exactly one non-zero
            if bundling.num_bundles == 1 {
                assert!(found_nonzero, "Row {} should have a non-zero value", row);
            }
        }
    }

    #[test]
    fn test_bin_offset_mapping() {
        let bundle = FeatureBundle::new(vec![0, 1], &[5, 3], &["a".to_string(), "b".to_string()]);

        // Feature 0: bins 0-4 (5 bins)
        // Feature 1: bins 5-7 (3 bins, offset by 5)
        assert_eq!(bundle.bin_offset(0), 0);
        assert_eq!(bundle.bin_offset(1), 5);
        assert_eq!(bundle.total_bins, 8);
    }

    #[test]
    fn test_bundler_disabled() {
        let dataset = create_exclusive_dataset();
        let config = BundlerConfig {
            enabled: false,
            ..Default::default()
        };
        let bundler = FeatureBundler::with_config(config);
        let result = bundler.find_bundles(&dataset);

        // When disabled, each feature becomes its own bundle
        assert_eq!(result.num_bundles, result.num_original_features);
    }

    #[test]
    fn test_high_conflict_threshold() {
        let dataset = create_exclusive_dataset();

        // With very low sparsity requirement, nothing gets bundled
        let config = BundlerConfig {
            min_sparsity: 0.99, // Require 99% sparsity
            ..Default::default()
        };
        let bundler = FeatureBundler::with_config(config);
        let result = bundler.find_bundles(&dataset);

        // One-hot has 75% sparsity, so shouldn't pass 99% threshold
        assert_eq!(result.num_bundles, result.num_original_features);
    }
}
