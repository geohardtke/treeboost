//! PyO3 bindings for TreeBoost GBDT
//!
//! This module provides Python wrappers around the core Rust GBDT implementation,
//! exposing a Python-friendly API while maintaining Rust performance.
//!
//! # Example
//!
//! ```python
//! import numpy as np
//! from treeboost import GBDTConfig, GBDTModel
//!
//! # Create configuration
//! config = GBDTConfig()
//! config.num_rounds = 100
//! config.max_depth = 6
//! config.learning_rate = 0.1
//!
//! # Train model (features: 2D array, targets: 1D array)
//! model = GBDTModel.train(features, targets, config)
//!
//! # Predict
//! predictions = model.predict(features)
//!
//! # Save/load model
//! model.save("model.rkyv")
//! loaded = GBDTModel.load("model.rkyv")
//! ```

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use crate::backend::{BackendType, GpuMode};
use crate::booster::{GBDTConfig, GBDTModel, LossType};
use crate::serialize;
use crate::tree::MonotonicConstraint;

/// Python wrapper for GBDT training configuration
#[pyclass(name = "GBDTConfig")]
#[derive(Clone)]
pub struct PyGBDTConfig {
    inner: GBDTConfig,
}

#[pymethods]
impl PyGBDTConfig {
    /// Create a new configuration with default values
    #[new]
    fn new() -> Self {
        Self { inner: GBDTConfig::default() }
    }

    // Ensemble parameters

    /// Number of boosting rounds (trees)
    #[getter]
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds
    }

    #[setter]
    fn set_num_rounds(&mut self, value: usize) {
        self.inner.num_rounds = value;
    }

    /// Learning rate (shrinkage)
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f32) {
        self.inner.learning_rate = value;
    }

    // Tree parameters

    /// Maximum depth of each tree
    #[getter]
    fn max_depth(&self) -> usize {
        self.inner.max_depth
    }

    #[setter]
    fn set_max_depth(&mut self, value: usize) {
        self.inner.max_depth = value;
    }

    /// Maximum number of leaves per tree
    #[getter]
    fn max_leaves(&self) -> usize {
        self.inner.max_leaves
    }

    #[setter]
    fn set_max_leaves(&mut self, value: usize) {
        self.inner.max_leaves = value;
    }

    /// Minimum samples required in a leaf
    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }

    #[setter]
    fn set_min_samples_leaf(&mut self, value: usize) {
        self.inner.min_samples_leaf = value;
    }

    /// Minimum hessian sum required in a leaf
    #[getter]
    fn min_hessian_leaf(&self) -> f32 {
        self.inner.min_hessian_leaf
    }

    #[setter]
    fn set_min_hessian_leaf(&mut self, value: f32) {
        self.inner.min_hessian_leaf = value;
    }

    /// Minimum gain to make a split
    #[getter]
    fn min_gain(&self) -> f32 {
        self.inner.min_gain
    }

    #[setter]
    fn set_min_gain(&mut self, value: f32) {
        self.inner.min_gain = value;
    }

    // Regularization

    /// L2 regularization (lambda)
    #[getter]
    fn lambda_(&self) -> f32 {
        self.inner.lambda
    }

    #[setter]
    fn set_lambda(&mut self, value: f32) {
        self.inner.lambda = value;
    }

    /// Shannon Entropy regularization weight (beta)
    #[getter]
    fn entropy_weight(&self) -> f32 {
        self.inner.entropy_weight
    }

    #[setter]
    fn set_entropy_weight(&mut self, value: f32) {
        self.inner.entropy_weight = value;
    }

    // Loss function

    /// Set loss function to MSE
    fn use_mse_loss(&mut self) {
        self.inner.loss_type = LossType::Mse;
    }

    /// Set loss function to Pseudo-Huber with given delta
    fn use_pseudo_huber_loss(&mut self, delta: f32) {
        self.inner.loss_type = LossType::PseudoHuber { delta };
    }

    /// Set loss function to Binary Log Loss (for binary classification)
    ///
    /// Uses sigmoid activation for probability output.
    /// Targets should be 0 or 1.
    fn use_binary_logloss(&mut self) {
        self.inner.loss_type = LossType::BinaryLogLoss;
    }

    /// Set loss function to Multi-class Log Loss (for multi-class classification)
    ///
    /// Uses softmax activation for probability output.
    /// Targets should be class indices: 0, 1, 2, ..., num_classes-1.
    ///
    /// This trains K trees per round (one per class) and combines predictions
    /// via softmax for final class probabilities.
    ///
    /// Args:
    ///     num_classes: Number of classes (K), must be >= 2
    fn use_multiclass_logloss(&mut self, num_classes: usize) -> PyResult<()> {
        if num_classes < 2 {
            return Err(PyValueError::new_err("num_classes must be >= 2"));
        }
        self.inner.loss_type = LossType::MultiClassLogLoss { num_classes };
        Ok(())
    }

    // Subsampling

    /// Row subsampling ratio (0.0-1.0)
    #[getter]
    fn subsample(&self) -> f32 {
        self.inner.subsample
    }

    #[setter]
    fn set_subsample(&mut self, value: f32) {
        self.inner.subsample = value;
    }

    /// Column subsampling ratio (0.0-1.0)
    #[getter]
    fn colsample(&self) -> f32 {
        self.inner.colsample
    }

    #[setter]
    fn set_colsample(&mut self, value: f32) {
        self.inner.colsample = value;
    }

    // Binning

    /// Number of histogram bins
    #[getter]
    fn num_bins(&self) -> usize {
        self.inner.num_bins
    }

    #[setter]
    fn set_num_bins(&mut self, value: usize) {
        self.inner.num_bins = value;
    }

    // Conformal prediction

    /// Calibration set ratio for conformal prediction (0.0 to disable)
    #[getter]
    fn calibration_ratio(&self) -> f32 {
        self.inner.calibration_ratio
    }

    #[setter]
    fn set_calibration_ratio(&mut self, value: f32) {
        self.inner.calibration_ratio = value;
    }

    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    #[getter]
    fn conformal_quantile(&self) -> f32 {
        self.inner.conformal_quantile
    }

    #[setter]
    fn set_conformal_quantile(&mut self, value: f32) {
        self.inner.conformal_quantile = value;
    }

    // Early stopping

    /// Number of rounds with no improvement before stopping (0 to disable)
    #[getter]
    fn early_stopping_rounds(&self) -> usize {
        self.inner.early_stopping_rounds
    }

    #[setter]
    fn set_early_stopping_rounds(&mut self, value: usize) {
        self.inner.early_stopping_rounds = value;
    }

    /// Ratio of data to use for validation (0.0 to disable early stopping)
    #[getter]
    fn validation_ratio(&self) -> f32 {
        self.inner.validation_ratio
    }

    #[setter]
    fn set_validation_ratio(&mut self, value: f32) {
        self.inner.validation_ratio = value;
    }

    // Performance optimizations

    /// Use parallel prediction via Rayon (default: true)
    #[getter]
    fn parallel_prediction(&self) -> bool {
        self.inner.parallel_prediction
    }

    #[setter]
    fn set_parallel_prediction(&mut self, value: bool) {
        self.inner.parallel_prediction = value;
    }

    /// Reorder columns by feature importance for cache locality (default: true)
    #[getter]
    fn column_reordering(&self) -> bool {
        self.inner.column_reordering
    }

    #[setter]
    fn set_column_reordering(&mut self, value: bool) {
        self.inner.column_reordering = value;
    }

    /// Use 4-bit packing for low-cardinality features (default: true)
    #[getter]
    fn packed_dataset(&self) -> bool {
        self.inner.packed_dataset
    }

    #[setter]
    fn set_packed_dataset(&mut self, value: bool) {
        self.inner.packed_dataset = value;
    }

    /// Use parallel gradient computation (default: false)
    /// Experimental: may not provide stable speedups, benchmark before enabling
    #[getter]
    fn parallel_gradient(&self) -> bool {
        self.inner.parallel_gradient
    }

    #[setter]
    fn set_parallel_gradient(&mut self, value: bool) {
        self.inner.parallel_gradient = value;
    }

    // Backend selection

    /// Enable GPU subgroup operations (default: True for Python)
    ///
    /// Subgroups can reduce atomic contention in GPU histogram building.
    /// Benchmarks show minimal benefit on modern NVIDIA GPUs (~1.0x),
    /// but may help on older AMD or Intel GPUs.
    #[getter]
    fn use_gpu_subgroups(&self) -> bool {
        self.inner.use_gpu_subgroups
    }

    #[setter]
    fn set_use_gpu_subgroups(&mut self, value: bool) {
        self.inner.use_gpu_subgroups = value;
    }

    /// Get current backend type as string
    #[getter]
    fn backend(&self) -> &'static str {
        match self.inner.backend_type {
            BackendType::Auto => "auto",
            BackendType::Scalar => "cpu",
            BackendType::Wgpu => "wgpu",
            BackendType::Avx512 => "avx512",
            BackendType::Sve2 => "sve2",
            BackendType::Cuda => "cuda",
            BackendType::Rocm => "rocm",
            BackendType::Metal => "metal",
        }
    }

    /// Set the backend for histogram building
    ///
    /// Args:
    ///     value: One of "auto", "cpu", "gpu", "wgpu", "cuda"
    ///         - "auto": Select best available (CUDA > WGPU > CPU)
    ///         - "cpu": Force CPU (AVX2/NEON optimized)
    ///         - "gpu": Select best GPU (CUDA > WGPU), same as "auto"
    ///         - "wgpu": Force WGPU (all GPUs via Vulkan/Metal/DX12)
    ///         - "cuda": Force CUDA (NVIDIA only, fastest)
    ///
    /// Example:
    ///     config.backend = "cuda"
    #[setter]
    fn set_backend(&mut self, value: &str) -> PyResult<()> {
        self.inner.backend_type = match value.to_lowercase().as_str() {
            "auto" | "gpu" => BackendType::Auto,  // gpu = auto-select best GPU (CUDA > WGPU)
            "cpu" | "scalar" => BackendType::Scalar,
            "wgpu" => BackendType::Wgpu,
            "cuda" => BackendType::Cuda,
            "rocm" => BackendType::Rocm,
            "metal" => BackendType::Metal,
            _ => return Err(PyValueError::new_err(
                "backend must be one of: 'auto', 'cpu', 'gpu', 'wgpu', 'cuda', 'rocm', 'metal'"
            )),
        };
        Ok(())
    }

    /// Get current GPU mode as string
    #[getter]
    fn gpu_mode(&self) -> &'static str {
        match self.inner.gpu_mode {
            GpuMode::Auto => "auto",
            GpuMode::Hybrid => "hybrid",
            GpuMode::Full => "full",
        }
    }

    /// Set the GPU execution mode
    ///
    /// Args:
    ///     value: One of "auto", "hybrid", "full"
    ///         - "auto": Select optimal mode per backend (CUDA→Full, WGPU→Hybrid)
    ///         - "hybrid": GPU histogram + CPU partition (best-first tree growth)
    ///         - "full": Full GPU pipeline (level-wise tree growth)
    ///
    /// Example:
    ///     config.gpu_mode = "full"
    #[setter]
    fn set_gpu_mode(&mut self, value: &str) -> PyResult<()> {
        self.inner.gpu_mode = match value.to_lowercase().as_str() {
            "auto" => GpuMode::Auto,
            "hybrid" => GpuMode::Hybrid,
            "full" => GpuMode::Full,
            _ => return Err(PyValueError::new_err(
                "gpu_mode must be one of: 'auto', 'hybrid', 'full'"
            )),
        };
        Ok(())
    }

    // Monotonic constraints

    /// Set monotonic constraints for features
    ///
    /// Args:
    ///     constraints: List of constraint values per feature
    ///         - 1 = Increasing (larger values -> larger predictions)
    ///         - -1 = Decreasing (larger values -> smaller predictions)
    ///         - 0 = None (no constraint)
    ///
    /// Example:
    ///     config.set_monotonic_constraints([1, -1, 0])  # Feature 0: inc, Feature 1: dec, Feature 2: none
    fn set_monotonic_constraints(&mut self, constraints: Vec<i32>) -> PyResult<()> {
        let parsed: Result<Vec<MonotonicConstraint>, _> = constraints
            .into_iter()
            .map(|c| match c {
                1 => Ok(MonotonicConstraint::Increasing),
                -1 => Ok(MonotonicConstraint::Decreasing),
                0 => Ok(MonotonicConstraint::None),
                _ => Err(PyValueError::new_err(
                    "Constraint must be 1 (increasing), -1 (decreasing), or 0 (none)",
                )),
            })
            .collect();

        self.inner.monotonic_constraints = parsed?;
        Ok(())
    }

    // Era splitting (Directional Era Splitting)

    /// Enable Directional Era Splitting (DES)
    ///
    /// When enabled, splits are only accepted if ALL eras agree on the
    /// split direction. This filters out spurious correlations that work
    /// in some time periods but not others.
    ///
    /// Requires passing era_indices to train_with_eras().
    #[getter]
    fn era_splitting(&self) -> bool {
        self.inner.era_splitting
    }

    #[setter]
    fn set_era_splitting(&mut self, value: bool) {
        self.inner.era_splitting = value;
    }

    // Interaction constraints

    /// Set feature interaction constraints
    ///
    /// Features in the same group can interact (appear together in a tree path).
    /// Features in different groups cannot be used together.
    /// Features not in any group can interact with all features.
    ///
    /// Args:
    ///     groups: List of feature index groups, e.g., [[0, 1, 2], [3, 4]]
    ///
    /// Example:
    ///     config.set_interaction_groups([[0, 1, 2], [3, 4]])
    fn set_interaction_groups(&mut self, groups: Vec<Vec<usize>>) {
        self.inner.interaction_groups = groups;
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTConfig(num_rounds={}, learning_rate={}, max_depth={}, max_leaves={}, backend='{}', gpu_mode='{}')",
            self.inner.num_rounds,
            self.inner.learning_rate,
            self.inner.max_depth,
            self.inner.max_leaves,
            self.backend(),
            self.gpu_mode()
        )
    }
}

/// Python wrapper for trained GBDT model
#[pyclass(name = "GBDTModel")]
pub struct PyGBDTModel {
    model: GBDTModel,
}

#[pymethods]
impl PyGBDTModel {
    /// Train a GBDT model from numpy arrays
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///     targets: 1D numpy array of shape (n_samples,)
    ///     config: GBDTConfig instance
    ///     feature_names: Optional list of feature names
    ///
    /// Returns:
    ///     Trained GBDTModel
    #[staticmethod]
    #[pyo3(signature = (features, targets, config, feature_names=None))]
    fn train<'py>(
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        targets: PyReadonlyArray1<'py, f32>,
        config: &PyGBDTConfig,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let features_arr = features.as_array();
        let targets_arr = targets.as_array();

        let num_rows = features_arr.nrows();
        let num_features = features_arr.ncols();

        // Convert to row-major flat Vec<f32> for Rust high-level API
        let mut features_flat: Vec<f32> = Vec::with_capacity(num_rows * num_features);
        for row in features_arr.rows() {
            features_flat.extend(row.iter().copied());
        }

        let targets_vec: Vec<f32> = targets_arr.to_vec();

        // Train model using high-level Rust API (release GIL during training)
        // Binning is now done in Rust with Rayon parallelization
        let model = py.allow_threads(|| {
            GBDTModel::train(
                &features_flat,
                num_features,
                &targets_vec,
                config.inner.clone(),
                feature_names,
            )
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { model })
    }

    /// Train a GBDT model with Directional Era Splitting (DES)
    ///
    /// Era splitting filters out spurious correlations by requiring all eras
    /// to agree on split direction. This is useful for time-series or financial
    /// data where patterns may not generalize across time periods.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///     targets: 1D numpy array of shape (n_samples,)
    ///     era_indices: 1D numpy array of era indices (uint16), shape (n_samples,)
    ///     config: GBDTConfig instance (must have era_splitting=True)
    ///     feature_names: Optional list of feature names
    ///
    /// Returns:
    ///     Trained GBDTModel
    ///
    /// Example:
    ///     config = GBDTConfig()
    ///     config.era_splitting = True
    ///     model = GBDTModel.train_with_eras(features, targets, era_indices, config)
    #[staticmethod]
    #[pyo3(signature = (features, targets, era_indices, config, feature_names=None))]
    fn train_with_eras<'py>(
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        targets: PyReadonlyArray1<'py, f32>,
        era_indices: PyReadonlyArray1<'py, u16>,
        config: &PyGBDTConfig,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let features_arr = features.as_array();
        let targets_arr = targets.as_array();
        let era_indices_arr = era_indices.as_array();

        let num_rows = features_arr.nrows();
        let num_features = features_arr.ncols();

        // Validate era_indices length
        if era_indices_arr.len() != num_rows {
            return Err(PyValueError::new_err(format!(
                "era_indices length {} doesn't match number of rows {}",
                era_indices_arr.len(),
                num_rows
            )));
        }

        // Validate era_splitting is enabled
        if !config.inner.era_splitting {
            return Err(PyValueError::new_err(
                "era_splitting must be True in config when using train_with_eras"
            ));
        }

        // Convert to row-major flat Vec<f32> for Rust high-level API
        let mut features_flat: Vec<f32> = Vec::with_capacity(num_rows * num_features);
        for row in features_arr.rows() {
            features_flat.extend(row.iter().copied());
        }

        let targets_vec: Vec<f32> = targets_arr.to_vec();
        let era_indices_vec: Vec<u16> = era_indices_arr.to_vec();

        // Train model using high-level Rust API (release GIL during training)
        let model = py.allow_threads(|| {
            GBDTModel::train_with_eras(
                &features_flat,
                num_features,
                &targets_vec,
                &era_indices_vec,
                config.inner.clone(),
                feature_names,
            )
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { model })
    }

    /// Predict for new data
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     1D numpy array of predictions
    #[pyo3(signature = (features))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Try f64 first (most common for numpy), then f32
        let (num_features, raw_features) = if let Ok(arr) =
            features.extract::<PyReadonlyArray2<'py, f64>>()
        {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().copied());
            }
            (num_cols, raw)
        } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().map(|&v| v as f64));
            }
            (num_cols, raw)
        } else {
            return Err(PyValueError::new_err(
                "features must be a 2D numpy array of float32 or float64",
            ));
        };

        if num_features != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.model.num_features(),
                num_features
            )));
        }

        // Predict using raw values (release GIL)
        let predictions = py.allow_threads(|| self.model.predict_raw(&raw_features));

        Ok(PyArray1::from_vec(py, predictions))
    }

    /// Predict with conformal intervals
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     Tuple of (predictions, lower_bounds, upper_bounds) as numpy arrays
    #[pyo3(signature = (features))]
    fn predict_with_intervals<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        // Try f64 first, then f32
        let (num_features, raw_features) = if let Ok(arr) =
            features.extract::<PyReadonlyArray2<'py, f64>>()
        {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().copied());
            }
            (num_cols, raw)
        } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().map(|&v| v as f64));
            }
            (num_cols, raw)
        } else {
            return Err(PyValueError::new_err(
                "features must be a 2D numpy array of float32 or float64",
            ));
        };

        if num_features != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.model.num_features(),
                num_features
            )));
        }

        // Predict with intervals (release GIL)
        let (preds, lower, upper) =
            py.allow_threads(|| self.model.predict_raw_with_intervals(&raw_features));

        Ok((
            PyArray1::from_vec(py, preds),
            PyArray1::from_vec(py, lower),
            PyArray1::from_vec(py, upper),
        ))
    }

    /// Predict class probabilities (for binary classification)
    ///
    /// Applies sigmoid to raw predictions to get probabilities in [0, 1].
    /// Only meaningful when trained with `use_binary_logloss()`.
    ///
    /// For multi-class models, use `predict_proba_multiclass()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     1D numpy array of probabilities (probability of class 1)
    #[pyo3(signature = (features))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Check if this is a multi-class model and provide helpful error
        if self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_proba() is for binary classification only. \
                 For multi-class models, use predict_proba_multiclass() instead."
            ));
        }

        // Try f64 first, then f32
        let (num_features, raw_features) = if let Ok(arr) =
            features.extract::<PyReadonlyArray2<'py, f64>>()
        {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().copied());
            }
            (num_cols, raw)
        } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().map(|&v| v as f64));
            }
            (num_cols, raw)
        } else {
            return Err(PyValueError::new_err(
                "features must be a 2D numpy array of float32 or float64",
            ));
        };

        if num_features != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.model.num_features(),
                num_features
            )));
        }

        // Predict probabilities (release GIL)
        let proba = py.allow_threads(|| self.model.predict_proba_raw(&raw_features));

        Ok(PyArray1::from_vec(py, proba))
    }

    /// Predict class labels (for binary classification)
    ///
    /// Applies sigmoid to raw predictions and thresholds.
    /// Only meaningful when trained with `use_binary_logloss()`.
    ///
    /// For multi-class models, use `predict_class_multiclass()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///     threshold: Classification threshold (default 0.5)
    ///
    /// Returns:
    ///     1D numpy array of class labels (0 or 1)
    #[pyo3(signature = (features, threshold = 0.5))]
    fn predict_class<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        // Check if this is a multi-class model and provide helpful error
        if self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_class() is for binary classification only. \
                 For multi-class models, use predict_class_multiclass() instead."
            ));
        }

        // Try f64 first, then f32
        let (num_features, raw_features) = if let Ok(arr) =
            features.extract::<PyReadonlyArray2<'py, f64>>()
        {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().copied());
            }
            (num_cols, raw)
        } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().map(|&v| v as f64));
            }
            (num_cols, raw)
        } else {
            return Err(PyValueError::new_err(
                "features must be a 2D numpy array of float32 or float64",
            ));
        };

        if num_features != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.model.num_features(),
                num_features
            )));
        }

        // Predict classes (release GIL)
        let classes = py.allow_threads(|| self.model.predict_class_raw(&raw_features, threshold));

        Ok(PyArray1::from_vec(py, classes))
    }

    // Multi-class classification methods

    /// Check if this is a multi-class model
    #[getter]
    fn is_multiclass(&self) -> bool {
        self.model.is_multiclass()
    }

    /// Get number of classes (0 for regression/binary)
    #[getter]
    fn num_classes(&self) -> usize {
        self.model.get_num_classes()
    }

    /// Predict class probabilities for multi-class classification
    ///
    /// Applies softmax to raw predictions to get probabilities for each class.
    /// Only meaningful when trained with `use_multiclass_logloss()`.
    ///
    /// For binary classification, use `predict_proba()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     2D numpy array of shape (n_samples, n_classes) with probabilities
    #[pyo3(signature = (features))]
    fn predict_proba_multiclass<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        use numpy::PyArray2;

        // Try f64 first, then f32
        let (num_features, raw_features) = if let Ok(arr) =
            features.extract::<PyReadonlyArray2<'py, f64>>()
        {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().copied());
            }
            (num_cols, raw)
        } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().map(|&v| v as f64));
            }
            (num_cols, raw)
        } else {
            return Err(PyValueError::new_err(
                "features must be a 2D numpy array of float32 or float64",
            ));
        };

        if num_features != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.model.num_features(),
                num_features
            )));
        }

        if !self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_proba_multiclass() is for multi-class models only. \
                 For binary classification, use predict_proba() instead."
            ));
        }

        let num_rows = raw_features.len() / num_features;
        let num_classes = self.model.get_num_classes();

        // Use the raw prediction method (no binning needed, release GIL)
        let proba = py.allow_threads(|| self.model.predict_proba_multiclass_raw(&raw_features));

        // Convert Vec<Vec<f32>> to 2D numpy array
        let flat: Vec<f32> = proba.into_iter().flatten().collect();
        PyArray2::from_vec(py, flat)
            .map_err(|e| PyValueError::new_err(format!("Failed to create numpy array: {:?}", e)))?
            .reshape([num_rows, num_classes])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape array: {:?}", e)))
    }

    /// Predict class labels for multi-class classification
    ///
    /// Returns the class with highest probability (argmax of softmax).
    /// Only meaningful when trained with `use_multiclass_logloss()`.
    ///
    /// For binary classification, use `predict_class()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     1D numpy array of class labels (0, 1, 2, ..., K-1)
    #[pyo3(signature = (features))]
    fn predict_class_multiclass<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        // Try f64 first, then f32
        let (num_features, raw_features) = if let Ok(arr) =
            features.extract::<PyReadonlyArray2<'py, f64>>()
        {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().copied());
            }
            (num_cols, raw)
        } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
            let arr = arr.as_array();
            let num_rows = arr.nrows();
            let num_cols = arr.ncols();
            let mut raw = Vec::with_capacity(num_rows * num_cols);
            for row in arr.rows() {
                raw.extend(row.iter().map(|&v| v as f64));
            }
            (num_cols, raw)
        } else {
            return Err(PyValueError::new_err(
                "features must be a 2D numpy array of float32 or float64",
            ));
        };

        if num_features != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.model.num_features(),
                num_features
            )));
        }

        if !self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_class_multiclass() is for multi-class models only. \
                 For binary classification, use predict_class() instead."
            ));
        }

        // Use the raw prediction method (no binning needed, release GIL)
        let classes = py.allow_threads(|| self.model.predict_class_multiclass_raw(&raw_features));

        Ok(PyArray1::from_vec(py, classes))
    }

    /// Get feature importances (gain-based)
    ///
    /// Returns:
    ///     1D numpy array of normalized feature importances
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let importances = self.model.feature_importances(self.model.num_features());
        PyArray1::from_vec(py, importances)
    }

    /// Save model to file (rkyv format)
    ///
    /// Args:
    ///     path: Path to save the model
    fn save(&self, path: &str) -> PyResult<()> {
        serialize::save_model(&self.model, path)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load model from file (rkyv format)
    ///
    /// Args:
    ///     path: Path to the model file
    ///
    /// Returns:
    ///     Loaded GBDTModel
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let model = serialize::load_model(path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(Self { model })
    }

    /// Get number of trees in the ensemble
    #[getter]
    fn num_trees(&self) -> usize {
        self.model.num_trees()
    }

    /// Get number of features
    #[getter]
    fn num_features(&self) -> usize {
        self.model.num_features()
    }

    /// Get base prediction value
    #[getter]
    fn base_prediction(&self) -> f32 {
        self.model.base_prediction()
    }

    /// Get conformal quantile (if calibrated)
    #[getter]
    fn conformal_quantile(&self) -> Option<f32> {
        self.model.conformal_quantile()
    }

    /// Get feature names
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model
            .feature_info()
            .iter()
            .map(|info| info.name.clone())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTModel(num_trees={}, num_features={}, base_prediction={:.4})",
            self.model.num_trees(),
            self.model.num_features(),
            self.model.base_prediction()
        )
    }
}

/// Register Python module classes and functions
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGBDTConfig>()?;
    m.add_class::<PyGBDTModel>()?;
    Ok(())
}
