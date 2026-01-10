//! Global registry for custom preprocessor types
//!
//! This module provides a thread-safe global registry for registering custom
//! preprocessor types. Custom preprocessors must be registered before loading
//! models that contain them.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::preprocessing::{register_preprocessor, PreprocessorTrait};
//!
//! // Define your custom preprocessor
//! #[derive(Clone, Default)]
//! struct MyScaler { /* ... */ }
//!
//! impl PreprocessorTrait for MyScaler {
//!     fn type_id(&self) -> &'static str { "my_crate::MyScaler" }
//!     // ... other methods
//! }
//!
//! fn main() {
//!     // Register at startup, before loading any models
//!     register_preprocessor::<MyScaler>();
//!
//!     // Now models with MyScaler can be loaded
//!     let model = UniversalModel::load_trb("model.trb")?;
//! }
//! ```

use std::collections::HashMap;
use std::sync::RwLock;

use super::traits::{CustomPreprocessorDyn, PreprocessorTrait, PreprocessorWrapper};
use crate::Result;

// =============================================================================
// Types
// =============================================================================

/// Constructor function type: takes serialized state, returns boxed trait object
pub(crate) type ConstructorFn = fn(&[u8]) -> Result<Box<dyn CustomPreprocessorDyn>>;

// =============================================================================
// Global Registry
// =============================================================================

/// Global registry for custom preprocessor types.
///
/// Thread-safe via RwLock. Registration happens once at startup,
/// lookups happen frequently during deserialization.
static REGISTRY: RwLock<Option<HashMap<&'static str, ConstructorFn>>> = RwLock::new(None);

// =============================================================================
// Public API
// =============================================================================

/// Register a custom preprocessor type with the global registry.
///
/// Call this once at program startup (e.g., in `main()`) before loading any
/// models that contain this custom preprocessor type.
///
/// # Type Requirements
///
/// - `T` must implement [`PreprocessorTrait`]
/// - `T` must implement [`Default`] (for constructing instances during deserialization)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::{register_preprocessor, PreprocessorTrait};
///
/// #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
/// struct MyScaler {
///     means: Vec<f32>,
///     fitted: bool,
/// }
///
/// impl PreprocessorTrait for MyScaler {
///     fn type_id(&self) -> &'static str { "my_crate::MyScaler" }
///     // ... implement other methods
/// }
///
/// fn main() {
///     // Register before loading models
///     register_preprocessor::<MyScaler>();
/// }
/// ```
///
/// # Thread Safety
///
/// This function is thread-safe. Multiple threads can register different types
/// concurrently. However, it's recommended to register all types at startup
/// before any model loading begins.
///
/// # Duplicate Registration
///
/// If the same `type_id` is registered twice, a warning is printed to stderr
/// and the new registration overwrites the previous one. This can happen if:
/// - Two different types have the same `type_id()` (a bug)
/// - The same type is registered multiple times (usually harmless)
pub fn register_preprocessor<T>()
where
    T: PreprocessorTrait + Default,
{
    let mut guard = REGISTRY.write().expect("Registry lock poisoned");
    let registry = guard.get_or_insert_with(HashMap::new);

    // Get type_id from a dummy instance
    let dummy = T::default();
    let type_id = dummy.type_id();

    // Warn on duplicate registration
    if registry.contains_key(type_id) {
        eprintln!(
            "Warning: Preprocessor type '{}' is already registered. \
             The previous registration will be overwritten.",
            type_id
        );
    }

    // Create constructor function
    let constructor: ConstructorFn = |data: &[u8]| {
        let mut instance = T::default();
        instance.deserialize_state(data)?;
        Ok(Box::new(PreprocessorWrapper(instance)) as Box<dyn CustomPreprocessorDyn>)
    };

    registry.insert(type_id, constructor);
}

/// Check if a type is registered in the global registry.
///
/// # Example
///
/// ```ignore
/// if is_registered("my_crate::MyScaler") {
///     println!("MyScaler is available");
/// }
/// ```
pub fn is_registered(type_id: &str) -> bool {
    let guard = REGISTRY.read().expect("Registry lock poisoned");
    guard
        .as_ref()
        .map(|r| r.contains_key(type_id))
        .unwrap_or(false)
}

/// Get the list of all registered type IDs.
///
/// Useful for debugging or listing available custom preprocessors.
pub fn registered_types() -> Vec<&'static str> {
    let guard = REGISTRY.read().expect("Registry lock poisoned");
    guard
        .as_ref()
        .map(|r| r.keys().copied().collect())
        .unwrap_or_default()
}

// =============================================================================
// Internal API
// =============================================================================

/// Look up a constructor by type ID.
///
/// Returns `None` if the type is not registered.
pub(crate) fn get_constructor(type_id: &str) -> Option<ConstructorFn> {
    let guard = REGISTRY.read().expect("Registry lock poisoned");
    guard.as_ref()?.get(type_id).copied()
}

/// Construct a custom preprocessor from its type ID and serialized state.
///
/// This is called during deserialization of the `Custom` variant.
pub(crate) fn construct_preprocessor(
    type_id: &str,
    serialized_state: &[u8],
) -> Result<Box<dyn CustomPreprocessorDyn>> {
    let constructor = get_constructor(type_id).ok_or_else(|| {
        crate::TreeBoostError::Serialization(format!(
            "Unknown preprocessor type: '{}'. \
             Did you call register_preprocessor::<YourType>() before loading this model? \
             Registered types: {:?}",
            type_id,
            registered_types()
        ))
    })?;

    constructor(serialized_state)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TreeBoostError;

    #[derive(Clone, Default)]
    struct TestPreprocessor {
        value: f32,
        fitted: bool,
    }

    impl PreprocessorTrait for TestPreprocessor {
        fn type_id(&self) -> &'static str {
            "test::TestPreprocessor"
        }

        fn supports_numerical(&self) -> bool {
            true
        }

        fn fit_numerical(&mut self, data: &[f32], _num_features: usize) -> Result<()> {
            self.value = data.iter().sum();
            self.fitted = true;
            Ok(())
        }

        fn transform_numerical(&self, data: &mut [f32], _num_features: usize) -> Result<()> {
            for x in data.iter_mut() {
                *x += self.value;
            }
            Ok(())
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn serialize_state(&self) -> Result<Vec<u8>> {
            Ok(format!("{}|{}", self.value, self.fitted).into_bytes())
        }

        fn deserialize_state(&mut self, data: &[u8]) -> Result<()> {
            let s = std::str::from_utf8(data)
                .map_err(|e| TreeBoostError::Serialization(e.to_string()))?;
            let parts: Vec<&str> = s.split('|').collect();
            if parts.len() != 2 {
                return Err(TreeBoostError::Serialization("Invalid format".into()));
            }
            self.value = parts[0].parse().map_err(|e: std::num::ParseFloatError| {
                TreeBoostError::Serialization(e.to_string())
            })?;
            self.fitted = parts[1] == "true";
            Ok(())
        }
    }

    #[test]
    fn test_register_and_lookup() {
        register_preprocessor::<TestPreprocessor>();
        assert!(is_registered("test::TestPreprocessor"));
        assert!(!is_registered("nonexistent::Type"));
    }

    #[test]
    fn test_construct_preprocessor() {
        register_preprocessor::<TestPreprocessor>();

        // Create serialized state
        let state = "42.5|true".as_bytes();

        // Construct from registry
        let instance = construct_preprocessor("test::TestPreprocessor", state).unwrap();

        assert_eq!(instance.type_id(), "test::TestPreprocessor");
        assert!(instance.is_fitted());
    }

    #[test]
    fn test_unknown_type_error() {
        let result = construct_preprocessor("unknown::Type", &[]);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unknown preprocessor type"));
        assert!(err.to_string().contains("register_preprocessor"));
    }

    #[test]
    fn test_registered_types() {
        register_preprocessor::<TestPreprocessor>();

        let types = registered_types();
        assert!(types.contains(&"test::TestPreprocessor"));
    }
}
