//! Polynomial feature generator
//!
//! Generates polynomial transformations of input features.

use super::FeatureGenerator;

/// Polynomial feature generator
///
/// Generates common polynomial transformations:
/// - x² (square)
/// - x³ (cube, if enabled)
/// - √|x| (square root)
/// - log(|x| + 1) (log1p)
///
/// # Example
///
/// ```ignore
/// let poly = PolynomialGenerator::new()
///     .with_square()
///     .with_sqrt()
///     .with_log1p();
///
/// let (features, names) = poly.generate(&data, num_features, &feature_names);
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialGenerator {
    /// Generate x²
    include_square: bool,
    /// Generate x³
    include_cube: bool,
    /// Generate √|x|
    include_sqrt: bool,
    /// Generate log(|x| + 1)
    include_log1p: bool,
}

impl PolynomialGenerator {
    /// Create a new polynomial generator with default settings
    ///
    /// By default, generates x² and √|x|.
    pub fn new() -> Self {
        Self {
            include_square: true,
            include_cube: false,
            include_sqrt: true,
            include_log1p: false,
        }
    }

    /// Enable all polynomial features
    pub fn all() -> Self {
        Self {
            include_square: true,
            include_cube: true,
            include_sqrt: true,
            include_log1p: true,
        }
    }

    /// Create with no features enabled
    pub fn none() -> Self {
        Self {
            include_square: false,
            include_cube: false,
            include_sqrt: false,
            include_log1p: false,
        }
    }

    /// Enable x² feature
    pub fn with_square(mut self) -> Self {
        self.include_square = true;
        self
    }

    /// Enable x³ feature
    pub fn with_cube(mut self) -> Self {
        self.include_cube = true;
        self
    }

    /// Enable √|x| feature
    pub fn with_sqrt(mut self) -> Self {
        self.include_sqrt = true;
        self
    }

    /// Enable log(|x| + 1) feature
    pub fn with_log1p(mut self) -> Self {
        self.include_log1p = true;
        self
    }

    /// Get the number of features generated per input feature
    pub fn features_per_input(&self) -> usize {
        let mut count = 0;
        if self.include_square {
            count += 1;
        }
        if self.include_cube {
            count += 1;
        }
        if self.include_sqrt {
            count += 1;
        }
        if self.include_log1p {
            count += 1;
        }
        count
    }
}

impl Default for PolynomialGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureGenerator for PolynomialGenerator {
    fn generate(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
    ) -> (Vec<f32>, Vec<String>) {
        if num_features == 0 || data.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_rows = data.len() / num_features;
        let features_per_input = self.features_per_input();

        if features_per_input == 0 {
            return (Vec::new(), Vec::new());
        }

        let total_new_features = num_features * features_per_input;
        let mut new_data = vec![0.0f32; num_rows * total_new_features];
        let mut new_names = Vec::with_capacity(total_new_features);

        // Generate features for each input feature
        for f in 0..num_features {
            let name = feature_names
                .get(f)
                .cloned()
                .unwrap_or_else(|| format!("f{}", f));

            let mut offset = f * features_per_input;

            // Extract column values
            let values: Vec<f32> = (0..num_rows).map(|r| data[r * num_features + f]).collect();

            // x²
            if self.include_square {
                for (r, &v) in values.iter().enumerate() {
                    new_data[r * total_new_features + offset] = v * v;
                }
                new_names.push(format!("{}_sq", name));
                offset += 1;
            }

            // x³
            if self.include_cube {
                for (r, &v) in values.iter().enumerate() {
                    new_data[r * total_new_features + offset] = v * v * v;
                }
                new_names.push(format!("{}_cb", name));
                offset += 1;
            }

            // √|x|
            if self.include_sqrt {
                for (r, &v) in values.iter().enumerate() {
                    new_data[r * total_new_features + offset] = v.abs().sqrt();
                }
                new_names.push(format!("{}_sqrt", name));
                offset += 1;
            }

            // log(|x| + 1)
            if self.include_log1p {
                for (r, &v) in values.iter().enumerate() {
                    new_data[r * total_new_features + offset] = (v.abs() + 1.0).ln();
                }
                new_names.push(format!("{}_log1p", name));
            }
        }

        (new_data, new_names)
    }

    fn name(&self) -> &'static str {
        "polynomial"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_default() {
        let poly = PolynomialGenerator::new();
        assert!(poly.include_square);
        assert!(!poly.include_cube);
        assert!(poly.include_sqrt);
        assert!(!poly.include_log1p);
        assert_eq!(poly.features_per_input(), 2);
    }

    #[test]
    fn test_polynomial_all() {
        let poly = PolynomialGenerator::all();
        assert_eq!(poly.features_per_input(), 4);
    }

    #[test]
    fn test_polynomial_none() {
        let poly = PolynomialGenerator::none();
        assert_eq!(poly.features_per_input(), 0);
    }

    #[test]
    fn test_generate_square() {
        let poly = PolynomialGenerator::none().with_square();

        // 2 rows, 2 features
        let data = vec![2.0, 3.0, 4.0, 5.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = poly.generate(&data, 2, &names);

        assert_eq!(new_names.len(), 2);
        assert_eq!(new_names[0], "a_sq");
        assert_eq!(new_names[1], "b_sq");

        // 2 new features, 2 rows
        assert_eq!(new_data.len(), 4);
        assert!((new_data[0] - 4.0).abs() < 1e-6); // 2² = 4
        assert!((new_data[1] - 9.0).abs() < 1e-6); // 3² = 9
        assert!((new_data[2] - 16.0).abs() < 1e-6); // 4² = 16
        assert!((new_data[3] - 25.0).abs() < 1e-6); // 5² = 25
    }

    #[test]
    fn test_generate_sqrt() {
        let poly = PolynomialGenerator::none().with_sqrt();

        let data = vec![4.0, 9.0, 16.0, 25.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = poly.generate(&data, 2, &names);

        assert_eq!(new_names.len(), 2);
        assert_eq!(new_names[0], "a_sqrt");

        assert!((new_data[0] - 2.0).abs() < 1e-6); // √4 = 2
        assert!((new_data[1] - 3.0).abs() < 1e-6); // √9 = 3
    }

    #[test]
    fn test_generate_negative_sqrt() {
        let poly = PolynomialGenerator::none().with_sqrt();

        // Negative values should use absolute value
        let data = vec![-4.0, 9.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, _) = poly.generate(&data, 2, &names);

        assert!((new_data[0] - 2.0).abs() < 1e-6); // √|-4| = 2
    }

    #[test]
    fn test_generate_log1p() {
        let poly = PolynomialGenerator::none().with_log1p();

        let data = vec![0.0, 1.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = poly.generate(&data, 2, &names);

        assert_eq!(new_names[0], "a_log1p");
        assert!((new_data[0] - 0.0).abs() < 1e-6); // log(|0| + 1) = log(1) = 0
        assert!((new_data[1] - 2.0f32.ln()).abs() < 1e-6); // log(|1| + 1) = log(2)
    }

    #[test]
    fn test_generate_empty() {
        let poly = PolynomialGenerator::new();
        let (new_data, new_names) = poly.generate(&[], 0, &[]);
        assert!(new_data.is_empty());
        assert!(new_names.is_empty());
    }

    #[test]
    fn test_generate_multiple() {
        let poly = PolynomialGenerator::new(); // square + sqrt

        let data = vec![4.0, 9.0]; // 1 row, 2 features
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = poly.generate(&data, 2, &names);

        // 2 features × 2 transforms = 4 new features
        assert_eq!(new_names.len(), 4);
        assert_eq!(new_data.len(), 4);

        // Check names
        assert!(new_names.contains(&"a_sq".to_string()));
        assert!(new_names.contains(&"a_sqrt".to_string()));
        assert!(new_names.contains(&"b_sq".to_string()));
        assert!(new_names.contains(&"b_sqrt".to_string()));
    }
}
