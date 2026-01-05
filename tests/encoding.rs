//! Integration tests for categorical encoding

use treeboost::encoding::{CategoryFilter, CategoryMapping, OrderedTargetEncoder};

#[test]
fn test_category_filter() {
    let mut filter = CategoryFilter::new(0.01, 0.99, 5);

    // Count categories
    for _ in 0..100 {
        filter.count("frequent_a");
        filter.count("frequent_b");
    }
    for _ in 0..10 {
        filter.count("medium");
    }
    for _ in 0..2 {
        filter.count("rare");
    }
    filter.count("very_rare");

    // Finalize
    filter.finalize(vec![
        "frequent_a".to_string(),
        "frequent_b".to_string(),
        "medium".to_string(),
        "rare".to_string(),
        "very_rare".to_string(),
    ]);

    // Frequent categories should be kept
    assert!(filter.is_frequent("frequent_a"));
    assert!(filter.is_frequent("frequent_b"));
    assert!(filter.is_frequent("medium")); // 10 > 5

    // Rare categories should be filtered
    assert!(!filter.is_frequent("rare")); // 2 < 5
    assert!(!filter.is_frequent("very_rare")); // 1 < 5
    assert!(!filter.is_frequent("unseen")); // 0 < 5

    // Filter function
    assert_eq!(filter.filter("frequent_a"), "frequent_a");
    assert_eq!(filter.filter("rare"), "unknown");
    assert_eq!(filter.filter("unseen"), "unknown");
}

#[test]
fn test_category_mapping() {
    let mut filter = CategoryFilter::new(0.01, 0.99, 3);

    for _ in 0..10 {
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

    // 3 frequent + 1 unknown
    assert_eq!(mapping.num_categories(), 4);

    // Indices should be unique and in range
    let idx_a = mapping.get_index("cat_a");
    let idx_b = mapping.get_index("cat_b");
    let idx_c = mapping.get_index("cat_c");
    let idx_rare = mapping.get_index("rare");

    assert!(idx_a < 3);
    assert!(idx_b < 3);
    assert!(idx_c < 3);
    assert_ne!(idx_a, idx_b);
    assert_ne!(idx_b, idx_c);
    assert_ne!(idx_a, idx_c);

    assert_eq!(idx_rare, mapping.unknown_idx);
    assert_eq!(mapping.get_index("unseen"), mapping.unknown_idx);
}

#[test]
fn test_ordered_target_encoder() {
    let categories = vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "C".to_string(),
    ];
    let targets = vec![10.0, 20.0, 12.0, 22.0, 14.0, 50.0];

    let mut encoder = OrderedTargetEncoder::new(5.0); // smoothing = 5

    let encoded = encoder.encode_column(&categories, &targets);

    // Ordered encoding: each row only sees PRIOR statistics
    // So first element gets 0 (no prior data), second gets mean of first, etc.
    assert_eq!(encoded.len(), 6);

    // All encoded values should be finite (not NaN or infinite)
    for &val in &encoded {
        assert!(val.is_finite(), "Encoded value should be finite");
    }

    // First element: no prior data -> global mean = 0
    assert_eq!(encoded[0], 0.0, "First element should be 0 (no prior data)");

    // Second element: global mean of first = 10.0
    assert!((encoded[1] - 10.0).abs() < 0.01, "Second should be ~10.0");

    // As more data accumulates, values become more meaningful
    // Check that later values are positive (non-trivial)
    assert!(encoded[5] > 0.0, "Later values should be positive");
}
