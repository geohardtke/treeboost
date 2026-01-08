use treeboost::{BackendConfig, BackendPreset, BackendType};

#[test]
fn gpu_required_disables_fallback() {
    let cfg = BackendConfig::default().with_preset(BackendPreset::GpuRequired);
    assert_eq!(cfg.preferred, BackendType::Wgpu);
    assert!(!cfg.fallback_to_scalar);
}

#[test]
fn cpu_only_uses_scalar() {
    let cfg = BackendConfig::default().with_preset(BackendPreset::CpuOnly);
    assert_eq!(cfg.preferred, BackendType::Scalar);
    assert!(cfg.fallback_to_scalar);
}
