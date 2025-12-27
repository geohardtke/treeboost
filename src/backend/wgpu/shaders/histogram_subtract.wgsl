// GPU Histogram Subtraction Shader
//
// Computes: larger_child = parent - smaller_child
// This enables the histogram subtraction trick on GPU.

struct SubtractParams {
    num_features: u32,
    num_bins: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: SubtractParams;
@group(0) @binding(1) var<storage, read> parent_hist: array<f32>;      // [feature][bin][3]
@group(0) @binding(2) var<storage, read> smaller_hist: array<f32>;     // [feature][bin][3]
@group(0) @binding(3) var<storage, read_write> larger_hist: array<f32>; // [feature][bin][3]

@compute @workgroup_size(256)
fn subtract_histograms(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.num_features * params.num_bins * 3u;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Element-wise subtraction: larger = parent - smaller
    larger_hist[idx] = parent_hist[idx] - smaller_hist[idx];
}
