// GPU Split Finding Shader
//
// Finds the best split for each node's histogram using parallel reduction.
// Each workgroup handles one node, threads scan features to find best split.

struct SplitParams {
    num_nodes: u32,
    num_features: u32,
    num_bins: u32,
    lambda: f32,
    min_samples_leaf: u32,
    min_hessian_leaf: f32,
    min_gain: f32,
    _padding: u32,
}

struct NodeInfo {
    total_gradient: f32,
    total_hessian: f32,
    total_count: u32,
    _padding: u32,
}

struct SplitResult {
    feature_idx: u32,
    bin_threshold: u32,
    gain: f32,
    left_gradient: f32,
    left_hessian: f32,
    left_count: u32,
    right_gradient: f32,
    right_hessian: f32,
    right_count: u32,
    is_valid: u32,
    _padding: array<u32, 2>,
}

@group(0) @binding(0) var<uniform> params: SplitParams;
@group(0) @binding(1) var<storage, read> node_info: array<NodeInfo>;
// Histograms: [node][feature][bin] -> (grad_sum, hess_sum, count)
// Layout: histograms[node * num_features * num_bins * 3 + feature * num_bins * 3 + bin * 3 + {0=grad, 1=hess, 2=count}]
@group(0) @binding(2) var<storage, read> histograms: array<f32>;
@group(0) @binding(3) var<storage, read_write> split_results: array<SplitResult>;

// Shared memory for per-thread best splits
var<workgroup> thread_best_gain: array<f32, 256>;
var<workgroup> thread_best_packed: array<u32, 256>;  // bin | (feature << 16)
var<workgroup> thread_best_left_g: array<f32, 256>;
var<workgroup> thread_best_left_h: array<f32, 256>;
var<workgroup> thread_best_left_c: array<u32, 256>;

const WORKGROUP_SIZE: u32 = 256u;

// Compute split gain using the standard GBDT formula
fn compute_gain(
    left_g: f32, left_h: f32,
    right_g: f32, right_h: f32,
    lambda: f32
) -> f32 {
    let left_term = (left_g * left_g) / (left_h + lambda);
    let right_term = (right_g * right_g) / (right_h + lambda);
    let parent_term = ((left_g + right_g) * (left_g + right_g)) / (left_h + right_h + lambda);
    return 0.5 * (left_term + right_term - parent_term);
}

@compute @workgroup_size(256)
fn find_best_splits(@builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let node_idx = workgroup_id.x;
    let thread_id = local_id.x;

    if (node_idx >= params.num_nodes) {
        return;
    }

    let info = node_info[node_idx];
    let total_g = info.total_gradient;
    let total_h = info.total_hessian;
    let total_c = info.total_count;

    // Each thread handles one or more features
    var my_best_gain = -1e30f;
    var my_best_feature = 0u;
    var my_best_bin = 0u;
    var my_best_left_g = 0.0f;
    var my_best_left_h = 0.0f;
    var my_best_left_c = 0u;

    // Process features assigned to this thread
    var feature = thread_id;
    while (feature < params.num_features) {
        // Scan bins for this feature, accumulating prefix sums
        var left_g = 0.0f;
        var left_h = 0.0f;
        var left_c = 0u;

        let feature_base = (node_idx * params.num_features + feature) * params.num_bins * 3u;

        // Scan through bins (threshold = bin means values <= bin go left)
        for (var bin = 0u; bin < params.num_bins - 1u; bin = bin + 1u) {
            let bin_base = feature_base + bin * 3u;
            let bin_g = histograms[bin_base];
            let bin_h = histograms[bin_base + 1u];
            let bin_c = bitcast<u32>(histograms[bin_base + 2u]);

            left_g += bin_g;
            left_h += bin_h;
            left_c += bin_c;

            let right_g = total_g - left_g;
            let right_h = total_h - left_h;
            let right_c = total_c - left_c;

            // Check constraints
            if (left_c < params.min_samples_leaf || right_c < params.min_samples_leaf) {
                continue;
            }
            if (left_h < params.min_hessian_leaf || right_h < params.min_hessian_leaf) {
                continue;
            }

            let gain = compute_gain(left_g, left_h, right_g, right_h, params.lambda);

            if (gain > my_best_gain && gain > params.min_gain) {
                my_best_gain = gain;
                my_best_feature = feature;
                my_best_bin = bin;
                my_best_left_g = left_g;
                my_best_left_h = left_h;
                my_best_left_c = left_c;
            }
        }

        feature += WORKGROUP_SIZE;
    }

    // Store in shared memory for reduction
    thread_best_gain[thread_id] = my_best_gain;
    thread_best_packed[thread_id] = my_best_bin | (my_best_feature << 16u);
    thread_best_left_g[thread_id] = my_best_left_g;
    thread_best_left_h[thread_id] = my_best_left_h;
    thread_best_left_c[thread_id] = my_best_left_c;

    workgroupBarrier();

    // Parallel reduction to find global best
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride) {
            if (thread_best_gain[thread_id + stride] > thread_best_gain[thread_id]) {
                thread_best_gain[thread_id] = thread_best_gain[thread_id + stride];
                thread_best_packed[thread_id] = thread_best_packed[thread_id + stride];
                thread_best_left_g[thread_id] = thread_best_left_g[thread_id + stride];
                thread_best_left_h[thread_id] = thread_best_left_h[thread_id + stride];
                thread_best_left_c[thread_id] = thread_best_left_c[thread_id + stride];
            }
        }
        workgroupBarrier();
    }

    // Thread 0 writes the result
    if (thread_id == 0u) {
        let best_gain = thread_best_gain[0];
        let packed = thread_best_packed[0];
        let best_bin = packed & 0xFFFFu;
        let best_feature = packed >> 16u;
        let left_g = thread_best_left_g[0];
        let left_h = thread_best_left_h[0];
        let left_c = thread_best_left_c[0];

        var result: SplitResult;
        if (best_gain > params.min_gain) {
            result.feature_idx = best_feature;
            result.bin_threshold = best_bin;
            result.gain = best_gain;
            result.left_gradient = left_g;
            result.left_hessian = left_h;
            result.left_count = left_c;
            result.right_gradient = total_g - left_g;
            result.right_hessian = total_h - left_h;
            result.right_count = total_c - left_c;
            result.is_valid = 1u;
        } else {
            result.is_valid = 0u;
        }

        split_results[node_idx] = result;
    }
}
