// TreeBoost GPU Histogram Building Shader
//
// Computes gradient/hessian histograms for GBDT training.
// One workgroup per feature, 256 threads per workgroup.
//
// Optimization: Uses workgroup shared memory to reduce global atomic contention.
// Each workgroup accumulates to shared memory first, then writes once to global.
//
// Data layout:
// - bins: row-major [row * num_features + feature] -> u32 (packed as u8)
// - grad_hess: [row * 2], [row * 2 + 1] -> gradient, hessian
// - histograms: [feature * 256 + bin] -> BinEntry

// Uniform buffer with histogram parameters
struct Params {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,  // Length of row_indices (0 = use all rows)
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins: array<u32>;  // Packed u8 bins
@group(0) @binding(2) var<storage, read> grad_hess: array<f32>;  // Interleaved [g0,h0,g1,h1,...]
@group(0) @binding(3) var<storage, read> row_indices: array<u32>;  // Optional row indices

// Output histogram bins: [sum_grad (as u32 bits), sum_hess (as u32 bits), count]
@group(0) @binding(4) var<storage, read_write> hist_grad_bits: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> hist_hess_bits: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> hist_counts: array<atomic<u32>>;

// Workgroup shared memory for local histogram accumulation
// 256 bins × 3 values = 768 atomic u32s = 3KB
var<workgroup> local_grad_bits: array<atomic<u32>, 256>;
var<workgroup> local_hess_bits: array<atomic<u32>, 256>;
var<workgroup> local_counts: array<atomic<u32>, 256>;

// Extract u8 bin value from packed u32
fn get_bin(packed: u32, byte_idx: u32) -> u32 {
    return (packed >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(256, 1, 1)
fn histogram_dense(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Initialize shared memory histogram to zero
    // Each thread initializes one bin
    atomicStore(&local_grad_bits[thread_id], 0u);
    atomicStore(&local_hess_bits[thread_id], 0u);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Determine row iteration bounds
    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    // Each thread processes a subset of rows
    // Accumulate to workgroup shared memory (much faster than global atomics)
    for (var i = thread_id; i < total_rows; i += num_threads) {
        // Get actual row index
        let row = select(i, row_indices[i], params.num_indices > 0u);

        // Get bin value for this feature
        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        // Get gradient and hessian (interleaved layout)
        let grad = grad_hess[row * 2u];
        let hess = grad_hess[row * 2u + 1u];

        // Accumulate to shared memory using workgroup atomics
        atomicAdd(&local_counts[bin], 1u);

        // CAS loop for gradient (to shared memory)
        {
            var old_bits = atomicLoad(&local_grad_bits[bin]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + grad;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&local_grad_bits[bin], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }

        // CAS loop for hessian (to shared memory)
        {
            var old_bits = atomicLoad(&local_hess_bits[bin]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + hess;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&local_hess_bits[bin], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }
    }

    // Barrier: wait for all threads to finish accumulating
    workgroupBarrier();

    // Write shared memory histogram to global memory
    // Each thread writes one bin (no contention between workgroups)
    let global_offset = feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        atomicAdd(&hist_counts[global_offset], local_count);

        // CAS loop for gradient (to global memory)
        let local_grad = bitcast<f32>(atomicLoad(&local_grad_bits[thread_id]));
        {
            var old_bits = atomicLoad(&hist_grad_bits[global_offset]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + local_grad;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&hist_grad_bits[global_offset], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }

        // CAS loop for hessian (to global memory)
        let local_hess = bitcast<f32>(atomicLoad(&local_hess_bits[thread_id]));
        {
            var old_bits = atomicLoad(&hist_hess_bits[global_offset]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + local_hess;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&hist_hess_bits[global_offset], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }
    }
}

// Zero out histogram buffers
@compute @workgroup_size(256, 1, 1)
fn zero_histograms(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let total_bins = params.num_features * 256u;

    if idx < total_bins {
        atomicStore(&hist_grad_bits[idx], 0u);
        atomicStore(&hist_hess_bits[idx], 0u);
        atomicStore(&hist_counts[idx], 0u);
    }
}

// ============================================================================
// Sparse Histogram Kernel
// ============================================================================
//
// For sparse features (90%+ zeros), uses the default-bin subtraction trick:
// 1. Compute total gradient/hessian sums for all rows
// 2. Accumulate only non-default entries
// 3. Default bin = totals - sum(non-default entries)
//
// Sparse data layout:
// - sparse_indices: array of (row_idx, feature_offset) pairs
// - sparse_values: corresponding bin values (non-default)
// - sparse_feature_info: [feature_idx, start_offset, count, default_bin] per sparse feature

// Additional bindings for sparse kernel
@group(0) @binding(7) var<storage, read> sparse_indices: array<u32>;   // Row indices
@group(0) @binding(8) var<storage, read> sparse_values: array<u32>;    // Bin values (packed u8)
@group(0) @binding(9) var<storage, read> sparse_info: array<u32>;      // Per-feature: [start, count, default_bin, _padding]
@group(0) @binding(10) var<storage, read> total_sums: array<f32>;      // [total_grad, total_hess, total_count] per feature

// Sparse histogram: processes only non-default entries, subtracts from totals
@compute @workgroup_size(256, 1, 1)
fn histogram_sparse(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Get sparse feature info
    let info_offset = feature * 4u;
    let sparse_start = sparse_info[info_offset];
    let sparse_count = sparse_info[info_offset + 1u];
    let default_bin = sparse_info[info_offset + 2u];

    // Get total sums for this feature (pre-computed on CPU)
    let total_grad = total_sums[feature * 3u];
    let total_hess = total_sums[feature * 3u + 1u];
    let total_count = bitcast<u32>(total_sums[feature * 3u + 2u]);

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad_bits[thread_id], 0u);
    atomicStore(&local_hess_bits[thread_id], 0u);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Each thread processes a subset of sparse entries
    for (var i = thread_id; i < sparse_count; i += num_threads) {
        let entry_idx = sparse_start + i;
        let row = sparse_indices[entry_idx];

        // Get bin value (packed as u8 in u32)
        let packed_idx = entry_idx / 4u;
        let byte_idx = entry_idx % 4u;
        let bin = get_bin(sparse_values[packed_idx], byte_idx);

        // Get gradient and hessian
        let grad = grad_hess[row * 2u];
        let hess = grad_hess[row * 2u + 1u];

        // Accumulate to shared memory
        atomicAdd(&local_counts[bin], 1u);

        // CAS loop for gradient
        {
            var old_bits = atomicLoad(&local_grad_bits[bin]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + grad;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&local_grad_bits[bin], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }

        // CAS loop for hessian
        {
            var old_bits = atomicLoad(&local_hess_bits[bin]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + hess;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&local_hess_bits[bin], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }
    }

    workgroupBarrier();

    // Write results to global memory with default-bin subtraction
    let global_offset = feature * 256u + thread_id;
    var local_count = atomicLoad(&local_counts[thread_id]);
    var local_grad = bitcast<f32>(atomicLoad(&local_grad_bits[thread_id]));
    var local_hess = bitcast<f32>(atomicLoad(&local_hess_bits[thread_id]));

    // Special handling for default bin: subtract accumulated values from totals
    if thread_id == default_bin {
        // Compute sum of all non-default entries
        var sum_grad = 0.0f;
        var sum_hess = 0.0f;
        var sum_count = 0u;

        for (var b = 0u; b < 256u; b++) {
            if b != default_bin {
                sum_grad += bitcast<f32>(atomicLoad(&local_grad_bits[b]));
                sum_hess += bitcast<f32>(atomicLoad(&local_hess_bits[b]));
                sum_count += atomicLoad(&local_counts[b]);
            }
        }

        // Default bin = totals - non-default sum
        local_grad = total_grad - sum_grad;
        local_hess = total_hess - sum_hess;
        local_count = total_count - sum_count;
    }

    if local_count > 0u {
        atomicAdd(&hist_counts[global_offset], local_count);

        // CAS loop for gradient
        {
            var old_bits = atomicLoad(&hist_grad_bits[global_offset]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + local_grad;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&hist_grad_bits[global_offset], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }

        // CAS loop for hessian
        {
            var old_bits = atomicLoad(&hist_hess_bits[global_offset]);
            loop {
                let old_val = bitcast<f32>(old_bits);
                let new_val = old_val + local_hess;
                let new_bits = bitcast<u32>(new_val);
                let result = atomicCompareExchangeWeak(&hist_hess_bits[global_offset], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }
    }
}
