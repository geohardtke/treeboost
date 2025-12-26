// TreeBoost GPU Histogram Building Shader
//
// Computes gradient/hessian histograms for GBDT training.
// One workgroup per feature, 256 threads per workgroup.
//
// Optimization: Uses fixed-point arithmetic with native atomicAdd.
// This eliminates expensive CAS loops required for f32 atomics.
//
// Fixed-point format:
// - Gradients/hessians are packed as i16 pairs in u32 (2x bandwidth reduction)
// - Each u32 contains: low 16 bits = gradient, high 16 bits = hessian
// - Scale factor: 2^10 = 1024 (allows values up to ±31.99 with 0.001 precision)
// - Shader unpacks to i32 and uses native atomicAdd
// - Results are de-scaled on CPU after download
//
// Data layout:
// - bins: row-major [row * num_features + feature] -> u32 (packed as u8)
// - grad_hess: [row] -> u32 (packed i16 gradient in low bits, i16 hessian in high bits)
// - histograms: [feature * 256 + bin] -> BinEntry (single batch)
// - histograms (batched): [batch * num_features * 256 + feature * 256 + bin]

// Uniform buffer with histogram parameters
struct Params {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,  // Length of row_indices (0 = use all rows) - for single batch mode
    num_batches: u32,  // Number of batches (0 or 1 = single batch mode, >1 = batched mode)
}

// Batch descriptor: start offset and count in the concatenated row_indices array
struct BatchInfo {
    start: u32,   // Start offset in row_indices
    count: u32,   // Number of rows in this batch
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins: array<u32>;  // Packed u8 bins
@group(0) @binding(2) var<storage, read> grad_hess: array<u32>;  // Packed i16 pairs: low=grad, high=hess
@group(0) @binding(3) var<storage, read> row_indices: array<u32>;  // Row indices (concatenated for batched mode)

// Output histogram bins: fixed-point i32 for grad/hess, u32 for count
@group(0) @binding(4) var<storage, read_write> hist_grad: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> hist_hess: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> hist_counts: array<atomic<u32>>;

// Batch info for batched mode: array of (start, count) pairs
@group(0) @binding(11) var<storage, read> batch_info: array<BatchInfo>;

// Workgroup shared memory for local histogram accumulation
// 256 bins × 3 values = 768 atomic i32/u32 = 3KB
var<workgroup> local_grad: array<atomic<i32>, 256>;
var<workgroup> local_hess: array<atomic<i32>, 256>;
var<workgroup> local_counts: array<atomic<u32>, 256>;

// Extract u8 bin value from packed u32
fn get_bin(packed: u32, byte_idx: u32) -> u32 {
    return (packed >> (byte_idx * 8u)) & 0xFFu;
}

// Unpack i16 gradient from low 16 bits of packed u32, with sign extension to i32
fn unpack_grad(packed: u32) -> i32 {
    let raw = packed & 0xFFFFu;
    // Sign extend: if bit 15 is set, extend to full i32
    if (raw & 0x8000u) != 0u {
        return i32(raw | 0xFFFF0000u);
    }
    return i32(raw);
}

// Unpack i16 hessian from high 16 bits of packed u32, with sign extension to i32
fn unpack_hess(packed: u32) -> i32 {
    let raw = (packed >> 16u) & 0xFFFFu;
    // Sign extend: if bit 15 is set, extend to full i32
    if (raw & 0x8000u) != 0u {
        return i32(raw | 0xFFFF0000u);
    }
    return i32(raw);
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
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Determine row iteration bounds
    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    // Each thread processes a subset of rows
    // Accumulate to workgroup shared memory using native atomicAdd (fast!)
    for (var i = thread_id; i < total_rows; i += num_threads) {
        // Get actual row index
        let row = select(i, row_indices[i], params.num_indices > 0u);

        // Get bin value for this feature
        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        // Get gradient and hessian (packed i16 in u32)
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Native atomicAdd for i32 - no CAS loops!
        atomicAdd(&local_grad[bin], grad);
        atomicAdd(&local_hess[bin], hess);
        atomicAdd(&local_counts[bin], 1u);
    }

    // Barrier: wait for all threads to finish accumulating
    workgroupBarrier();

    // Write shared memory histogram to global memory
    // Each thread writes one bin (no contention between workgroups)
    let global_offset = feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        // Native atomicAdd for global memory too
        atomicAdd(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
        atomicAdd(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
        atomicAdd(&hist_counts[global_offset], local_count);
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
        atomicStore(&hist_grad[idx], 0i);
        atomicStore(&hist_hess[idx], 0i);
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
@group(0) @binding(10) var<storage, read> total_sums: array<i32>;      // [total_grad, total_hess, total_count] per feature (fixed-point)

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

    // Get total sums for this feature (pre-computed on CPU, fixed-point)
    let total_grad = total_sums[feature * 3u];
    let total_hess = total_sums[feature * 3u + 1u];
    let total_count = bitcast<u32>(total_sums[feature * 3u + 2u]);

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
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

        // Get gradient and hessian (packed i16 in u32)
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Native atomicAdd
        atomicAdd(&local_grad[bin], grad);
        atomicAdd(&local_hess[bin], hess);
        atomicAdd(&local_counts[bin], 1u);
    }

    workgroupBarrier();

    // Write results to global memory with default-bin subtraction
    let global_offset = feature * 256u + thread_id;
    var local_count_val = atomicLoad(&local_counts[thread_id]);
    var local_grad_val = atomicLoad(&local_grad[thread_id]);
    var local_hess_val = atomicLoad(&local_hess[thread_id]);

    // Special handling for default bin: subtract accumulated values from totals
    if thread_id == default_bin {
        // Compute sum of all non-default entries
        var sum_grad = 0i;
        var sum_hess = 0i;
        var sum_count = 0u;

        for (var b = 0u; b < 256u; b++) {
            if b != default_bin {
                sum_grad += atomicLoad(&local_grad[b]);
                sum_hess += atomicLoad(&local_hess[b]);
                sum_count += atomicLoad(&local_counts[b]);
            }
        }

        // Default bin = totals - non-default sum
        local_grad_val = total_grad - sum_grad;
        local_hess_val = total_hess - sum_hess;
        local_count_val = total_count - sum_count;
    }

    if local_count_val > 0u {
        atomicAdd(&hist_grad[global_offset], local_grad_val);
        atomicAdd(&hist_hess[global_offset], local_hess_val);
        atomicAdd(&hist_counts[global_offset], local_count_val);
    }
}

// ============================================================================
// Batched Histogram Kernel
// ============================================================================
//
// Processes multiple batches of row indices in a single dispatch.
// Dispatch: (num_features, num_batches, 1)
// Each workgroup handles one (feature, batch) pair.
//
// Output layout: [batch * num_features * 256 + feature * 256 + bin]

@compute @workgroup_size(256, 1, 1)
fn histogram_batched(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let batch = wg_id.y;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Get batch info
    let batch_start = batch_info[batch].start;
    let batch_count = batch_info[batch].count;

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Each thread processes a subset of rows from this batch
    for (var i = thread_id; i < batch_count; i += num_threads) {
        // Get actual row index from the batch's portion of row_indices
        let row = row_indices[batch_start + i];

        // Get bin value for this feature
        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        // Get gradient and hessian (packed i16 in u32)
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Native atomicAdd
        atomicAdd(&local_grad[bin], grad);
        atomicAdd(&local_hess[bin], hess);
        atomicAdd(&local_counts[bin], 1u);
    }

    // Barrier: wait for all threads to finish accumulating
    workgroupBarrier();

    // Write shared memory histogram to global memory
    // Output layout: [batch * num_features * 256 + feature * 256 + bin]
    let hist_stride = params.num_features * 256u;
    let global_offset = batch * hist_stride + feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        // Direct store (no atomics needed - each workgroup writes to unique location)
        atomicStore(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
        atomicStore(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
        atomicStore(&hist_counts[global_offset], local_count);
    }
}

// Zero out batched histogram buffers
@compute @workgroup_size(256, 1, 1)
fn zero_histograms_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let total_bins = params.num_batches * params.num_features * 256u;

    if idx < total_bins {
        atomicStore(&hist_grad[idx], 0i);
        atomicStore(&hist_hess[idx], 0i);
        atomicStore(&hist_counts[idx], 0u);
    }
}
