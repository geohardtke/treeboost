// TreeBoost GPU Histogram Building Shader - Subgroup Optimized
//
// Computes gradient/hessian histograms for GBDT training.
// One workgroup per feature, 256 threads per workgroup.
//
// Optimization: Uses subgroup operations to reduce atomic contention.
// When multiple threads in a subgroup write to the SAME bin, we use
// subgroupAdd to combine their values before doing a single atomic write.
//
// This helps when data has locality - adjacent rows often have the same
// bin value for a feature, so threads processing adjacent rows hit the same bin.
//
// Requires: wgpu::Features::SUBGROUP
//
// Fixed-point format (same as base shader):
// - Gradients/hessians packed as i16 pairs in u32 (2x bandwidth reduction)
// - Scale factor: 2^10 = 1024
//
// Note: The `enable subgroups;` directive is not used because naga (wgpu's WGSL
// compiler) hasn't fully implemented the WGSL subgroups extension yet.
// The subgroup builtins work directly when the SUBGROUP feature is requested.

// Uniform buffer with histogram parameters
struct Params {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,
    num_batches: u32,
}

struct BatchInfo {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> grad_hess: array<u32>;
@group(0) @binding(3) var<storage, read> row_indices: array<u32>;

@group(0) @binding(4) var<storage, read_write> hist_grad: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> hist_hess: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> hist_counts: array<atomic<u32>>;

@group(0) @binding(11) var<storage, read> batch_info: array<BatchInfo>;

var<workgroup> local_grad: array<atomic<i32>, 256>;
var<workgroup> local_hess: array<atomic<i32>, 256>;
var<workgroup> local_counts: array<atomic<u32>, 256>;

fn get_bin(packed: u32, byte_idx: u32) -> u32 {
    return (packed >> (byte_idx * 8u)) & 0xFFu;
}

fn unpack_grad(packed: u32) -> i32 {
    let raw = packed & 0xFFFFu;
    if (raw & 0x8000u) != 0u {
        return i32(raw | 0xFFFF0000u);
    }
    return i32(raw);
}

fn unpack_hess(packed: u32) -> i32 {
    let raw = (packed >> 16u) & 0xFFFFu;
    if (raw & 0x8000u) != 0u {
        return i32(raw | 0xFFFF0000u);
    }
    return i32(raw);
}

// Subgroup-optimized histogram building
//
// Key optimization: Use subgroup broadcast and reduction to combine
// values from threads that hit the same bin.
//
// For each row:
// 1. Each thread gets its bin and values
// 2. Use subgroupBroadcastFirst to get the "pivot" bin from first active thread
// 3. Threads with matching bin contribute to subgroupAdd
// 4. First thread does one atomic for all matching threads
// 5. Repeat for non-matching threads (process in waves)
//
// Best case: All threads hit same bin -> 1 atomic instead of subgroup_size
// Worst case: All different bins -> same as baseline (each thread does atomic)

@compute @workgroup_size(256, 1, 1)
fn histogram_dense_subgroups(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let feature = wg_id.x;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Initialize shared memory
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    // Process rows
    for (var i = thread_id; i < total_rows; i += num_threads) {
        let row = select(i, row_indices[i], params.num_indices > 0u);

        // Get bin value
        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        // Get gradient and hessian
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Subgroup optimization: Check if multiple threads have the same bin
        // Use ballot to find threads with same bin as first thread's bin
        let first_bin = subgroupBroadcastFirst(bin);
        let matches_first = (bin == first_bin);

        if matches_first {
            // Multiple threads might have this bin - use reduction
            let sum_grad = subgroupAdd(grad);
            let sum_hess = subgroupAdd(hess);
            let sum_count = subgroupAdd(1u);

            // First thread in subgroup does the atomic
            if sg_id == 0u {
                atomicAdd(&local_grad[first_bin], sum_grad);
                atomicAdd(&local_hess[first_bin], sum_hess);
                atomicAdd(&local_counts[first_bin], sum_count);
            }
        } else {
            // This thread has a different bin - do direct atomic
            // (These threads were excluded from the subgroup reduction above)
            atomicAdd(&local_grad[bin], grad);
            atomicAdd(&local_hess[bin], hess);
            atomicAdd(&local_counts[bin], 1u);
        }
    }

    workgroupBarrier();

    // Write to global memory
    let global_offset = feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        atomicAdd(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
        atomicAdd(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
        atomicAdd(&hist_counts[global_offset], local_count);
    }
}

// Zero histograms (same as base shader)
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

// Batched histogram with subgroups
@compute @workgroup_size(256, 1, 1)
fn histogram_batched_subgroups(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let feature = wg_id.x;
    let batch = wg_id.y;
    let thread_id = lid.x;
    let num_threads = 256u;

    let batch_start = batch_info[batch].start;
    let batch_count = batch_info[batch].count;

    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    for (var i = thread_id; i < batch_count; i += num_threads) {
        let row = row_indices[batch_start + i];

        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        let first_bin = subgroupBroadcastFirst(bin);
        let matches_first = (bin == first_bin);

        if matches_first {
            let sum_grad = subgroupAdd(grad);
            let sum_hess = subgroupAdd(hess);
            let sum_count = subgroupAdd(1u);

            if sg_id == 0u {
                atomicAdd(&local_grad[first_bin], sum_grad);
                atomicAdd(&local_hess[first_bin], sum_hess);
                atomicAdd(&local_counts[first_bin], sum_count);
            }
        } else {
            atomicAdd(&local_grad[bin], grad);
            atomicAdd(&local_hess[bin], hess);
            atomicAdd(&local_counts[bin], 1u);
        }
    }

    workgroupBarrier();

    let hist_stride = params.num_features * 256u;
    let global_offset = batch * hist_stride + feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        atomicStore(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
        atomicStore(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
        atomicStore(&hist_counts[global_offset], local_count);
    }
}

// Zero batched histograms
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
