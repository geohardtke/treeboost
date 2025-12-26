// TreeBoost GPU Histogram Building Shader - 4-bit Bin Packing
//
// Optimized for datasets where all features have <=16 bins.
// Uses nibble (4-bit) packing for 50% memory bandwidth reduction.
//
// Data layout:
// - bins_4bit: row-major, 2 features per byte (nibble packed)
//   byte[i] = (feature[2i+1] << 4) | feature[2i]
// - grad_hess: [row] -> u32 (packed i16 gradient in low bits, i16 hessian in high bits)
// - histograms: [feature * 16 + bin] or [feature * 256 + bin] depending on mode

// Uniform buffer with histogram parameters
struct Params {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,  // Length of row_indices (0 = use all rows)
    num_batches: u32,  // Number of batches (0 or 1 = single batch mode)
}

// Batch descriptor for batched mode
struct BatchInfo {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins_4bit: array<u32>;  // Nibble-packed bins
@group(0) @binding(2) var<storage, read> grad_hess: array<u32>;  // Packed i16 pairs
@group(0) @binding(3) var<storage, read> row_indices: array<u32>;

// Output histogram bins: fixed-point i32 for grad/hess, u32 for count
// Note: Still using 256 bins per feature for compatibility with 8-bit path
@group(0) @binding(4) var<storage, read_write> hist_grad: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> hist_hess: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> hist_counts: array<atomic<u32>>;

// Batch info for batched mode
@group(0) @binding(11) var<storage, read> batch_info: array<BatchInfo>;

// Workgroup shared memory for local histogram accumulation
// Only need 16 bins for 4-bit, but use 256 for compatibility
var<workgroup> local_grad: array<atomic<i32>, 256>;
var<workgroup> local_hess: array<atomic<i32>, 256>;
var<workgroup> local_counts: array<atomic<u32>, 256>;

// Extract 4-bit bin value from nibble-packed u32
// nibble_idx: 0-7, selects which nibble within the u32
fn get_bin_4bit(packed: u32, nibble_idx: u32) -> u32 {
    return (packed >> (nibble_idx * 4u)) & 0xFu;
}

// Get bin value for a specific row and feature in 4-bit packed format
// bins_4bit layout: row-major, 2 features per byte
// For row R with F features: bytes_per_row = ceil(F/2)
// Byte index = R * bytes_per_row + feature/2
// Nibble index within byte = feature % 2 (low nibble = even feature)
fn get_bin_for_feature_4bit(row: u32, feature: u32, bytes_per_row: u32) -> u32 {
    let byte_offset = row * bytes_per_row + (feature / 2u);
    let u32_idx = byte_offset / 4u;
    let byte_in_u32 = byte_offset % 4u;
    let nibble_in_byte = feature % 2u;
    let nibble_idx = byte_in_u32 * 2u + nibble_in_byte;

    return get_bin_4bit(bins_4bit[u32_idx], nibble_idx);
}

// Unpack i16 gradient from low 16 bits of packed u32, with sign extension to i32
fn unpack_grad(packed: u32) -> i32 {
    let raw = packed & 0xFFFFu;
    if (raw & 0x8000u) != 0u {
        return i32(raw | 0xFFFF0000u);
    }
    return i32(raw);
}

// Unpack i16 hessian from high 16 bits of packed u32, with sign extension to i32
fn unpack_hess(packed: u32) -> i32 {
    let raw = (packed >> 16u) & 0xFFFFu;
    if (raw & 0x8000u) != 0u {
        return i32(raw | 0xFFFF0000u);
    }
    return i32(raw);
}

@compute @workgroup_size(256, 1, 1)
fn histogram_dense_4bit(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Calculate bytes per row for 4-bit packed format
    let bytes_per_row = (params.num_features + 1u) / 2u;

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Determine row iteration bounds
    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    // Each thread processes a subset of rows
    for (var i = thread_id; i < total_rows; i += num_threads) {
        // Get actual row index
        let row = select(i, row_indices[i], params.num_indices > 0u);

        // Get bin value for this feature (4-bit)
        let bin = get_bin_for_feature_4bit(row, feature, bytes_per_row);

        // Get gradient and hessian (packed i16 in u32)
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Native atomicAdd for i32
        atomicAdd(&local_grad[bin], grad);
        atomicAdd(&local_hess[bin], hess);
        atomicAdd(&local_counts[bin], 1u);
    }

    workgroupBarrier();

    // Write shared memory histogram to global memory
    // Only write first 16 bins (4-bit max), but use 256-bin stride for compatibility
    if thread_id < 16u {
        let global_offset = feature * 256u + thread_id;
        let local_count = atomicLoad(&local_counts[thread_id]);

        if local_count > 0u {
            atomicAdd(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
            atomicAdd(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
            atomicAdd(&hist_counts[global_offset], local_count);
        }
    }
}

// Zero out histogram buffers (same as 8-bit)
@compute @workgroup_size(256, 1, 1)
fn zero_histograms_4bit(
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

// Batched histogram kernel for 4-bit bins
@compute @workgroup_size(256, 1, 1)
fn histogram_batched_4bit(
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

    // Calculate bytes per row for 4-bit packed format
    let bytes_per_row = (params.num_features + 1u) / 2u;

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Each thread processes a subset of rows from this batch
    for (var i = thread_id; i < batch_count; i += num_threads) {
        let row = row_indices[batch_start + i];

        // Get bin value for this feature (4-bit)
        let bin = get_bin_for_feature_4bit(row, feature, bytes_per_row);

        // Get gradient and hessian
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Native atomicAdd
        atomicAdd(&local_grad[bin], grad);
        atomicAdd(&local_hess[bin], hess);
        atomicAdd(&local_counts[bin], 1u);
    }

    workgroupBarrier();

    // Write shared memory histogram to global memory
    // Only need to write first 16 bins for 4-bit, but use 256-bin stride
    if thread_id < 16u {
        let hist_stride = params.num_features * 256u;
        let global_offset = batch * hist_stride + feature * 256u + thread_id;
        let local_count = atomicLoad(&local_counts[thread_id]);

        if local_count > 0u {
            atomicStore(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
            atomicStore(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
            atomicStore(&hist_counts[global_offset], local_count);
        }
    }
}

// Zero out batched histogram buffers (same as 8-bit)
@compute @workgroup_size(256, 1, 1)
fn zero_histograms_batched_4bit(
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
