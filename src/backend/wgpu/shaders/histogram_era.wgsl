// TreeBoost GPU Era Histogram Building Shader
//
// Computes era-stratified gradient/hessian histograms for Directional Era Splitting (DES).
// Each row belongs to an era, and histograms are accumulated per (era, feature, bin).
//
// Dispatch: (num_features, num_eras, 1)
// Each workgroup handles one (feature, era) pair.
//
// Output layout: [era * num_features * 256 + feature * 256 + bin]

// Uniform buffer with histogram parameters
struct Params {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,  // Length of row_indices (0 = use all rows)
    num_eras: u32,     // Number of unique eras
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins: array<u32>;  // Packed u8 bins [row][feature]
@group(0) @binding(2) var<storage, read> grad_hess: array<u32>;  // Packed i16 pairs: low=grad, high=hess
@group(0) @binding(3) var<storage, read> row_indices: array<u32>;  // Row indices (optional subset)
@group(0) @binding(7) var<storage, read> era_indices: array<u32>;  // Era index per row (packed as u16 in u32)

// Output histogram bins: fixed-point i32 for grad/hess, u32 for count
// Layout: [era * num_features * 256 + feature * 256 + bin]
@group(0) @binding(4) var<storage, read_write> hist_grad: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> hist_hess: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> hist_counts: array<atomic<u32>>;

// Workgroup shared memory for local histogram accumulation
// 256 bins x 3 values = 768 atomic i32/u32 = 3KB
var<workgroup> local_grad: array<atomic<i32>, 256>;
var<workgroup> local_hess: array<atomic<i32>, 256>;
var<workgroup> local_counts: array<atomic<u32>, 256>;

// Extract u8 bin value from packed u32
fn get_bin(packed: u32, byte_idx: u32) -> u32 {
    return (packed >> (byte_idx * 8u)) & 0xFFu;
}

// Extract u16 era index from packed u32
fn get_era(packed: u32, half_idx: u32) -> u32 {
    return (packed >> (half_idx * 16u)) & 0xFFFFu;
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
fn histogram_era(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let target_era = wg_id.y;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Determine row iteration bounds
    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    // Each thread processes a subset of rows, only accumulating rows in target_era
    for (var i = thread_id; i < total_rows; i += num_threads) {
        // Get actual row index
        let row = select(i, row_indices[i], params.num_indices > 0u);

        // Get era index for this row
        let era_packed_idx = row / 2u;
        let era_half_idx = row % 2u;
        let row_era = get_era(era_indices[era_packed_idx], era_half_idx);

        // Only process rows that belong to target_era
        if row_era != target_era {
            continue;
        }

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
    // Output layout: [era * num_features * 256 + feature * 256 + bin]
    let hist_stride = params.num_features * 256u;
    let global_offset = target_era * hist_stride + feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        // Direct store (each workgroup writes to unique location)
        atomicStore(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
        atomicStore(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
        atomicStore(&hist_counts[global_offset], local_count);
    }
}

// Zero out era histogram buffers
@compute @workgroup_size(256, 1, 1)
fn zero_histograms_era(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let total_bins = params.num_eras * params.num_features * 256u;

    if idx < total_bins {
        atomicStore(&hist_grad[idx], 0i);
        atomicStore(&hist_hess[idx], 0i);
        atomicStore(&hist_counts[idx], 0u);
    }
}
