// TreeBoost GPU Histogram Building Shader - Register Cache Optimization
//
// Reduces shared memory atomic contention by caching recent bins in registers.
// When consecutive rows hit the same bin, we accumulate locally in registers
// and only flush to shared memory when the bin changes.
//
// This is particularly effective for sorted or semi-sorted data where
// consecutive rows often have similar bin values.
//
// Based on CatBoost's register-resident accumulation pattern.

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

// Register cache size - trade-off between register pressure and cache hit rate
// Using 4 slots as a balance (CatBoost uses 4-8)
const CACHE_SIZE: u32 = 4u;
const INVALID_BIN: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn histogram_dense_register_cache(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let thread_id = lid.x;
    let num_threads = 256u;

    // Initialize shared memory histogram to zero
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Register cache: stores (bin, accumulated_grad, accumulated_hess, count)
    // Using 4 cache slots per thread
    var cache_bins: array<u32, 4>;
    var cache_grads: array<i32, 4>;
    var cache_hess: array<i32, 4>;
    var cache_counts: array<u32, 4>;

    // Initialize cache as empty
    cache_bins[0] = INVALID_BIN;
    cache_bins[1] = INVALID_BIN;
    cache_bins[2] = INVALID_BIN;
    cache_bins[3] = INVALID_BIN;
    cache_grads[0] = 0i;
    cache_grads[1] = 0i;
    cache_grads[2] = 0i;
    cache_grads[3] = 0i;
    cache_hess[0] = 0i;
    cache_hess[1] = 0i;
    cache_hess[2] = 0i;
    cache_hess[3] = 0i;
    cache_counts[0] = 0u;
    cache_counts[1] = 0u;
    cache_counts[2] = 0u;
    cache_counts[3] = 0u;

    // Next cache slot to use (round-robin eviction)
    var next_slot: u32 = 0u;

    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    // Process rows with register caching
    for (var i = thread_id; i < total_rows; i += num_threads) {
        let row = select(i, row_indices[i], params.num_indices > 0u);

        // Get bin value for this feature
        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        // Get gradient and hessian
        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Check if bin is in cache (linear search through 4 slots)
        var found = false;
        if cache_bins[0] == bin {
            cache_grads[0] += grad;
            cache_hess[0] += hess;
            cache_counts[0] += 1u;
            found = true;
        } else if cache_bins[1] == bin {
            cache_grads[1] += grad;
            cache_hess[1] += hess;
            cache_counts[1] += 1u;
            found = true;
        } else if cache_bins[2] == bin {
            cache_grads[2] += grad;
            cache_hess[2] += hess;
            cache_counts[2] += 1u;
            found = true;
        } else if cache_bins[3] == bin {
            cache_grads[3] += grad;
            cache_hess[3] += hess;
            cache_counts[3] += 1u;
            found = true;
        }

        if !found {
            // Cache miss - evict current slot and use it for new bin
            let evict_slot = next_slot;
            next_slot = (next_slot + 1u) % CACHE_SIZE;

            // Flush evicted entry to shared memory (if valid)
            let evict_bin = cache_bins[evict_slot];
            if evict_bin != INVALID_BIN {
                atomicAdd(&local_grad[evict_bin], cache_grads[evict_slot]);
                atomicAdd(&local_hess[evict_bin], cache_hess[evict_slot]);
                atomicAdd(&local_counts[evict_bin], cache_counts[evict_slot]);
            }

            // Store new entry in cache
            cache_bins[evict_slot] = bin;
            cache_grads[evict_slot] = grad;
            cache_hess[evict_slot] = hess;
            cache_counts[evict_slot] = 1u;
        }
    }

    // Flush remaining cache entries to shared memory
    for (var slot = 0u; slot < CACHE_SIZE; slot++) {
        let bin = cache_bins[slot];
        if bin != INVALID_BIN {
            atomicAdd(&local_grad[bin], cache_grads[slot]);
            atomicAdd(&local_hess[bin], cache_hess[slot]);
            atomicAdd(&local_counts[bin], cache_counts[slot]);
        }
    }

    workgroupBarrier();

    // Write shared memory histogram to global memory
    let global_offset = feature * 256u + thread_id;
    let local_count = atomicLoad(&local_counts[thread_id]);

    if local_count > 0u {
        atomicAdd(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
        atomicAdd(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
        atomicAdd(&hist_counts[global_offset], local_count);
    }
}

// Zero out histogram buffers
@compute @workgroup_size(256, 1, 1)
fn zero_histograms_register_cache(
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

// Batched version with register caching
@compute @workgroup_size(256, 1, 1)
fn histogram_batched_register_cache(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let feature = wg_id.x;
    let batch = wg_id.y;
    let thread_id = lid.x;
    let num_threads = 256u;

    let batch_start = batch_info[batch].start;
    let batch_count = batch_info[batch].count;

    // Initialize shared memory
    atomicStore(&local_grad[thread_id], 0i);
    atomicStore(&local_hess[thread_id], 0i);
    atomicStore(&local_counts[thread_id], 0u);

    workgroupBarrier();

    // Register cache
    var cache_bins: array<u32, 4>;
    var cache_grads: array<i32, 4>;
    var cache_hess: array<i32, 4>;
    var cache_counts: array<u32, 4>;

    cache_bins[0] = INVALID_BIN;
    cache_bins[1] = INVALID_BIN;
    cache_bins[2] = INVALID_BIN;
    cache_bins[3] = INVALID_BIN;
    cache_grads[0] = 0i;
    cache_grads[1] = 0i;
    cache_grads[2] = 0i;
    cache_grads[3] = 0i;
    cache_hess[0] = 0i;
    cache_hess[1] = 0i;
    cache_hess[2] = 0i;
    cache_hess[3] = 0i;
    cache_counts[0] = 0u;
    cache_counts[1] = 0u;
    cache_counts[2] = 0u;
    cache_counts[3] = 0u;

    var next_slot: u32 = 0u;

    for (var i = thread_id; i < batch_count; i += num_threads) {
        let row = row_indices[batch_start + i];

        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        // Check cache
        var found = false;
        if cache_bins[0] == bin {
            cache_grads[0] += grad;
            cache_hess[0] += hess;
            cache_counts[0] += 1u;
            found = true;
        } else if cache_bins[1] == bin {
            cache_grads[1] += grad;
            cache_hess[1] += hess;
            cache_counts[1] += 1u;
            found = true;
        } else if cache_bins[2] == bin {
            cache_grads[2] += grad;
            cache_hess[2] += hess;
            cache_counts[2] += 1u;
            found = true;
        } else if cache_bins[3] == bin {
            cache_grads[3] += grad;
            cache_hess[3] += hess;
            cache_counts[3] += 1u;
            found = true;
        }

        if !found {
            let evict_slot = next_slot;
            next_slot = (next_slot + 1u) % CACHE_SIZE;

            let evict_bin = cache_bins[evict_slot];
            if evict_bin != INVALID_BIN {
                atomicAdd(&local_grad[evict_bin], cache_grads[evict_slot]);
                atomicAdd(&local_hess[evict_bin], cache_hess[evict_slot]);
                atomicAdd(&local_counts[evict_bin], cache_counts[evict_slot]);
            }

            cache_bins[evict_slot] = bin;
            cache_grads[evict_slot] = grad;
            cache_hess[evict_slot] = hess;
            cache_counts[evict_slot] = 1u;
        }
    }

    // Flush remaining cache
    for (var slot = 0u; slot < CACHE_SIZE; slot++) {
        let bin = cache_bins[slot];
        if bin != INVALID_BIN {
            atomicAdd(&local_grad[bin], cache_grads[slot]);
            atomicAdd(&local_hess[bin], cache_hess[slot]);
            atomicAdd(&local_counts[bin], cache_counts[slot]);
        }
    }

    workgroupBarrier();

    // Write to global memory
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
fn zero_histograms_batched_register_cache(
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
