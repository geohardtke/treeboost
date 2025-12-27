// TreeBoost GPU Row Partitioning Shader
//
// Partitions row indices based on a split condition (bin <= threshold).
// Used for fully-GPU tree building without CPU histogram downloads.
//
// Algorithm: 3-pass parallel stream compaction
//   Pass 1: Compute flags, local prefix sums, block totals
//   Pass 2: Scan block totals for global offsets
//   Pass 3: Scatter rows to left/right arrays using global positions
//
// Each pass is a separate kernel dispatch.

struct PartitionParams {
    num_indices: u32,       // Number of row indices to partition
    split_feature: u32,     // Feature index for split
    split_threshold: u32,   // Bin threshold (rows with bin <= threshold go left)
    num_features: u32,      // Total features (for bin indexing)
    num_blocks: u32,        // Number of workgroups
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PartitionParams;
@group(0) @binding(1) var<storage, read> bins: array<u32>;           // Packed bin data
@group(0) @binding(2) var<storage, read> input_indices: array<u32>;  // Row indices to partition
@group(0) @binding(3) var<storage, read_write> left_indices: array<u32>;   // Output: left partition
@group(0) @binding(4) var<storage, read_write> right_indices: array<u32>;  // Output: right partition
@group(0) @binding(5) var<storage, read_write> block_totals: array<u32>;   // Block sums [left_0, right_0, left_1, right_1, ...]
@group(0) @binding(6) var<storage, read_write> global_offsets: array<u32>; // Global prefix sums of block totals
@group(0) @binding(7) var<storage, read_write> flags: array<u32>;          // Temporary: 1 if left, 0 if right
@group(0) @binding(8) var<storage, read_write> local_positions: array<u32>; // Temporary: local prefix sum positions

// Workgroup shared memory for prefix sums
var<workgroup> shared_flags: array<u32, 256>;
var<workgroup> shared_prefix: array<u32, 256>;

const WORKGROUP_SIZE: u32 = 256u;

// Extract bin value from packed data
fn get_bin(row: u32, feature: u32) -> u32 {
    let bin_offset = row * params.num_features + feature;
    let packed_idx = bin_offset / 4u;
    let byte_idx = bin_offset % 4u;
    return (bins[packed_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// ============================================================================
// Pass 1: Compute flags, local prefix sums, and block totals
// ============================================================================
// Each workgroup processes WORKGROUP_SIZE elements
// Outputs:
//   - flags[i] = 1 if row goes left, 0 if right
//   - local_positions[i] = local prefix sum (position within block)
//   - block_totals[block*2] = total left count for block
//   - block_totals[block*2+1] = total right count for block

@compute @workgroup_size(256, 1, 1)
fn partition_pass1_flags_and_local_scan(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let block_id = wg_id.x;
    let thread_id = lid.x;
    let global_id = block_id * WORKGROUP_SIZE + thread_id;

    // Load and compute flag
    var my_flag: u32 = 0u;
    if global_id < params.num_indices {
        let row = input_indices[global_id];
        let bin = get_bin(row, params.split_feature);
        my_flag = select(0u, 1u, bin <= params.split_threshold);
        flags[global_id] = my_flag;
    }

    // Load flag into shared memory for prefix sum
    shared_flags[thread_id] = my_flag;

    workgroupBarrier();

    // Hillis-Steele inclusive prefix sum
    var sum = my_flag;
    for (var stride = 1u; stride < WORKGROUP_SIZE; stride *= 2u) {
        workgroupBarrier();
        var add_val = 0u;
        if thread_id >= stride {
            add_val = shared_flags[thread_id - stride];
        }
        workgroupBarrier();
        sum += add_val;
        shared_flags[thread_id] = sum;
        workgroupBarrier();
    }

    // shared_flags now contains inclusive prefix sum
    // Convert to exclusive by shifting
    var exclusive_sum = 0u;
    if thread_id > 0u {
        exclusive_sum = shared_flags[thread_id - 1u];
    }

    // Store local position (exclusive prefix sum)
    if global_id < params.num_indices {
        local_positions[global_id] = exclusive_sum;
    }

    // Last thread writes block total (inclusive sum of last element)
    if thread_id == WORKGROUP_SIZE - 1u {
        let left_total = shared_flags[WORKGROUP_SIZE - 1u];
        let count_in_block = min(WORKGROUP_SIZE, params.num_indices - block_id * WORKGROUP_SIZE);
        let right_total = count_in_block - left_total;

        block_totals[block_id * 2u] = left_total;
        block_totals[block_id * 2u + 1u] = right_total;
    }
}

// ============================================================================
// Pass 2: Scan block totals to compute global offsets
// ============================================================================
// Single workgroup scans all block totals
// Input: block_totals[block*2] = left count, block_totals[block*2+1] = right count
// Output: global_offsets[block*2] = left offset, global_offsets[block*2+1] = right offset

var<workgroup> shared_left: array<u32, 256>;
var<workgroup> shared_right: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn partition_pass2_scan_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let thread_id = lid.x;

    // Load block totals (each thread handles one block)
    var left_val = 0u;
    var right_val = 0u;
    if thread_id < params.num_blocks {
        left_val = block_totals[thread_id * 2u];
        right_val = block_totals[thread_id * 2u + 1u];
    }

    shared_left[thread_id] = left_val;
    shared_right[thread_id] = right_val;

    workgroupBarrier();

    // Inclusive prefix sum for left counts
    var sum_left = left_val;
    var sum_right = right_val;
    for (var stride = 1u; stride < WORKGROUP_SIZE; stride *= 2u) {
        workgroupBarrier();
        var add_left = 0u;
        var add_right = 0u;
        if thread_id >= stride {
            add_left = shared_left[thread_id - stride];
            add_right = shared_right[thread_id - stride];
        }
        workgroupBarrier();
        sum_left += add_left;
        sum_right += add_right;
        shared_left[thread_id] = sum_left;
        shared_right[thread_id] = sum_right;
        workgroupBarrier();
    }

    // Convert to exclusive prefix sum
    var exclusive_left = 0u;
    var exclusive_right = 0u;
    if thread_id > 0u {
        exclusive_left = shared_left[thread_id - 1u];
        exclusive_right = shared_right[thread_id - 1u];
    }

    // Store global offsets
    if thread_id < params.num_blocks {
        global_offsets[thread_id * 2u] = exclusive_left;
        global_offsets[thread_id * 2u + 1u] = exclusive_right;
    }
}

// ============================================================================
// Pass 3: Scatter rows to output arrays using global positions
// ============================================================================
// Each thread computes its global position and writes to output

@compute @workgroup_size(256, 1, 1)
fn partition_pass3_scatter(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let block_id = wg_id.x;
    let thread_id = lid.x;
    let global_id = block_id * WORKGROUP_SIZE + thread_id;

    if global_id >= params.num_indices {
        return;
    }

    let row = input_indices[global_id];
    let flag = flags[global_id];
    let local_pos = local_positions[global_id];

    // Get global offset for this block
    let left_offset = global_offsets[block_id * 2u];
    let right_offset = global_offsets[block_id * 2u + 1u];

    if flag == 1u {
        // Goes left
        let global_pos = left_offset + local_pos;
        left_indices[global_pos] = row;
    } else {
        // Goes right
        // For right, we need local position among rights only
        // local_pos is position among lefts, so we compute right position as:
        // position_in_block - left_count_before_me
        let position_in_block = thread_id;
        let right_local_pos = position_in_block - local_pos;
        let global_pos = right_offset + right_local_pos;
        right_indices[global_pos] = row;
    }
}

// ============================================================================
// Alternative: Single-pass atomic version (simpler but slower for large arrays)
// ============================================================================
// Uses atomic counters - simpler but has contention issues
// Good for small arrays or as reference implementation

struct AtomicCounters {
    left_count: atomic<u32>,
    right_count: atomic<u32>,
}

@group(0) @binding(9) var<storage, read_write> counters: AtomicCounters;

@compute @workgroup_size(256, 1, 1)
fn partition_atomic(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let global_id = gid.x;

    if global_id >= params.num_indices {
        return;
    }

    let row = input_indices[global_id];
    let bin = get_bin(row, params.split_feature);
    let goes_left = bin <= params.split_threshold;

    if goes_left {
        let pos = atomicAdd(&counters.left_count, 1u);
        left_indices[pos] = row;
    } else {
        let pos = atomicAdd(&counters.right_count, 1u);
        right_indices[pos] = row;
    }
}

// Zero the atomic counters before partition_atomic
@compute @workgroup_size(1, 1, 1)
fn zero_counters() {
    atomicStore(&counters.left_count, 0u);
    atomicStore(&counters.right_count, 0u);
}

// ============================================================================
// Batched partitioning for level-wise tree building
// ============================================================================
// Partitions multiple node's rows in a single dispatch
// Used when building all nodes at a tree level simultaneously

struct BatchedPartitionParams {
    num_nodes: u32,         // Number of nodes to partition
    num_features: u32,      // Total features
    _pad0: u32,
    _pad1: u32,
}

struct NodeSplit {
    input_start: u32,       // Start index in input_indices
    input_count: u32,       // Number of rows for this node
    output_left_start: u32, // Start index for left output
    output_right_start: u32,// Start index for right output
    split_feature: u32,     // Feature to split on
    split_threshold: u32,   // Bin threshold
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(10) var<uniform> batched_params: BatchedPartitionParams;
@group(0) @binding(11) var<storage, read> node_splits: array<NodeSplit>;
@group(0) @binding(12) var<storage, read_write> node_counters: array<atomic<u32>>; // [left_0, right_0, left_1, right_1, ...]

// Extract bin value using batched_params.num_features
fn get_bin_batched(row: u32, feature: u32) -> u32 {
    let bin_offset = row * batched_params.num_features + feature;
    let packed_idx = bin_offset / 4u;
    let byte_idx = bin_offset % 4u;
    return (bins[packed_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// Batched atomic partition - one workgroup per node
@compute @workgroup_size(256, 1, 1)
fn partition_batched_atomic(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let node_id = wg_id.x;
    let thread_id = lid.x;

    if node_id >= batched_params.num_nodes {
        return;
    }

    let split = node_splits[node_id];
    let num_threads = WORKGROUP_SIZE;

    // Each thread processes multiple rows if needed
    for (var i = thread_id; i < split.input_count; i += num_threads) {
        let input_idx = split.input_start + i;
        let row = input_indices[input_idx];
        let bin = get_bin_batched(row, split.split_feature);
        let goes_left = bin <= split.split_threshold;

        if goes_left {
            let pos = atomicAdd(&node_counters[node_id * 2u], 1u);
            left_indices[split.output_left_start + pos] = row;
        } else {
            let pos = atomicAdd(&node_counters[node_id * 2u + 1u], 1u);
            right_indices[split.output_right_start + pos] = row;
        }
    }
}

// Zero node counters for batched partition
@compute @workgroup_size(256, 1, 1)
fn zero_node_counters(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx < batched_params.num_nodes * 2u {
        atomicStore(&node_counters[idx], 0u);
    }
}
