// CUDA histogram kernels for TreeBoost GBDT
// Build histograms for gradient boosting tree construction

// Standard histogram building kernel
// Uses 2D grid: grid.x = features, grid.y = row_tiles
// Based on WarpGBM's tiled approach for better GPU utilization
extern "C" __global__ void build_histogram(
    const unsigned char* __restrict__ bins,      // Row-major bins [num_rows * num_features]
    const float* __restrict__ gradients,         // Gradients [num_rows]
    const float* __restrict__ hessians,          // Hessians [num_rows]
    const unsigned int* __restrict__ row_indices,
    float* __restrict__ grad_hist,               // Output [num_features * 256]
    float* __restrict__ hess_hist,               // Output [num_features * 256]
    unsigned int* __restrict__ count_hist,       // Output [num_features * 256]
    unsigned int num_rows,
    unsigned int num_features,
    unsigned int num_indices
) {
    // 2D grid: blockIdx.x = feature, blockIdx.y = row tile
    unsigned int feature_idx = blockIdx.x;
    if (feature_idx >= num_features) return;

    // Calculate row range for this tile
    unsigned int rows_per_tile = blockDim.x;
    unsigned int tile_start = blockIdx.y * rows_per_tile;
    unsigned int my_row = tile_start + threadIdx.x;

    // Shared memory for local histogram accumulation
    extern __shared__ char shared_mem[];
    float* sh_grad = (float*)shared_mem;
    float* sh_hess = sh_grad + 256;
    unsigned int* sh_count = (unsigned int*)(sh_hess + 256);

    // Initialize shared memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
        sh_count[i] = 0;
    }
    __syncthreads();

    // Process one row per thread in this tile
    if (my_row < num_indices) {
        unsigned int row = row_indices[my_row];
        unsigned int bin = bins[row * num_features + feature_idx];

        float grad = gradients[row];
        float hess = hessians[row];

        // Atomic add to shared memory
        atomicAdd(&sh_grad[bin], grad);
        atomicAdd(&sh_hess[bin], hess);
        atomicAdd(&sh_count[bin], 1u);
    }
    __syncthreads();

    // Write shared memory to global memory (atomic for multi-tile reduction)
    float* feat_grad = grad_hist + feature_idx * 256;
    float* feat_hess = hess_hist + feature_idx * 256;
    unsigned int* feat_count = count_hist + feature_idx * 256;

    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        if (sh_grad[b] != 0.0f) atomicAdd(&feat_grad[b], sh_grad[b]);
        if (sh_hess[b] != 0.0f) atomicAdd(&feat_hess[b], sh_hess[b]);
        if (sh_count[b] != 0) atomicAdd(&feat_count[b], sh_count[b]);
    }
}

// Zero histogram buffers
extern "C" __global__ void zero_histograms(
    float* __restrict__ grad_hist,
    float* __restrict__ hess_hist,
    unsigned int* __restrict__ count_hist,
    unsigned int total_bins
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_bins) {
        grad_hist[idx] = 0.0f;
        hess_hist[idx] = 0.0f;
        count_hist[idx] = 0;
    }
}

// Era histogram: build histograms stratified by era for Directional Era Splitting (DES)
// Grid: (num_features, num_eras)
// Each block handles one (feature, era) pair
extern "C" __global__ void build_histogram_era(
    const unsigned char* __restrict__ bins,       // Row-major bins [num_rows * num_features]
    const float* __restrict__ gradients,          // Gradients [num_rows]
    const float* __restrict__ hessians,           // Hessians [num_rows]
    const unsigned int* __restrict__ row_indices, // Row indices for this node
    const unsigned short* __restrict__ era_indices, // Era index per row
    float* __restrict__ grad_hist,                // Output [num_eras * num_features * 256]
    float* __restrict__ hess_hist,                // Output [num_eras * num_features * 256]
    unsigned int* __restrict__ count_hist,        // Output [num_eras * num_features * 256]
    unsigned int num_rows,
    unsigned int num_features,
    unsigned int num_indices,
    unsigned int num_eras
) {
    // blockIdx.x = feature_idx, blockIdx.y = era_idx
    unsigned int feature_idx = blockIdx.x;
    unsigned int target_era = blockIdx.y;
    if (feature_idx >= num_features || target_era >= num_eras) return;

    // Shared memory for local histogram accumulation
    extern __shared__ char shared_mem[];
    float* sh_grad = (float*)shared_mem;
    float* sh_hess = sh_grad + 256;
    unsigned int* sh_count = (unsigned int*)(sh_hess + 256);

    // Initialize shared memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
        sh_count[i] = 0;
    }
    __syncthreads();

    // Each thread processes multiple rows, only accumulating rows in target_era
    for (unsigned int i = threadIdx.x; i < num_indices; i += blockDim.x) {
        unsigned int row = row_indices[i];
        unsigned int row_era = era_indices[row];

        // Only process rows that belong to target_era
        if (row_era != target_era) continue;

        unsigned int bin = bins[row * num_features + feature_idx];
        float grad = gradients[row];
        float hess = hessians[row];

        atomicAdd(&sh_grad[bin], grad);
        atomicAdd(&sh_hess[bin], hess);
        atomicAdd(&sh_count[bin], 1u);
    }
    __syncthreads();

    // Write shared memory to global memory
    // Output layout: [era * num_features * 256 + feature * 256 + bin]
    unsigned int hist_offset = (target_era * num_features + feature_idx) * 256;
    float* feat_grad = grad_hist + hist_offset;
    float* feat_hess = hess_hist + hist_offset;
    unsigned int* feat_count = count_hist + hist_offset;

    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        if (sh_grad[b] != 0.0f) atomicAdd(&feat_grad[b], sh_grad[b]);
        if (sh_hess[b] != 0.0f) atomicAdd(&feat_hess[b], sh_hess[b]);
        if (sh_count[b] != 0) atomicAdd(&feat_count[b], sh_count[b]);
    }
}

// Batched histogram: build histograms for multiple nodes in one launch
// Grid: (num_features, num_nodes * max_tiles_per_node)
// Each block processes one tile of one node's rows for one feature
extern "C" __global__ void build_histogram_batched(
    const unsigned char* __restrict__ bins,       // Row-major bins [total_rows * num_features]
    const float* __restrict__ gradients,          // Gradients [total_rows]
    const float* __restrict__ hessians,           // Hessians [total_rows]
    const unsigned int* __restrict__ row_indices, // All row indices concatenated
    const unsigned int* __restrict__ node_starts, // Start offset in row_indices for each node
    const unsigned int* __restrict__ node_counts, // Number of rows for each node
    float* __restrict__ grad_hist,                // Output [num_nodes * num_features * 256]
    float* __restrict__ hess_hist,                // Output [num_nodes * num_features * 256]
    unsigned int* __restrict__ count_hist,        // Output [num_nodes * num_features * 256]
    unsigned int num_features,
    unsigned int num_nodes,
    unsigned int max_tiles_per_node
) {
    // blockIdx.x = feature_idx
    // blockIdx.y = node_idx * max_tiles_per_node + tile_idx
    unsigned int feature_idx = blockIdx.x;
    if (feature_idx >= num_features) return;

    unsigned int node_idx = blockIdx.y / max_tiles_per_node;
    unsigned int tile_idx = blockIdx.y % max_tiles_per_node;
    if (node_idx >= num_nodes) return;

    unsigned int node_start = node_starts[node_idx];
    unsigned int node_count = node_counts[node_idx];

    unsigned int rows_per_tile = blockDim.x;
    unsigned int tile_start = tile_idx * rows_per_tile;
    unsigned int my_local_row = tile_start + threadIdx.x;

    // Shared memory for local histogram
    extern __shared__ char shared_mem[];
    float* sh_grad = (float*)shared_mem;
    float* sh_hess = sh_grad + 256;
    unsigned int* sh_count = (unsigned int*)(sh_hess + 256);

    // Initialize shared memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
        sh_count[i] = 0;
    }
    __syncthreads();

    // Process one row per thread if within bounds
    if (my_local_row < node_count) {
        unsigned int row = row_indices[node_start + my_local_row];
        unsigned int bin = bins[row * num_features + feature_idx];
        float grad = gradients[row];
        float hess = hessians[row];

        atomicAdd(&sh_grad[bin], grad);
        atomicAdd(&sh_hess[bin], hess);
        atomicAdd(&sh_count[bin], 1u);
    }
    __syncthreads();

    // Write to global memory - output indexed by [node_idx][feature_idx][bin]
    unsigned int hist_base = (node_idx * num_features + feature_idx) * 256;
    float* feat_grad = grad_hist + hist_base;
    float* feat_hess = hess_hist + hist_base;
    unsigned int* feat_count = count_hist + hist_base;

    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        if (sh_grad[b] != 0.0f) atomicAdd(&feat_grad[b], sh_grad[b]);
        if (sh_hess[b] != 0.0f) atomicAdd(&feat_hess[b], sh_hess[b]);
        if (sh_count[b] != 0) atomicAdd(&feat_count[b], sh_count[b]);
    }
}
