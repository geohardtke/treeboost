//! Benchmark different GPU workgroup sizes for histogram building.
//!
//! Run with: cargo run --release --features gpu --example workgroup_size_benchmark

use std::time::{Duration, Instant};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages, ComputePipeline};

/// Fixed-point scale factor (matching kernels.rs)
const FIXED_POINT_SCALE: f32 = 1024.0;

/// Parameters passed to histogram shader via uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramParams {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,
    num_batches: u32,
}

/// Generate histogram shader with specific workgroup size.
fn generate_shader(workgroup_size: u32) -> String {
    format!(
        r#"
// Histogram shader with workgroup size {}

struct Params {{
    num_rows: u32,
    num_features: u32,
    num_indices: u32,
    num_batches: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> grad_hess: array<u32>;
@group(0) @binding(3) var<storage, read> row_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> hist_grad: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> hist_hess: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> hist_counts: array<atomic<u32>>;

var<workgroup> local_grad: array<atomic<i32>, 256>;
var<workgroup> local_hess: array<atomic<i32>, 256>;
var<workgroup> local_counts: array<atomic<u32>, 256>;

fn get_bin(packed: u32, byte_idx: u32) -> u32 {{
    return (packed >> (byte_idx * 8u)) & 0xFFu;
}}

fn unpack_grad(packed: u32) -> i32 {{
    let raw = packed & 0xFFFFu;
    if (raw & 0x8000u) != 0u {{
        return i32(raw | 0xFFFF0000u);
    }}
    return i32(raw);
}}

fn unpack_hess(packed: u32) -> i32 {{
    let raw = (packed >> 16u) & 0xFFFFu;
    if (raw & 0x8000u) != 0u {{
        return i32(raw | 0xFFFF0000u);
    }}
    return i32(raw);
}}

@compute @workgroup_size({}, 1, 1)
fn histogram_dense(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let feature = wg_id.x;
    let thread_id = lid.x;
    let num_threads = {}u;

    // Initialize shared memory (only first 256 threads)
    if thread_id < 256u {{
        atomicStore(&local_grad[thread_id], 0i);
        atomicStore(&local_hess[thread_id], 0i);
        atomicStore(&local_counts[thread_id], 0u);
    }}

    workgroupBarrier();

    let total_rows = select(params.num_rows, params.num_indices, params.num_indices > 0u);

    for (var i = thread_id; i < total_rows; i += num_threads) {{
        let row = select(i, row_indices[i], params.num_indices > 0u);

        let bin_offset = row * params.num_features + feature;
        let packed_idx = bin_offset / 4u;
        let byte_idx = bin_offset % 4u;
        let bin = get_bin(bins[packed_idx], byte_idx);

        let packed_gh = grad_hess[row];
        let grad = unpack_grad(packed_gh);
        let hess = unpack_hess(packed_gh);

        atomicAdd(&local_grad[bin], grad);
        atomicAdd(&local_hess[bin], hess);
        atomicAdd(&local_counts[bin], 1u);
    }}

    workgroupBarrier();

    // Write to global memory (only first 256 threads)
    if thread_id < 256u {{
        let global_offset = feature * 256u + thread_id;
        let local_count = atomicLoad(&local_counts[thread_id]);

        if local_count > 0u {{
            atomicAdd(&hist_grad[global_offset], atomicLoad(&local_grad[thread_id]));
            atomicAdd(&hist_hess[global_offset], atomicLoad(&local_hess[thread_id]));
            atomicAdd(&hist_counts[global_offset], local_count);
        }}
    }}
}}

@compute @workgroup_size({}, 1, 1)
fn zero_histograms(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {{
    let idx = gid.x;
    let total_bins = params.num_features * 256u;

    if idx < total_bins {{
        atomicStore(&hist_grad[idx], 0i);
        atomicStore(&hist_hess[idx], 0i);
        atomicStore(&hist_counts[idx], 0u);
    }}
}}
"#,
        workgroup_size, workgroup_size, workgroup_size, workgroup_size
    )
}

struct WorkgroupBenchmark {
    device: wgpu::Device,
    queue: wgpu::Queue,
    workgroup_size: u32,
    pipeline_dense: ComputePipeline,
    pipeline_zero: ComputePipeline,
}

impl WorkgroupBenchmark {
    fn new(device: wgpu::Device, queue: wgpu::Queue, workgroup_size: u32) -> Self {
        let shader_source = generate_shader(workgroup_size);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("histogram_wg{}_shader", workgroup_size)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_dense = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("histogram_wg{}_pipeline", workgroup_size)),
            layout: None,
            module: &shader_module,
            entry_point: Some("histogram_dense"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_zero = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("zero_wg{}_pipeline", workgroup_size)),
            layout: None,
            module: &shader_module,
            entry_point: Some("zero_histograms"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            workgroup_size,
            pipeline_dense,
            pipeline_zero,
        }
    }

    fn benchmark(
        &self,
        bins_packed: &[u32],
        grad_hess_packed: &[u32],
        indices: &[u32],
        num_rows: usize,
        num_features: usize,
        iterations: usize,
    ) -> Duration {
        // Create buffers
        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: indices.len() as u32,
            num_batches: 0,
        };

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params_buffer"),
            size: std::mem::size_of::<HistogramParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bins_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bins_buffer"),
            size: (bins_packed.len() * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grad_hess_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_hess_buffer"),
            size: (grad_hess_packed.len() * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indices_size = if indices.is_empty() { 4 } else { indices.len() * 4 };
        let indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indices_buffer"),
            size: indices_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hist_size = (num_features * 256 * 4) as u64;
        let hist_grad = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_grad"),
            size: hist_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hist_hess = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_hess"),
            size: hist_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hist_count = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_count"),
            size: hist_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data
        self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&bins_buffer, 0, bytemuck::cast_slice(bins_packed));
        self.queue.write_buffer(&grad_hess_buffer, 0, bytemuck::cast_slice(grad_hess_packed));
        if !indices.is_empty() {
            self.queue.write_buffer(&indices_buffer, 0, bytemuck::cast_slice(indices));
        }

        // Create bind groups
        let bind_group_layout_zero = self.pipeline_zero.get_bind_group_layout(0);
        let bind_group_zero = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group"),
            layout: &bind_group_layout_zero,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: hist_grad.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: hist_hess.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: hist_count.as_entire_binding() },
            ],
        });

        let bind_group_layout_dense = self.pipeline_dense.get_bind_group_layout(0);
        let bind_group_dense = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("histogram_bind_group"),
            layout: &bind_group_layout_dense,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: bins_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: grad_hess_buffer.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: indices_buffer.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: hist_grad.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: hist_hess.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: hist_count.as_entire_binding() },
            ],
        });

        // Warmup
        for _ in 0..3 {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("warmup_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zero_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_zero);
                pass.set_bind_group(0, &bind_group_zero, &[]);
                let workgroups = ((num_features * 256) as u32 + self.workgroup_size - 1) / self.workgroup_size;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("histogram_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dense);
                pass.set_bind_group(0, &bind_group_dense, &[]);
                pass.dispatch_workgroups(num_features as u32, 1, 1);
            }

            let submission = self.queue.submit(std::iter::once(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(60)),
            });
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("benchmark_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zero_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_zero);
                pass.set_bind_group(0, &bind_group_zero, &[]);
                let workgroups = ((num_features * 256) as u32 + self.workgroup_size - 1) / self.workgroup_size;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("histogram_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dense);
                pass.set_bind_group(0, &bind_group_dense, &[]);
                pass.dispatch_workgroups(num_features as u32, 1, 1);
            }

            let submission = self.queue.submit(std::iter::once(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(60)),
            });
        }

        start.elapsed() / iterations as u32
    }
}

fn create_test_data(num_rows: usize, num_features: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // Create row-major bins (as packed u32)
    let mut bins_raw = vec![0u8; num_rows * num_features];
    for r in 0..num_rows {
        for f in 0..num_features {
            bins_raw[r * num_features + f] = ((r * 7 + f * 13) % 256) as u8;
        }
    }
    let bins_packed: Vec<u32> = bins_raw
        .chunks(4)
        .map(|chunk| {
            let mut val = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                val |= (b as u32) << (i * 8);
            }
            val
        })
        .collect();

    // Create packed grad/hess
    let grad_hess_packed: Vec<u32> = (0..num_rows)
        .map(|i| {
            let grad = ((i as f32 * 0.01).sin() * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0) as i16;
            let hess = (FIXED_POINT_SCALE).clamp(-32767.0, 32767.0) as i16;
            (grad as u16 as u32) | ((hess as u16 as u32) << 16)
        })
        .collect();

    // Create indices (use all rows)
    let indices: Vec<u32> = (0..num_rows as u32).collect();

    (bins_packed, grad_hess_packed, indices)
}

fn main() {
    println!("GPU Workgroup Size Benchmark");
    println!("============================\n");

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No GPU adapter found");

    println!("GPU: {}", adapter.get_info().name);
    println!("Backend: {:?}", adapter.get_info().backend);
    println!("Driver: {}", adapter.get_info().driver);

    // Query limits
    let limits = adapter.limits();
    println!("Max workgroup size X: {}", limits.max_compute_workgroup_size_x);
    println!("Max workgroup invocations: {}", limits.max_compute_invocations_per_workgroup);
    println!();

    let (_device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("benchmark_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::default(),
        },
    ))
    .expect("Failed to create device");

    // Get device limits - use default limits since that's what we'll use for the benchmark devices
    // Note: Limits::default() is more conservative than adapter.limits()
    let default_limits = wgpu::Limits::default();
    let max_invocations = default_limits.max_compute_invocations_per_workgroup;

    // Test configurations - filter to valid sizes only
    let all_workgroup_sizes = [64u32, 128, 256, 512, 1024];
    let workgroup_sizes: Vec<u32> = all_workgroup_sizes
        .into_iter()
        .filter(|&size| size <= max_invocations)
        .collect();
    let row_counts = [10_000, 50_000, 100_000, 250_000, 500_000];
    let num_features = 20;
    let iterations = 50;

    println!("Configuration:");
    println!("  Features: {}", num_features);
    println!("  Iterations: {}", iterations);
    println!("  Workgroup sizes: {:?}", workgroup_sizes);
    println!("  Max invocations: {}", max_invocations);
    println!();

    // Header
    print!("| {:>8} |", "Rows");
    for wg in &workgroup_sizes {
        print!(" {:>8} |", format!("WG{}", wg));
    }
    println!(" {:>8} |", "Best");

    print!("|{:-<10}|", "");
    for _ in &workgroup_sizes {
        print!("{:-<10}|", "");
    }
    println!("{:-<10}|", "");

    for &num_rows in &row_counts {
        let (bins_packed, grad_hess_packed, indices) = create_test_data(num_rows, num_features);

        print!("| {:>8} |", num_rows);

        let mut times = Vec::new();
        let mut best_time = Duration::MAX;
        let mut best_wg = 0;

        for &wg_size in &workgroup_sizes {
            // Create new device for each workgroup size since pipelines are tied to device
            let (dev, q) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("benchmark_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                },
            ))
            .expect("Failed to create device");

            let benchmark = WorkgroupBenchmark::new(dev, q, wg_size);
            let time = benchmark.benchmark(
                &bins_packed,
                &grad_hess_packed,
                &indices,
                num_rows,
                num_features,
                iterations,
            );

            times.push((wg_size, time));
            print!(" {:>7.3} |", time.as_secs_f64() * 1000.0);

            if time < best_time {
                best_time = time;
                best_wg = wg_size;
            }
        }

        println!(" {:>8} |", format!("WG{}", best_wg));
    }

    println!();
    println!("Legend: Time in milliseconds. Lower is better.");
    println!("Best column shows the optimal workgroup size for each row count.");
    println!();

    // Summary
    println!("Analysis:");
    println!("- Workgroup size 256 is the default and often optimal for histogram building");
    println!("- Smaller workgroup sizes (64, 128) may be better for small datasets");
    println!("- Larger workgroup sizes (512) may help with very large datasets");
    println!("- Optimal size depends on GPU architecture and memory bandwidth");
}
