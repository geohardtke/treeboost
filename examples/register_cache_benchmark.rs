//! Benchmark comparing register-cached vs baseline histogram shaders.
//!
//! Run with: cargo run --release --features gpu --example register_cache_benchmark

use std::time::{Duration, Instant};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages, ComputePipeline};

const FIXED_POINT_SCALE: f32 = 1024.0;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramParams {
    num_rows: u32,
    num_features: u32,
    num_indices: u32,
    num_batches: u32,
}

#[allow(dead_code)]
struct ShaderBenchmark {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_dense: ComputePipeline,
    pipeline_zero: ComputePipeline,
    name: String,
}

impl ShaderBenchmark {
    fn new(device: wgpu::Device, queue: wgpu::Queue, shader_source: &str, name: &str, entry_dense: &str, entry_zero: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", name)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_dense = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_dense_pipeline", name)),
            layout: None,
            module: &shader_module,
            entry_point: Some(entry_dense),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_zero = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_zero_pipeline", name)),
            layout: None,
            module: &shader_module,
            entry_point: Some(entry_zero),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline_dense,
            pipeline_zero,
            name: name.to_string(),
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

        self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&bins_buffer, 0, bytemuck::cast_slice(bins_packed));
        self.queue.write_buffer(&grad_hess_buffer, 0, bytemuck::cast_slice(grad_hess_packed));
        if !indices.is_empty() {
            self.queue.write_buffer(&indices_buffer, 0, bytemuck::cast_slice(indices));
        }

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
                let workgroups = ((num_features * 256) as u32 + 255) / 256;
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
                let workgroups = ((num_features * 256) as u32 + 255) / 256;
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

fn create_test_data(num_rows: usize, num_features: usize, sorted: bool) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut bins_raw = vec![0u8; num_rows * num_features];

    if sorted {
        // Sorted data - consecutive rows have similar bin values
        // This maximizes register cache hit rate
        for r in 0..num_rows {
            for f in 0..num_features {
                // Bins increase slowly with row index (sorted pattern)
                bins_raw[r * num_features + f] = ((r * 256 / num_rows) % 256) as u8;
            }
        }
    } else {
        // Random-ish data - bins vary more randomly
        for r in 0..num_rows {
            for f in 0..num_features {
                bins_raw[r * num_features + f] = ((r * 7 + f * 13) % 256) as u8;
            }
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

    let grad_hess_packed: Vec<u32> = (0..num_rows)
        .map(|i| {
            let grad = ((i as f32 * 0.01).sin() * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0) as i16;
            let hess = (FIXED_POINT_SCALE).clamp(-32767.0, 32767.0) as i16;
            (grad as u16 as u32) | ((hess as u16 as u32) << 16)
        })
        .collect();

    let indices: Vec<u32> = (0..num_rows as u32).collect();

    (bins_packed, grad_hess_packed, indices)
}

fn main() {
    println!("Register Cache vs Baseline Histogram Benchmark");
    println!("===============================================\n");

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
    println!();

    // Load shaders
    let baseline_shader = include_str!("../src/backend/wgpu/shaders/histogram.wgsl");
    let register_cache_shader = include_str!("../src/backend/wgpu/shaders/histogram_register_cache.wgsl");

    let row_counts = [10_000, 50_000, 100_000, 250_000, 500_000];
    let num_features = 20;
    let iterations = 50;

    println!("Configuration:");
    println!("  Features: {}", num_features);
    println!("  Iterations: {}", iterations);
    println!();

    // Test with random data pattern
    println!("=== Random Data Pattern ===");
    println!("(Consecutive rows have different bins - low cache hit rate expected)");
    println!();
    println!("| {:>8} | {:>12} | {:>12} | {:>8} |", "Rows", "Baseline", "RegCache", "Speedup");
    println!("|{:-<10}|{:-<14}|{:-<14}|{:-<10}|", "", "", "", "");

    for &num_rows in &row_counts {
        let (bins_packed, grad_hess_packed, indices) = create_test_data(num_rows, num_features, false);

        // Baseline
        let (dev1, q1) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("baseline_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        )).expect("Failed to create device");

        let baseline = ShaderBenchmark::new(dev1, q1, baseline_shader, "baseline", "histogram_dense", "zero_histograms");
        let baseline_time = baseline.benchmark(&bins_packed, &grad_hess_packed, &indices, num_rows, num_features, iterations);

        // Register cache
        let (dev2, q2) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("regcache_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        )).expect("Failed to create device");

        let regcache = ShaderBenchmark::new(dev2, q2, register_cache_shader, "regcache", "histogram_dense_register_cache", "zero_histograms_register_cache");
        let regcache_time = regcache.benchmark(&bins_packed, &grad_hess_packed, &indices, num_rows, num_features, iterations);

        let speedup = baseline_time.as_secs_f64() / regcache_time.as_secs_f64();

        println!("| {:>8} | {:>11.3}ms | {:>11.3}ms | {:>7.2}x |",
            num_rows,
            baseline_time.as_secs_f64() * 1000.0,
            regcache_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!();
    println!("=== Sorted Data Pattern ===");
    println!("(Consecutive rows have similar bins - high cache hit rate expected)");
    println!();
    println!("| {:>8} | {:>12} | {:>12} | {:>8} |", "Rows", "Baseline", "RegCache", "Speedup");
    println!("|{:-<10}|{:-<14}|{:-<14}|{:-<10}|", "", "", "", "");

    for &num_rows in &row_counts {
        let (bins_packed, grad_hess_packed, indices) = create_test_data(num_rows, num_features, true);

        // Baseline
        let (dev1, q1) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("baseline_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        )).expect("Failed to create device");

        let baseline = ShaderBenchmark::new(dev1, q1, baseline_shader, "baseline", "histogram_dense", "zero_histograms");
        let baseline_time = baseline.benchmark(&bins_packed, &grad_hess_packed, &indices, num_rows, num_features, iterations);

        // Register cache
        let (dev2, q2) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("regcache_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        )).expect("Failed to create device");

        let regcache = ShaderBenchmark::new(dev2, q2, register_cache_shader, "regcache", "histogram_dense_register_cache", "zero_histograms_register_cache");
        let regcache_time = regcache.benchmark(&bins_packed, &grad_hess_packed, &indices, num_rows, num_features, iterations);

        let speedup = baseline_time.as_secs_f64() / regcache_time.as_secs_f64();

        println!("| {:>8} | {:>11.3}ms | {:>11.3}ms | {:>7.2}x |",
            num_rows,
            baseline_time.as_secs_f64() * 1000.0,
            regcache_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!();
    println!("Analysis:");
    println!("- Register caching is effective when consecutive rows hit the same bins");
    println!("- Sorted/semi-sorted data benefits most from this optimization");
    println!("- Random data may see overhead from cache management without benefit");
    println!("- The optimization trades register pressure for reduced atomic contention");
}
