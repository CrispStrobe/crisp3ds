//! `wgpu` device handling and a small kernel-dispatch layer.
//!
//! Everything here is plain WebGPU: WGSL compute shaders, storage and uniform
//! buffers, no backend-specific calls. The device is requested with the default
//! WebGPU limits, so validation fails here whenever a kernel would not run in a
//! browser: at most 8 storage buffers per stage, 128 MiB per storage binding,
//! 256 MiB per buffer, 65535 workgroups per dimension. Callers size their work
//! from [`Gpu::binding_budget`] instead of assuming a large adapter.

use std::borrow::Cow;

use anyhow::{anyhow, Context};
use bytemuck::Pod;
use wgpu::util::DeviceExt;

/// Invocations per workgroup of every kernel (`@workgroup_size(64)`).
pub const WORKGROUP: u32 = 64;

/// Workgroups along x before a dispatch spills into y. The kernels rebuild the
/// linear index as `id.x + id.y * ROW`.
pub const ROW_GROUPS: u32 = 16384;
pub const ROW: u32 = ROW_GROUPS * WORKGROUP;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    peak: std::sync::atomic::AtomicU64,
}

impl Gpu {
    /// The system's preferred adapter with default WebGPU limits.
    /// `CRISP3DS_GPU_FALLBACK=1` asks for a software adapter instead (for CI).
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let fallback = std::env::var("CRISP3DS_GPU_FALLBACK").map(|v| v == "1").unwrap_or(false);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: fallback,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        let limits = wgpu::Limits::default();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("crisp3ds-dense"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("GPU device with default WebGPU limits")?;
        Ok(Gpu { device, queue, info: adapter.get_info(), limits, peak: std::sync::atomic::AtomicU64::new(0) })
    }

    pub fn describe(&self) -> String {
        format!("{} ({:?})", self.info.name, self.info.backend)
    }

    /// Largest buffer bound to a kernel so far, in bytes (128 MiB is the WebGPU default limit).
    pub fn peak_binding(&self) -> u64 {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Bytes one storage binding may hold, with headroom below the device limit.
    pub fn binding_budget(&self) -> u64 {
        let limit = self.limits.max_storage_buffer_binding_size.min(self.limits.max_buffer_size);
        limit - limit / 8
    }

    /// Read-only (to kernels) storage buffer with the given content.
    pub fn upload<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.is_empty() {
            // Zero-sized bindings are invalid; keep one word.
            return self.zeroed(label, 4);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Zero-initialised storage buffer of `bytes` bytes (rounded up to 4).
    pub fn zeroed(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4).div_ceil(4) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn uniform<T: Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn write<T: Pod>(&self, buffer: &wgpu::Buffer, data: &[T]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn clear(&self, buffer: &wgpu::Buffer) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(buffer, 0, None);
        self.queue.submit([encoder.finish()]);
    }

    /// Copies the first `count` elements of a storage buffer back to the CPU.
    pub fn read<T: Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> anyhow::Result<Vec<T>> {
        let bytes = (count * std::mem::size_of::<T>()) as u64;
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let padded = bytes.div_ceil(4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, padded);
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU poll: {e}"))?;
        receiver.recv()?.map_err(|e| anyhow!("GPU readback: {e}"))?;
        let view = staging.slice(..).get_mapped_range().map_err(|e| anyhow!("GPU readback: {e}"))?;
        let mut out = vec![T::zeroed(); count];
        bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&view[..bytes as usize]);
        drop(view);
        staging.unmap();
        Ok(out)
    }

    /// Blocks until everything submitted so far has run.
    pub fn wait(&self) -> anyhow::Result<()> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU poll: {e}"))?;
        Ok(())
    }

    /// Compiles one compute entry point. Validation errors become `Err`.
    pub fn kernel(&self, label: &str, source: &str, entry: &str) -> anyhow::Result<Kernel> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!("const ROW: u32 = {ROW}u;\n{source}"))),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(anyhow!("kernel {label}: {error}"));
        }
        Ok(Kernel { label: label.to_string(), pipeline })
    }

    /// Runs `kernel` for at least `count` invocations with buffers bound at
    /// group 0, bindings 0, 1, 2, ... Kernels must ignore indices `>= count`.
    pub fn run(&self, kernel: &Kernel, buffers: &[&wgpu::Buffer], count: u64) -> anyhow::Result<()> {
        if count == 0 {
            return Ok(());
        }
        for buffer in buffers {
            self.peak.fetch_max(buffer.size(), std::sync::atomic::Ordering::Relaxed);
        }
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(n, buffer)| wgpu::BindGroupEntry { binding: n as u32, resource: buffer.as_entire_binding() })
            .collect();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&kernel.label),
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let groups = count.div_ceil(WORKGROUP as u64);
        let (x, y) = if groups <= ROW_GROUPS as u64 { (groups as u32, 1) } else { (ROW_GROUPS, groups.div_ceil(ROW_GROUPS as u64) as u32) };
        if y > self.limits.max_compute_workgroups_per_dimension {
            return Err(anyhow!("kernel {}: {count} invocations exceed one dispatch", kernel.label));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit([encoder.finish()]);
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(anyhow!("kernel {}: {error}", kernel.label));
        }
        Ok(())
    }
}

pub struct Kernel {
    label: String,
    pipeline: wgpu::ComputePipeline,
}

/// Packs booleans into 32-bit words, least significant bit first.
pub fn pack_bits(values: impl ExactSizeIterator<Item = bool>) -> Vec<u32> {
    let mut words = vec![0u32; values.len().div_ceil(32).max(1)];
    for (n, value) in values.enumerate() {
        if value {
            words[n / 32] |= 1 << (n % 32);
        }
    }
    words
}

/// True when GPU tests should run (`CRISP3DS_GPU_TESTS=1`); they need an adapter.
pub fn tests_enabled() -> bool {
    std::env::var("CRISP3DS_GPU_TESTS").map(|v| v == "1").unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use wgpu::naga;

    const SHADERS: [(&str, &str); 8] = [
        ("carve", include_str!("../shaders/carve.wgsl")),
        ("cover", include_str!("../shaders/cover.wgsl")),
        ("fuse", include_str!("../shaders/fuse.wgsl")),
        ("warp", include_str!("../shaders/warp.wgsl")),
        ("ncc_rows", include_str!("../shaders/ncc_rows.wgsl")),
        ("ncc_columns", include_str!("../shaders/ncc_columns.wgsl")),
        ("aggregate", include_str!("../shaders/aggregate.wgsl")),
        ("peak", include_str!("../shaders/peak.wgsl")),
    ];

    /// Every kernel parses and validates without a GPU, uses plain WebGPU
    /// features only and stays within the default limit of 8 storage buffers.
    #[test]
    fn kernels_are_valid_webgpu() {
        for (name, source) in SHADERS {
            let source = format!("const ROW: u32 = {}u;\n{source}", super::ROW);
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
            let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
            validator.validate(&module).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let storage = module.global_variables.iter().filter(|(_, v)| matches!(v.space, naga::AddressSpace::Storage { .. })).count();
            assert!(storage <= 8, "{name} binds {storage} storage buffers");
            assert!(!module.entry_points.is_empty(), "{name} has no entry point");
            for entry in &module.entry_points {
                assert_eq!(entry.stage, naga::ShaderStage::Compute, "{name}");
                assert_eq!(entry.workgroup_size, [super::WORKGROUP, 1, 1], "{name}");
            }
        }
    }

    /// Names the adapter the GPU tests run on (CRISP3DS_GPU_TESTS=1), for CI logs.
    #[test]
    fn adapter() {
        if !super::tests_enabled() {
            return;
        }
        let gpu = super::Gpu::new().expect("a GPU adapter");
        println!("adapter: {} ({:?}, {:?})", gpu.info.name, gpu.info.backend, gpu.info.device_type);
        assert!(gpu.limits.max_storage_buffers_per_shader_stage >= 8);
    }

    #[test]
    fn bits_pack_least_significant_first() {
        let mut flags = vec![false; 34];
        (flags[0], flags[2], flags[33]) = (true, true, true);
        assert_eq!(super::pack_bits(flags.into_iter()), vec![0b101, 0b10]);
        assert_eq!(super::pack_bits(Vec::<bool>::new().into_iter()), vec![0]);
    }
}
