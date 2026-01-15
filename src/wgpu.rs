//! WebGPU Runtime for HVM2
//!
//! This module provides GPU execution of HVM programs using WebGPU,
//! enabling cross-platform GPU acceleration including web browsers.

use crate::hvm::{Book, Port, Pair, Numb, GNet, VAR, REF, ROOT, NONE};
use std::sync::atomic::Ordering;

/// Configuration for WebGPU execution
#[derive(Clone, Debug)]
pub struct WgpuConfig {
    /// Number of workgroups to dispatch
    pub workgroups: u32,
    /// Maximum iterations per kernel invocation
    pub max_iterations: u32,
    /// Node buffer size (in nodes)
    pub node_len: u32,
    /// Vars buffer size (in vars)
    pub vars_len: u32,
}

impl Default for WgpuConfig {
    fn default() -> Self {
        WgpuConfig {
            workgroups: 256,
            max_iterations: 1000,
            node_len: 1 << 20, // 1M nodes
            vars_len: 1 << 20, // 1M vars
        }
    }
}

/// NetConfig structure matching WGSL layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NetConfig {
    node_len: u32,
    vars_len: u32,
    rbag_len: u32,
    defs_len: u32,
    mode: u32,
    done: u32,
    itrs: u32,
    rwts: u32,
}

/// Result of WebGPU execution
pub struct WgpuResult {
    /// Total number of interactions performed
    pub interactions: u64,
    /// Execution time in seconds
    pub time_secs: f64,
    /// The result port (if readable)
    pub result: Option<Port>,
}

/// Run HVM on WebGPU
pub fn run_wgpu(book: &Book) -> Result<WgpuResult, String> {
    run_wgpu_with_config(book, WgpuConfig::default())
}

/// Run HVM on WebGPU with custom configuration
pub fn run_wgpu_with_config(book: &Book, config: WgpuConfig) -> Result<WgpuResult, String> {
    pollster::block_on(run_wgpu_async(book, config))
}

/// Async implementation of WebGPU execution
async fn run_wgpu_async(book: &Book, config: WgpuConfig) -> Result<WgpuResult, String> {
    // Initialize wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to find an appropriate GPU adapter")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("HVM Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| format!("Failed to create device: {}", e))?;

    println!("WebGPU device: {}", adapter.get_info().name);

    // Serialize book to buffer
    let mut book_data: Vec<u8> = Vec::new();
    book.to_buffer(&mut book_data);

    // Align to 4 bytes
    while book_data.len() % 4 != 0 {
        book_data.push(0);
    }

    // Calculate buffer sizes
    let node_buf_size = (config.node_len as usize * 2 * 4) as u64; // 2 u32s per node
    let vars_buf_size = (config.vars_len as usize * 4) as u64;
    let rbag_size = (config.workgroups as usize * 256 * 2 * 4) as u64; // 256 redexes per thread, 2 u32s each
    let rbag_len_size = (config.workgroups as usize * 4) as u64;

    // Create buffers
    let node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Buffer"),
        size: node_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let vars_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vars Buffer"),
        size: vars_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rbag_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Redex Bag Buffer"),
        size: rbag_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rbag_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Redex Length Buffer"),
        size: rbag_len_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let book_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Book Buffer"),
        contents: &book_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let net_config = NetConfig {
        node_len: config.node_len,
        vars_len: config.vars_len,
        rbag_len: config.workgroups * 256,
        defs_len: book.defs.len() as u32,
        mode: 0,
        done: 0,
        itrs: 0,
        rwts: 0,
    };

    let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Config Buffer"),
        contents: bytemuck::bytes_of(&net_config),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    // Staging buffer for reading results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: std::cmp::max(vars_buf_size, 32),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let config_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Staging"),
        size: std::mem::size_of::<NetConfig>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load shader
    let shader_source = include_str!("hvm.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("HVM Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("HVM Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HVM Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: node_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vars_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rbag_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rbag_len_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: book_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: config_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("HVM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create pipelines
    let boot_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Boot Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "boot",
        compilation_options: Default::default(),
    });

    let main_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Main Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
        compilation_options: Default::default(),
    });

    let start_time = std::time::Instant::now();

    // Run boot kernel to initialize
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Boot Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Boot Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&boot_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    // Main evaluation loop
    let mut total_itrs: u64 = 0;
    let mut iterations = 0;

    loop {
        iterations += 1;

        // Reset done counter
        queue.write_buffer(&config_buffer, 20, &0u32.to_ne_bytes()); // offset of 'done' field

        // Run main kernel
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Main Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&main_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(config.workgroups, 1, 1);
            }

            // Copy config to staging for reading
            encoder.copy_buffer_to_buffer(
                &config_buffer,
                0,
                &config_staging,
                0,
                std::mem::size_of::<NetConfig>() as u64,
            );

            queue.submit(Some(encoder.finish()));
        }

        // Read results
        {
            let buffer_slice = config_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().map_err(|e| format!("Failed to map buffer: {:?}", e))?;

            let data = buffer_slice.get_mapped_range();
            let config_result: NetConfig = *bytemuck::from_bytes(&data);
            drop(data);
            config_staging.unmap();

            total_itrs += config_result.itrs as u64;

            // Check if done (all threads finished with no redexes)
            if config_result.done == config.workgroups || iterations >= config.max_iterations {
                break;
            }
        }

        // Update iteration counter in config
        queue.write_buffer(&config_buffer, 24, &0u32.to_ne_bytes()); // Reset itrs
    }

    let elapsed = start_time.elapsed();

    // Read result from vars buffer (ROOT location)
    let result = {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Result Encoder"),
        });

        // ROOT is at 0x1FFFFFFF, but we'll read from the beginning for simplicity
        // In a real implementation, we'd need to properly read the result
        encoder.copy_buffer_to_buffer(&vars_buffer, 0, &staging_buffer, 0, 4);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(0..4);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().ok();

        let data = buffer_slice.get_mapped_range();
        let port_val: u32 = *bytemuck::from_bytes(&data[0..4]);
        drop(data);
        staging_buffer.unmap();

        if port_val != NONE.0 && port_val != 0 {
            Some(Port(port_val))
        } else {
            None
        }
    };

    Ok(WgpuResult {
        interactions: total_itrs,
        time_secs: elapsed.as_secs_f64(),
        result,
    })
}

/// Generate standalone WebGPU/WASM code for a book
pub fn gen_wgpu(book: &Book, with_io: bool) -> String {
    let mut output = String::new();

    // Generate book data as JavaScript array
    let mut book_data: Vec<u8> = Vec::new();
    book.to_buffer(&mut book_data);

    output.push_str("// HVM2 WebGPU Generated Code\n");
    output.push_str("// Generated for standalone execution in browsers or Node.js\n\n");

    // Include the WGSL shader
    output.push_str("const HVM_SHADER = `\n");
    output.push_str(include_str!("hvm.wgsl"));
    output.push_str("`;\n\n");

    // Book data as Uint8Array
    output.push_str("const BOOK_DATA = new Uint8Array([");
    for (i, byte) in book_data.iter().enumerate() {
        if i > 0 {
            output.push_str(",");
        }
        if i % 32 == 0 {
            output.push_str("\n  ");
        }
        output.push_str(&format!("{}", byte));
    }
    output.push_str("\n]);\n\n");

    // Runtime JavaScript code
    output.push_str(r#"
// HVM WebGPU Runtime
class HVMRuntime {
  constructor() {
    this.device = null;
    this.queue = null;
    this.buffers = {};
    this.pipelines = {};
    this.bindGroup = null;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });
    if (!adapter) {
      throw new Error("Failed to get GPU adapter");
    }

    this.device = await adapter.requestDevice();
    this.queue = this.device.queue;

    console.log("WebGPU initialized");
    return this;
  }

  async run(bookData = BOOK_DATA, config = {}) {
    const {
      workgroups = 256,
      nodeLen = 1 << 20,
      varsLen = 1 << 20,
      maxIterations = 1000
    } = config;

    // Ensure book data is 4-byte aligned
    const alignedBookData = new Uint8Array(Math.ceil(bookData.length / 4) * 4);
    alignedBookData.set(bookData);

    // Create buffers
    const nodeBufSize = nodeLen * 2 * 4;
    const varsBufSize = varsLen * 4;
    const rbagSize = workgroups * 256 * 2 * 4;
    const rbagLenSize = workgroups * 4;

    this.buffers.node = this.device.createBuffer({
      size: nodeBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.vars = this.device.createBuffer({
      size: varsBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.rbag = this.device.createBuffer({
      size: rbagSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.rbagLen = this.device.createBuffer({
      size: rbagLenSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.book = this.device.createBuffer({
      size: alignedBookData.length,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Uint8Array(this.buffers.book.getMappedRange()).set(alignedBookData);
    this.buffers.book.unmap();

    const configData = new Uint32Array([
      nodeLen,   // node_len
      varsLen,   // vars_len
      workgroups * 256, // rbag_len
      0,         // defs_len (will be read from book)
      0,         // mode
      0,         // done
      0,         // itrs
      0          // rwts
    ]);

    this.buffers.config = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(this.buffers.config.getMappedRange()).set(configData);
    this.buffers.config.unmap();

    // Staging buffer
    this.buffers.staging = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // Create shader module
    const shaderModule = this.device.createShaderModule({
      code: HVM_SHADER
    });

    // Create bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
      ]
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers.node } },
        { binding: 1, resource: { buffer: this.buffers.vars } },
        { binding: 2, resource: { buffer: this.buffers.rbag } },
        { binding: 3, resource: { buffer: this.buffers.rbagLen } },
        { binding: 4, resource: { buffer: this.buffers.book } },
        { binding: 5, resource: { buffer: this.buffers.config } }
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });

    this.pipelines.boot = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "boot" }
    });

    this.pipelines.main = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "main" }
    });

    const startTime = performance.now();

    // Run boot kernel
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.boot);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      this.queue.submit([encoder.finish()]);
    }

    // Main evaluation loop
    let totalItrs = 0;
    let iterations = 0;

    while (iterations < maxIterations) {
      iterations++;

      // Reset done counter
      this.queue.writeBuffer(this.buffers.config, 20, new Uint32Array([0]));

      // Run main kernel
      {
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.main);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(this.buffers.config, 0, this.buffers.staging, 0, 32);
        this.queue.submit([encoder.finish()]);
      }

      // Read results
      await this.buffers.staging.mapAsync(GPUMapMode.READ);
      const data = new Uint32Array(this.buffers.staging.getMappedRange());
      const done = data[5];
      const itrs = data[6];
      this.buffers.staging.unmap();

      totalItrs += itrs;

      if (done >= workgroups) {
        break;
      }

      // Reset iteration counter
      this.queue.writeBuffer(this.buffers.config, 24, new Uint32Array([0]));
    }

    const elapsed = (performance.now() - startTime) / 1000;

    return {
      interactions: totalItrs,
      time: elapsed,
      mips: totalItrs / elapsed / 1e6
    };
  }
}

// Main entry point
async function main() {
  try {
    const hvm = await new HVMRuntime().init();
    console.log("Running HVM program...");
    const result = await hvm.run();
    console.log(`Result: ${result.interactions} interactions in ${result.time.toFixed(2)}s (${result.mips.toFixed(2)} MIPS)`);
  } catch (e) {
    console.error("Error:", e);
  }
}

// Auto-run if in browser
if (typeof window !== "undefined") {
  main();
}

// Export for Node.js
if (typeof module !== "undefined") {
  module.exports = { HVMRuntime, BOOK_DATA, HVM_SHADER };
}
"#);

    output
}

/// Trait for wgpu buffer initialization
trait BufferInitDescriptor {
    fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer;
}

impl BufferInitDescriptor for wgpu::Device {
    fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer {
        wgpu::util::DeviceExt::create_buffer_init(self, desc)
    }
}
