//! CUDA device wrapper using cudarc.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Wrapper around cudarc's CudaContext for TreeBoost operations.
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaDevice {
    /// Create a new CUDA device (uses device 0).
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();

        Some(Self { ctx, stream })
    }

    /// Get the device name.
    pub fn name(&self) -> String {
        "CUDA Device 0".to_string()
    }

    /// Get the underlying context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Get the default stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Allocate device memory initialized to zero.
    pub fn alloc_zeros<T: DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, len: usize) -> CudaSlice<T> {
        self.stream.alloc_zeros::<T>(len).expect("CUDA alloc failed")
    }

    /// Copy data from host to device.
    pub fn htod_copy<T: DeviceRepr + Clone>(&self, data: &[T]) -> CudaSlice<T> {
        self.stream.clone_htod(data).expect("CUDA htod failed")
    }

    /// Copy data from device to host.
    pub fn dtoh_copy<T: DeviceRepr + Clone>(&self, slice: &CudaSlice<T>) -> Vec<T> {
        self.stream.clone_dtoh(slice).expect("CUDA dtoh failed")
    }

    /// Copy data from host to existing device buffer (no allocation).
    pub fn htod_copy_into<T: DeviceRepr + Clone>(&self, data: &[T], dst: &mut CudaSlice<T>) {
        self.stream.memcpy_htod(data, dst).expect("CUDA memcpy_htod failed");
    }

    /// Load a PTX module.
    pub fn load_module(&self, ptx: Ptx) -> Arc<CudaModule> {
        self.ctx.load_module(ptx).expect("Failed to load PTX module")
    }

    /// Load a function from a module.
    pub fn load_function(module: &Arc<CudaModule>, name: &str) -> CudaFunction {
        module.load_function(name).expect("Function not found")
    }

    /// Synchronize the device (wait for all operations to complete).
    pub fn synchronize(&self) {
        self.stream.synchronize().expect("CUDA sync failed");
    }
}
