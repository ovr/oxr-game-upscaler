pub mod debug_view;
pub mod rcas;

use crate::fsr3_types::*;
use crate::gpu_pipeline::GpuState;
use windows::Win32::Graphics::Direct3D12::*;

/// Shared context passed to post-processing effects.
pub struct PostContext<'a> {
    pub cmd_list: &'a ID3D12GraphicsCommandList,
    pub gpu: &'a GpuState,
    pub d: &'a FfxFsr3UpscalerDispatchDescription,
    pub output_res: &'a ID3D12Resource,
    pub output_w: u32,
    pub output_h: u32,
}
