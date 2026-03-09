pub mod sgsr2_three_pass;
pub mod sgsr2_two_pass;
pub mod simple;

use crate::fsr3_types::*;
use crate::gpu_pipeline::{self, GpuState};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Shared context passed to every upscaler dispatch.
pub struct DispatchContext<'a> {
    pub cmd_list: &'a ID3D12GraphicsCommandList,
    pub gpu: &'a GpuState,
    pub d: &'a FfxFsr3UpscalerDispatchDescription,
    pub color_res: &'a ID3D12Resource,
    pub output_res: &'a ID3D12Resource,
    pub render_w: u32,
    pub render_h: u32,
    pub output_w: u32,
    pub output_h: u32,
}

impl<'a> DispatchContext<'a> {
    pub unsafe fn srv_cpu(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        gpu_pipeline::get_srv_cpu_handle(self.gpu, slot)
    }

    pub unsafe fn srv_gpu(&self, slot: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        gpu_pipeline::get_srv_gpu_handle(self.gpu, slot)
    }

    pub unsafe fn rtv_cpu(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        gpu_pipeline::get_rtv_cpu_handle(self.gpu, slot)
    }
}

/// Create an SRV with an explicit typed format descriptor.
/// Converts FFX format -> DXGI, then typeless -> typed, so resources like R32_TYPELESS
/// (depth buffers) get a valid SRV format instead of relying on D3D12 auto-inference.
/// Returns `false` if the format is UNKNOWN and the SRV was not created.
pub unsafe fn create_typed_srv(
    gpu: &GpuState,
    resource: &ID3D12Resource,
    ffx_format: u32,
    slot: u32,
) -> bool {
    let res_format = resource.GetDesc().Format;
    let dxgi_format = if res_format != DXGI_FORMAT_UNKNOWN {
        res_format
    } else {
        gpu_pipeline::ffx_format_to_dxgi(ffx_format)
    };
    let typed_format = gpu_pipeline::dxgi_typeless_to_typed(dxgi_format);
    let typed_format = if typed_format != dxgi_format {
        gpu_pipeline::dxgi_to_filterable(typed_format)
    } else {
        if gpu_pipeline::dxgi_to_filterable(typed_format) != typed_format {
            return false;
        }
        typed_format
    };

    if typed_format == DXGI_FORMAT_UNKNOWN {
        tracing::warn!(
            "create_typed_srv: slot {} has UNKNOWN format after all fallbacks, skipping",
            slot
        );
        return false;
    }

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: typed_format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };

    gpu.device.CreateShaderResourceView(
        resource,
        Some(&srv_desc),
        gpu_pipeline::get_srv_cpu_handle(gpu, slot),
    );
    true
}

/// Borrow a COM resource from a raw FFX pointer, returning None if null.
pub unsafe fn borrow_resource(raw: *mut core::ffi::c_void) -> Option<ID3D12Resource> {
    if raw.is_null() {
        return None;
    }
    <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&raw).map(|b| b.clone())
}
