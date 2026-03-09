use std::sync::Mutex;

use crate::gpu_pipeline;
use crate::post_processing::PostContext;
use crate::upscaler_type;
use tracing::{error, info};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::dispatch::{apply_barriers, resource_barrier_transition_d3d12};

/// SRV slot for RCAS input.
const SRV_RCAS: u32 = 9;

struct RcasResources {
    temp_rt: ID3D12Resource,
    width: u32,
    height: u32,
}

static RCAS_TEMP: Mutex<Option<RcasResources>> = Mutex::new(None);

unsafe fn ensure_rcas_temp(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource, String> {
    let mut guard = RCAS_TEMP.lock().map_err(|_| "RCAS_TEMP mutex poisoned")?;

    if let Some(res) = guard.as_ref() {
        if res.width == w && res.height == h {
            return Ok(res.temp_rt.clone());
        }
    }

    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: w as u64,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    };

    let mut resource: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            None,
            &mut resource,
        )
        .map_err(|e| format!("CreateCommittedResource for RCAS temp RT failed: {}", e))?;

    let resource = resource.ok_or("CreateCommittedResource returned null")?;
    info!("RCAS temp RT created: {}x{} format={:?}", w, h, format);

    *guard = Some(RcasResources {
        temp_rt: resource.clone(),
        width: w,
        height: h,
    });

    Ok(resource)
}

/// Returns true if RCAS should be applied for the current upscaler.
pub fn is_enabled() -> bool {
    upscaler_type::rcas_get()
}

/// Apply RCAS sharpening: copy output -> temp_rt, RCAS temp_rt -> output.
/// Assumes output is in RENDER_TARGET state with RTV at slot 0.
pub unsafe fn apply(ctx: &PostContext) {
    let gpu = ctx.gpu;
    let cmd_list = ctx.cmd_list;

    let temp_rt = match ensure_rcas_temp(&gpu.device, ctx.output_w, ctx.output_h, gpu.rt_format) {
        Ok(rt) => rt,
        Err(e) => {
            error!("RCAS: temp RT creation failed: {}, skipping", e);
            return;
        }
    };

    // Barrier: temp_rt PSR -> RT
    let barrier_temp_to_rt = resource_barrier_transition_d3d12(
        &temp_rt,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    cmd_list.ResourceBarrier(&[barrier_temp_to_rt]);

    // Barrier: output RT -> COPY_SOURCE, temp_rt RT -> COPY_DEST
    apply_barriers(
        cmd_list,
        &[
            resource_barrier_transition_d3d12(
                ctx.output_res,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
            resource_barrier_transition_d3d12(
                &temp_rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
        ],
    );

    // Copy output -> temp_rt
    cmd_list.CopyResource(&temp_rt, ctx.output_res);

    // Barrier: temp_rt COPY_DEST -> PSR, output COPY_SOURCE -> RT
    apply_barriers(
        cmd_list,
        &[
            resource_barrier_transition_d3d12(
                &temp_rt,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
            resource_barrier_transition_d3d12(
                ctx.output_res,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
        ],
    );

    // Create SRV for temp_rt at RCAS slot
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: gpu.rt_format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    gpu.device.CreateShaderResourceView(
        &temp_rt,
        Some(&srv_desc),
        gpu_pipeline::get_srv_cpu_handle(gpu, SRV_RCAS),
    );

    // Create RTV for output at slot 0
    gpu.device.CreateRenderTargetView(
        ctx.output_res,
        None,
        gpu_pipeline::get_rtv_cpu_handle(gpu, 0),
    );

    // Set RCAS pipeline
    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetPipelineState(&gpu.pso_rcas);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

    let sharpness: f32 = 1.0;
    cmd_list.SetGraphicsRoot32BitConstant(0, sharpness.to_bits(), 0);

    cmd_list.SetGraphicsRootDescriptorTable(1, gpu_pipeline::get_srv_gpu_handle(gpu, SRV_RCAS));

    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: ctx.output_w as f32,
        Height: ctx.output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: ctx.output_w as i32,
        bottom: ctx.output_h as i32,
    };
    cmd_list.RSSetViewports(&[viewport]);
    cmd_list.RSSetScissorRects(&[scissor]);

    let rtv_output = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_output), false, None);

    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd_list.DrawInstanced(3, 1, 0, 0);
    gpu_pipeline::log_device_removed_reason(&gpu.device);

    info!("RCAS sharpening applied");
}
