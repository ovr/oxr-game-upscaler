use std::sync::Mutex;

use crate::gpu_pipeline;
use crate::upscalers::{borrow_resource, create_typed_srv, DispatchContext};
use tracing::{error, info};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::super::dispatch::{
    ffx_state_to_d3d12, resource_barrier_transition, resource_barrier_transition_d3d12,
};

// --- Persistent state ---

pub struct Sgsr2ThreePassState {
    pub ycocg_color: ID3D12Resource,
    pub motion_depth_alpha: ID3D12Resource,
    pub motion_depth_clip_alpha: ID3D12Resource,
    pub luma_history: [ID3D12Resource; 2],
    pub history: [ID3D12Resource; 2],
    pub frame_idx: u32,
    pub initialized: bool,
    pub render_w: u32,
    pub render_h: u32,
    pub output_w: u32,
    pub output_h: u32,
}

static STATE: Mutex<Option<Sgsr2ThreePassState>> = Mutex::new(None);

unsafe fn get_or_create_state(
    device: &ID3D12Device,
    render_w: u32,
    render_h: u32,
    output_w: u32,
    output_h: u32,
    _output_format: DXGI_FORMAT,
) -> Result<std::sync::MutexGuard<'static, Option<Sgsr2ThreePassState>>, String> {
    let mut guard = STATE
        .lock()
        .map_err(|_| "sgsr2 3-pass state mutex poisoned")?;

    let needs_recreate = match guard.as_ref() {
        Some(s) => {
            s.render_w != render_w
                || s.render_h != render_h
                || s.output_w != output_w
                || s.output_h != output_h
        }
        None => true,
    };

    if needs_recreate {
        let ycocg_color = create_texture(device, render_w, render_h, DXGI_FORMAT_R32_UINT)?;
        let motion_depth_alpha =
            create_texture(device, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let motion_depth_clip_alpha =
            create_texture(device, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let luma_history_0 = create_texture(device, render_w, render_h, DXGI_FORMAT_R32_UINT)?;
        let luma_history_1 = create_texture(device, render_w, render_h, DXGI_FORMAT_R32_UINT)?;

        let history0 = create_texture(device, output_w, output_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let history1 = create_texture(device, output_w, output_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;

        info!(
            "sgsr2_3pass: created textures render={}x{} output={}x{}",
            render_w, render_h, output_w, output_h
        );

        *guard = Some(Sgsr2ThreePassState {
            ycocg_color,
            motion_depth_alpha,
            motion_depth_clip_alpha,
            luma_history: [luma_history_0, luma_history_1],
            history: [history0, history1],
            frame_idx: 0,
            initialized: false,
            render_w,
            render_h,
            output_w,
            output_h,
        });
    }

    Ok(guard)
}

unsafe fn create_texture(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource, String> {
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
        .map_err(|e| format!("sgsr2_3pass CreateCommittedResource failed: {}", e))?;

    resource.ok_or_else(|| "sgsr2_3pass CreateCommittedResource returned null".to_string())
}

// --- SRV/RTV slot constants ---
const SRV_CONVERT_DEPTH: u32 = 15;
const SRV_CONVERT_VELOCITY: u32 = 16;
const SRV_CONVERT_COLOR: u32 = 17;
const SRV_ACTIVATE_YCOCG: u32 = 18;
const SRV_ACTIVATE_MDA: u32 = 19;
const SRV_ACTIVATE_PREV_LUMA: u32 = 20;
const SRV_UPSCALE_PREV_HISTORY: u32 = 21;
const SRV_UPSCALE_MDCA: u32 = 22;
const SRV_UPSCALE_YCOCG: u32 = 23;
const RTV_CONVERT_YCOCG: u32 = 4;
const RTV_CONVERT_MDA: u32 = 5;
const RTV_ACTIVATE_MDCA: u32 = 6;
const RTV_ACTIVATE_LUMA: u32 = 7;
const RTV_UPSCALE_HISTORY: u32 = 8;
const RTV_UPSCALE_OUTPUT: u32 = 9;

/// SGSRv2 temporal upscaler: 3-pass dispatch (convert + activate + upscale) with history ping-pong.
pub unsafe fn dispatch(ctx: &DispatchContext) -> u32 {
    let gpu = ctx.gpu;
    let cmd_list = ctx.cmd_list;
    let d = ctx.d;
    let render_w = ctx.render_w;
    let render_h = ctx.render_h;
    let output_w = ctx.output_w;
    let output_h = ctx.output_h;

    let mut state_guard = match get_or_create_state(
        &gpu.device,
        render_w,
        render_h,
        output_w,
        output_h,
        gpu.rt_format,
    ) {
        Ok(g) => g,
        Err(e) => {
            error!("dispatch_sgsr2_3pass: state creation failed: {}", e);
            return 1;
        }
    };
    let state = state_guard.as_mut().unwrap();

    let is_reset = d.reset || !state.initialized;

    let depth_res = match borrow_resource(d.depth.resource) {
        Some(r) => r,
        None => {
            error!("dispatch_sgsr2_3pass: null depth resource");
            return 1;
        }
    };
    let mv_res = match borrow_resource(d.motion_vectors.resource) {
        Some(r) => r,
        None => {
            error!("dispatch_sgsr2_3pass: null motion_vectors resource");
            return 1;
        }
    };

    let root_constants = build_root_constants(d, render_w, render_h, output_w, output_h, is_reset);

    let prev_idx = state.frame_idx as usize;
    let curr_idx = 1 - prev_idx;

    // ============================================================
    // Pass 1: Convert (render resolution)
    // ============================================================
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition(
            &depth_res,
            d.depth.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition(
            &mv_res,
            d.motion_vectors.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition(
            ctx.color_res,
            d.color.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.ycocg_color,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_alpha,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    create_typed_srv(
        gpu,
        &depth_res,
        d.depth.description.format,
        SRV_CONVERT_DEPTH,
    );
    create_typed_srv(
        gpu,
        &mv_res,
        d.motion_vectors.description.format,
        SRV_CONVERT_VELOCITY,
    );
    create_typed_srv(
        gpu,
        ctx.color_res,
        d.color.description.format,
        SRV_CONVERT_COLOR,
    );

    // RTVs for convert MRT
    {
        let rtv_desc_uint = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        gpu.device.CreateRenderTargetView(
            &state.ycocg_color,
            Some(&rtv_desc_uint),
            ctx.rtv_cpu(RTV_CONVERT_YCOCG),
        );
    }
    gpu.device.CreateRenderTargetView(
        &state.motion_depth_alpha,
        None,
        ctx.rtv_cpu(RTV_CONVERT_MDA),
    );

    cmd_list.SetGraphicsRootSignature(&gpu.sgsr2_3pass_root_signature);
    cmd_list.SetPipelineState(&gpu.pso_sgsr2_3p_convert);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);
    cmd_list.SetGraphicsRoot32BitConstants(
        0,
        root_constants.len() as u32,
        root_constants.as_ptr() as *const core::ffi::c_void,
        0,
    );
    cmd_list.SetGraphicsRootDescriptorTable(1, ctx.srv_gpu(SRV_CONVERT_DEPTH));

    let render_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: render_w as f32,
        Height: render_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let render_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: render_w as i32,
        bottom: render_h as i32,
    };
    cmd_list.RSSetViewports(&[render_viewport]);
    cmd_list.RSSetScissorRects(&[render_scissor]);

    let rtv0 = ctx.rtv_cpu(RTV_CONVERT_YCOCG);
    let rtv1 = ctx.rtv_cpu(RTV_CONVERT_MDA);
    let rtvs = [rtv0, rtv1];
    cmd_list.OMSetRenderTargets(2, Some(rtvs.as_ptr()), false, None);

    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd_list.DrawInstanced(3, 1, 0, 0);

    // ============================================================
    // Pass 2: Activate (render resolution)
    // ============================================================
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.ycocg_color,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_alpha,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_clip_alpha,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.luma_history[curr_idx],
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // SRVs for activate
    {
        let srv_uint = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
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
            &state.ycocg_color,
            Some(&srv_uint),
            ctx.srv_cpu(SRV_ACTIVATE_YCOCG),
        );

        let srv_float = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
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
            &state.motion_depth_alpha,
            Some(&srv_float),
            ctx.srv_cpu(SRV_ACTIVATE_MDA),
        );

        gpu.device.CreateShaderResourceView(
            &state.luma_history[prev_idx],
            Some(&srv_uint),
            ctx.srv_cpu(SRV_ACTIVATE_PREV_LUMA),
        );
    }

    // RTVs for activate MRT
    gpu.device.CreateRenderTargetView(
        &state.motion_depth_clip_alpha,
        None,
        ctx.rtv_cpu(RTV_ACTIVATE_MDCA),
    );
    {
        let rtv_desc_uint = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        gpu.device.CreateRenderTargetView(
            &state.luma_history[curr_idx],
            Some(&rtv_desc_uint),
            ctx.rtv_cpu(RTV_ACTIVATE_LUMA),
        );
    }

    cmd_list.SetPipelineState(&gpu.pso_sgsr2_3p_activate);
    cmd_list.SetGraphicsRootDescriptorTable(1, ctx.srv_gpu(SRV_ACTIVATE_YCOCG));

    cmd_list.RSSetViewports(&[render_viewport]);
    cmd_list.RSSetScissorRects(&[render_scissor]);

    let act_rtv0 = ctx.rtv_cpu(RTV_ACTIVATE_MDCA);
    let act_rtv1 = ctx.rtv_cpu(RTV_ACTIVATE_LUMA);
    let act_rtvs = [act_rtv0, act_rtv1];
    cmd_list.OMSetRenderTargets(2, Some(act_rtvs.as_ptr()), false, None);

    cmd_list.DrawInstanced(3, 1, 0, 0);

    // ============================================================
    // Pass 3: Upscale (display resolution)
    // ============================================================
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_clip_alpha,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.luma_history[curr_idx],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if is_reset {
            if let Some(b) = resource_barrier_transition_d3d12(
                &state.history[prev_idx],
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ) {
                barriers.push(b);
            }
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // Clear history on reset
    if is_reset {
        let black = [0.0_f32, 0.0, 0.0, 0.0];
        for idx in [prev_idx, curr_idx] {
            gpu.device.CreateRenderTargetView(
                &state.history[idx],
                None,
                ctx.rtv_cpu(RTV_UPSCALE_OUTPUT),
            );
            cmd_list.ClearRenderTargetView(ctx.rtv_cpu(RTV_UPSCALE_OUTPUT), &black, None);
        }
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[prev_idx],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // SRVs for upscale
    {
        let srv_history = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
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
            &state.history[prev_idx],
            Some(&srv_history),
            ctx.srv_cpu(SRV_UPSCALE_PREV_HISTORY),
        );

        let srv_float = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
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
            &state.motion_depth_clip_alpha,
            Some(&srv_float),
            ctx.srv_cpu(SRV_UPSCALE_MDCA),
        );

        let srv_uint = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
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
            &state.ycocg_color,
            Some(&srv_uint),
            ctx.srv_cpu(SRV_UPSCALE_YCOCG),
        );
    }

    // RTVs for upscale MRT
    gpu.device.CreateRenderTargetView(
        &state.history[curr_idx],
        None,
        ctx.rtv_cpu(RTV_UPSCALE_HISTORY),
    );
    gpu.device
        .CreateRenderTargetView(ctx.output_res, None, ctx.rtv_cpu(RTV_UPSCALE_OUTPUT));

    // Barrier: output_res -> RENDER_TARGET for MRT
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition(
            ctx.output_res,
            d.output.state,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    cmd_list.SetPipelineState(&gpu.pso_sgsr2_3p_upscale);
    cmd_list.SetGraphicsRootDescriptorTable(1, ctx.srv_gpu(SRV_UPSCALE_PREV_HISTORY));

    let upscale_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: output_w as f32,
        Height: output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let upscale_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: output_w as i32,
        bottom: output_h as i32,
    };
    cmd_list.RSSetViewports(&[upscale_viewport]);
    cmd_list.RSSetScissorRects(&[upscale_scissor]);

    let rtv0 = ctx.rtv_cpu(RTV_UPSCALE_HISTORY);
    let rtv1 = ctx.rtv_cpu(RTV_UPSCALE_OUTPUT);
    let up_rtvs = [rtv0, rtv1];
    cmd_list.OMSetRenderTargets(2, Some(up_rtvs.as_ptr()), false, None);

    cmd_list.DrawInstanced(3, 1, 0, 0);

    // ============================================================
    // Post-upscale: transition history, prepare output for overlay
    // ============================================================

    // history[curr]: RENDER_TARGET -> SRV for next frame
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // output_res is already RENDER_TARGET from MRT — create RTV at slot 0 for overlay
    gpu.device
        .CreateRenderTargetView(ctx.output_res, None, ctx.rtv_cpu(0));

    // Restore input resource barriers
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &depth_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.depth.state),
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &mv_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.motion_vectors.state),
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            ctx.color_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.color.state),
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    gpu_pipeline::log_device_removed_reason(&gpu.device);

    // Advance ping-pong
    state.frame_idx = curr_idx as u32;
    state.initialized = true;

    info!(
        render = format_args!("{}x{}", render_w, render_h),
        output = format_args!("{}x{}", output_w, output_h),
        reset = is_reset,
        jitter = format_args!("({:.4}, {:.4})", d.jitter_offset.x, d.jitter_offset.y),
        frame_idx = state.frame_idx,
        pre_exposure = d.pre_exposure,
        "SGSRv2 temporal upscale (3-pass)"
    );

    0 // FFX_OK
}

fn build_root_constants(
    d: &crate::fsr3_types::FfxFsr3UpscalerDispatchDescription,
    render_w: u32,
    render_h: u32,
    output_w: u32,
    output_h: u32,
    is_reset: bool,
) -> [u32; 32] {
    let scale_ratio_x = output_w as f32 / render_w as f32;
    let scale_ratio_y = output_h as f32 / render_h as f32;
    [
        // clipToPrevClip (identity 4x4)
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        // renderSize
        (render_w as f32).to_bits(),
        (render_h as f32).to_bits(),
        // outputSize
        (output_w as f32).to_bits(),
        (output_h as f32).to_bits(),
        // renderSizeRcp
        (1.0 / render_w as f32).to_bits(),
        (1.0 / render_h as f32).to_bits(),
        // outputSizeRcp
        (1.0 / output_w as f32).to_bits(),
        (1.0 / output_h as f32).to_bits(),
        // jitterOffset
        d.jitter_offset.x.to_bits(),
        d.jitter_offset.y.to_bits(),
        // scaleRatio
        scale_ratio_x.to_bits(),
        scale_ratio_y.to_bits(),
        // cameraFovAngleHor
        (d.camera_fov_angle_vertical * 16.0 / 9.0).to_bits(),
        // minLerpContribution
        0.25_f32.to_bits(),
        // reset (ValidReset)
        (if is_reset { 1.0_f32 } else { 0.0_f32 }).to_bits(),
        // preExposure (replaces bSameCamera)
        d.pre_exposure.to_bits(),
    ]
}
