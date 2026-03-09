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

pub struct Sgsr2State {
    pub motion_depth_clip: ID3D12Resource,
    pub history: [ID3D12Resource; 2],
    pub frame_idx: u32,
    pub initialized: bool,
    pub render_w: u32,
    pub render_h: u32,
    pub output_w: u32,
    pub output_h: u32,
}

static STATE: Mutex<Option<Sgsr2State>> = Mutex::new(None);

unsafe fn get_or_create_state(
    device: &ID3D12Device,
    render_w: u32,
    render_h: u32,
    output_w: u32,
    output_h: u32,
    output_format: DXGI_FORMAT,
) -> Result<std::sync::MutexGuard<'static, Option<Sgsr2State>>, String> {
    let mut guard = STATE.lock().map_err(|_| "sgsr2 state mutex poisoned")?;

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
        let motion_depth_clip =
            create_texture(device, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let history0 = create_texture(device, output_w, output_h, output_format)?;
        let history1 = create_texture(device, output_w, output_h, output_format)?;

        info!(
            "sgsr2: created textures render={}x{} output={}x{} format={:?}",
            render_w, render_h, output_w, output_h, output_format
        );

        *guard = Some(Sgsr2State {
            motion_depth_clip,
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
        .map_err(|e| format!("sgsr2 CreateCommittedResource failed: {}", e))?;

    resource.ok_or_else(|| "sgsr2 CreateCommittedResource returned null".to_string())
}

// --- SRV/RTV slots ---
const SRV_DEPTH: u32 = 10;
const SRV_VELOCITY: u32 = 11;
const SRV_PREV_HISTORY: u32 = 12;
const SRV_MDC: u32 = 13;
const SRV_COLOR: u32 = 14;
const RTV_MDC: u32 = 2;
const RTV_HISTORY: u32 = 3;

/// SGSRv2 temporal upscaler: 2-pass dispatch (convert + upscale) with history ping-pong.
pub unsafe fn dispatch(ctx: &DispatchContext) -> u32 {
    let gpu = ctx.gpu;
    let cmd_list = ctx.cmd_list;
    let d = ctx.d;
    let render_w = ctx.render_w;
    let render_h = ctx.render_h;
    let output_w = ctx.output_w;
    let output_h = ctx.output_h;

    // Get/create persistent SGSRv2 textures
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
            error!("dispatch_sgsr2: state creation failed: {}", e);
            return 1;
        }
    };
    let state = state_guard.as_mut().unwrap();

    let is_reset = d.reset || !state.initialized;

    // Borrow depth and motion_vectors resources
    let depth_res = match borrow_resource(d.depth.resource) {
        Some(r) => r,
        None => {
            error!("dispatch_sgsr2: null depth resource");
            return 1;
        }
    };
    let mv_res = match borrow_resource(d.motion_vectors.resource) {
        Some(r) => r,
        None => {
            error!("dispatch_sgsr2: null motion_vectors resource");
            return 1;
        }
    };

    // === Pass 1: Convert (render resolution) ===
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
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_clip,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // Create SRVs for convert pass
    create_typed_srv(gpu, &depth_res, d.depth.description.format, SRV_DEPTH);
    create_typed_srv(
        gpu,
        &mv_res,
        d.motion_vectors.description.format,
        SRV_VELOCITY,
    );

    // Create RTV for motion_depth_clip
    gpu.device
        .CreateRenderTargetView(&state.motion_depth_clip, None, ctx.rtv_cpu(RTV_MDC));

    // Build root constants (32 DWORDs)
    let root_constants = build_root_constants(d, render_w, render_h, output_w, output_h, is_reset);

    // Set pipeline for convert pass
    cmd_list.SetGraphicsRootSignature(&gpu.sgsr2_root_signature);
    cmd_list.SetPipelineState(&gpu.pso_sgsr2_convert);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

    cmd_list.SetGraphicsRoot32BitConstants(
        0,
        root_constants.len() as u32,
        root_constants.as_ptr() as *const core::ffi::c_void,
        0,
    );

    cmd_list.SetGraphicsRootDescriptorTable(1, ctx.srv_gpu(SRV_DEPTH));

    let convert_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: render_w as f32,
        Height: render_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let convert_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: render_w as i32,
        bottom: render_h as i32,
    };
    cmd_list.RSSetViewports(&[convert_viewport]);
    cmd_list.RSSetScissorRects(&[convert_scissor]);

    let rtv_mdc = ctx.rtv_cpu(RTV_MDC);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_mdc), false, None);

    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd_list.DrawInstanced(3, 1, 0, 0);

    // === Barrier: motion_depth_clip RT -> SRV, color -> SRV ===
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_clip,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
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
        let prev_idx = state.frame_idx as usize;
        let curr_idx = 1 - prev_idx;
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

    let prev_idx = state.frame_idx as usize;
    let curr_idx = 1 - prev_idx;

    // Clear history buffers on reset
    if is_reset {
        let black = [0.0_f32, 0.0, 0.0, 0.0];
        for idx in [prev_idx, curr_idx] {
            gpu.device
                .CreateRenderTargetView(&state.history[idx], None, ctx.rtv_cpu(RTV_HISTORY));
            cmd_list.ClearRenderTargetView(ctx.rtv_cpu(RTV_HISTORY), &black, None);
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

    // === Pass 2: Upscale (display resolution) ===
    {
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
            &state.history[prev_idx],
            Some(&srv_desc),
            ctx.srv_cpu(SRV_PREV_HISTORY),
        );

        let mdc_srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
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
            &state.motion_depth_clip,
            Some(&mdc_srv_desc),
            ctx.srv_cpu(SRV_MDC),
        );
    }
    create_typed_srv(gpu, ctx.color_res, d.color.description.format, SRV_COLOR);

    // Create RTV for history write
    gpu.device
        .CreateRenderTargetView(&state.history[curr_idx], None, ctx.rtv_cpu(RTV_HISTORY));

    cmd_list.SetPipelineState(&gpu.pso_sgsr2_upscale);
    cmd_list.SetGraphicsRootDescriptorTable(1, ctx.srv_gpu(SRV_PREV_HISTORY));

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

    let rtv_history = ctx.rtv_cpu(RTV_HISTORY);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_history), false, None);

    cmd_list.DrawInstanced(3, 1, 0, 0);

    // === Copy history[curr] -> output ===
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition(
            ctx.output_res,
            d.output.state,
            D3D12_RESOURCE_STATE_COPY_DEST,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    cmd_list.CopyResource(ctx.output_res, &state.history[curr_idx]);

    // === Transition output to RENDER_TARGET for overlay ===
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            ctx.output_res,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // Create RTV for output (slot 0) — needed for overlay and post-fx
    gpu.device
        .CreateRenderTargetView(ctx.output_res, None, ctx.rtv_cpu(0));

    // === Restore barriers ===
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
        // History[curr] stays as COPY_SOURCE -> transition to SRV for next frame
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
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
        prev_idx = prev_idx,
        curr_idx = curr_idx,
        d_reset = d.reset,
        "SGSRv2 temporal upscale (2-pass, two_pass)"
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
        // reset
        (if is_reset { 1.0_f32 } else { 0.0_f32 }).to_bits(),
        // bSameCamera
        0u32,
    ]
}
