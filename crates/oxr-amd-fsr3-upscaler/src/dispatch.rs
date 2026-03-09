use crate::fsr3_types::*;
use crate::gpu_pipeline;
use crate::overlay;
use crate::post_processing::{self, PostContext};
use crate::upscaler_type;
use crate::upscalers::{self, DispatchContext};
use tracing::{error, warn};
use windows::Win32::Graphics::Direct3D12::*;

/// Main upscale dispatch: extract resources, run upscaler, post-fx chain, overlay.
pub unsafe fn dispatch_upscale(d: &FfxFsr3UpscalerDispatchDescription) -> u32 {
    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        warn!("dispatch_upscale: null command list");
        return 1;
    }

    let color_raw = d.color.resource;
    let output_raw = d.output.resource;
    if color_raw.is_null() || output_raw.is_null() {
        warn!("dispatch_upscale: null color or output resource");
        return 1;
    }

    let cmd_list: ID3D12GraphicsCommandList =
        match <ID3D12GraphicsCommandList as windows::core::Interface>::from_raw_borrowed(
            &cmd_list_raw,
        ) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_upscale: from_raw_borrowed failed for command list");
                return 1;
            }
        };

    let color_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&color_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_upscale: from_raw_borrowed failed for color resource");
                return 1;
            }
        };

    let output_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&output_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_upscale: from_raw_borrowed failed for output resource");
                return 1;
            }
        };

    // Determine output format and initialize GPU pipeline.
    let output_format = gpu_pipeline::ffx_format_to_dxgi(d.output.description.format);
    let gpu = match gpu_pipeline::get_or_init(&cmd_list, output_format) {
        Some(g) => g,
        None => {
            error!("dispatch_upscale: GPU pipeline init failed, skipping blit");
            return 1;
        }
    };

    // Render dimensions.
    let render_w = if d.render_size.width > 0 {
        d.render_size.width
    } else {
        d.color.description.width
    };
    let render_h = if d.render_size.height > 0 {
        d.render_size.height
    } else {
        d.color.description.height
    };

    let output_w = d.output.description.width;
    let output_h = d.output.description.height;

    if d.color.description.width == 0
        || d.color.description.height == 0
        || output_w == 0
        || output_h == 0
    {
        error!(
            "dispatch_upscale: zero dimensions: color={}x{}, output={}x{}",
            d.color.description.width, d.color.description.height, output_w, output_h
        );
        return 1;
    }

    let current_upscaler = upscaler_type::get();

    // Build dispatch context
    let ctx = DispatchContext {
        cmd_list: &cmd_list,
        gpu,
        d,
        color_res: &color_res,
        output_res: &output_res,
        render_w,
        render_h,
        output_w,
        output_h,
    };

    // --- Dispatch upscaler ---
    // Each upscaler manages its own barriers internally.
    // Contract: receives resources in original FFX states, leaves output in RENDER_TARGET.
    let result = match current_upscaler {
        upscaler_type::UpscalerType::SGSRv2TwoPass => upscalers::sgsr2_two_pass::dispatch(&ctx),
        upscaler_type::UpscalerType::SGSRv2 => upscalers::sgsr2_three_pass::dispatch(&ctx),
        upscaler_type::UpscalerType::Bilinear
        | upscaler_type::UpscalerType::Lanczos
        | upscaler_type::UpscalerType::SGSR => upscalers::simple::dispatch(&ctx),
    };

    if result != 0 {
        return result;
    }

    // --- Post-processing chain ---
    // After upscaler dispatch, output is in RENDER_TARGET state with RTV at slot 0.
    let post_ctx = PostContext {
        cmd_list: &cmd_list,
        gpu,
        d,
        output_res: &output_res,
        output_w,
        output_h,
    };

    if post_processing::rcas::is_enabled() {
        post_processing::rcas::apply(&post_ctx);
    }

    if post_processing::debug_view::is_enabled() {
        post_processing::debug_view::apply(&post_ctx);
    }

    // --- Render imgui overlay ---
    let full_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: output_w as f32,
        Height: output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let full_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: output_w as i32,
        bottom: output_h as i32,
    };
    cmd_list.RSSetViewports(&[full_viewport]);
    cmd_list.RSSetScissorRects(&[full_scissor]);
    overlay::render_frame(&cmd_list, gpu, output_w, output_h);

    // --- Restore output barrier ---
    // All upscalers leave output in RENDER_TARGET state. Restore to original FFX state.
    let barrier = resource_barrier_transition_d3d12(
        &output_res,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        ffx_state_to_d3d12(d.output.state),
    );
    if let Some(b) = barrier {
        cmd_list.ResourceBarrier(&[b]);
    }

    0 // FFX_OK
}

/// Color -> output copy for AA mode (render_size == output_size) with overlay support.
pub unsafe fn dispatch_anti_aliasing(d: &FfxFsr3UpscalerDispatchDescription) -> u32 {
    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        warn!("dispatch_aa_copy: null command list");
        return 1;
    }

    let color_raw = d.color.resource;
    let output_raw = d.output.resource;
    if color_raw.is_null() || output_raw.is_null() {
        warn!("dispatch_aa_copy: null color or output resource");
        return 1;
    }

    let cmd_list: ID3D12GraphicsCommandList =
        match <ID3D12GraphicsCommandList as windows::core::Interface>::from_raw_borrowed(
            &cmd_list_raw,
        ) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_aa_copy: from_raw_borrowed failed for command list");
                return 1;
            }
        };

    let color_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&color_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_aa_copy: from_raw_borrowed failed for color resource");
                return 1;
            }
        };

    let output_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&output_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_aa_copy: from_raw_borrowed failed for output resource");
                return 1;
            }
        };

    let output_w = d.output.description.width;
    let output_h = d.output.description.height;

    // Barriers: color -> COPY_SOURCE, output -> COPY_DEST
    let barriers_before = [
        resource_barrier_transition(&color_res, d.color.state, D3D12_RESOURCE_STATE_COPY_SOURCE),
        resource_barrier_transition(&output_res, d.output.state, D3D12_RESOURCE_STATE_COPY_DEST),
    ];
    let valid_before: Vec<_> = barriers_before.iter().flatten().cloned().collect();
    if !valid_before.is_empty() {
        cmd_list.ResourceBarrier(&valid_before);
    }

    cmd_list.CopyResource(&output_res, &color_res);

    let output_format = gpu_pipeline::ffx_format_to_dxgi(d.output.description.format);
    let gpu = gpu_pipeline::get_or_init(&cmd_list, output_format);

    if let Some(gpu) = gpu {
        let barriers_mid = [
            resource_barrier_transition_d3d12(
                &color_res,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                ffx_state_to_d3d12(d.color.state),
            ),
            resource_barrier_transition_d3d12(
                &output_res,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
        ];
        let valid_mid: Vec<_> = barriers_mid.iter().flatten().cloned().collect();
        if !valid_mid.is_empty() {
            cmd_list.ResourceBarrier(&valid_mid);
        }

        gpu.device.CreateRenderTargetView(
            &output_res,
            None,
            gpu_pipeline::get_rtv_cpu_handle(gpu, 0),
        );

        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: output_w as f32,
            Height: output_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: output_w as i32,
            bottom: output_h as i32,
        };
        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);

        let rtv_handle = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
        cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

        overlay::render_frame(&cmd_list, gpu, output_w, output_h);

        let barriers_after = [resource_barrier_transition_d3d12(
            &output_res,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            ffx_state_to_d3d12(d.output.state),
        )];
        let valid_after: Vec<_> = barriers_after.iter().flatten().cloned().collect();
        if !valid_after.is_empty() {
            cmd_list.ResourceBarrier(&valid_after);
        }
    } else {
        let barriers_after = [
            resource_barrier_transition_d3d12(
                &color_res,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                ffx_state_to_d3d12(d.color.state),
            ),
            resource_barrier_transition_d3d12(
                &output_res,
                D3D12_RESOURCE_STATE_COPY_DEST,
                ffx_state_to_d3d12(d.output.state),
            ),
        ];
        let valid_after: Vec<_> = barriers_after.iter().flatten().cloned().collect();
        if !valid_after.is_empty() {
            cmd_list.ResourceBarrier(&valid_after);
        }
    }

    0 // FFX_OK
}

// ============================================================
// Barrier helpers (pub(crate) — used by upscalers and post-fx)
// ============================================================

// Old SDK FfxResourceStates bit values
const FFX_RESOURCE_STATE_COMMON: u32 = 1 << 0;
const FFX_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_RESOURCE_STATE_COMPUTE_READ: u32 = 1 << 2;
const FFX_RESOURCE_STATE_PIXEL_READ: u32 = 1 << 3;
const FFX_RESOURCE_STATE_COPY_SRC: u32 = 1 << 4;
const FFX_RESOURCE_STATE_COPY_DEST: u32 = 1 << 5;
const FFX_RESOURCE_STATE_INDIRECT_ARGUMENT: u32 = 1 << 6;
const FFX_RESOURCE_STATE_PRESENT: u32 = 1 << 7;
const FFX_RESOURCE_STATE_RENDER_TARGET: u32 = 1 << 8;

/// Map FFX resource state to D3D12 resource state.
pub(crate) fn ffx_state_to_d3d12(state: u32) -> D3D12_RESOURCE_STATES {
    let mut d3d_state = D3D12_RESOURCE_STATES(0);

    if state & FFX_RESOURCE_STATE_UNORDERED_ACCESS != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if state & FFX_RESOURCE_STATE_COMPUTE_READ != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_RESOURCE_STATE_PIXEL_READ != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_RESOURCE_STATE_COPY_SRC != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if state & FFX_RESOURCE_STATE_COPY_DEST != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if state & FFX_RESOURCE_STATE_RENDER_TARGET != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    if state & FFX_RESOURCE_STATE_PRESENT != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_PRESENT;
    }
    if state & FFX_RESOURCE_STATE_INDIRECT_ARGUMENT != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    }

    if d3d_state.0 == 0 && state & FFX_RESOURCE_STATE_COMMON != 0 {
        return D3D12_RESOURCE_STATE_COMMON;
    }

    d3d_state
}

/// Build a transition barrier from FFX state -> D3D12 target state.
pub(crate) fn resource_barrier_transition(
    resource: &ID3D12Resource,
    state_before_ffx: u32,
    state_after: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_BARRIER> {
    let state_before = ffx_state_to_d3d12(state_before_ffx);
    resource_barrier_transition_d3d12(resource, state_before, state_after)
}

/// Build a transition barrier between two D3D12 states, returning None if they're equal.
pub(crate) fn resource_barrier_transition_d3d12(
    resource: &ID3D12Resource,
    state_before: D3D12_RESOURCE_STATES,
    state_after: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_BARRIER> {
    if state_before == state_after {
        return None;
    }

    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        ..Default::default()
    };
    barrier.Anonymous.Transition = std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: state_before,
        StateAfter: state_after,
    });

    Some(barrier)
}
