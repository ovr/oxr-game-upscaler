use crate::fsr3_types::*;
use crate::gpu_pipeline;
use crate::overlay;
use crate::upscaler_type::{self, UpscalerType};
use tracing::{error, info, warn};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;

/// Bilinear-upscale dispatch: draw a fullscreen triangle sampling the color texture
/// with bilinear filtering, writing to the output as a render target.
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

    // Cast raw void* pointers to windows-rs COM interfaces (borrowed, no AddRef).
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

    // Determine output format and initialize GPU pipeline (extracts device on first call).
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

    let color_tex_w = d.color.description.width;
    let color_tex_h = d.color.description.height;
    let output_w = d.output.description.width;
    let output_h = d.output.description.height;

    if color_tex_w == 0 || color_tex_h == 0 || output_w == 0 || output_h == 0 {
        error!(
            "dispatch_upscale: zero dimensions: color={}x{}, output={}x{}",
            color_tex_w, color_tex_h, output_w, output_h
        );
        return 1;
    }

    // UV scale: render region may be smaller than the color texture.
    let uv_scale_x = render_w as f32 / color_tex_w as f32;
    let uv_scale_y = render_h as f32 / color_tex_h as f32;

    // --- Barriers: color → PIXEL_SHADER_RESOURCE, output → RENDER_TARGET ---
    let barriers_before = [
        resource_barrier_transition(
            &color_res,
            d.color.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ),
        resource_barrier_transition(
            &output_res,
            d.output.state,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ),
    ];
    let valid_before: Vec<_> = barriers_before.iter().flatten().cloned().collect();
    if !valid_before.is_empty() {
        cmd_list.ResourceBarrier(&valid_before);
    }

    // --- Create SRV for color texture ---
    gpu.device.CreateShaderResourceView(
        &color_res,
        None, // default SRV desc
        gpu.srv_heap.GetCPUDescriptorHandleForHeapStart(),
    );

    // --- Create RTV for output texture ---
    gpu.device.CreateRenderTargetView(
        &output_res,
        None, // default RTV desc
        gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart(),
    );

    // --- Set pipeline state ---
    let pso = match upscaler_type::get() {
        UpscalerType::Bilinear => &gpu.pso_bilinear,
        UpscalerType::Lanczos => &gpu.pso_lanczos,
    };
    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetPipelineState(pso);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

    // Root constants: uvScale (slots 0–1) + inputSize (slots 2–3)
    cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_x.to_bits(), 0);
    cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_y.to_bits(), 1);
    cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_w as f32).to_bits(), 2);
    cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_h as f32).to_bits(), 3);

    // Descriptor table: SRV
    cmd_list.SetGraphicsRootDescriptorTable(1, gpu.srv_heap.GetGPUDescriptorHandleForHeapStart());

    // --- Viewport + scissor ---
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

    // --- Set render target ---
    let rtv_handle = gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart();
    cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

    // --- Draw fullscreen triangle ---
    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd_list.DrawInstanced(3, 1, 0, 0);

    info!(
        render = format_args!("{}x{}", render_w, render_h),
        output = format_args!("{}x{}", output_w, output_h),
        uv_scale = format_args!("({:.3}, {:.3})", uv_scale_x, uv_scale_y),
        "DrawInstanced upscale blit"
    );

    // --- Render imgui overlay (RTV still bound, output still in RENDER_TARGET state) ---
    overlay::render_frame(&cmd_list, gpu, output_w, output_h);

    // --- Barriers: restore original states ---
    let barriers_after = [
        resource_barrier_transition_d3d12(
            &color_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.color.state),
        ),
        resource_barrier_transition_d3d12(
            &output_res,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            ffx_state_to_d3d12(d.output.state),
        ),
    ];
    let valid_after: Vec<_> = barriers_after.iter().flatten().cloned().collect();
    if !valid_after.is_empty() {
        cmd_list.ResourceBarrier(&valid_after);
    }

    0 // FFX_OK
}

// Old SDK FfxResourceStates bit values (identical to FFX_API_RESOURCE_STATE_*):
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
fn ffx_state_to_d3d12(state: u32) -> D3D12_RESOURCE_STATES {
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

    // FFX_RESOURCE_STATE_COMMON maps to D3D12 COMMON (0)
    if d3d_state.0 == 0 && state & FFX_RESOURCE_STATE_COMMON != 0 {
        return D3D12_RESOURCE_STATE_COMMON;
    }

    d3d_state
}

/// Build a transition barrier from FFX state → D3D12 target state.
fn resource_barrier_transition(
    resource: &ID3D12Resource,
    state_before_ffx: u32,
    state_after: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_BARRIER> {
    let state_before = ffx_state_to_d3d12(state_before_ffx);
    resource_barrier_transition_d3d12(resource, state_before, state_after)
}

/// Build a transition barrier between two D3D12 states, returning None if they're equal.
fn resource_barrier_transition_d3d12(
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
