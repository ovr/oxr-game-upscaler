use fsr_sys::*;
use tracing::{info, warn};
use windows::Win32::Graphics::Direct3D12::*;

pub unsafe fn handle_dispatch(
    _context: *mut ffxContext,
    desc: *const ffxDispatchDescHeader,
) -> ffxReturnCode_t {
    if desc.is_null() {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    let type_ = (*desc).type_;

    match type_ {
        FFX_API_DISPATCH_DESC_TYPE_UPSCALE => dispatch_upscale(desc),
        FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK => {
            info!("ffxDispatch: GenerateReactiveMask (no-op passthrough)");
            FFX_API_RETURN_OK
        }
        _ => {
            warn!(type_ = type_, "ffxDispatch: unknown descriptor type");
            FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE
        }
    }
}

unsafe fn dispatch_upscale(desc: *const ffxDispatchDescHeader) -> ffxReturnCode_t {
    let d = &*(desc as *const ffxDispatchDescUpscale);

    info!(
        render_size = format_args!("{}x{}", d.render_size.width, d.render_size.height),
        upscale_size = format_args!("{}x{}", d.upscale_size.width, d.upscale_size.height),
        jitter = format_args!("({:.4}, {:.4})", d.jitter_offset.x, d.jitter_offset.y),
        frame_time_delta = d.frame_time_delta,
        sharpness = d.sharpness,
        reset = d.reset,
        "ffxDispatch: Upscale"
    );

    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        warn!("ffxDispatch: null command list");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    let color_raw = d.color.resource;
    let output_raw = d.output.resource;
    if color_raw.is_null() || output_raw.is_null() {
        warn!("ffxDispatch: null color or output resource");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    // Cast raw void* pointers to windows-rs COM interfaces.
    // The game passes raw COM pointers; we wrap them without AddRef via from_raw_borrowed + clone.
    let cmd_list: ID3D12GraphicsCommandList = {
        let borrowed: &ID3D12GraphicsCommandList =
            windows::core::Interface::from_raw_borrowed(&cmd_list_raw)
                .expect("invalid command list pointer");
        borrowed.clone()
    };

    let color_res: ID3D12Resource = {
        let borrowed: &ID3D12Resource =
            windows::core::Interface::from_raw_borrowed(&color_raw)
                .expect("invalid color resource pointer");
        borrowed.clone()
    };

    let output_res: ID3D12Resource = {
        let borrowed: &ID3D12Resource =
            windows::core::Interface::from_raw_borrowed(&output_raw)
                .expect("invalid output resource pointer");
        borrowed.clone()
    };

    // Transition: color → COPY_SOURCE, output → COPY_DEST
    let barriers_before = [
        resource_barrier_transition(&color_res, d.color.state, D3D12_RESOURCE_STATE_COPY_SOURCE),
        resource_barrier_transition(&output_res, d.output.state, D3D12_RESOURCE_STATE_COPY_DEST),
    ];

    let valid_before: Vec<_> = barriers_before.iter().flatten().cloned().collect();
    if !valid_before.is_empty() {
        cmd_list.ResourceBarrier(&valid_before);
    }

    // Passthrough: copy render-region of color → top-left corner of output.
    // CopyResource requires identical dimensions and silently fails when sizes differ
    // (e.g. 1080p render target into a 4K output buffer).  Use CopyTextureRegion
    // instead so only the rendered region is copied.
    let src_w = if d.render_size.width > 0 {
        d.render_size.width
    } else {
        d.color.description.width
    };
    let src_h = if d.render_size.height > 0 {
        d.render_size.height
    } else {
        d.color.description.height
    };

    // Build copy locations without AddRef (transmute_copy borrows the pointer).
    let src_loc = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(unsafe {
            std::mem::transmute_copy(&color_res)
        }),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };
    let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(unsafe {
            std::mem::transmute_copy(&output_res)
        }),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };
    let src_box = D3D12_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: src_w,
        bottom: src_h,
        back: 1,
    };

    info!(
        src = format_args!("{}x{}", src_w, src_h),
        "ffxDispatch: CopyTextureRegion render→output"
    );
    cmd_list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, Some(&src_box));

    // Transition back to original states.
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

    FFX_API_RETURN_OK
}

/// Map FFX resource state to D3D12 resource state.
fn ffx_state_to_d3d12(state: u32) -> D3D12_RESOURCE_STATES {
    let mut d3d_state = D3D12_RESOURCE_STATES(0);

    if state & FFX_API_RESOURCE_STATE_UNORDERED_ACCESS != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if state & FFX_API_RESOURCE_STATE_COMPUTE_READ != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_API_RESOURCE_STATE_PIXEL_READ != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_API_RESOURCE_STATE_COPY_SRC != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if state & FFX_API_RESOURCE_STATE_COPY_DEST != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if state & FFX_API_RESOURCE_STATE_RENDER_TARGET != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    if state & FFX_API_RESOURCE_STATE_PRESENT != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_PRESENT;
    }
    if state & FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    }

    // FFX_API_RESOURCE_STATE_COMMON maps to D3D12 COMMON (0)
    if d3d_state.0 == 0 && state & FFX_API_RESOURCE_STATE_COMMON != 0 {
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
