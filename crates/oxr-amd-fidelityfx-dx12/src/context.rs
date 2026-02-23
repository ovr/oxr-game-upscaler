use std::ffi::c_void;

use fsr_sys::*;
use tracing::{error, info};

/// Internal context state stored behind the opaque `ffxContext` pointer.
#[allow(dead_code)]
pub struct OxrContext {
    pub max_render_size: FfxApiDimensions2D,
    pub max_upscale_size: FfxApiDimensions2D,
    pub flags: u32,
    /// Raw `ID3D12Device*` obtained from the DX12 backend descriptor.
    pub device: *mut c_void,
}

pub unsafe fn create_context(
    context: *mut ffxContext,
    desc: *mut ffxCreateContextDescHeader,
    _mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t {
    if context.is_null() || desc.is_null() {
        error!("ffxCreateContext: null pointer argument");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    // Walk the descriptor chain to find the upscale create descriptor.
    let upscale_desc = find_desc(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE);
    if upscale_desc.is_null() {
        error!("ffxCreateContext: no upscale descriptor in chain");
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
    let upscale_desc = &*(upscale_desc as *const ffxCreateContextDescUpscale);

    info!(
        flags = upscale_desc.flags,
        max_render = ?upscale_desc.max_render_size,
        max_upscale = ?upscale_desc.max_upscale_size,
        "ffxCreateContext: upscale"
    );

    // Try to find DX12 backend descriptor for the device pointer.
    let dx12_desc = find_desc(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12);
    let device = if !dx12_desc.is_null() {
        let dx12 = &*(dx12_desc as *const ffxCreateBackendDX12Desc);
        info!(device = ?dx12.device, "ffxCreateContext: DX12 backend");
        dx12.device
    } else {
        info!("ffxCreateContext: no DX12 backend descriptor");
        std::ptr::null_mut()
    };

    let ctx = Box::new(OxrContext {
        max_render_size: upscale_desc.max_render_size,
        max_upscale_size: upscale_desc.max_upscale_size,
        flags: upscale_desc.flags,
        device,
    });

    // Store the boxed context as the opaque ffxContext handle.
    *context = Box::into_raw(ctx) as *mut c_void;

    FFX_API_RETURN_OK
}

pub unsafe fn destroy_context(
    context: *mut ffxContext,
    _mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t {
    if context.is_null() || (*context).is_null() {
        error!("ffxDestroyContext: null context");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    let ctx = Box::from_raw(*context as *mut OxrContext);
    info!(
        max_render = ?ctx.max_render_size,
        max_upscale = ?ctx.max_upscale_size,
        "ffxDestroyContext"
    );

    // ctx is dropped here, freeing the memory.
    *context = std::ptr::null_mut();

    FFX_API_RETURN_OK
}
