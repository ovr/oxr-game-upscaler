#![allow(non_snake_case)]

mod context;
mod dispatch;
mod logging;
mod query;

use fsr_sys::*;
use tracing::info;
use windows::Win32::Foundation::HINSTANCE;
use windows::Win32::System::SystemServices::{DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH};

#[no_mangle]
unsafe extern "system" fn DllMain(
    _hinst: HINSTANCE,
    call_reason: u32,
    _reserved: *mut (),
) -> bool {
    match call_reason {
        DLL_PROCESS_ATTACH => {
            logging::init();
            info!("OXR upscaler proxy loaded (passthrough mode)");
            true
        }
        DLL_PROCESS_DETACH => {
            info!("OXR upscaler proxy unloading");
            true
        }
        _ => true,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ffxCreateContext(
    context: *mut ffxContext,
    desc: *mut ffxCreateContextDescHeader,
    mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t {
    context::create_context(context, desc, mem_cb)
}

#[no_mangle]
pub unsafe extern "C" fn ffxDestroyContext(
    context: *mut ffxContext,
    mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t {
    context::destroy_context(context, mem_cb)
}

#[no_mangle]
pub unsafe extern "C" fn ffxConfigure(
    _context: *mut ffxContext,
    desc: *const ffxConfigureDescHeader,
) -> ffxReturnCode_t {
    if !desc.is_null() {
        info!(type_ = (*desc).type_, "ffxConfigure");
    }
    FFX_API_RETURN_OK
}

#[no_mangle]
pub unsafe extern "C" fn ffxQuery(
    context: *mut ffxContext,
    desc: *mut ffxQueryDescHeader,
) -> ffxReturnCode_t {
    query::handle_query(context, desc)
}

#[no_mangle]
pub unsafe extern "C" fn ffxDispatch(
    context: *mut ffxContext,
    desc: *const ffxDispatchDescHeader,
) -> ffxReturnCode_t {
    dispatch::handle_dispatch(context, desc)
}
