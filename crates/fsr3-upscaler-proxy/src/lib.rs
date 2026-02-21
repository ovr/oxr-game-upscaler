#![allow(non_snake_case)]
mod logging;
use core::ffi::c_void;
use std::sync::OnceLock;
use tracing::info;
use windows::core::PCSTR;
use windows::Win32::Foundation::HINSTANCE;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::SystemServices::{DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH};

struct FnTable {
    ContextCreate: unsafe extern "C" fn(*mut c_void, *const c_void) -> u32,
    ContextDestroy: unsafe extern "C" fn(*mut c_void) -> u32,
    ContextDispatch: unsafe extern "C" fn(*mut c_void, *const c_void) -> u32,
    GenReactiveMask: unsafe extern "C" fn(*mut c_void, *const c_void) -> u32,
    GetJitterOffset: unsafe extern "C" fn(*mut f32, *mut f32, i32, i32) -> u32,
    GetJitterPhCount: unsafe extern "C" fn(i32, i32) -> i32,
    GetRenderRes: unsafe extern "C" fn(*mut u32, *mut u32, u32, u32, u32) -> u32,
    GetSharedResDesc: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32,
    GetUpscaleRatio: unsafe extern "C" fn(u32) -> f32,
    ResourceIsNull: unsafe extern "C" fn(*const c_void) -> u32,
    SafeRelCopy: unsafe extern "C" fn(*mut c_void, *const c_void, u32),
    SafeRelPipeline: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
    SafeRelResource: unsafe extern "C" fn(*mut c_void, *const c_void, u32),
    AssertReport: unsafe extern "C" fn(*const i8, u32, *const i8, *const i8),
    AssertSetCb: unsafe extern "C" fn(*mut c_void),
}

static FN_TABLE: OnceLock<FnTable> = OnceLock::new();

unsafe fn resolve(module: windows::Win32::Foundation::HMODULE, name: &[u8]) -> *const c_void {
    GetProcAddress(module, PCSTR(name.as_ptr())).unwrap() as *const c_void
}

#[no_mangle]
unsafe extern "system" fn DllMain(_: HINSTANCE, reason: u32, _: *mut ()) -> bool {
    match reason {
        DLL_PROCESS_ATTACH => {
            logging::init();
            info!("fsr3-upscaler-proxy: loading original");
            let wname: Vec<u16> = "ffx_fsr3upscaler_x64_original.dll\0"
                .encode_utf16()
                .collect();
            let hmod = LoadLibraryW(windows::core::PCWSTR(wname.as_ptr()))
                .expect("failed to load ffx_fsr3upscaler_x64_original.dll");
            FN_TABLE
                .set(FnTable {
                    ContextCreate: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerContextCreate\0",
                    )),
                    ContextDestroy: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerContextDestroy\0",
                    )),
                    ContextDispatch: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerContextDispatch\0",
                    )),
                    GenReactiveMask: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerContextGenerateReactiveMask\0",
                    )),
                    GetJitterOffset: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerGetJitterOffset\0",
                    )),
                    GetJitterPhCount: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerGetJitterPhaseCount\0",
                    )),
                    GetRenderRes: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerGetRenderResolutionFromQualityMode\0",
                    )),
                    GetSharedResDesc: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerGetSharedResourceDescriptions\0",
                    )),
                    GetUpscaleRatio: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerGetUpscaleRatioFromQualityMode\0",
                    )),
                    ResourceIsNull: std::mem::transmute(resolve(
                        hmod,
                        b"ffxFsr3UpscalerResourceIsNull\0",
                    )),
                    SafeRelCopy: std::mem::transmute(resolve(
                        hmod,
                        b"ffxSafeReleaseCopyResource\0",
                    )),
                    SafeRelPipeline: std::mem::transmute(resolve(
                        hmod,
                        b"ffxSafeReleasePipeline\0",
                    )),
                    SafeRelResource: std::mem::transmute(resolve(
                        hmod,
                        b"ffxSafeReleaseResource\0",
                    )),
                    AssertReport: std::mem::transmute(resolve(hmod, b"ffxAssertReport\0")),
                    AssertSetCb: std::mem::transmute(resolve(
                        hmod,
                        b"ffxAssertSetPrintingCallback\0",
                    )),
                })
                .ok();
            info!("fsr3-upscaler-proxy: ready");
            true
        }
        DLL_PROCESS_DETACH => {
            info!("fsr3-upscaler-proxy: unloading");
            true
        }
        _ => true,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerContextCreate(
    ctx: *mut c_void,
    desc: *const c_void,
) -> u32 {
    info!("ffxFsr3UpscalerContextCreate");
    (FN_TABLE.get().unwrap().ContextCreate)(ctx, desc)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerContextDestroy(ctx: *mut c_void) -> u32 {
    info!("ffxFsr3UpscalerContextDestroy");
    (FN_TABLE.get().unwrap().ContextDestroy)(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerContextDispatch(
    ctx: *mut c_void,
    desc: *const c_void,
) -> u32 {
    if !desc.is_null() {
        // FfxFsr3UpscalerDispatchDescription layout:
        //   offset 0:   commandList (*mut c_void, 8 bytes)
        //   offset 8:   10 × FfxApiResource (each 48 bytes) = 480 bytes
        //   offset 488: jitterOffset (FfxApiFloatCoords2D, 8 bytes)
        //   offset 496: motionVectorScale (FfxApiFloatCoords2D, 8 bytes)
        //   offset 504: renderSize (FfxApiDimensions2D = 2×u32)
        //   offset 512: upscaleSize (FfxApiDimensions2D = 2×u32)
        let base = desc as *const u8;
        let rw = *(base.add(504) as *const u32);
        let rh = *(base.add(508) as *const u32);
        let uw = *(base.add(512) as *const u32);
        let uh = *(base.add(516) as *const u32);
        info!(
            render = format_args!("{}x{}", rw, rh),
            upscale = format_args!("{}x{}", uw, uh),
            "ffxFsr3UpscalerContextDispatch"
        );
    }
    (FN_TABLE.get().unwrap().ContextDispatch)(ctx, desc)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerContextGenerateReactiveMask(
    ctx: *mut c_void,
    desc: *const c_void,
) -> u32 {
    (FN_TABLE.get().unwrap().GenReactiveMask)(ctx, desc)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerGetJitterOffset(
    ox: *mut f32,
    oy: *mut f32,
    idx: i32,
    pc: i32,
) -> u32 {
    (FN_TABLE.get().unwrap().GetJitterOffset)(ox, oy, idx, pc)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerGetJitterPhaseCount(rw: i32, dw: i32) -> i32 {
    (FN_TABLE.get().unwrap().GetJitterPhCount)(rw, dw)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerGetRenderResolutionFromQualityMode(
    ow: *mut u32,
    oh: *mut u32,
    dw: u32,
    dh: u32,
    qm: u32,
) -> u32 {
    (FN_TABLE.get().unwrap().GetRenderRes)(ow, oh, dw, dh, qm)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerGetSharedResourceDescriptions(
    ctx: *mut c_void,
    desc: *mut c_void,
) -> u32 {
    (FN_TABLE.get().unwrap().GetSharedResDesc)(ctx, desc)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerGetUpscaleRatioFromQualityMode(qm: u32) -> f32 {
    (FN_TABLE.get().unwrap().GetUpscaleRatio)(qm)
}

#[no_mangle]
pub unsafe extern "C" fn ffxFsr3UpscalerResourceIsNull(res: *const c_void) -> u32 {
    (FN_TABLE.get().unwrap().ResourceIsNull)(res)
}

#[no_mangle]
pub unsafe extern "C" fn ffxSafeReleaseCopyResource(
    iface: *mut c_void,
    res: *const c_void,
    fid: u32,
) {
    (FN_TABLE.get().unwrap().SafeRelCopy)(iface, res, fid)
}

#[no_mangle]
pub unsafe extern "C" fn ffxSafeReleasePipeline(iface: *mut c_void, pipe: *mut c_void, fid: u32) {
    (FN_TABLE.get().unwrap().SafeRelPipeline)(iface, pipe, fid)
}

#[no_mangle]
pub unsafe extern "C" fn ffxSafeReleaseResource(iface: *mut c_void, res: *const c_void, fid: u32) {
    (FN_TABLE.get().unwrap().SafeRelResource)(iface, res, fid)
}

#[no_mangle]
pub unsafe extern "C" fn ffxAssertReport(
    file: *const i8,
    line: u32,
    cond: *const i8,
    msg: *const i8,
) {
    (FN_TABLE.get().unwrap().AssertReport)(file, line, cond, msg)
}

#[no_mangle]
pub unsafe extern "C" fn ffxAssertSetPrintingCallback(cb: *mut c_void) {
    (FN_TABLE.get().unwrap().AssertSetCb)(cb)
}
