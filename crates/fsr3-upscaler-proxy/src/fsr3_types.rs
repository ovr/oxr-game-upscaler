//! Rust bindings for FSR3 Upscaler host types.
//!
//! Each type is annotated with the SDK header path and line number of its C definition.
//! All types use `#[repr(C)]` to match the game's binary ABI exactly.
//!
//! NOTE: `FfxFsr3UpscalerDispatchDescription` intentionally omits the `upscaleSize`
//! field that was added in SDK v1.1.4 (ffx_fsr3upscaler.h:L205). Cyberpunk 2077 was
//! compiled against an older SDK revision; empirical scan confirmed that
//! `enableSharpening` sits at offset 1792 (immediately after `renderSize` at 1784),
//! with no `upscaleSize` gap between them.

#![allow(dead_code, non_camel_case_types)]

use core::ffi::c_void;

// ── From ffx_types.h ─────────────────────────────────────────────────────────

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L278
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FfxSurfaceFormat {
    Unknown = 0,
    R32G32B32A32Typeless = 1,
    R32G32B32A32Uint = 2,
    R32G32B32A32Float = 3,
    R16G16B16A16Float = 4,
    R32G32B32Float = 5,
    R32G32Float = 6,
    R8Uint = 7,
    R32Uint = 8,
    R8G8B8A8Typeless = 9,
    R8G8B8A8Unorm = 10,
    R8G8B8A8Snorm = 11,
    R8G8B8A8Srgb = 12,
    B8G8R8A8Typeless = 13,
    B8G8R8A8Unorm = 14,
    B8G8R8A8Srgb = 15,
    R11G11B10Float = 16,
    R10G10B10A2Unorm = 17,
    R16G16Float = 18,
    R16G16Uint = 19,
    R16G16Sint = 20,
    R16Float = 21,
    R16Uint = 22,
    R16Unorm = 23,
    R16Snorm = 24,
    R8Unorm = 25,
    R8G8Unorm = 26,
    R8G8Uint = 27,
    R32Float = 28,
    R9G9B9E5Sharedexp = 29,
    R16G16B16A16Typeless = 30,
    R32G32Typeless = 31,
    R10G10B10A2Typeless = 32,
    R16G16Typeless = 33,
    R16Typeless = 34,
    R8Typeless = 35,
    R8G8Typeless = 36,
    R32Typeless = 37,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L330
// Bitflags — represented as u32 to avoid UB when the game passes combined values.
pub type FfxResourceUsage = u32;

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L345
// Bitflags — represented as u32 to avoid UB when the game passes combined values.
pub type FfxResourceStates = u32;

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L386
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FfxResourceFlags {
    None = 0,
    Aliasable = 1 << 0,
    Undefined = 1 << 1,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L443
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FfxResourceType {
    Buffer = 0,
    Texture1D = 1,
    Texture2D = 2,
    TextureCube = 3,
    Texture3D = 4,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L676
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FfxDimensions2D {
    pub width: u32,
    pub height: u32,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L705
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FfxFloatCoords2D {
    pub x: f32,
    pub y: f32,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L714
// C unions (width/size, height/stride, depth/alignment) are collapsed to the
// first member name; the layout is identical (all branches are u32).
// 8 × u32 = 32 bytes, alignment 4.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FfxResourceDescription {
    pub type_: u32,  // FfxResourceType
    pub format: u32, // FfxSurfaceFormat
    pub width: u32,  // union: width (texture) / size (buffer)
    pub height: u32, // union: height (texture) / stride (buffer)
    pub depth: u32,  // union: depth (texture) / alignment (buffer)
    pub mip_count: u32,
    pub flags: u32, // FfxResourceFlags
    pub usage: u32, // FfxResourceUsage (bitflags)
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L741
// Memory layout (x64 Windows, wchar_t = 2 bytes):
//   offset  0: resource  (*mut c_void, 8)
//   offset  8: description (FfxResourceDescription, 32)
//   offset 40: state      (u32, 4)
//   offset 44: name       ([u16; 64], 128)
//   offset 172: [4 bytes trailing padding to reach 176, struct alignment = 8]
//   total: 176 bytes
#[repr(C)]
#[derive(Copy, Clone)]
pub struct FfxResource {
    pub resource: *mut c_void,
    pub description: FfxResourceDescription,
    pub state: FfxResourceStates,
    pub name: [u16; 64], // wchar_t[FFX_RESOURCE_NAME_SIZE]
                         // 4 bytes of trailing padding implicit in repr(C)
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_interface.h:L622
// Opaque: 26 fn-ptrs + scratchBuffer (void*) + scratchBufferSize (size_t) + device (void*)
// = 29 × 8 bytes = 232 bytes, alignment 8.
// We never inspect the fields — all callers pass/receive a pointer.
#[repr(C, align(8))]
pub struct FfxInterface {
    _opaque: [u64; 29],
}

// ── From ffx_fsr3upscaler.h ──────────────────────────────────────────────────

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L119
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FfxFsr3UpscalerQualityMode {
    NativeAA = 0,
    Quality = 1,
    Balanced = 2,
    Performance = 3,
    UltraPerformance = 4,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L131
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FfxFsr3UpscalerInitializationFlagBits {
    EnableHighDynamicRange = 1 << 0,
    EnableDisplayResolutionMotionVectors = 1 << 1,
    EnableMotionVectorsJitterCancellation = 1 << 2,
    EnableDepthInverted = 1 << 3,
    EnableDepthInfinite = 1 << 4,
    EnableAutoExposure = 1 << 5,
    EnableDynamicResolution = 1 << 6,
    EnableTexture1DUsage = 1 << 7,
    EnableDebugChecking = 1 << 8,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L171
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FfxFsr3UpscalerDispatchFlags {
    DrawDebugView = 1 << 0,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L161
// Memory layout (x64 Windows):
//   offset  0: flags          (u32, 4)
//   offset  4: max_render_size (FfxDimensions2D, 8)
//   offset 12: max_upscale_size (FfxDimensions2D, 8)
//   offset 20: [4 bytes implicit padding — fp_message needs 8-byte alignment]
//   offset 24: fp_message     (Option<fn>, 8)
//   offset 32: backend_interface (FfxInterface, 232)
//   total: 264 bytes
#[repr(C)]
pub struct FfxFsr3UpscalerContextDescription {
    pub flags: u32,
    pub max_render_size: FfxDimensions2D,
    pub max_upscale_size: FfxDimensions2D,
    // 4 bytes implicit padding here (fn ptr requires 8-byte alignment)
    pub fp_message: Option<unsafe extern "C" fn(u32, *const u16)>,
    pub backend_interface: FfxInterface,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L189
//
// *** GAME ABI — does NOT include `upscaleSize` ***
//
// The SDK v1.1.4 header (line 205) adds `upscaleSize: FfxDimensions2D` between
// `renderSize` and `enableSharpening`. Cyberpunk 2077 was compiled against an older
// revision that lacks this field. Empirical scan confirmed `enableSharpening = 1`
// at byte offset 1792, immediately following `renderSize` at offset 1784.
// The upscale output size is read from `output.description.{width,height}` instead.
//
// Verified field offsets (FfxResource = 176 bytes each):
//   offset    0: command_list                     (*mut c_void, 8)
//   offset    8: color                            (FfxResource, 176)
//   offset  184: depth                            (FfxResource, 176)
//   offset  360: motion_vectors                   (FfxResource, 176)
//   offset  536: exposure                         (FfxResource, 176)
//   offset  712: reactive                         (FfxResource, 176)
//   offset  888: transparency_and_composition     (FfxResource, 176)
//   offset 1064: dilated_depth                    (FfxResource, 176)
//   offset 1240: dilated_motion_vectors           (FfxResource, 176)
//   offset 1416: reconstructed_prev_nearest_depth (FfxResource, 176)
//   offset 1592: output                           (FfxResource, 176)
//   offset 1768: jitter_offset                    (FfxFloatCoords2D, 8)
//   offset 1776: motion_vector_scale              (FfxFloatCoords2D, 8)
//   offset 1784: render_size                      (FfxDimensions2D, 8)  ← confirmed
//   offset 1792: enable_sharpening                (bool, 1 + 3 pad)
//   offset 1796: sharpness                        (f32, 4)
//   offset 1800: frame_time_delta                 (f32, 4)
//   offset 1804: pre_exposure                     (f32, 4)
//   offset 1808: reset                            (bool, 1 + 3 pad)
//   offset 1812: camera_near                      (f32, 4)
//   offset 1816: camera_far                       (f32, 4)
//   offset 1820: camera_fov_angle_vertical        (f32, 4)
//   offset 1824: view_space_to_meters_factor      (f32, 4)
//   offset 1828: flags                            (u32, 4)
//   total: 1832 bytes
#[repr(C)]
pub struct FfxFsr3UpscalerDispatchDescription {
    pub command_list: *mut c_void,
    pub color: FfxResource,
    pub depth: FfxResource,
    pub motion_vectors: FfxResource,
    pub exposure: FfxResource,
    pub reactive: FfxResource,
    pub transparency_and_composition: FfxResource,
    pub dilated_depth: FfxResource,
    pub dilated_motion_vectors: FfxResource,
    pub reconstructed_prev_nearest_depth: FfxResource,
    pub output: FfxResource,
    pub jitter_offset: FfxFloatCoords2D,
    pub motion_vector_scale: FfxFloatCoords2D,
    pub render_size: FfxDimensions2D,
    // NOTE: no `upscale_size` field — see module-level comment
    pub enable_sharpening: bool,
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L221
#[repr(C)]
pub struct FfxFsr3UpscalerGenerateReactiveDescription {
    pub command_list: *mut c_void,
    pub color_opaque_only: FfxResource,
    pub color_pre_upscale: FfxResource,
    pub out_reactive: FfxResource,
    pub render_size: FfxDimensions2D,
    pub scale: f32,
    pub cutoff_threshold: f32,
    pub binary_value: f32,
    pub flags: u32,
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_types.h:L931
// Complex struct (contains FfxResourceInitData with a union). We never read its
// fields in the proxy — only pass pointers through.
pub struct FfxCreateResourceDescription {
    _opaque: [u8; 0],
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L237
// Contains 3 × FfxCreateResourceDescription; passed only as a raw pointer.
pub struct FfxFsr3UpscalerSharedResourceDescriptions {
    _opaque: [u8; 0],
}

// vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h:L256
// FFX_FSR3UPSCALER_CONTEXT_SIZE = FFX_SDK_DEFAULT_CONTEXT_SIZE = 1024 × 128 = 131072
// 131072 × 4 = 524288 bytes (512 KB). Never stack-allocate this.
#[repr(C)]
pub struct FfxFsr3UpscalerContext {
    pub data: [u32; 131072],
}
