use crate::api::*;
use crate::types::*;
use core::ffi::c_void;

// ---- Descriptor type constants ----
// FFX_API_MAKE_EFFECT_SUB_ID(FFX_API_EFFECT_ID_UPSCALE, sub)
// = (0x00010000 & 0x00ff0000) | (sub & ~0x00ff0000) = 0x00010000 | sub

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: ffxStructType_t = 0x0001_0000;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: ffxStructType_t = 0x0001_0001;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: ffxStructType_t =
    0x0001_0002;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: ffxStructType_t =
    0x0001_0003;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: ffxStructType_t = 0x0001_0004;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: ffxStructType_t = 0x0001_0005;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK: ffxStructType_t = 0x0001_0006;
pub const FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE: ffxStructType_t = 0x0001_0007;

// ---- Quality modes ----

pub type FfxApiUpscaleQualityMode = u32;

pub const FFX_UPSCALE_QUALITY_MODE_NATIVEAA: FfxApiUpscaleQualityMode = 0;
pub const FFX_UPSCALE_QUALITY_MODE_QUALITY: FfxApiUpscaleQualityMode = 1;
pub const FFX_UPSCALE_QUALITY_MODE_BALANCED: FfxApiUpscaleQualityMode = 2;
pub const FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: FfxApiUpscaleQualityMode = 3;
pub const FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: FfxApiUpscaleQualityMode = 4;

// ---- Create context flags ----

pub const FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
pub const FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;
pub const FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION: u32 = 1 << 2;
pub const FFX_UPSCALE_ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
pub const FFX_UPSCALE_ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
pub const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;
pub const FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION: u32 = 1 << 6;
pub const FFX_UPSCALE_ENABLE_DEBUG_CHECKING: u32 = 1 << 7;

// ---- Dispatch flags ----

pub const FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW: u32 = 1 << 0;

// ---- Create context descriptor ----

#[repr(C)]
#[derive(Debug)]
pub struct ffxCreateContextDescUpscale {
    pub header: ffxCreateContextDescHeader,
    pub flags: u32,
    // 4 bytes padding on 64-bit due to alignment after u32
    pub max_render_size: FfxApiDimensions2D,
    pub max_upscale_size: FfxApiDimensions2D,
    pub fp_message: FfxApiMessage,
}

// ---- Dispatch descriptor ----

#[repr(C)]
pub struct ffxDispatchDescUpscale {
    pub header: ffxDispatchDescHeader,
    pub command_list: *mut c_void,
    pub color: FfxApiResource,
    pub depth: FfxApiResource,
    pub motion_vectors: FfxApiResource,
    pub exposure: FfxApiResource,
    pub reactive: FfxApiResource,
    pub transparency_and_composition: FfxApiResource,
    pub output: FfxApiResource,
    pub jitter_offset: FfxApiFloatCoords2D,
    pub motion_vector_scale: FfxApiFloatCoords2D,
    pub render_size: FfxApiDimensions2D,
    pub upscale_size: FfxApiDimensions2D,
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

// ---- Query descriptors ----

#[repr(C)]
pub struct ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode {
    pub header: ffxQueryDescHeader,
    pub quality_mode: u32,
    pub p_out_upscale_ratio: *mut f32,
}

#[repr(C)]
pub struct ffxQueryDescUpscaleGetRenderResolutionFromQualityMode {
    pub header: ffxQueryDescHeader,
    pub display_width: u32,
    pub display_height: u32,
    pub quality_mode: u32,
    pub p_out_render_width: *mut u32,
    pub p_out_render_height: *mut u32,
}

#[repr(C)]
pub struct ffxQueryDescUpscaleGetJitterPhaseCount {
    pub header: ffxQueryDescHeader,
    pub render_width: u32,
    pub display_width: u32,
    pub p_out_phase_count: *mut i32,
}

#[repr(C)]
pub struct ffxQueryDescUpscaleGetJitterOffset {
    pub header: ffxQueryDescHeader,
    pub index: i32,
    pub phase_count: i32,
    pub p_out_x: *mut f32,
    pub p_out_y: *mut f32,
}

// ---- Generate reactive mask dispatch descriptor ----

#[repr(C)]
pub struct ffxDispatchDescUpscaleGenerateReactiveMask {
    pub header: ffxDispatchDescHeader,
    pub command_list: *mut c_void,
    pub color_opaque_only: FfxApiResource,
    pub color_pre_upscale: FfxApiResource,
    pub out_reactive: FfxApiResource,
    pub render_size: FfxApiDimensions2D,
    pub scale: f32,
    pub cutoff_threshold: f32,
    pub binary_value: f32,
    pub flags: u32,
}

// ---- Configure key-value descriptor ----

#[repr(C)]
pub struct ffxConfigureDescUpscaleKeyValue {
    pub header: ffxConfigureDescHeader,
    pub key: u64,
    pub u64_val: u64,
    pub ptr: *mut c_void,
}
