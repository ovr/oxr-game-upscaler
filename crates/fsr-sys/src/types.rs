use core::ffi::c_void;

// ---- Surface format ----

pub type FfxApiSurfaceFormat = u32;

pub const FFX_API_SURFACE_FORMAT_UNKNOWN: FfxApiSurfaceFormat = 0;
pub const FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS: FfxApiSurfaceFormat = 1;
pub const FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT: FfxApiSurfaceFormat = 2;
pub const FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT: FfxApiSurfaceFormat = 3;
pub const FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: FfxApiSurfaceFormat = 4;
pub const FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT: FfxApiSurfaceFormat = 5;
pub const FFX_API_SURFACE_FORMAT_R32G32_FLOAT: FfxApiSurfaceFormat = 6;
pub const FFX_API_SURFACE_FORMAT_R8_UINT: FfxApiSurfaceFormat = 7;
pub const FFX_API_SURFACE_FORMAT_R32_UINT: FfxApiSurfaceFormat = 8;
pub const FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS: FfxApiSurfaceFormat = 9;
pub const FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM: FfxApiSurfaceFormat = 10;
pub const FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM: FfxApiSurfaceFormat = 11;
pub const FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB: FfxApiSurfaceFormat = 12;
pub const FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS: FfxApiSurfaceFormat = 13;
pub const FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM: FfxApiSurfaceFormat = 14;
pub const FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB: FfxApiSurfaceFormat = 15;
pub const FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT: FfxApiSurfaceFormat = 16;
pub const FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM: FfxApiSurfaceFormat = 17;
pub const FFX_API_SURFACE_FORMAT_R16G16_FLOAT: FfxApiSurfaceFormat = 18;
pub const FFX_API_SURFACE_FORMAT_R16G16_UINT: FfxApiSurfaceFormat = 19;
pub const FFX_API_SURFACE_FORMAT_R16G16_SINT: FfxApiSurfaceFormat = 20;
pub const FFX_API_SURFACE_FORMAT_R16_FLOAT: FfxApiSurfaceFormat = 21;
pub const FFX_API_SURFACE_FORMAT_R16_UINT: FfxApiSurfaceFormat = 22;
pub const FFX_API_SURFACE_FORMAT_R16_UNORM: FfxApiSurfaceFormat = 23;
pub const FFX_API_SURFACE_FORMAT_R16_SNORM: FfxApiSurfaceFormat = 24;
pub const FFX_API_SURFACE_FORMAT_R8_UNORM: FfxApiSurfaceFormat = 25;
pub const FFX_API_SURFACE_FORMAT_R8G8_UNORM: FfxApiSurfaceFormat = 26;
pub const FFX_API_SURFACE_FORMAT_R8G8_UINT: FfxApiSurfaceFormat = 27;
pub const FFX_API_SURFACE_FORMAT_R32_FLOAT: FfxApiSurfaceFormat = 28;
pub const FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: FfxApiSurfaceFormat = 29;
pub const FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS: FfxApiSurfaceFormat = 30;
pub const FFX_API_SURFACE_FORMAT_R32G32_TYPELESS: FfxApiSurfaceFormat = 31;
pub const FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS: FfxApiSurfaceFormat = 32;
pub const FFX_API_SURFACE_FORMAT_R16G16_TYPELESS: FfxApiSurfaceFormat = 33;
pub const FFX_API_SURFACE_FORMAT_R16_TYPELESS: FfxApiSurfaceFormat = 34;
pub const FFX_API_SURFACE_FORMAT_R8_TYPELESS: FfxApiSurfaceFormat = 35;
pub const FFX_API_SURFACE_FORMAT_R8G8_TYPELESS: FfxApiSurfaceFormat = 36;
pub const FFX_API_SURFACE_FORMAT_R32_TYPELESS: FfxApiSurfaceFormat = 37;
pub const FFX_API_SURFACE_FORMAT_R32G32_UINT: FfxApiSurfaceFormat = 38;
pub const FFX_API_SURFACE_FORMAT_R8_SNORM: FfxApiSurfaceFormat = 39;

// ---- Resource usage flags ----

pub type FfxApiResourceUsage = u32;

pub const FFX_API_RESOURCE_USAGE_READ_ONLY: FfxApiResourceUsage = 0;
pub const FFX_API_RESOURCE_USAGE_RENDERTARGET: FfxApiResourceUsage = 1 << 0;
pub const FFX_API_RESOURCE_USAGE_UAV: FfxApiResourceUsage = 1 << 1;
pub const FFX_API_RESOURCE_USAGE_DEPTHTARGET: FfxApiResourceUsage = 1 << 2;
pub const FFX_API_RESOURCE_USAGE_INDIRECT: FfxApiResourceUsage = 1 << 3;
pub const FFX_API_RESOURCE_USAGE_ARRAYVIEW: FfxApiResourceUsage = 1 << 4;
pub const FFX_API_RESOURCE_USAGE_STENCILTARGET: FfxApiResourceUsage = 1 << 5;

// ---- Resource state flags ----

pub type FfxApiResourceState = u32;

pub const FFX_API_RESOURCE_STATE_COMMON: FfxApiResourceState = 1 << 0;
pub const FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: FfxApiResourceState = 1 << 1;
pub const FFX_API_RESOURCE_STATE_COMPUTE_READ: FfxApiResourceState = 1 << 2;
pub const FFX_API_RESOURCE_STATE_PIXEL_READ: FfxApiResourceState = 1 << 3;
pub const FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ: FfxApiResourceState =
    FFX_API_RESOURCE_STATE_PIXEL_READ | FFX_API_RESOURCE_STATE_COMPUTE_READ;
pub const FFX_API_RESOURCE_STATE_COPY_SRC: FfxApiResourceState = 1 << 4;
pub const FFX_API_RESOURCE_STATE_COPY_DEST: FfxApiResourceState = 1 << 5;
pub const FFX_API_RESOURCE_STATE_GENERIC_READ: FfxApiResourceState =
    FFX_API_RESOURCE_STATE_COPY_SRC | FFX_API_RESOURCE_STATE_COMPUTE_READ;
pub const FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT: FfxApiResourceState = 1 << 6;
pub const FFX_API_RESOURCE_STATE_PRESENT: FfxApiResourceState = 1 << 7;
pub const FFX_API_RESOURCE_STATE_RENDER_TARGET: FfxApiResourceState = 1 << 8;
pub const FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: FfxApiResourceState = 1 << 9;

// ---- Resource type ----

pub type FfxApiResourceType = u32;

pub const FFX_API_RESOURCE_TYPE_BUFFER: FfxApiResourceType = 0;
pub const FFX_API_RESOURCE_TYPE_TEXTURE1D: FfxApiResourceType = 1;
pub const FFX_API_RESOURCE_TYPE_TEXTURE2D: FfxApiResourceType = 2;
pub const FFX_API_RESOURCE_TYPE_TEXTURE_CUBE: FfxApiResourceType = 3;
pub const FFX_API_RESOURCE_TYPE_TEXTURE3D: FfxApiResourceType = 4;

// ---- Resource flags ----

pub type FfxApiResourceFlags = u32;

pub const FFX_API_RESOURCE_FLAGS_NONE: FfxApiResourceFlags = 0;
pub const FFX_API_RESOURCE_FLAGS_ALIASABLE: FfxApiResourceFlags = 1 << 0;

// ---- Structs ----

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfxApiDimensions2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfxApiFloatCoords2D {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfxApiResourceDescription {
    pub type_: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub flags: u32,
    pub usage: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfxApiResource {
    pub resource: *mut c_void,
    pub description: FfxApiResourceDescription,
    pub state: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfxApiEffectMemoryUsage {
    pub total_usage_in_bytes: u64,
    pub aliasable_usage_in_bytes: u64,
}
