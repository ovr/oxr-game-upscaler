use core::ffi::c_void;

// ---- Return codes ----

pub type ffxReturnCode_t = u32;

pub const FFX_API_RETURN_OK: ffxReturnCode_t = 0;
pub const FFX_API_RETURN_ERROR: ffxReturnCode_t = 1;
pub const FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: ffxReturnCode_t = 2;
pub const FFX_API_RETURN_ERROR_RUNTIME_ERROR: ffxReturnCode_t = 3;
pub const FFX_API_RETURN_NO_PROVIDER: ffxReturnCode_t = 4;
pub const FFX_API_RETURN_ERROR_MEMORY: ffxReturnCode_t = 5;
pub const FFX_API_RETURN_ERROR_PARAMETER: ffxReturnCode_t = 6;

// ---- Message types ----

pub const FFX_API_MESSAGE_TYPE_ERROR: u32 = 0;
pub const FFX_API_MESSAGE_TYPE_WARNING: u32 = 1;

// ---- Context type ----

pub type ffxContext = *mut c_void;

// ---- Struct type (u64 in the SDK) ----

pub type ffxStructType_t = u64;

// ---- Effect / backend masks and IDs ----

pub const FFX_API_EFFECT_MASK: u64 = 0x00ff_0000;
pub const FFX_API_BACKEND_MASK: u64 = 0xff00_0000;

pub const FFX_API_EFFECT_ID_GENERAL: u64 = 0x0000_0000;
pub const FFX_API_EFFECT_ID_UPSCALE: u64 = 0x0001_0000;
pub const FFX_API_EFFECT_ID_FRAMEGENERATION: u64 = 0x0002_0000;

pub const FFX_API_BACKEND_ID_DX12: u64 = 0x0000_0000;

// ---- General descriptor type constants ----

pub const FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1: ffxStructType_t = 0x0000_0001;
pub const FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: ffxStructType_t = 4;
pub const FFX_API_DESC_TYPE_OVERRIDE_VERSION: ffxStructType_t = 5;

// ---- DX12 backend descriptor types ----

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12: ffxStructType_t = 0x0000_0002;

// ---- Header struct (polymorphic descriptor base) ----

#[repr(C)]
#[derive(Debug)]
pub struct ffxApiHeader {
    pub type_: ffxStructType_t,
    pub p_next: *mut ffxApiHeader,
}

pub type ffxCreateContextDescHeader = ffxApiHeader;
pub type ffxConfigureDescHeader = ffxApiHeader;
pub type ffxQueryDescHeader = ffxApiHeader;
pub type ffxDispatchDescHeader = ffxApiHeader;

// ---- Allocation callbacks ----

pub type FfxAlloc = Option<unsafe extern "C" fn(p_user_data: *mut c_void, size: u64) -> *mut c_void>;
pub type FfxDealloc = Option<unsafe extern "C" fn(p_user_data: *mut c_void, p_mem: *mut c_void)>;

#[repr(C)]
#[derive(Debug)]
pub struct ffxAllocationCallbacks {
    pub p_user_data: *mut c_void,
    pub alloc: FfxAlloc,
    pub dealloc: FfxDealloc,
}

// ---- Message callback ----

pub type FfxApiMessage = Option<unsafe extern "C" fn(type_: u32, message: *const u16)>;

// ---- DX12 backend descriptor ----

#[repr(C)]
#[derive(Debug)]
pub struct ffxCreateBackendDX12Desc {
    pub header: ffxCreateContextDescHeader,
    pub device: *mut c_void, // ID3D12Device*
}

// ---- Global debug configure descriptor ----

#[repr(C)]
#[derive(Debug)]
pub struct ffxConfigureDescGlobalDebug1 {
    pub header: ffxConfigureDescHeader,
    pub fp_message: FfxApiMessage,
    pub debug_level: u32,
}

// ---- Helper: walk pNext chain ----

/// Walk the descriptor chain and find the first header with the given type.
///
/// # Safety
/// All pointers in the chain must be valid.
pub unsafe fn find_desc(mut header: *const ffxApiHeader, type_: ffxStructType_t) -> *const ffxApiHeader {
    while !header.is_null() {
        if (*header).type_ == type_ {
            return header;
        }
        header = (*header).p_next as *const ffxApiHeader;
    }
    core::ptr::null()
}
