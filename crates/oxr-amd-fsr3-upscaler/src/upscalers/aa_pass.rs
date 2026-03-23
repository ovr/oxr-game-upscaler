//! AA model inference via compute shaders.
//!
//! Port of AAPass.cpp from imba. 17 compute shaders, ~57K params,
//! 26 dispatches per frame (prev encoder cached after frame 1).

use std::sync::Mutex;

use tracing::{error, info};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gpu_pipeline::{self, GpuState};
use crate::logging;

// ── Constants ────────────────────────────────────────────────────────────────

const AA_TOTAL_WEIGHTS: u32 = 57352;
const AA_NO_OFFSET: u32 = 0xFFFF_FFFF;

// Pass type indices — must match shader file order in gpu_pipeline AA_PSO array
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PassType {
    PixelUnshuffle = 0,
    Conv = 1,
    GNStats = 2,
    GNApply = 3,
    BackwardWarp = 4,
    Attention = 5,
    NearestUpsample = 6,
    SkipConcatConv = 7,
    PixelShuffleOut = 8,
    ScaleMV = 9,
    GNStatsReduce = 10,
    Conv3x3_16x16 = 11,
    Conv3x3_32x32 = 12,
    Conv3x3_32x16 = 13,
    Conv3x3_16x12 = 14,
    Conv3x3_12x12 = 15,
    Conv3x3S2_16x32 = 16,
}

pub const PASS_COUNT: usize = 17;

const ACT_NONE: u32 = 0;
const ACT_SILU: u32 = 1;
const ACT_TANH: u32 = 3;

const FLAG_HAS_SKIP: u32 = 1;
const FLAG_IS_PREV: u32 = 2;

// ── Weight offsets (from AAWeightOffsets.h) ──────────────────────────────────

struct WeightEntry {
    offset: u32,
    _count: u32,
}

#[allow(dead_code)]
#[repr(usize)]
enum W {
    EncCompressConv = 0,
    EncCompressGnGamma,
    EncCompressGnBeta,
    EncRes1C1Conv,
    EncRes1C1GnGamma,
    EncRes1C1GnBeta,
    EncRes1C2Conv,
    EncRes1Gn2Gamma,
    EncRes1Gn2Beta,
    EncRes2C1Conv,
    EncRes2C1GnGamma,
    EncRes2C1GnBeta,
    EncRes2C2Conv,
    EncRes2Gn2Gamma,
    EncRes2Gn2Beta,
    EncDownConv,
    EncDownGnGamma,
    EncDownGnBeta,
    EncRes3C1Conv,
    EncRes3C1GnGamma,
    EncRes3C1GnBeta,
    EncRes3C2Conv,
    EncRes3Gn2Gamma,
    EncRes3Gn2Beta,
    TempAttnConv,
    TempAttnBias,
    TempMergeConv,
    TempMergeGnGamma,
    TempMergeGnBeta,
    DecUpConv,
    DecUpGnGamma,
    DecUpGnBeta,
    DecSkipConv,
    DecResC1Conv,
    DecResC1GnGamma,
    DecResC1GnBeta,
    DecResC2Conv,
    DecResGn2Gamma,
    DecResGn2Beta,
    HeadExpandConv,
    HeadExpandBias,
    HeadRefineConv,
    HeadRefineBias,
}

static WEIGHTS: &[WeightEntry] = &[
    WeightEntry {
        offset: 0,
        _count: 512,
    }, // EncCompressConv [16,32,1,1]
    WeightEntry {
        offset: 512,
        _count: 16,
    }, // EncCompressGnGamma
    WeightEntry {
        offset: 528,
        _count: 16,
    }, // EncCompressGnBeta
    WeightEntry {
        offset: 544,
        _count: 2304,
    }, // EncRes1C1Conv [16,16,3,3]
    WeightEntry {
        offset: 2848,
        _count: 16,
    }, // EncRes1C1GnGamma
    WeightEntry {
        offset: 2864,
        _count: 16,
    }, // EncRes1C1GnBeta
    WeightEntry {
        offset: 2880,
        _count: 2304,
    }, // EncRes1C2Conv
    WeightEntry {
        offset: 5184,
        _count: 16,
    }, // EncRes1Gn2Gamma
    WeightEntry {
        offset: 5200,
        _count: 16,
    }, // EncRes1Gn2Beta
    WeightEntry {
        offset: 5216,
        _count: 2304,
    }, // EncRes2C1Conv
    WeightEntry {
        offset: 7520,
        _count: 16,
    }, // EncRes2C1GnGamma
    WeightEntry {
        offset: 7536,
        _count: 16,
    }, // EncRes2C1GnBeta
    WeightEntry {
        offset: 7552,
        _count: 2304,
    }, // EncRes2C2Conv
    WeightEntry {
        offset: 9856,
        _count: 16,
    }, // EncRes2Gn2Gamma
    WeightEntry {
        offset: 9872,
        _count: 16,
    }, // EncRes2Gn2Beta
    WeightEntry {
        offset: 9888,
        _count: 4608,
    }, // EncDownConv [32,16,3,3]
    WeightEntry {
        offset: 14496,
        _count: 32,
    }, // EncDownGnGamma
    WeightEntry {
        offset: 14528,
        _count: 32,
    }, // EncDownGnBeta
    WeightEntry {
        offset: 14560,
        _count: 9216,
    }, // EncRes3C1Conv [32,32,3,3]
    WeightEntry {
        offset: 23776,
        _count: 32,
    }, // EncRes3C1GnGamma
    WeightEntry {
        offset: 23808,
        _count: 32,
    }, // EncRes3C1GnBeta
    WeightEntry {
        offset: 23840,
        _count: 9216,
    }, // EncRes3C2Conv
    WeightEntry {
        offset: 33056,
        _count: 32,
    }, // EncRes3Gn2Gamma
    WeightEntry {
        offset: 33088,
        _count: 32,
    }, // EncRes3Gn2Beta
    WeightEntry {
        offset: 33120,
        _count: 2048,
    }, // TempAttnConv [32,64,1,1]
    WeightEntry {
        offset: 35168,
        _count: 32,
    }, // TempAttnBias
    WeightEntry {
        offset: 35200,
        _count: 9216,
    }, // TempMergeConv [32,32,3,3]
    WeightEntry {
        offset: 44416,
        _count: 32,
    }, // TempMergeGnGamma
    WeightEntry {
        offset: 44448,
        _count: 32,
    }, // TempMergeGnBeta
    WeightEntry {
        offset: 44480,
        _count: 4608,
    }, // DecUpConv [16,32,3,3]
    WeightEntry {
        offset: 49088,
        _count: 16,
    }, // DecUpGnGamma
    WeightEntry {
        offset: 49104,
        _count: 16,
    }, // DecUpGnBeta
    WeightEntry {
        offset: 49120,
        _count: 512,
    }, // DecSkipConv [16,32,1,1]
    WeightEntry {
        offset: 49632,
        _count: 2304,
    }, // DecResC1Conv [16,16,3,3]
    WeightEntry {
        offset: 51936,
        _count: 16,
    }, // DecResC1GnGamma
    WeightEntry {
        offset: 51952,
        _count: 16,
    }, // DecResC1GnBeta
    WeightEntry {
        offset: 51968,
        _count: 2304,
    }, // DecResC2Conv
    WeightEntry {
        offset: 54272,
        _count: 16,
    }, // DecResGn2Gamma
    WeightEntry {
        offset: 54288,
        _count: 16,
    }, // DecResGn2Beta
    WeightEntry {
        offset: 54304,
        _count: 1728,
    }, // HeadExpandConv [12,16,3,3]
    WeightEntry {
        offset: 56032,
        _count: 12,
    }, // HeadExpandBias
    WeightEntry {
        offset: 56044,
        _count: 1296,
    }, // HeadRefineConv [12,12,3,3]
    WeightEntry {
        offset: 57340,
        _count: 12,
    }, // HeadRefineBias
];

fn w(idx: W) -> u32 {
    WEIGHTS[idx as usize].offset
}

// ── AAConstants (root constants, 28 DWORDs = 112 bytes) ─────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AAConstants {
    pub pass_type: u32,
    pub in_buf: u32,
    pub out_buf: u32,
    pub aux_buf: u32,
    pub width: u32,
    pub height: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub kernel_size: u32,
    pub stride: u32,
    pub weight_off: u32,
    pub bias_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub num_groups: u32,
    pub activation: u32,
    pub flags: u32,
    pub in_width: u32,
    pub in_height: u32,
    pub buf_stride: u32,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub prev_jitter_x: f32,
    pub prev_jitter_y: f32,
    pub debug_mode: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<AAConstants>() == 112);

impl Default for AAConstants {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ── Persistent AA state ─────────────────────────────────────────────────────

pub struct AAState {
    weight_buffer: ID3D12Resource,
    feature_buffer: ID3D12Resource,
    gn_stats_buffer: ID3D12Resource,
    gn_partials_buffer: ID3D12Resource,
    cached_temporal_buffer: ID3D12Resource,
    pub output_texture: ID3D12Resource,
    pub prev_color: ID3D12Resource,
    pub prev_depth: ID3D12Resource,
    pub prev_motion: ID3D12Resource,

    dispatch_table: Vec<AAConstants>,
    buf_stride: u32,
    feature_buf_total_elements: u32,
    gn_partials_total_elements: u32,
    temporal_cache_bytes: u64,

    prev_encoder_start: u32,
    prev_encoder_end: u32,
    temporal_slot: u32,
    prev_temporal_slot: u32,

    pub has_cached_temporal: bool,
    pub prev_frame_valid: bool,
    pub prev_jitter_x: f32,
    pub prev_jitter_y: f32,
    render_w: u32,
    render_h: u32,
}

static AA_STATE: Mutex<Option<AAState>> = Mutex::new(None);

/// SRV/UAV heap slot assignments for AA pass.
pub const AA_SRV_START: u32 = 25; // t0-t6: slots 25-31
pub const AA_UAV_START: u32 = 32; // u0-u3: slots 32-35

/// Get or create AA state. Returns `None` on error.
pub unsafe fn get_or_create(
    device: &ID3D12Device,
    render_w: u32,
    render_h: u32,
    color_format: DXGI_FORMAT,
) -> std::sync::MutexGuard<'static, Option<AAState>> {
    let mut guard = AA_STATE.lock().unwrap();

    let need_create = match guard.as_ref() {
        None => true,
        Some(s) => s.render_w != render_w || s.render_h != render_h,
    };

    if need_create {
        match create_state(device, render_w, render_h, color_format) {
            Ok(state) => {
                info!(
                    "aa_pass: created resources for {}x{} (features={} floats)",
                    render_w, render_h, state.feature_buf_total_elements
                );
                *guard = Some(state);
            }
            Err(e) => {
                error!("aa_pass: failed to create resources: {}", e);
                *guard = None;
            }
        }
    }

    guard
}

unsafe fn create_state(
    device: &ID3D12Device,
    render_w: u32,
    render_h: u32,
    color_format: DXGI_FORMAT,
) -> Result<AAState, String> {
    let half_w = render_w / 2;
    let half_h = render_h / 2;
    let qtr_w = half_w / 2;
    let qtr_h = half_h / 2;

    let buf_stride = 32 * half_w * half_h;
    let feature_buf_total_elements = 6 * buf_stride;

    let default_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let upload_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        ..Default::default()
    };

    // ── Weight buffer (upload heap) ──
    let weight_buffer = {
        let weight_data = load_weights()?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: weight_data.len() as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &upload_heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut buf,
            )
            .map_err(|e| format!("weight buffer: {}", e))?;
        let buf = buf.ok_or("weight buffer: null")?;

        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut mapped))
            .map_err(|e| format!("weight buffer Map: {}", e))?;
        std::ptr::copy_nonoverlapping(weight_data.as_ptr(), mapped as *mut u8, weight_data.len());
        buf.Unmap(0, None);

        info!(
            "aa_pass: loaded {} weight floats ({} bytes)",
            weight_data.len() / 4,
            weight_data.len()
        );
        buf
    };

    // Helper: create buffer
    let make_buf = |bytes: u64, name: &str| -> Result<ID3D12Resource, String> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &default_heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|e| format!("{}: {}", name, e))?;
        resource.ok_or_else(|| format!("{}: null", name))
    };

    let feature_buffer = make_buf(feature_buf_total_elements as u64 * 4, "feature buffer")?;

    let gn_stats_buffer = make_buf(64 * 4, "gn_stats buffer")?;

    let gn_partials_total_elements = 4 * 64 * 2; // NumGroups * TILES_PER_GROUP * 2
    let gn_partials_buffer = make_buf(gn_partials_total_elements as u64 * 4, "gn_partials buffer")?;

    let temporal_cache_bytes = 32u64 * qtr_w as u64 * qtr_h as u64 * 4;
    let cached_temporal_buffer = make_buf(temporal_cache_bytes, "cached_temporal buffer")?;

    // Helper: create texture
    let make_tex = |w: u32,
                    h: u32,
                    fmt: DXGI_FORMAT,
                    flags: D3D12_RESOURCE_FLAGS,
                    name: &str|
     -> Result<ID3D12Resource, String> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w as u64,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: fmt,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &default_heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|e| format!("{}: {}", name, e))?;
        resource.ok_or_else(|| format!("{}: null", name))
    };

    let output_texture = make_tex(
        render_w,
        render_h,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        "aa output",
    )?;

    let prev_color = make_tex(
        render_w,
        render_h,
        color_format,
        D3D12_RESOURCE_FLAG_NONE,
        "aa prev_color",
    )?;
    let prev_depth = make_tex(
        render_w,
        render_h,
        DXGI_FORMAT_R32_FLOAT,
        D3D12_RESOURCE_FLAG_NONE,
        "aa prev_depth",
    )?;
    let prev_motion = make_tex(
        render_w,
        render_h,
        DXGI_FORMAT_R16G16_FLOAT,
        D3D12_RESOURCE_FLAG_NONE,
        "aa prev_motion",
    )?;

    // Build dispatch table
    let (dispatch_table, prev_encoder_start, prev_encoder_end) =
        build_dispatch_table(render_w, render_h, buf_stride);

    info!(
        "aa_pass: {} dispatches, prev_encoder=[{}..{})",
        dispatch_table.len(),
        prev_encoder_start,
        prev_encoder_end
    );

    Ok(AAState {
        weight_buffer,
        feature_buffer,
        gn_stats_buffer,
        gn_partials_buffer,
        cached_temporal_buffer,
        output_texture,
        prev_color,
        prev_depth,
        prev_motion,
        dispatch_table,
        buf_stride,
        feature_buf_total_elements,
        gn_partials_total_elements,
        temporal_cache_bytes,
        prev_encoder_start,
        prev_encoder_end,
        temporal_slot: 3,
        prev_temporal_slot: 4,
        has_cached_temporal: false,
        prev_frame_valid: false,
        prev_jitter_x: 0.0,
        prev_jitter_y: 0.0,
        render_w,
        render_h,
    })
}

fn load_weights() -> Result<Vec<u8>, String> {
    let dll_dir = logging::dll_directory().unwrap_or_else(|| std::path::PathBuf::from("."));
    let path = dll_dir.join("aa_weights.bin");
    std::fs::read(&path).map_err(|e| format!("failed to read {}: {}", path.display(), e))
}

// ── Descriptor setup ────────────────────────────────────────────────────────

pub unsafe fn setup_descriptors(
    gpu: &GpuState,
    state: &AAState,
    color_res: &ID3D12Resource,
    depth_res: &ID3D12Resource,
    mv_res: &ID3D12Resource,
    output_format: DXGI_FORMAT,
) {
    let device = &gpu.device;

    // t0: weights (StructuredBuffer<float>)
    {
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: AA_TOTAL_WEIGHTS,
                    StructureByteStride: 4,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        device.CreateShaderResourceView(
            &state.weight_buffer,
            Some(&srv),
            gpu_pipeline::get_srv_cpu_handle(gpu, AA_SRV_START),
        );
    }

    // t1: currColor
    create_texture_srv(device, color_res, output_format, gpu, AA_SRV_START + 1);
    // t2: currDepth
    create_texture_srv(
        device,
        depth_res,
        DXGI_FORMAT_R32_FLOAT,
        gpu,
        AA_SRV_START + 2,
    );
    // t3: currMotion
    create_texture_srv(
        device,
        mv_res,
        DXGI_FORMAT_R16G16_FLOAT,
        gpu,
        AA_SRV_START + 3,
    );
    // t4: prevColor
    create_texture_srv(
        device,
        &state.prev_color,
        output_format,
        gpu,
        AA_SRV_START + 4,
    );
    // t5: prevDepth
    create_texture_srv(
        device,
        &state.prev_depth,
        DXGI_FORMAT_R32_FLOAT,
        gpu,
        AA_SRV_START + 5,
    );
    // t6: prevMotion
    create_texture_srv(
        device,
        &state.prev_motion,
        DXGI_FORMAT_R16G16_FLOAT,
        gpu,
        AA_SRV_START + 6,
    );

    // u0: features (RWStructuredBuffer<float>)
    create_buffer_uav(
        device,
        &state.feature_buffer,
        state.feature_buf_total_elements,
        gpu,
        AA_UAV_START,
    );
    // u1: gnStats
    create_buffer_uav(device, &state.gn_stats_buffer, 64, gpu, AA_UAV_START + 1);
    // u2: output (RWTexture2D<float4>)
    {
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        device.CreateUnorderedAccessView(
            &state.output_texture,
            None,
            Some(&uav),
            gpu_pipeline::get_srv_cpu_handle(gpu, AA_UAV_START + 2),
        );
    }
    // u3: gnPartials
    create_buffer_uav(
        device,
        &state.gn_partials_buffer,
        state.gn_partials_total_elements,
        gpu,
        AA_UAV_START + 3,
    );
}

unsafe fn create_texture_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    format: DXGI_FORMAT,
    gpu: &GpuState,
    slot: u32,
) {
    let res_format = resource.GetDesc().Format;
    let typed = gpu_pipeline::dxgi_typeless_to_typed(res_format);
    let final_format = if typed != DXGI_FORMAT_UNKNOWN && typed != res_format {
        // Was typeless, use typed version
        gpu_pipeline::dxgi_to_filterable(typed)
    } else if res_format != DXGI_FORMAT_UNKNOWN {
        res_format
    } else {
        format
    };

    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: final_format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    device.CreateShaderResourceView(
        resource,
        Some(&srv),
        gpu_pipeline::get_srv_cpu_handle(gpu, slot),
    );
}

unsafe fn create_buffer_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    num_elements: u32,
    gpu: &GpuState,
    slot: u32,
) {
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: num_elements,
                StructureByteStride: 4,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_NONE,
            },
        },
    };
    device.CreateUnorderedAccessView(
        resource,
        None,
        Some(&uav),
        gpu_pipeline::get_srv_cpu_handle(gpu, slot),
    );
}

// ── Execute ─────────────────────────────────────────────────────────────────

pub unsafe fn execute(
    cmd_list: &ID3D12GraphicsCommandList,
    gpu: &GpuState,
    state: &mut AAState,
    jitter_x: f32,
    jitter_y: f32,
    prev_jitter_x: f32,
    prev_jitter_y: f32,
) {
    if state.dispatch_table.is_empty() {
        return;
    }

    cmd_list.SetComputeRootSignature(&gpu.aa_root_signature);
    cmd_list.SetComputeRootDescriptorTable(1, gpu_pipeline::get_srv_gpu_handle(gpu, AA_SRV_START));
    cmd_list.SetComputeRootDescriptorTable(2, gpu_pipeline::get_srv_gpu_handle(gpu, AA_UAV_START));

    let use_cache = state.has_cached_temporal;

    // If using cache, copy cached prev_temporal → buf4
    if use_cache {
        let dst_offset = state.prev_temporal_slot as u64 * state.buf_stride as u64 * 4;

        let barrier_pre = make_uav_to_copy_src(&state.cached_temporal_buffer);
        cmd_list.ResourceBarrier(&[barrier_pre]);

        cmd_list.CopyBufferRegion(
            &state.feature_buffer,
            dst_offset,
            &state.cached_temporal_buffer,
            0,
            state.temporal_cache_bytes,
        );

        let barrier_post = make_copy_src_to_common(&state.cached_temporal_buffer);
        cmd_list.ResourceBarrier(&[barrier_post]);
    }

    let mut current_pso = u32::MAX;

    for i in 0..state.dispatch_table.len() {
        // Skip prev encoder dispatches when using cached result
        if use_cache && i as u32 >= state.prev_encoder_start && (i as u32) < state.prev_encoder_end
        {
            continue;
        }

        let d = &mut state.dispatch_table[i];
        let pass_type = d.pass_type;

        if pass_type >= PASS_COUNT as u32 {
            continue;
        }

        // Switch PSO only when pass type changes
        if pass_type != current_pso {
            cmd_list.SetPipelineState(&gpu.aa_psos[pass_type as usize]);
            current_pso = pass_type;
        }

        // Patch per-frame data
        d.jitter_x = jitter_x;
        d.jitter_y = jitter_y;
        d.prev_jitter_x = prev_jitter_x;
        d.prev_jitter_y = prev_jitter_y;
        d.debug_mode = 0;

        // Set root constants (28 DWORDs)
        cmd_list.SetComputeRoot32BitConstants(
            0,
            28,
            d as *const AAConstants as *const std::ffi::c_void,
            0,
        );

        // Calculate thread groups
        let (groups_x, groups_y) = if pass_type == PassType::GNStats as u32 {
            (d.num_groups * 64, 1) // TILES_PER_GROUP = 64
        } else if pass_type == PassType::GNStatsReduce as u32 {
            (d.num_groups, 1)
        } else {
            ((d.width + 7) / 8, (d.height + 7) / 8)
        };

        cmd_list.Dispatch(groups_x, groups_y, 1);

        // Targeted UAV barriers on our resources only (not NULL — NULL flushes ALL GPU UAV work
        // including the game's, causing massive stalls under concurrent rendering).
        let make_uav = |res: &ID3D12Resource| {
            let mut b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            };
            b.Anonymous.UAV = std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::transmute_copy(res),
            });
            b
        };
        cmd_list.ResourceBarrier(&[
            make_uav(&state.feature_buffer),
            make_uav(&state.gn_stats_buffer),
            make_uav(&state.gn_partials_buffer),
        ]);
    }

    // Save curr_temporal (buf3) → cache for next frame
    {
        let src_offset = state.temporal_slot as u64 * state.buf_stride as u64 * 4;

        let barriers_pre = [
            make_barrier(
                &state.feature_buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
            make_barrier(
                &state.cached_temporal_buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
        ];
        cmd_list.ResourceBarrier(&barriers_pre);

        cmd_list.CopyBufferRegion(
            &state.cached_temporal_buffer,
            0,
            &state.feature_buffer,
            src_offset,
            state.temporal_cache_bytes,
        );

        let barriers_post = [
            make_barrier(
                &state.feature_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            make_barrier(
                &state.cached_temporal_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            ),
        ];
        cmd_list.ResourceBarrier(&barriers_post);

        state.has_cached_temporal = true;
    }
}

// ── Barrier helpers ─────────────────────────────────────────────────────────

unsafe fn make_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        ..Default::default()
    };
    barrier.Anonymous.Transition = std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: std::mem::transmute_copy(resource),
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: before,
        StateAfter: after,
    });
    barrier
}

unsafe fn make_uav_to_copy_src(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    make_barrier(
        resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    )
}

unsafe fn make_copy_src_to_common(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    make_barrier(
        resource,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    )
}

// ── Dispatch table builder ──────────────────────────────────────────────────

fn build_dispatch_table(
    render_w: u32,
    render_h: u32,
    buf_stride: u32,
) -> (Vec<AAConstants>, u32, u32) {
    let half_w = render_w / 2;
    let half_h = render_h / 2;
    let qtr_w = half_w / 2;
    let qtr_h = half_h / 2;

    let mut dispatches: Vec<AAConstants> = Vec::with_capacity(64);

    // ── Helper closures ──

    let add_pixel_unshuffle = |dispatches: &mut Vec<AAConstants>, is_prev: bool, out_buf: u32| {
        let mut c = AAConstants::default();
        c.pass_type = PassType::PixelUnshuffle as u32;
        c.out_buf = out_buf;
        c.width = half_w;
        c.height = half_h;
        c.in_width = render_w;
        c.in_height = render_h;
        c.flags = if is_prev { FLAG_IS_PREV } else { 0 };
        c.buf_stride = buf_stride;
        dispatches.push(c);
    };

    let select_conv_pass = |ks: u32, stride: u32, in_ch: u32, out_ch: u32| -> u32 {
        if ks == 3 && stride == 1 && in_ch == 16 && out_ch == 16 {
            PassType::Conv3x3_16x16 as u32
        } else if ks == 3 && stride == 1 && in_ch == 32 && out_ch == 32 {
            PassType::Conv3x3_32x32 as u32
        } else if ks == 3 && stride == 1 && in_ch == 32 && out_ch == 16 {
            PassType::Conv3x3_32x16 as u32
        } else if ks == 3 && stride == 1 && in_ch == 16 && out_ch == 12 {
            PassType::Conv3x3_16x12 as u32
        } else if ks == 3 && stride == 1 && in_ch == 12 && out_ch == 12 {
            PassType::Conv3x3_12x12 as u32
        } else if ks == 3 && stride == 2 && in_ch == 16 && out_ch == 32 {
            PassType::Conv3x3S2_16x32 as u32
        } else {
            PassType::Conv as u32
        }
    };

    #[allow(clippy::too_many_arguments)]
    let add_conv = |dispatches: &mut Vec<AAConstants>,
                    in_buf: u32,
                    out_buf: u32,
                    in_ch: u32,
                    out_ch: u32,
                    ks: u32,
                    stride: u32,
                    in_w: u32,
                    in_h: u32,
                    w_idx: W,
                    bias_off: u32,
                    activation: u32| {
        let mut c = AAConstants::default();
        c.pass_type = select_conv_pass(ks, stride, in_ch, out_ch);
        c.in_buf = in_buf;
        c.out_buf = out_buf;
        c.in_channels = in_ch;
        c.out_channels = out_ch;
        c.kernel_size = ks;
        c.stride = stride;
        c.in_width = in_w;
        c.in_height = in_h;
        c.width = if stride == 2 { in_w / 2 } else { in_w };
        c.height = if stride == 2 { in_h / 2 } else { in_h };
        c.weight_off = w(w_idx);
        c.bias_off = bias_off;
        c.activation = activation;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    };

    let add_gn_stats = |dispatches: &mut Vec<AAConstants>,
                        buf: u32,
                        ch: u32,
                        groups: u32,
                        gn_w: u32,
                        gn_h: u32| {
        // Pass A: partial reduction
        let mut c = AAConstants::default();
        c.pass_type = PassType::GNStats as u32;
        c.in_buf = buf;
        c.out_channels = ch;
        c.num_groups = groups;
        c.width = gn_w;
        c.height = gn_h;
        c.buf_stride = buf_stride;
        dispatches.push(c);

        // Pass B: final reduction
        let mut r = AAConstants::default();
        r.pass_type = PassType::GNStatsReduce as u32;
        r.in_buf = buf;
        r.out_channels = ch;
        r.num_groups = groups;
        r.width = gn_w;
        r.height = gn_h;
        r.buf_stride = buf_stride;
        dispatches.push(r);
    };

    #[allow(clippy::too_many_arguments)]
    let add_gn_apply = |dispatches: &mut Vec<AAConstants>,
                        in_buf: u32,
                        out_buf: u32,
                        ch: u32,
                        groups: u32,
                        gn_w: u32,
                        gn_h: u32,
                        gamma_idx: W,
                        beta_idx: W,
                        activation: u32,
                        has_skip: bool,
                        skip_buf: u32| {
        let mut c = AAConstants::default();
        c.pass_type = PassType::GNApply as u32;
        c.in_buf = in_buf;
        c.out_buf = out_buf;
        c.out_channels = ch;
        c.num_groups = groups;
        c.width = gn_w;
        c.height = gn_h;
        c.gamma_off = w(gamma_idx);
        c.beta_off = w(beta_idx);
        c.activation = activation;
        c.flags = if has_skip { FLAG_HAS_SKIP } else { 0 };
        c.aux_buf = skip_buf;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    };

    // ── Macro helpers for common patterns ──

    // ConvBlock: Conv → GN → SiLU
    macro_rules! conv_block {
        ($d:expr, $in_buf:expr, $out_buf:expr, $in_ch:expr, $out_ch:expr,
         $ks:expr, $stride:expr, $iw:expr, $ih:expr,
         $conv_w:expr, $gn_gamma:expr, $gn_beta:expr) => {{
            let ow = if $stride == 2 { $iw / 2 } else { $iw };
            let oh = if $stride == 2 { $ih / 2 } else { $ih };
            add_conv(
                $d,
                $in_buf,
                $out_buf,
                $in_ch,
                $out_ch,
                $ks,
                $stride,
                $iw,
                $ih,
                $conv_w,
                AA_NO_OFFSET,
                ACT_NONE,
            );
            add_gn_stats($d, $out_buf, $out_ch, 4, ow, oh);
            add_gn_apply(
                $d, $out_buf, $out_buf, $out_ch, 4, ow, oh, $gn_gamma, $gn_beta, ACT_SILU, false, 0,
            );
        }};
    }

    // ResBlock: ConvBlock → Conv → GN → SiLU+skip
    macro_rules! res_block {
        ($d:expr, $in_buf:expr, $temp_buf:expr, $out_buf:expr, $skip_buf:expr,
         $ch:expr, $bw:expr, $bh:expr,
         $c1_conv:expr, $c1_gamma:expr, $c1_beta:expr,
         $c2_conv:expr, $gn2_gamma:expr, $gn2_beta:expr) => {{
            conv_block!(
                $d, $in_buf, $temp_buf, $ch, $ch, 3, 1, $bw, $bh, $c1_conv, $c1_gamma, $c1_beta
            );
            add_conv(
                $d,
                $temp_buf,
                $out_buf,
                $ch,
                $ch,
                3,
                1,
                $bw,
                $bh,
                $c2_conv,
                AA_NO_OFFSET,
                ACT_NONE,
            );
            add_gn_stats($d, $out_buf, $ch, 4, $bw, $bh);
            add_gn_apply(
                $d, $out_buf, $out_buf, $ch, 4, $bw, $bh, $gn2_gamma, $gn2_beta, ACT_SILU, true,
                $skip_buf,
            );
        }};
    }

    // =====================================================
    // ENCODE CURRENT FRAME
    // =====================================================

    // PixelUnshuffle curr → buf0 [32, halfW, halfH]
    add_pixel_unshuffle(&mut dispatches, false, 0);

    // encoder.compress: Conv1x1 32→16 + GN + SiLU → buf1
    conv_block!(
        &mut dispatches,
        0,
        1,
        32,
        16,
        1,
        1,
        half_w,
        half_h,
        W::EncCompressConv,
        W::EncCompressGnGamma,
        W::EncCompressGnBeta
    );

    // encoder.res1: buf1→buf2(temp), buf0(out), skip=buf1
    res_block!(
        &mut dispatches,
        1,
        2,
        0,
        1,
        16,
        half_w,
        half_h,
        W::EncRes1C1Conv,
        W::EncRes1C1GnGamma,
        W::EncRes1C1GnBeta,
        W::EncRes1C2Conv,
        W::EncRes1Gn2Gamma,
        W::EncRes1Gn2Beta
    );

    // encoder.res2: buf0→buf1(temp), buf2(out), skip=buf0
    res_block!(
        &mut dispatches,
        0,
        1,
        2,
        0,
        16,
        half_w,
        half_h,
        W::EncRes2C1Conv,
        W::EncRes2C1GnGamma,
        W::EncRes2C1GnBeta,
        W::EncRes2C2Conv,
        W::EncRes2Gn2Gamma,
        W::EncRes2Gn2Beta
    );
    // buf2 = curr_spatial [16, halfH, halfW]

    // encoder.downsample: Conv3x3 stride=2 16→32 + GN + SiLU → buf0
    conv_block!(
        &mut dispatches,
        2,
        0,
        16,
        32,
        3,
        2,
        half_w,
        half_h,
        W::EncDownConv,
        W::EncDownGnGamma,
        W::EncDownGnBeta
    );

    // encoder.res3: buf0→buf1(temp), buf3(out), skip=buf0
    res_block!(
        &mut dispatches,
        0,
        1,
        3,
        0,
        32,
        qtr_w,
        qtr_h,
        W::EncRes3C1Conv,
        W::EncRes3C1GnGamma,
        W::EncRes3C1GnBeta,
        W::EncRes3C2Conv,
        W::EncRes3Gn2Gamma,
        W::EncRes3Gn2Beta
    );
    // buf2 = curr_spatial, buf3 = curr_temporal

    // =====================================================
    // ENCODE PREVIOUS FRAME
    // =====================================================
    let prev_encoder_start = dispatches.len() as u32;

    add_pixel_unshuffle(&mut dispatches, true, 0);

    conv_block!(
        &mut dispatches,
        0,
        1,
        32,
        16,
        1,
        1,
        half_w,
        half_h,
        W::EncCompressConv,
        W::EncCompressGnGamma,
        W::EncCompressGnBeta
    );

    res_block!(
        &mut dispatches,
        1,
        4,
        0,
        1,
        16,
        half_w,
        half_h,
        W::EncRes1C1Conv,
        W::EncRes1C1GnGamma,
        W::EncRes1C1GnBeta,
        W::EncRes1C2Conv,
        W::EncRes1Gn2Gamma,
        W::EncRes1Gn2Beta
    );

    res_block!(
        &mut dispatches,
        0,
        1,
        4,
        0,
        16,
        half_w,
        half_h,
        W::EncRes2C1Conv,
        W::EncRes2C1GnGamma,
        W::EncRes2C1GnBeta,
        W::EncRes2C2Conv,
        W::EncRes2Gn2Gamma,
        W::EncRes2Gn2Beta
    );

    conv_block!(
        &mut dispatches,
        4,
        0,
        16,
        32,
        3,
        2,
        half_w,
        half_h,
        W::EncDownConv,
        W::EncDownGnGamma,
        W::EncDownGnBeta
    );

    res_block!(
        &mut dispatches,
        0,
        1,
        4,
        0,
        32,
        qtr_w,
        qtr_h,
        W::EncRes3C1Conv,
        W::EncRes3C1GnGamma,
        W::EncRes3C1GnBeta,
        W::EncRes3C2Conv,
        W::EncRes3Gn2Gamma,
        W::EncRes3Gn2Beta
    );

    let prev_encoder_end = dispatches.len() as u32;

    // =====================================================
    // TEMPORAL FUSION
    // =====================================================

    // Scale motion vectors
    {
        let mut c = AAConstants::default();
        c.pass_type = PassType::ScaleMV as u32;
        c.out_buf = 0;
        c.width = qtr_w;
        c.height = qtr_h;
        c.in_width = render_w;
        c.in_height = render_h;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    }

    // Backward warp: warp buf4 (prev temporal) using buf0 (scaled MV) → buf1
    {
        let mut c = AAConstants::default();
        c.pass_type = PassType::BackwardWarp as u32;
        c.in_buf = 4;
        c.out_buf = 1;
        c.aux_buf = 0;
        c.width = qtr_w;
        c.height = qtr_h;
        c.in_channels = 32;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    }

    // Attention: concat(buf3=curr, buf1=warped) → blend → buf0
    {
        let mut c = AAConstants::default();
        c.pass_type = PassType::Attention as u32;
        c.in_buf = 3;
        c.aux_buf = 1;
        c.out_buf = 0;
        c.width = qtr_w;
        c.height = qtr_h;
        c.in_channels = 32;
        c.weight_off = w(W::TempAttnConv);
        c.bias_off = w(W::TempAttnBias);
        c.buf_stride = buf_stride;
        dispatches.push(c);
    }

    // Merge conv + GN + SiLU: buf0 → buf1
    add_conv(
        &mut dispatches,
        0,
        1,
        32,
        32,
        3,
        1,
        qtr_w,
        qtr_h,
        W::TempMergeConv,
        AA_NO_OFFSET,
        ACT_NONE,
    );
    add_gn_stats(&mut dispatches, 1, 32, 4, qtr_w, qtr_h);
    add_gn_apply(
        &mut dispatches,
        1,
        1,
        32,
        4,
        qtr_w,
        qtr_h,
        W::TempMergeGnGamma,
        W::TempMergeGnBeta,
        ACT_SILU,
        false,
        0,
    );

    // =====================================================
    // DECODER
    // =====================================================

    // Nearest upsample 2×: buf1 [32, qtrH, qtrW] → buf0 [32, halfH, halfW]
    {
        let mut c = AAConstants::default();
        c.pass_type = PassType::NearestUpsample as u32;
        c.in_buf = 1;
        c.out_buf = 0;
        c.width = half_w;
        c.height = half_h;
        c.in_width = qtr_w;
        c.in_height = qtr_h;
        c.in_channels = 32;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    }

    // decoder.upsample_conv: Conv3x3 32→16 + GN + SiLU → buf3
    conv_block!(
        &mut dispatches,
        0,
        3,
        32,
        16,
        3,
        1,
        half_w,
        half_h,
        W::DecUpConv,
        W::DecUpGnGamma,
        W::DecUpGnBeta
    );

    // Skip concat conv: concat(buf3, buf2) → 1×1 conv 32→16 → buf0
    {
        let mut c = AAConstants::default();
        c.pass_type = PassType::SkipConcatConv as u32;
        c.in_buf = 3;
        c.aux_buf = 2;
        c.out_buf = 0;
        c.width = half_w;
        c.height = half_h;
        c.in_channels = 32;
        c.out_channels = 16;
        c.weight_off = w(W::DecSkipConv);
        c.bias_off = AA_NO_OFFSET;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    }

    // decoder.res: buf0→buf1(temp), buf2(out), skip=buf0
    res_block!(
        &mut dispatches,
        0,
        1,
        2,
        0,
        16,
        half_w,
        half_h,
        W::DecResC1Conv,
        W::DecResC1GnGamma,
        W::DecResC1GnBeta,
        W::DecResC2Conv,
        W::DecResGn2Gamma,
        W::DecResGn2Beta
    );

    // =====================================================
    // AA HEAD
    // =====================================================

    // Conv3x3 16→12 + bias + SiLU → buf0
    add_conv(
        &mut dispatches,
        2,
        0,
        16,
        12,
        3,
        1,
        half_w,
        half_h,
        W::HeadExpandConv,
        w(W::HeadExpandBias),
        ACT_SILU,
    );

    // Conv3x3 12→12 + bias + Tanh → buf1
    add_conv(
        &mut dispatches,
        0,
        1,
        12,
        12,
        3,
        1,
        half_w,
        half_h,
        W::HeadRefineConv,
        w(W::HeadRefineBias),
        ACT_TANH,
    );

    // PixelShuffle + add to input color → output texture
    {
        let mut c = AAConstants::default();
        c.pass_type = PassType::PixelShuffleOut as u32;
        c.in_buf = 1;
        c.width = render_w;
        c.height = render_h;
        c.in_width = half_w;
        c.in_height = half_h;
        c.buf_stride = buf_stride;
        dispatches.push(c);
    }

    (dispatches, prev_encoder_start, prev_encoder_end)
}
