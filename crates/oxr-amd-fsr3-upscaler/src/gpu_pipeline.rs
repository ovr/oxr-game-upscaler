use std::sync::OnceLock;
use tracing::{error, info};
use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::fsr3_types::FfxSurfaceFormat;

const VS_SOURCE: &[u8] = include_bytes!("alg-scale/blit_vs.hlsl");
const PS_SOURCE: &[u8] = include_bytes!("alg-scale/blit_ps.hlsl");
const LANCZOS_PS_SOURCE: &[u8] = include_bytes!("alg-scale/lanczos_ps.hlsl");

pub struct GpuState {
    pub device: ID3D12Device,
    pub root_signature: ID3D12RootSignature,
    pub pso_bilinear: ID3D12PipelineState,
    pub pso_lanczos: ID3D12PipelineState,
    pub srv_heap: ID3D12DescriptorHeap,
    pub rtv_heap: ID3D12DescriptorHeap,
}

// Stores Option<GpuState> so init failure doesn't poison the lock.
static GPU_STATE: OnceLock<Option<GpuState>> = OnceLock::new();

pub unsafe fn get_or_init(
    cmd_list: &ID3D12GraphicsCommandList,
    output_format: DXGI_FORMAT,
) -> Option<&'static GpuState> {
    GPU_STATE
        .get_or_init(|| match try_init(cmd_list, output_format) {
            Ok(state) => Some(state),
            Err(e) => {
                error!("gpu_pipeline: init failed: {}", e);
                None
            }
        })
        .as_ref()
}

unsafe fn try_init(
    cmd_list: &ID3D12GraphicsCommandList,
    output_format: DXGI_FORMAT,
) -> Result<GpuState, String> {
    info!("gpu_pipeline: initializing GPU state");

    let mut device: Option<ID3D12Device> = None;
    cmd_list
        .GetDevice(&mut device)
        .map_err(|e| format!("GetDevice failed: {}", e))?;
    let device = device.ok_or_else(|| "GetDevice returned null".to_string())?;

    let typed_format = dxgi_typeless_to_typed(output_format);
    info!(
        "gpu_pipeline: output DXGI format={:?} (0x{:04X}), typed={:?} (0x{:04X})",
        output_format, output_format.0, typed_format, typed_format.0
    );

    let vs_blob = compile_shader(VS_SOURCE, b"VS\0", b"vs_5_0\0")?;
    let ps_blob = compile_shader(PS_SOURCE, b"PS\0", b"ps_5_0\0")?;
    let lanczos_ps_blob = compile_shader(LANCZOS_PS_SOURCE, b"PS\0", b"ps_5_0\0")?;
    info!("gpu_pipeline: shaders compiled");

    let root_signature = create_root_signature(&device)?;
    info!("gpu_pipeline: root signature created");

    let pso_bilinear = create_pso(&device, &root_signature, &vs_blob, &ps_blob, typed_format)?;
    info!(
        "gpu_pipeline: PSO created (bilinear, format={:?})",
        typed_format
    );

    let pso_lanczos = create_pso(
        &device,
        &root_signature,
        &vs_blob,
        &lanczos_ps_blob,
        typed_format,
    )?;
    info!(
        "gpu_pipeline: PSO created (lanczos, format={:?})",
        typed_format
    );

    let srv_heap =
        create_descriptor_heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true)?;
    let rtv_heap = create_descriptor_heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, false)?;
    info!("gpu_pipeline: descriptor heaps created");

    Ok(GpuState {
        device,
        root_signature,
        pso_bilinear,
        pso_lanczos,
        srv_heap,
        rtv_heap,
    })
}

unsafe fn compile_shader(source: &[u8], entry: &[u8], target: &[u8]) -> Result<ID3DBlob, String> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;

    let hr = D3DCompile(
        source.as_ptr() as *const _,
        source.len(),
        None,
        None,
        None,
        PCSTR(entry.as_ptr()),
        PCSTR(target.as_ptr()),
        0,
        0,
        &mut code,
        Some(&mut errors),
    );

    if let Some(err_blob) = &errors {
        let err_ptr = err_blob.GetBufferPointer() as *const u8;
        let err_len = err_blob.GetBufferSize();
        let err_msg = std::str::from_utf8_unchecked(std::slice::from_raw_parts(err_ptr, err_len));
        let trimmed = err_msg.trim_end_matches('\0');
        if hr.is_err() {
            error!("Shader compile error: {}", trimmed);
            return Err(format!("D3DCompile failed: {}", trimmed));
        }
    }

    if let Err(e) = hr {
        return Err(format!("D3DCompile failed: {}", e));
    }

    code.ok_or_else(|| "D3DCompile produced no code".to_string())
}

unsafe fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let constants = D3D12_ROOT_CONSTANTS {
        ShaderRegister: 0,
        RegisterSpace: 0,
        Num32BitValues: 4,
    };

    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };

    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: constants,
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    let static_sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MipLODBias: 0.0,
        MaxAnisotropy: 0,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };

    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &static_sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };

    let mut blob: Option<ID3DBlob> = None;
    let mut error_blob: Option<ID3DBlob> = None;

    if let Err(e) = D3D12SerializeRootSignature(
        &desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &mut blob,
        Some(&mut error_blob),
    ) {
        if let Some(err_blob) = &error_blob {
            let err_ptr = err_blob.GetBufferPointer() as *const u8;
            let err_len = err_blob.GetBufferSize();
            let err_msg =
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(err_ptr, err_len));
            error!(
                "D3D12SerializeRootSignature error: {}",
                err_msg.trim_end_matches('\0')
            );
        }
        return Err(format!("D3D12SerializeRootSignature failed: {}", e));
    }

    let blob = blob.ok_or_else(|| "D3D12SerializeRootSignature produced no blob".to_string())?;

    device
        .CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
        )
        .map_err(|e| format!("CreateRootSignature failed: {}", e))
}

unsafe fn create_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs_blob: &ID3DBlob,
    ps_blob: &ID3DBlob,
    rt_format: DXGI_FORMAT,
) -> Result<ID3D12PipelineState, String> {
    let vs_bytecode = D3D12_SHADER_BYTECODE {
        pShaderBytecode: vs_blob.GetBufferPointer(),
        BytecodeLength: vs_blob.GetBufferSize(),
    };

    let ps_bytecode = D3D12_SHADER_BYTECODE {
        pShaderBytecode: ps_blob.GetBufferPointer(),
        BytecodeLength: ps_blob.GetBufferSize(),
    };

    let mut rt_formats = [DXGI_FORMAT_UNKNOWN; 8];
    rt_formats[0] = rt_format;

    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::transmute_copy(root_sig),
        VS: vs_bytecode,
        PS: ps_bytecode,
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [
                D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    LogicOpEnable: false.into(),
                    SrcBlend: D3D12_BLEND_ONE,
                    DestBlend: D3D12_BLEND_ZERO,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ONE,
                    DestBlendAlpha: D3D12_BLEND_ZERO,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
                    LogicOp: D3D12_LOGIC_OP_NOOP,
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                },
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            ],
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: false.into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: true.into(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            ..Default::default()
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: std::ptr::null(),
            NumElements: 0,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: rt_formats,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };

    device
        .CreateGraphicsPipelineState(&desc)
        .map_err(|e| format!("CreateGraphicsPipelineState failed: {}", e))
}

unsafe fn create_descriptor_heap(
    device: &ID3D12Device,
    heap_type: D3D12_DESCRIPTOR_HEAP_TYPE,
    num_descriptors: u32,
    shader_visible: bool,
) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: heap_type,
        NumDescriptors: num_descriptors,
        Flags: if shader_visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        },
        NodeMask: 0,
    };

    device
        .CreateDescriptorHeap(&desc)
        .map_err(|e| format!("CreateDescriptorHeap({:?}) failed: {}", heap_type, e))
}

/// Convert typeless DXGI formats to their most common typed equivalent.
/// D3D12 PSOs require typed formats in RTVFormats[].
fn dxgi_typeless_to_typed(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32G32B32A32_TYPELESS => DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R32G32_TYPELESS => DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R16G16_TYPELESS => DXGI_FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R32_TYPELESS => DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R16_TYPELESS => DXGI_FORMAT_R16_FLOAT,
        DXGI_FORMAT_R8_TYPELESS => DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8_TYPELESS => DXGI_FORMAT_R8G8_UNORM,
        other => other,
    }
}

pub fn ffx_format_to_dxgi(ffx_format: u32) -> DXGI_FORMAT {
    match ffx_format {
        x if x == FfxSurfaceFormat::R32G32B32A32Typeless as u32 => {
            DXGI_FORMAT_R32G32B32A32_TYPELESS
        }
        x if x == FfxSurfaceFormat::R32G32B32A32Uint as u32 => DXGI_FORMAT_R32G32B32A32_UINT,
        x if x == FfxSurfaceFormat::R32G32B32A32Float as u32 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        x if x == FfxSurfaceFormat::R16G16B16A16Float as u32 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        x if x == FfxSurfaceFormat::R32G32B32Float as u32 => DXGI_FORMAT_R32G32B32_FLOAT,
        x if x == FfxSurfaceFormat::R32G32Float as u32 => DXGI_FORMAT_R32G32_FLOAT,
        x if x == FfxSurfaceFormat::R8Uint as u32 => DXGI_FORMAT_R8_UINT,
        x if x == FfxSurfaceFormat::R32Uint as u32 => DXGI_FORMAT_R32_UINT,
        x if x == FfxSurfaceFormat::R8G8B8A8Typeless as u32 => DXGI_FORMAT_R8G8B8A8_TYPELESS,
        x if x == FfxSurfaceFormat::R8G8B8A8Unorm as u32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        x if x == FfxSurfaceFormat::R8G8B8A8Snorm as u32 => DXGI_FORMAT_R8G8B8A8_SNORM,
        x if x == FfxSurfaceFormat::R8G8B8A8Srgb as u32 => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        x if x == FfxSurfaceFormat::B8G8R8A8Typeless as u32 => DXGI_FORMAT_B8G8R8A8_TYPELESS,
        x if x == FfxSurfaceFormat::B8G8R8A8Unorm as u32 => DXGI_FORMAT_B8G8R8A8_UNORM,
        x if x == FfxSurfaceFormat::B8G8R8A8Srgb as u32 => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        x if x == FfxSurfaceFormat::R11G11B10Float as u32 => DXGI_FORMAT_R11G11B10_FLOAT,
        x if x == FfxSurfaceFormat::R10G10B10A2Unorm as u32 => DXGI_FORMAT_R10G10B10A2_UNORM,
        x if x == FfxSurfaceFormat::R16G16Float as u32 => DXGI_FORMAT_R16G16_FLOAT,
        x if x == FfxSurfaceFormat::R16G16Uint as u32 => DXGI_FORMAT_R16G16_UINT,
        x if x == FfxSurfaceFormat::R16G16Sint as u32 => DXGI_FORMAT_R16G16_SINT,
        x if x == FfxSurfaceFormat::R16Float as u32 => DXGI_FORMAT_R16_FLOAT,
        x if x == FfxSurfaceFormat::R16Uint as u32 => DXGI_FORMAT_R16_UINT,
        x if x == FfxSurfaceFormat::R16Unorm as u32 => DXGI_FORMAT_R16_UNORM,
        x if x == FfxSurfaceFormat::R16Snorm as u32 => DXGI_FORMAT_R16_SNORM,
        x if x == FfxSurfaceFormat::R8Unorm as u32 => DXGI_FORMAT_R8_UNORM,
        x if x == FfxSurfaceFormat::R8G8Unorm as u32 => DXGI_FORMAT_R8G8_UNORM,
        x if x == FfxSurfaceFormat::R8G8Uint as u32 => DXGI_FORMAT_R8G8_UINT,
        x if x == FfxSurfaceFormat::R32Float as u32 => DXGI_FORMAT_R32_FLOAT,
        x if x == FfxSurfaceFormat::R9G9B9E5Sharedexp as u32 => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        x if x == FfxSurfaceFormat::R16G16B16A16Typeless as u32 => {
            DXGI_FORMAT_R16G16B16A16_TYPELESS
        }
        x if x == FfxSurfaceFormat::R32G32Typeless as u32 => DXGI_FORMAT_R32G32_TYPELESS,
        x if x == FfxSurfaceFormat::R10G10B10A2Typeless as u32 => DXGI_FORMAT_R10G10B10A2_TYPELESS,
        x if x == FfxSurfaceFormat::R16G16Typeless as u32 => DXGI_FORMAT_R16G16_TYPELESS,
        x if x == FfxSurfaceFormat::R16Typeless as u32 => DXGI_FORMAT_R16_TYPELESS,
        x if x == FfxSurfaceFormat::R8Typeless as u32 => DXGI_FORMAT_R8_TYPELESS,
        x if x == FfxSurfaceFormat::R8G8Typeless as u32 => DXGI_FORMAT_R8G8_TYPELESS,
        x if x == FfxSurfaceFormat::R32Typeless as u32 => DXGI_FORMAT_R32_TYPELESS,
        _ => DXGI_FORMAT_UNKNOWN,
    }
}
