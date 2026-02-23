/// Minimal Dear ImGui DX12 renderer.
///
/// Follows the pattern of `imgui_impl_dx12.cpp` from the Dear ImGui repository.
/// Uses our existing `windows = "0.58"` COM types to avoid dependency conflicts.
use std::mem;
use tracing::{error, info};
use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

// --- HLSL shaders (inline) ---------------------------------------------------

const IMGUI_VS: &[u8] = b"
cbuffer vertexBuffer : register(b0)
{
    float4x4 ProjectionMatrix;
};
struct VS_INPUT { float2 pos : POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
struct PS_INPUT { float4 pos : SV_POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
PS_INPUT VS(VS_INPUT input)
{
    PS_INPUT output;
    output.pos = mul(ProjectionMatrix, float4(input.pos.xy, 0.f, 1.f));
    output.col = input.col;
    output.uv  = input.uv;
    return output;
}
\0";

const IMGUI_PS: &[u8] = b"
struct PS_INPUT { float4 pos : SV_POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
SamplerState sampler0 : register(s0);
Texture2D texture0    : register(t0);
float4 PS(PS_INPUT input) : SV_Target
{
    return input.col * texture0.Sample(sampler0, input.uv);
}
\0";

// --- Per-frame resources (UPLOAD heap VB/IB) ---------------------------------

struct FrameResources {
    vertex_buf: Option<ID3D12Resource>,
    index_buf: Option<ID3D12Resource>,
    vertex_cap: usize,
    index_cap: usize,
}

impl FrameResources {
    fn new() -> Self {
        Self {
            vertex_buf: None,
            index_buf: None,
            vertex_cap: 0,
            index_cap: 0,
        }
    }
}

// --- Renderer ----------------------------------------------------------------

pub struct ImguiDx12Renderer {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    frames: [FrameResources; 2],
    font_tex: Option<ID3D12Resource>,
    font_staging: Option<ID3D12Resource>, // kept alive until GPU executes the copy
    font_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    // We need the device to (re)create buffers each frame
    device: ID3D12Device,
}

impl ImguiDx12Renderer {
    /// Create the renderer.  `font_cpu_slot1` / `font_gpu_slot1` are the
    /// handles at slot 1 of the shared shader-visible SRV heap where the font
    /// atlas SRV will be written.
    pub unsafe fn new(device: &ID3D12Device, rt_format: DXGI_FORMAT) -> Result<Self, String> {
        let root_sig = create_imgui_root_signature(device)?;
        let pso = create_imgui_pso(device, &root_sig, rt_format)?;
        info!("imgui_renderer: PSO created");

        Ok(Self {
            root_sig,
            pso,
            frames: [FrameResources::new(), FrameResources::new()],
            font_tex: None,
            font_staging: None,
            font_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 },
            device: device.clone(),
        })
    }

    /// Upload the font atlas on first call (lazy init).
    pub unsafe fn ensure_fonts(
        &mut self,
        ctx: &mut imgui::Context,
        cmd_list: &ID3D12GraphicsCommandList,
        font_cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
        font_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<(), String> {
        if self.font_tex.is_some() {
            return Ok(());
        }

        let fonts = ctx.fonts();
        let atlas = fonts.build_rgba32_texture();
        let width = atlas.width;
        let height = atlas.height;
        let data = atlas.data; // &[u8], RGBA8

        let row_pitch = (width as usize * 4 + 255) & !255; // 256-byte aligned
        let upload_size = row_pitch * height as usize;

        // Default-heap texture
        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let heap_props_default = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut font_tex: Option<ID3D12Resource> = None;
        self.device
            .CreateCommittedResource(
                &heap_props_default,
                D3D12_HEAP_FLAG_NONE,
                &tex_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut font_tex,
            )
            .map_err(|e| format!("font tex CreateCommittedResource: {e}"))?;
        let font_tex = font_tex.unwrap();

        // Upload-heap staging buffer
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: upload_size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let heap_props_upload = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let mut staging: Option<ID3D12Resource> = None;
        self.device
            .CreateCommittedResource(
                &heap_props_upload,
                D3D12_HEAP_FLAG_NONE,
                &buf_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut staging,
            )
            .map_err(|e| format!("font staging CreateCommittedResource: {e}"))?;
        let staging = staging.unwrap();

        // Map and copy pixel rows (source pitch = width*4, dst pitch = row_pitch)
        let mut mapped: *mut u8 = std::ptr::null_mut();
        staging
            .Map(0, None, Some(&mut mapped as *mut *mut u8 as *mut *mut _))
            .map_err(|e| format!("staging Map: {e}"))?;
        for row in 0..height as usize {
            let src = data.as_ptr().add(row * width as usize * 4);
            let dst = mapped.add(row * row_pitch);
            std::ptr::copy_nonoverlapping(src, dst, width as usize * 4);
        }
        staging.Unmap(0, None);

        // CopyTextureRegion: staging buffer → font texture
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::transmute_copy(&font_tex),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::transmute_copy(&staging),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: row_pitch as u32,
                    },
                },
            },
        };
        cmd_list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);

        // Barrier: COPY_DEST → PIXEL_SHADER_RESOURCE
        let barrier = transition_barrier(
            &font_tex,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        cmd_list.ResourceBarrier(&[barrier]);

        // Create SRV at slot 1
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        self.device
            .CreateShaderResourceView(&font_tex, Some(&srv_desc), font_cpu_handle);

        self.font_tex = Some(font_tex);
        self.font_staging = Some(staging); // keep alive until GPU executes the copy
        self.font_gpu_handle = font_gpu_handle;

        // Store the font GPU handle as the imgui font texture ID
        let tex_id = imgui::TextureId::new(font_gpu_handle.ptr as usize);
        ctx.fonts().tex_id = tex_id;

        info!("imgui_renderer: font atlas uploaded ({}x{})", width, height);
        Ok(())
    }

    pub unsafe fn render(
        &mut self,
        draw_data: &imgui::DrawData,
        cmd_list: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        srv_heap: &ID3D12DescriptorHeap,
        rtv_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) -> Result<(), String> {
        let display_w = draw_data.display_size[0];
        let display_h = draw_data.display_size[1];
        if display_w <= 0.0 || display_h <= 0.0 {
            return Ok(());
        }

        let total_vtx = draw_data.total_vtx_count as usize;
        let total_idx = draw_data.total_idx_count as usize;
        if total_vtx == 0 || total_idx == 0 {
            return Ok(());
        }

        let frame = &mut self.frames[frame_idx % 2];

        // Ensure vertex buffer capacity
        if frame.vertex_cap < total_vtx {
            frame.vertex_buf = None;
            let new_cap = (total_vtx + 5000).next_power_of_two();
            frame.vertex_buf = Some(create_upload_buffer(
                &self.device,
                new_cap * mem::size_of::<imgui::DrawVert>(),
            )?);
            frame.vertex_cap = new_cap;
        }

        // Ensure index buffer capacity
        if frame.index_cap < total_idx {
            frame.index_buf = None;
            let new_cap = (total_idx + 10000).next_power_of_two();
            frame.index_buf = Some(create_upload_buffer(
                &self.device,
                new_cap * mem::size_of::<imgui::DrawIdx>(),
            )?);
            frame.index_cap = new_cap;
        }

        // Upload vertices + indices
        let vb = frame.vertex_buf.as_ref().unwrap();
        let ib = frame.index_buf.as_ref().unwrap();

        let vtx_stride = mem::size_of::<imgui::DrawVert>();
        let idx_stride = mem::size_of::<imgui::DrawIdx>();

        let mut vtx_ptr: *mut imgui::DrawVert = std::ptr::null_mut();
        let mut idx_ptr: *mut imgui::DrawIdx = std::ptr::null_mut();

        vb.Map(
            0,
            None,
            Some(&mut vtx_ptr as *mut *mut imgui::DrawVert as *mut *mut _),
        )
        .map_err(|e| format!("vb Map: {e}"))?;
        ib.Map(
            0,
            None,
            Some(&mut idx_ptr as *mut *mut imgui::DrawIdx as *mut *mut _),
        )
        .map_err(|e| format!("ib Map: {e}"))?;

        for draw_list in draw_data.draw_lists() {
            let verts = draw_list.vtx_buffer();
            std::ptr::copy_nonoverlapping(verts.as_ptr(), vtx_ptr, verts.len());
            vtx_ptr = vtx_ptr.add(verts.len());

            let indices = draw_list.idx_buffer();
            std::ptr::copy_nonoverlapping(indices.as_ptr(), idx_ptr, indices.len());
            idx_ptr = idx_ptr.add(indices.len());
        }

        vb.Unmap(0, None);
        ib.Unmap(0, None);

        // Build orthographic projection matrix
        let l = draw_data.display_pos[0];
        let r = draw_data.display_pos[0] + draw_data.display_size[0];
        let t = draw_data.display_pos[1];
        let b = draw_data.display_pos[1] + draw_data.display_size[1];
        // Column-major 4x4
        #[rustfmt::skip]
        let mvp: [f32; 16] = [
            2.0/(r-l),    0.0,          0.0,  0.0,
            0.0,          2.0/(t-b),    0.0,  0.0,
            0.0,          0.0,          0.5,  0.0,
            (r+l)/(l-r),  (t+b)/(b-t),  0.5,  1.0,
        ];

        // Set pipeline
        cmd_list.SetGraphicsRootSignature(&self.root_sig);
        cmd_list.SetPipelineState(&self.pso);
        cmd_list.SetDescriptorHeaps(&[Some(srv_heap.clone())]);

        // Root constants = MVP (16 floats)
        for (i, &v) in mvp.iter().enumerate() {
            cmd_list.SetGraphicsRoot32BitConstant(0, v.to_bits(), i as u32);
        }

        // Render target (must re-bind after root signature change)
        cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

        // Viewport
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: display_w,
            Height: display_h,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        cmd_list.RSSetViewports(&[viewport]);

        // IA
        let vb_desc = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vb.GetGPUVirtualAddress(),
            SizeInBytes: (frame.vertex_cap * vtx_stride) as u32,
            StrideInBytes: vtx_stride as u32,
        };
        let ib_desc = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib.GetGPUVirtualAddress(),
            SizeInBytes: (frame.index_cap * idx_stride) as u32,
            Format: if idx_stride == 2 {
                DXGI_FORMAT_R16_UINT
            } else {
                DXGI_FORMAT_R32_UINT
            },
        };
        cmd_list.IASetVertexBuffers(0, Some(&[vb_desc]));
        cmd_list.IASetIndexBuffer(Some(&ib_desc));
        cmd_list.IASetPrimitiveTopology(
            windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        );

        // Draw
        let clip_off = draw_data.display_pos;
        let mut global_vtx_offset = 0i32;
        let mut global_idx_offset = 0u32;

        for draw_list in draw_data.draw_lists() {
            for cmd in draw_list.commands() {
                match cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        let clip_rect = windows::Win32::Foundation::RECT {
                            left: (cmd_params.clip_rect[0] - clip_off[0]) as i32,
                            top: (cmd_params.clip_rect[1] - clip_off[1]) as i32,
                            right: (cmd_params.clip_rect[2] - clip_off[0]) as i32,
                            bottom: (cmd_params.clip_rect[3] - clip_off[1]) as i32,
                        };
                        cmd_list.RSSetScissorRects(&[clip_rect]);

                        // The texture ID holds the GPU descriptor handle ptr
                        let tex_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: cmd_params.texture_id.id() as u64,
                        };
                        cmd_list.SetGraphicsRootDescriptorTable(1, tex_gpu);

                        cmd_list.DrawIndexedInstanced(
                            count as u32,
                            1,
                            global_idx_offset + cmd_params.idx_offset as u32,
                            global_vtx_offset + cmd_params.vtx_offset as i32,
                            0,
                        );
                    }
                    imgui::DrawCmd::ResetRenderState => {}
                    imgui::DrawCmd::RawCallback { .. } => {}
                }
            }
            global_vtx_offset += draw_list.vtx_buffer().len() as i32;
            global_idx_offset += draw_list.idx_buffer().len() as u32;
        }

        Ok(())
    }
}

// --- Helpers -----------------------------------------------------------------

unsafe fn create_upload_buffer(
    device: &ID3D12Device,
    size: usize,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut res: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut res,
        )
        .map_err(|e| format!("create_upload_buffer: {e}"))?;
    Ok(res.unwrap())
}

fn transition_barrier(
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
        pResource: unsafe { mem::transmute_copy(resource) },
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: before,
        StateAfter: after,
    });
    barrier
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
    if let Some(err) = &errors {
        let ptr = err.GetBufferPointer() as *const u8;
        let len = err.GetBufferSize();
        let msg = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
        if hr.is_err() {
            error!("imgui shader compile error: {}", msg.trim_end_matches('\0'));
            return Err(format!("D3DCompile failed: {}", msg.trim_end_matches('\0')));
        }
    }
    hr.map_err(|e| format!("D3DCompile: {e}"))?;
    Ok(code.unwrap())
}

unsafe fn create_imgui_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    // Slot 0: 16 root constants (4x4 MVP)
    let constants = D3D12_ROOT_CONSTANTS {
        ShaderRegister: 0,
        RegisterSpace: 0,
        Num32BitValues: 16,
    };

    // Slot 1: SRV descriptor table (1 descriptor for font/texture)
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: constants,
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
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
    D3D12SerializeRootSignature(
        &desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &mut blob,
        Some(&mut error_blob),
    )
    .map_err(|e| format!("imgui SerializeRootSignature: {e}"))?;

    let blob = blob.unwrap();
    device
        .CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
        )
        .map_err(|e| format!("imgui CreateRootSignature: {e}"))
}

unsafe fn create_imgui_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    rt_format: DXGI_FORMAT,
) -> Result<ID3D12PipelineState, String> {
    let vs_blob = compile_shader(IMGUI_VS, b"VS\0", b"vs_5_0\0")?;
    let ps_blob = compile_shader(IMGUI_PS, b"PS\0", b"ps_5_0\0")?;

    let vs_bytecode = D3D12_SHADER_BYTECODE {
        pShaderBytecode: vs_blob.GetBufferPointer(),
        BytecodeLength: vs_blob.GetBufferSize(),
    };
    let ps_bytecode = D3D12_SHADER_BYTECODE {
        pShaderBytecode: ps_blob.GetBufferPointer(),
        BytecodeLength: ps_blob.GetBufferSize(),
    };

    // Input layout matching imgui::DrawVert
    let position_name = b"POSITION\0";
    let texcoord_name = b"TEXCOORD\0";
    let color_name = b"COLOR\0";
    let input_elements = [
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(position_name.as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(texcoord_name.as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 8,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(color_name.as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            InputSlot: 0,
            AlignedByteOffset: 16,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ];

    let mut rt_formats = [DXGI_FORMAT_UNKNOWN; 8];
    rt_formats[0] = rt_format;

    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: mem::transmute_copy(root_sig),
        VS: vs_bytecode,
        PS: ps_bytecode,
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [
                D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: true.into(),
                    LogicOpEnable: false.into(),
                    SrcBlend: D3D12_BLEND_SRC_ALPHA,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ONE,
                    DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
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
            StencilEnable: false.into(),
            ..Default::default()
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_elements.as_ptr(),
            NumElements: input_elements.len() as u32,
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
        .map_err(|e| format!("imgui CreateGraphicsPipelineState: {e}"))
}
