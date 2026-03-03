use std::sync::Mutex;

use crate::fsr3_types::*;
use crate::gpu_pipeline::{self, GpuState};
use crate::overlay;
use crate::sgsr2_state;
use crate::upscaler_type;
use tracing::{error, info, warn};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

struct RcasResources {
    temp_rt: ID3D12Resource,
    width: u32,
    height: u32,
}

static RCAS_TEMP: Mutex<Option<RcasResources>> = Mutex::new(None);

/// Ensure the RCAS temp render target exists at the given dimensions and format.
/// Returns a clone of the ID3D12Resource.
unsafe fn ensure_rcas_temp(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource, String> {
    let mut guard = RCAS_TEMP.lock().map_err(|_| "RCAS_TEMP mutex poisoned")?;

    if let Some(res) = guard.as_ref() {
        if res.width == w && res.height == h {
            return Ok(res.temp_rt.clone());
        }
    }

    // Create or recreate
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: w as u64,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    };

    let mut resource: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            None,
            &mut resource,
        )
        .map_err(|e| format!("CreateCommittedResource for RCAS temp RT failed: {}", e))?;

    let resource = resource.ok_or("CreateCommittedResource returned null")?;
    info!("RCAS temp RT created: {}x{} format={:?}", w, h, format);

    *guard = Some(RcasResources {
        temp_rt: resource.clone(),
        width: w,
        height: h,
    });

    Ok(resource)
}

/// Create an SRV with an explicit typed format descriptor.
/// Converts FFX format → DXGI, then typeless → typed, so resources like R32_TYPELESS
/// (depth buffers) get a valid SRV format instead of relying on D3D12 auto-inference.
/// Returns `false` if the format is UNKNOWN and the SRV was not created.
unsafe fn create_typed_srv(
    gpu: &GpuState,
    resource: &ID3D12Resource,
    ffx_format: u32,
    slot: u32,
) -> bool {
    // Use actual D3D12 resource format (always correct family), fall back to FFX format
    let res_format = resource.GetDesc().Format;
    let dxgi_format = if res_format != DXGI_FORMAT_UNKNOWN {
        res_format
    } else {
        gpu_pipeline::ffx_format_to_dxgi(ffx_format)
    };
    let typed_format = gpu_pipeline::dxgi_typeless_to_typed(dxgi_format);
    let typed_format = if typed_format != dxgi_format {
        // Resource was typeless/depth → cross-format views are allowed → make filterable
        gpu_pipeline::dxgi_to_filterable(typed_format)
    } else {
        // Resource is already typed → must use exact format for the SRV
        // If non-filterable (UINT/SINT), skip — linear sampling on integer formats causes TDR
        if gpu_pipeline::dxgi_to_filterable(typed_format) != typed_format {
            return false;
        }
        typed_format
    };

    // If still unknown, log and bail — don't create an invalid SRV
    if typed_format == DXGI_FORMAT_UNKNOWN {
        warn!(
            "create_typed_srv: slot {} has UNKNOWN format after all fallbacks, skipping",
            slot
        );
        return false;
    }

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: typed_format,
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

    gpu.device.CreateShaderResourceView(
        resource,
        Some(&srv_desc),
        gpu_pipeline::get_srv_cpu_handle(gpu, slot),
    );
    true
}

/// Bilinear-upscale dispatch: draw a fullscreen triangle sampling the color texture
/// with bilinear filtering, writing to the output as a render target.
pub unsafe fn dispatch_upscale(d: &FfxFsr3UpscalerDispatchDescription) -> u32 {
    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        warn!("dispatch_upscale: null command list");
        return 1;
    }

    let color_raw = d.color.resource;
    let output_raw = d.output.resource;
    if color_raw.is_null() || output_raw.is_null() {
        warn!("dispatch_upscale: null color or output resource");
        return 1;
    }

    // Cast raw void* pointers to windows-rs COM interfaces (borrowed, no AddRef).
    let cmd_list: ID3D12GraphicsCommandList =
        match <ID3D12GraphicsCommandList as windows::core::Interface>::from_raw_borrowed(
            &cmd_list_raw,
        ) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_upscale: from_raw_borrowed failed for command list");
                return 1;
            }
        };

    let color_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&color_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_upscale: from_raw_borrowed failed for color resource");
                return 1;
            }
        };

    let output_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&output_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_upscale: from_raw_borrowed failed for output resource");
                return 1;
            }
        };

    // Determine output format and initialize GPU pipeline (extracts device on first call).
    let output_format = gpu_pipeline::ffx_format_to_dxgi(d.output.description.format);
    let gpu = match gpu_pipeline::get_or_init(&cmd_list, output_format) {
        Some(g) => g,
        None => {
            error!("dispatch_upscale: GPU pipeline init failed, skipping blit");
            return 1;
        }
    };

    // Render dimensions.
    let render_w = if d.render_size.width > 0 {
        d.render_size.width
    } else {
        d.color.description.width
    };
    let render_h = if d.render_size.height > 0 {
        d.render_size.height
    } else {
        d.color.description.height
    };

    let color_tex_w = d.color.description.width;
    let color_tex_h = d.color.description.height;
    let output_w = d.output.description.width;
    let output_h = d.output.description.height;

    if color_tex_w == 0 || color_tex_h == 0 || output_w == 0 || output_h == 0 {
        error!(
            "dispatch_upscale: zero dimensions: color={}x{}, output={}x{}",
            color_tex_w, color_tex_h, output_w, output_h
        );
        return 1;
    }

    // UV scale: render region may be smaller than the color texture.
    let uv_scale_x = render_w as f32 / color_tex_w as f32;
    let uv_scale_y = render_h as f32 / color_tex_h as f32;

    // --- SGSRv2 temporal upscaler: separate multi-pass path ---
    let current_upscaler = upscaler_type::get();
    if current_upscaler == upscaler_type::UpscalerType::SGSRv2 {
        return dispatch_sgsr2(
            &cmd_list,
            gpu,
            d,
            &color_res,
            &output_res,
            render_w,
            render_h,
            output_w,
            output_h,
        );
    }

    // --- Barriers: color → PIXEL_SHADER_RESOURCE, output → RENDER_TARGET ---
    let barriers_before = [
        resource_barrier_transition(
            &color_res,
            d.color.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ),
        resource_barrier_transition(
            &output_res,
            d.output.state,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ),
    ];
    let valid_before: Vec<_> = barriers_before.iter().flatten().cloned().collect();
    if !valid_before.is_empty() {
        cmd_list.ResourceBarrier(&valid_before);
    }

    // --- Create SRV for color texture (explicit typed format) ---
    create_typed_srv(gpu, &color_res, d.color.description.format, 0);

    // --- Create RTV for output texture (slot 0) ---
    gpu.device
        .CreateRenderTargetView(&output_res, None, gpu_pipeline::get_rtv_cpu_handle(gpu, 0));

    // --- Determine if RCAS post-pass is needed ---
    let use_rcas = upscaler_type::rcas_get()
        && matches!(
            current_upscaler,
            upscaler_type::UpscalerType::Bilinear | upscaler_type::UpscalerType::Lanczos
        );

    // --- Set pipeline state ---
    let pso = match current_upscaler {
        upscaler_type::UpscalerType::Bilinear => &gpu.pso_bilinear,
        upscaler_type::UpscalerType::Lanczos => &gpu.pso_lanczos,
        upscaler_type::UpscalerType::SGSR => &gpu.pso_sgsr,
        upscaler_type::UpscalerType::SGSRv2 => unreachable!("SGSRv2 handled above"),
    };

    // --- Viewport + scissor (shared by both passes) ---
    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: output_w as f32,
        Height: output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: output_w as i32,
        bottom: output_h as i32,
    };

    // Try to get temp RT for RCAS; fall back to single pass on failure
    let rcas_temp = if use_rcas {
        match ensure_rcas_temp(&gpu.device, output_w, output_h, gpu.rt_format) {
            Ok(rt) => Some(rt),
            Err(e) => {
                error!("dispatch_upscale: RCAS temp RT creation failed: {}, falling back to single pass", e);
                None
            }
        }
    } else {
        None
    };

    if let Some(temp_rt) = rcas_temp {
        // --- Pass 1: upscale color → temp_rt ---
        // Barrier: temp_rt PIXEL_SHADER_RESOURCE → RENDER_TARGET
        let barrier_temp_to_rt = resource_barrier_transition_d3d12(
            &temp_rt,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        if let Some(b) = &barrier_temp_to_rt {
            cmd_list.ResourceBarrier(&[b.clone()]);
        }

        // Create RTV for temp_rt at slot 1
        gpu.device
            .CreateRenderTargetView(&temp_rt, None, gpu_pipeline::get_rtv_cpu_handle(gpu, 1));

        cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
        cmd_list.SetPipelineState(pso);
        cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

        // Root constants: uvScale + inputSize
        cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_x.to_bits(), 0);
        cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_y.to_bits(), 1);
        cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_w as f32).to_bits(), 2);
        cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_h as f32).to_bits(), 3);

        // SRV: color texture at slot 0
        cmd_list
            .SetGraphicsRootDescriptorTable(1, gpu.srv_heap.GetGPUDescriptorHandleForHeapStart());

        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);

        let rtv_temp = gpu_pipeline::get_rtv_cpu_handle(gpu, 1);
        cmd_list.OMSetRenderTargets(1, Some(&rtv_temp), false, None);

        cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd_list.DrawInstanced(3, 1, 0, 0);

        // --- Barrier: temp_rt RENDER_TARGET → PIXEL_SHADER_RESOURCE ---
        let barrier_temp_to_srv = resource_barrier_transition_d3d12(
            &temp_rt,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        if let Some(b) = &barrier_temp_to_srv {
            cmd_list.ResourceBarrier(&[b.clone()]);
        }

        // --- Pass 2: RCAS temp_rt → output ---
        // Create SRV for temp_rt at slot 9
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: gpu.rt_format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        gpu.device.CreateShaderResourceView(
            &temp_rt,
            Some(&srv_desc),
            gpu_pipeline::get_srv_cpu_handle(gpu, 9),
        );

        // Create RTV for output at slot 0
        gpu.device.CreateRenderTargetView(
            &output_res,
            None,
            gpu_pipeline::get_rtv_cpu_handle(gpu, 0),
        );

        cmd_list.SetPipelineState(&gpu.pso_rcas);

        // Root constant: sharpness = exp2(-0.0) = 1.0 (maximum sharpness)
        let sharpness: f32 = 1.0;
        cmd_list.SetGraphicsRoot32BitConstant(0, sharpness.to_bits(), 0);

        // Bind temp_rt SRV at slot 9
        cmd_list.SetGraphicsRootDescriptorTable(1, gpu_pipeline::get_srv_gpu_handle(gpu, 9));

        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);

        let rtv_output = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
        cmd_list.OMSetRenderTargets(1, Some(&rtv_output), false, None);

        cmd_list.DrawInstanced(3, 1, 0, 0);
        gpu_pipeline::log_device_removed_reason(&gpu.device);

        info!(
            render = format_args!("{}x{}", render_w, render_h),
            output = format_args!("{}x{}", output_w, output_h),
            uv_scale = format_args!("({:.3}, {:.3})", uv_scale_x, uv_scale_y),
            "DrawInstanced upscale+RCAS (2-pass)"
        );
    } else {
        // === Single pass: upscale → output ===
        cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
        cmd_list.SetPipelineState(pso);
        cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

        // Root constants: uvScale (slots 0–1) + inputSize (slots 2–3)
        cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_x.to_bits(), 0);
        cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_y.to_bits(), 1);
        cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_w as f32).to_bits(), 2);
        cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_h as f32).to_bits(), 3);

        // Descriptor table: SRV
        cmd_list
            .SetGraphicsRootDescriptorTable(1, gpu.srv_heap.GetGPUDescriptorHandleForHeapStart());

        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);

        // --- Set render target ---
        let rtv_handle = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
        cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

        // --- Draw fullscreen triangle ---
        cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd_list.DrawInstanced(3, 1, 0, 0);
        gpu_pipeline::log_device_removed_reason(&gpu.device);

        info!(
            render = format_args!("{}x{}", render_w, render_h),
            output = format_args!("{}x{}", output_w, output_h),
            uv_scale = format_args!("({:.3}, {:.3})", uv_scale_x, uv_scale_y),
            "DrawInstanced upscale blit"
        );
    }

    // --- Debug view overlay (draws debug tiles on top of the upscaled image) ---
    if upscaler_type::debug_view_get() {
        draw_debug_tiles(&cmd_list, gpu, d, output_w, output_h);
    }

    // --- Render imgui overlay (RTV still bound, output still in RENDER_TARGET state) ---
    // Reset viewport/scissor to full output in case debug tiles changed them.
    let full_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: output_w as f32,
        Height: output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let full_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: output_w as i32,
        bottom: output_h as i32,
    };
    cmd_list.RSSetViewports(&[full_viewport]);
    cmd_list.RSSetScissorRects(&[full_scissor]);
    overlay::render_frame(&cmd_list, gpu, output_w, output_h);

    // --- Barriers: restore original states ---
    let barriers_after = [
        resource_barrier_transition_d3d12(
            &color_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.color.state),
        ),
        resource_barrier_transition_d3d12(
            &output_res,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            ffx_state_to_d3d12(d.output.state),
        ),
    ];
    let valid_after: Vec<_> = barriers_after.iter().flatten().cloned().collect();
    if !valid_after.is_empty() {
        cmd_list.ResourceBarrier(&valid_after);
    }

    0 // FFX_OK
}

/// SGSRv2 temporal upscaler: 2-pass dispatch (convert + upscale) with history ping-pong.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_sgsr2(
    cmd_list: &ID3D12GraphicsCommandList,
    gpu: &GpuState,
    d: &FfxFsr3UpscalerDispatchDescription,
    color_res: &ID3D12Resource,
    output_res: &ID3D12Resource,
    render_w: u32,
    render_h: u32,
    output_w: u32,
    output_h: u32,
) -> u32 {
    // Get/create persistent SGSRv2 textures
    let mut state_guard = match sgsr2_state::get_or_create(
        &gpu.device,
        render_w,
        render_h,
        output_w,
        output_h,
        gpu.rt_format,
    ) {
        Ok(g) => g,
        Err(e) => {
            error!("dispatch_sgsr2: state creation failed: {}", e);
            return 1;
        }
    };
    let state = state_guard.as_mut().unwrap();

    let is_reset = d.reset || !state.initialized;

    // Borrow depth and motion_vectors resources
    let depth_res = match borrow_resource(d.depth.resource) {
        Some(r) => r,
        None => {
            error!("dispatch_sgsr2: null depth resource");
            return 1;
        }
    };
    let mv_res = match borrow_resource(d.motion_vectors.resource) {
        Some(r) => r,
        None => {
            error!("dispatch_sgsr2: null motion_vectors resource");
            return 1;
        }
    };

    // === Pass 1: Convert (render resolution) ===
    // Barriers: depth → SRV, motion_vectors → SRV, color → SRV, motion_depth_clip → RT
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition(
            &depth_res,
            d.depth.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition(
            &mv_res,
            d.motion_vectors.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_clip,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // Create SRVs for convert pass: depth (slot 10), velocity (slot 11)
    create_typed_srv(gpu, &depth_res, d.depth.description.format, 10);
    create_typed_srv(gpu, &mv_res, d.motion_vectors.description.format, 11);

    // Create RTV for motion_depth_clip (slot 2)
    gpu.device.CreateRenderTargetView(
        &state.motion_depth_clip,
        None,
        gpu_pipeline::get_rtv_cpu_handle(gpu, 2),
    );

    // Build root constants (32 DWORDs)
    let scale_ratio_x = output_w as f32 / render_w as f32;
    let scale_ratio_y = output_h as f32 / render_h as f32;
    let render_size_rcp_x = 1.0 / render_w as f32;
    let render_size_rcp_y = 1.0 / render_h as f32;
    let output_size_rcp_x = 1.0 / output_w as f32;
    let output_size_rcp_y = 1.0 / output_h as f32;

    // clipToPrevClip = identity (4x4 column-major, stored as 4 vec4s)
    // For identity: col0=(1,0,0,0), col1=(0,1,0,0), col2=(0,0,1,0), col3=(0,0,0,1)
    let root_constants: [u32; 32] = [
        // clipToPrevClip[0] = col0
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        // clipToPrevClip[1] = col1
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        // clipToPrevClip[2] = col2
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        // clipToPrevClip[3] = col3
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        // renderSize
        (render_w as f32).to_bits(),
        (render_h as f32).to_bits(),
        // outputSize
        (output_w as f32).to_bits(),
        (output_h as f32).to_bits(),
        // renderSizeRcp
        render_size_rcp_x.to_bits(),
        render_size_rcp_y.to_bits(),
        // outputSizeRcp
        output_size_rcp_x.to_bits(),
        output_size_rcp_y.to_bits(),
        // jitterOffset
        d.jitter_offset.x.to_bits(),
        d.jitter_offset.y.to_bits(),
        // scaleRatio
        scale_ratio_x.to_bits(),
        scale_ratio_y.to_bits(),
        // cameraFovAngleHor (approximate from vertical FOV)
        (d.camera_fov_angle_vertical * 16.0 / 9.0).to_bits(),
        // minLerpContribution
        0.25_f32.to_bits(),
        // reset
        (if is_reset { 1.0_f32 } else { 0.0_f32 }).to_bits(),
        // bSameCamera
        0u32,
    ];

    // Set pipeline for convert pass
    cmd_list.SetGraphicsRootSignature(&gpu.sgsr2_root_signature);
    cmd_list.SetPipelineState(&gpu.pso_sgsr2_convert);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

    // Upload root constants
    cmd_list.SetGraphicsRoot32BitConstants(
        0,
        root_constants.len() as u32,
        root_constants.as_ptr() as *const core::ffi::c_void,
        0,
    );

    // SRV table: slots 10-11 (depth, velocity) — offset from heap start
    cmd_list.SetGraphicsRootDescriptorTable(1, gpu_pipeline::get_srv_gpu_handle(gpu, 10));

    // Viewport = render resolution (convert pass operates at render res)
    let convert_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: render_w as f32,
        Height: render_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let convert_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: render_w as i32,
        bottom: render_h as i32,
    };
    cmd_list.RSSetViewports(&[convert_viewport]);
    cmd_list.RSSetScissorRects(&[convert_scissor]);

    let rtv_mdc = gpu_pipeline::get_rtv_cpu_handle(gpu, 2);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_mdc), false, None);

    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd_list.DrawInstanced(3, 1, 0, 0);

    // === Barrier: motion_depth_clip RT → SRV, color → SRV ===
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.motion_depth_clip,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) = resource_barrier_transition(
            color_res,
            d.color.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        let prev_idx = state.frame_idx as usize;
        let curr_idx = 1 - prev_idx;
        // On reset: transition both history buffers to RT for clearing
        if is_reset {
            if let Some(b) = resource_barrier_transition_d3d12(
                &state.history[prev_idx],
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ) {
                barriers.push(b);
            }
        }
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    let prev_idx = state.frame_idx as usize;
    let curr_idx = 1 - prev_idx;

    // Clear history buffers on reset so uninitialized GPU memory can't leak in
    if is_reset {
        let black = [0.0_f32, 0.0, 0.0, 0.0];
        for idx in [prev_idx, curr_idx] {
            gpu.device.CreateRenderTargetView(
                &state.history[idx],
                None,
                gpu_pipeline::get_rtv_cpu_handle(gpu, 3),
            );
            cmd_list.ClearRenderTargetView(gpu_pipeline::get_rtv_cpu_handle(gpu, 3), &black, None);
        }
        // Transition prev history back to SRV (curr stays as RT for upscale pass)
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[prev_idx],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // === Pass 2: Upscale (display resolution) ===

    // Create SRVs: PrevHistory (slot 12), MotionDepthClip (slot 13), InputColor (slot 14)
    {
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: gpu.rt_format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        gpu.device.CreateShaderResourceView(
            &state.history[prev_idx],
            Some(&srv_desc),
            gpu_pipeline::get_srv_cpu_handle(gpu, 12),
        );

        let mdc_srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        gpu.device.CreateShaderResourceView(
            &state.motion_depth_clip,
            Some(&mdc_srv_desc),
            gpu_pipeline::get_srv_cpu_handle(gpu, 13),
        );
    }
    create_typed_srv(gpu, color_res, d.color.description.format, 14);

    // Create RTV for history write (slot 3)
    gpu.device.CreateRenderTargetView(
        &state.history[curr_idx],
        None,
        gpu_pipeline::get_rtv_cpu_handle(gpu, 3),
    );

    // Set pipeline for upscale pass
    cmd_list.SetPipelineState(&gpu.pso_sgsr2_upscale);

    // Root constants are already set (same cbuffer layout)
    // SRV table: slots 12-14 (prev_history, motion_depth_clip, color)
    cmd_list.SetGraphicsRootDescriptorTable(1, gpu_pipeline::get_srv_gpu_handle(gpu, 12));

    // Viewport = display resolution
    let upscale_viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: output_w as f32,
        Height: output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let upscale_scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: output_w as i32,
        bottom: output_h as i32,
    };
    cmd_list.RSSetViewports(&[upscale_viewport]);
    cmd_list.RSSetScissorRects(&[upscale_scissor]);

    let rtv_history = gpu_pipeline::get_rtv_cpu_handle(gpu, 3);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_history), false, None);

    cmd_list.DrawInstanced(3, 1, 0, 0);

    // === Copy history[curr] → output ===
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        ) {
            barriers.push(b);
        }
        if let Some(b) =
            resource_barrier_transition(output_res, d.output.state, D3D12_RESOURCE_STATE_COPY_DEST)
        {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    cmd_list.CopyResource(output_res, &state.history[curr_idx]);

    // === Render overlay on output ===
    {
        let mut barriers = Vec::new();
        if let Some(b) = resource_barrier_transition_d3d12(
            output_res,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    // Create RTV for output (slot 0) and render overlay
    gpu.device
        .CreateRenderTargetView(output_res, None, gpu_pipeline::get_rtv_cpu_handle(gpu, 0));
    cmd_list.RSSetViewports(&[upscale_viewport]);
    cmd_list.RSSetScissorRects(&[upscale_scissor]);
    let rtv_output = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_output), false, None);
    overlay::render_frame(cmd_list, gpu, output_w, output_h);

    // === Restore barriers ===
    {
        let mut barriers = Vec::new();
        // Restore depth
        if let Some(b) = resource_barrier_transition_d3d12(
            &depth_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.depth.state),
        ) {
            barriers.push(b);
        }
        // Restore motion vectors
        if let Some(b) = resource_barrier_transition_d3d12(
            &mv_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.motion_vectors.state),
        ) {
            barriers.push(b);
        }
        // Restore color
        if let Some(b) = resource_barrier_transition_d3d12(
            color_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.color.state),
        ) {
            barriers.push(b);
        }
        // Restore output
        if let Some(b) = resource_barrier_transition_d3d12(
            output_res,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            ffx_state_to_d3d12(d.output.state),
        ) {
            barriers.push(b);
        }
        // History[curr] stays as COPY_SOURCE → transition to SRV for next frame
        if let Some(b) = resource_barrier_transition_d3d12(
            &state.history[curr_idx],
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers.push(b);
        }
        if !barriers.is_empty() {
            cmd_list.ResourceBarrier(&barriers);
        }
    }

    gpu_pipeline::log_device_removed_reason(&gpu.device);

    // Advance ping-pong
    state.frame_idx = curr_idx as u32;
    state.initialized = true;

    info!(
        render = format_args!("{}x{}", render_w, render_h),
        output = format_args!("{}x{}", output_w, output_h),
        reset = is_reset,
        jitter = format_args!("({:.4}, {:.4})", d.jitter_offset.x, d.jitter_offset.y),
        frame_idx = state.frame_idx,
        prev_idx = prev_idx,
        curr_idx = curr_idx,
        d_reset = d.reset,
        "SGSRv2 temporal upscale (2-pass)"
    );

    0 // FFX_OK
}

/// Draw 6 debug tiles (FSR3 internal textures) on top of the current render target.
/// Layout: left column (3 tiles) | center (untouched) | right column (3 tiles).
/// Each side column is 1/5 output width.
///
/// Assumes: output is already in RENDER_TARGET state with RTV bound, pipeline state
/// (root signature, descriptor heaps, topology) is already set on `cmd_list`.
unsafe fn draw_debug_tiles(
    cmd_list: &ID3D12GraphicsCommandList,
    gpu: &GpuState,
    d: &FfxFsr3UpscalerDispatchDescription,
    output_w: u32,
    output_h: u32,
) {
    struct DebugTile {
        resource_ptr: *mut core::ffi::c_void,
        ffx_state: u32,
        ffx_format: u32,
        viz_mode: u32,
        srv_slot: u32,
    }

    // Left column: motion_vectors, transparency_and_composition, dilated_depth
    // Right column: reconstructed_prev_nearest_depth, reactive, depth
    let tiles = [
        DebugTile {
            resource_ptr: d.motion_vectors.resource,
            ffx_state: d.motion_vectors.state,
            ffx_format: d.motion_vectors.description.format,
            viz_mode: 2,
            srv_slot: 2,
        },
        DebugTile {
            resource_ptr: d.transparency_and_composition.resource,
            ffx_state: d.transparency_and_composition.state,
            ffx_format: d.transparency_and_composition.description.format,
            viz_mode: 0,
            srv_slot: 3,
        },
        DebugTile {
            resource_ptr: d.dilated_depth.resource,
            ffx_state: d.dilated_depth.state,
            ffx_format: d.dilated_depth.description.format,
            viz_mode: 1,
            srv_slot: 4,
        },
        DebugTile {
            resource_ptr: d.reconstructed_prev_nearest_depth.resource,
            ffx_state: d.reconstructed_prev_nearest_depth.state,
            ffx_format: d.reconstructed_prev_nearest_depth.description.format,
            viz_mode: 1,
            srv_slot: 5,
        },
        DebugTile {
            resource_ptr: d.reactive.resource,
            ffx_state: d.reactive.state,
            ffx_format: d.reactive.description.format,
            viz_mode: 3,
            srv_slot: 6,
        },
        DebugTile {
            resource_ptr: d.depth.resource,
            ffx_state: d.depth.state,
            ffx_format: d.depth.description.format,
            viz_mode: 1,
            srv_slot: 7,
        },
    ];

    // Borrow COM resources and barrier them to PIXEL_SHADER_RESOURCE.
    let mut tile_resources: Vec<Option<ID3D12Resource>> = Vec::new();
    let mut barriers_before = Vec::new();

    for tile in &tiles {
        let res = borrow_resource(tile.resource_ptr);
        if let Some(r) = &res {
            if let Some(b) = resource_barrier_transition(
                r,
                tile.ffx_state,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ) {
                barriers_before.push(b);
            }
        }
        tile_resources.push(res);
    }

    if !barriers_before.is_empty() {
        cmd_list.ResourceBarrier(&barriers_before);
    }

    // Grid layout
    let side_w = output_w / 5;
    let tile_h = output_h / 3;
    let center_w = output_w - 2 * side_w;

    // Re-bind shared pipeline state for debug PSO.
    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);
    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    for (i, tile) in tiles.iter().enumerate() {
        let res = match &tile_resources[i] {
            Some(r) => r,
            None => continue,
        };

        let (tile_x, tile_y) = if i < 3 {
            (0u32, (i as u32) * tile_h)
        } else {
            (side_w + center_w, ((i - 3) as u32) * tile_h)
        };

        if !create_typed_srv(gpu, res, tile.ffx_format, tile.srv_slot) {
            continue;
        }

        cmd_list.SetPipelineState(&gpu.pso_debug);
        cmd_list.SetGraphicsRoot32BitConstant(0, 1.0_f32.to_bits(), 0);
        cmd_list.SetGraphicsRoot32BitConstant(0, 1.0_f32.to_bits(), 1);
        cmd_list.SetGraphicsRoot32BitConstant(0, tile.viz_mode, 2);
        cmd_list.SetGraphicsRoot32BitConstant(0, 0u32, 3);
        cmd_list.SetGraphicsRootDescriptorTable(
            1,
            gpu_pipeline::get_srv_gpu_handle(gpu, tile.srv_slot),
        );

        let viewport = D3D12_VIEWPORT {
            TopLeftX: tile_x as f32,
            TopLeftY: tile_y as f32,
            Width: side_w as f32,
            Height: tile_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: tile_x as i32,
            top: tile_y as i32,
            right: (tile_x + side_w) as i32,
            bottom: (tile_y + tile_h) as i32,
        };
        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);
        cmd_list.DrawInstanced(3, 1, 0, 0);
        gpu_pipeline::log_device_removed_reason(&gpu.device);
    }

    // Restore tile resource barriers to their original FFX states.
    let tile_ffx_states = [
        d.motion_vectors.state,
        d.transparency_and_composition.state,
        d.dilated_depth.state,
        d.reconstructed_prev_nearest_depth.state,
        d.reactive.state,
        d.depth.state,
    ];
    let mut barriers_after = Vec::new();
    for (idx, ffx_state) in tile_ffx_states.iter().enumerate() {
        if let Some(r) = &tile_resources[idx] {
            if let Some(b) = resource_barrier_transition_d3d12(
                r,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ffx_state_to_d3d12(*ffx_state),
            ) {
                barriers_after.push(b);
            }
        }
    }
    if !barriers_after.is_empty() {
        cmd_list.ResourceBarrier(&barriers_after);
    }

    info!("debug tiles rendered");
}

/// Borrow a COM resource from a raw FFX pointer, returning None if null.
pub(crate) unsafe fn borrow_resource(raw: *mut core::ffi::c_void) -> Option<ID3D12Resource> {
    if raw.is_null() {
        return None;
    }
    <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&raw).map(|b| b.clone())
}

/// Color → output copy for AA mode (render_size == output_size) with overlay support.
/// Copies via CopyResource, then renders the imgui overlay on top.
pub unsafe fn dispatch_anti_aliasing(d: &FfxFsr3UpscalerDispatchDescription) -> u32 {
    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        warn!("dispatch_aa_copy: null command list");
        return 1;
    }

    let color_raw = d.color.resource;
    let output_raw = d.output.resource;
    if color_raw.is_null() || output_raw.is_null() {
        warn!("dispatch_aa_copy: null color or output resource");
        return 1;
    }

    let cmd_list: ID3D12GraphicsCommandList =
        match <ID3D12GraphicsCommandList as windows::core::Interface>::from_raw_borrowed(
            &cmd_list_raw,
        ) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_aa_copy: from_raw_borrowed failed for command list");
                return 1;
            }
        };

    let color_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&color_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_aa_copy: from_raw_borrowed failed for color resource");
                return 1;
            }
        };

    let output_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&output_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_aa_copy: from_raw_borrowed failed for output resource");
                return 1;
            }
        };

    let output_w = d.output.description.width;
    let output_h = d.output.description.height;

    // Barriers: color → COPY_SOURCE, output → COPY_DEST
    let barriers_before = [
        resource_barrier_transition(&color_res, d.color.state, D3D12_RESOURCE_STATE_COPY_SOURCE),
        resource_barrier_transition(&output_res, d.output.state, D3D12_RESOURCE_STATE_COPY_DEST),
    ];
    let valid_before: Vec<_> = barriers_before.iter().flatten().cloned().collect();
    if !valid_before.is_empty() {
        cmd_list.ResourceBarrier(&valid_before);
    }

    // CopyResource — valid because sizes are identical in AA mode.
    cmd_list.CopyResource(&output_res, &color_res);

    // Try to init GPU pipeline for overlay rendering.
    let output_format = gpu_pipeline::ffx_format_to_dxgi(d.output.description.format);
    let gpu = gpu_pipeline::get_or_init(&cmd_list, output_format);

    if let Some(gpu) = gpu {
        // Transition: color COPY_SOURCE → original, output COPY_DEST → RENDER_TARGET
        let barriers_mid = [
            resource_barrier_transition_d3d12(
                &color_res,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                ffx_state_to_d3d12(d.color.state),
            ),
            resource_barrier_transition_d3d12(
                &output_res,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
        ];
        let valid_mid: Vec<_> = barriers_mid.iter().flatten().cloned().collect();
        if !valid_mid.is_empty() {
            cmd_list.ResourceBarrier(&valid_mid);
        }

        // Create RTV for output
        gpu.device.CreateRenderTargetView(
            &output_res,
            None,
            gpu_pipeline::get_rtv_cpu_handle(gpu, 0),
        );

        // Set viewport + scissor to full output
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: output_w as f32,
            Height: output_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: output_w as i32,
            bottom: output_h as i32,
        };
        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);

        // Bind RTV
        let rtv_handle = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
        cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

        // Render imgui overlay
        overlay::render_frame(&cmd_list, gpu, output_w, output_h);

        // Final barrier: output RENDER_TARGET → original FFX state
        let barriers_after = [resource_barrier_transition_d3d12(
            &output_res,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            ffx_state_to_d3d12(d.output.state),
        )];
        let valid_after: Vec<_> = barriers_after.iter().flatten().cloned().collect();
        if !valid_after.is_empty() {
            cmd_list.ResourceBarrier(&valid_after);
        }
    } else {
        // GPU pipeline unavailable — just restore original states directly.
        let barriers_after = [
            resource_barrier_transition_d3d12(
                &color_res,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                ffx_state_to_d3d12(d.color.state),
            ),
            resource_barrier_transition_d3d12(
                &output_res,
                D3D12_RESOURCE_STATE_COPY_DEST,
                ffx_state_to_d3d12(d.output.state),
            ),
        ];
        let valid_after: Vec<_> = barriers_after.iter().flatten().cloned().collect();
        if !valid_after.is_empty() {
            cmd_list.ResourceBarrier(&valid_after);
        }
    }

    0 // FFX_OK
}

// Old SDK FfxResourceStates bit values (identical to FFX_API_RESOURCE_STATE_*):
const FFX_RESOURCE_STATE_COMMON: u32 = 1 << 0;
const FFX_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_RESOURCE_STATE_COMPUTE_READ: u32 = 1 << 2;
const FFX_RESOURCE_STATE_PIXEL_READ: u32 = 1 << 3;
const FFX_RESOURCE_STATE_COPY_SRC: u32 = 1 << 4;
const FFX_RESOURCE_STATE_COPY_DEST: u32 = 1 << 5;
const FFX_RESOURCE_STATE_INDIRECT_ARGUMENT: u32 = 1 << 6;
const FFX_RESOURCE_STATE_PRESENT: u32 = 1 << 7;
const FFX_RESOURCE_STATE_RENDER_TARGET: u32 = 1 << 8;

/// Map FFX resource state to D3D12 resource state.
pub(crate) fn ffx_state_to_d3d12(state: u32) -> D3D12_RESOURCE_STATES {
    let mut d3d_state = D3D12_RESOURCE_STATES(0);

    if state & FFX_RESOURCE_STATE_UNORDERED_ACCESS != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if state & FFX_RESOURCE_STATE_COMPUTE_READ != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_RESOURCE_STATE_PIXEL_READ != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_RESOURCE_STATE_COPY_SRC != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if state & FFX_RESOURCE_STATE_COPY_DEST != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if state & FFX_RESOURCE_STATE_RENDER_TARGET != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    if state & FFX_RESOURCE_STATE_PRESENT != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_PRESENT;
    }
    if state & FFX_RESOURCE_STATE_INDIRECT_ARGUMENT != 0 {
        d3d_state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    }

    // FFX_RESOURCE_STATE_COMMON maps to D3D12 COMMON (0)
    if d3d_state.0 == 0 && state & FFX_RESOURCE_STATE_COMMON != 0 {
        return D3D12_RESOURCE_STATE_COMMON;
    }

    d3d_state
}

/// Build a transition barrier from FFX state → D3D12 target state.
pub(crate) fn resource_barrier_transition(
    resource: &ID3D12Resource,
    state_before_ffx: u32,
    state_after: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_BARRIER> {
    let state_before = ffx_state_to_d3d12(state_before_ffx);
    resource_barrier_transition_d3d12(resource, state_before, state_after)
}

/// Build a transition barrier between two D3D12 states, returning None if they're equal.
pub(crate) fn resource_barrier_transition_d3d12(
    resource: &ID3D12Resource,
    state_before: D3D12_RESOURCE_STATES,
    state_after: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_BARRIER> {
    if state_before == state_after {
        return None;
    }

    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        ..Default::default()
    };
    barrier.Anonymous.Transition = std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: state_before,
        StateAfter: state_after,
    });

    Some(barrier)
}
