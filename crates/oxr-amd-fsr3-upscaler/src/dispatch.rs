use crate::fsr3_types::*;
use crate::gpu_pipeline::{self, GpuState};
use crate::overlay;
use crate::upscaler_type::{self, UpscalerType};
use tracing::{error, info, warn};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;

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

    // --- Create RTV for output texture ---
    gpu.device.CreateRenderTargetView(
        &output_res,
        None, // default RTV desc
        gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart(),
    );

    // --- Set pipeline state ---
    let pso = match upscaler_type::get() {
        UpscalerType::Bilinear => &gpu.pso_bilinear,
        UpscalerType::Lanczos | UpscalerType::DebugView => &gpu.pso_lanczos,
    };
    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetPipelineState(pso);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

    // Root constants: uvScale (slots 0–1) + inputSize (slots 2–3)
    cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_x.to_bits(), 0);
    cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_y.to_bits(), 1);
    cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_w as f32).to_bits(), 2);
    cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_h as f32).to_bits(), 3);

    // Descriptor table: SRV
    cmd_list.SetGraphicsRootDescriptorTable(1, gpu.srv_heap.GetGPUDescriptorHandleForHeapStart());

    // --- Viewport + scissor ---
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

    // --- Set render target ---
    let rtv_handle = gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart();
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

    // --- Render imgui overlay (RTV still bound, output still in RENDER_TARGET state) ---
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

/// Debug view dispatch: renders a 3×1×3 grid showing internal FSR3 textures.
/// Layout: left column (3 tiles) | center (upscaled color) | right column (3 tiles)
/// Each side column is 1/5 output width; center is 3/5 output width.
pub unsafe fn dispatch_debug_view(d: &FfxFsr3UpscalerDispatchDescription) -> u32 {
    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        warn!("dispatch_debug_view: null command list");
        return 1;
    }

    let output_raw = d.output.resource;
    if output_raw.is_null() {
        warn!("dispatch_debug_view: null output resource");
        return 1;
    }

    let cmd_list: ID3D12GraphicsCommandList =
        match <ID3D12GraphicsCommandList as windows::core::Interface>::from_raw_borrowed(
            &cmd_list_raw,
        ) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_debug_view: from_raw_borrowed failed for command list");
                return 1;
            }
        };

    let output_res: ID3D12Resource =
        match <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&output_raw) {
            Some(borrowed) => borrowed.clone(),
            None => {
                error!("dispatch_debug_view: from_raw_borrowed failed for output resource");
                return 1;
            }
        };

    let output_format = gpu_pipeline::ffx_format_to_dxgi(d.output.description.format);
    let gpu = match gpu_pipeline::get_or_init(&cmd_list, output_format) {
        Some(g) => g,
        None => {
            error!("dispatch_debug_view: GPU pipeline init failed");
            return 1;
        }
    };

    let output_w = d.output.description.width;
    let output_h = d.output.description.height;
    if output_w == 0 || output_h == 0 {
        error!("dispatch_debug_view: zero output dimensions");
        return 1;
    }

    // Collect all input resources we want to visualize, with their FFX states and viz modes.
    // (resource_ptr, ffx_state, viz_mode, srv_slot)
    struct DebugTile {
        resource_ptr: *mut core::ffi::c_void,
        ffx_state: u32,
        ffx_format: u32,
        viz_mode: u32,
        srv_slot: u32,
    }

    // Left column (top to bottom): motion_vectors, transparency_and_composition, dilated_depth
    // Right column (top to bottom): reconstructed_prev_nearest_depth, reactive, depth
    let left_tiles = [
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
    ];
    let right_tiles = [
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

    // --- Barrier all input resources → PIXEL_SHADER_RESOURCE, output → RENDER_TARGET ---
    let color_res = borrow_resource(d.color.resource);
    let mut barriers_before = Vec::new();

    if let Some(r) = &color_res {
        if let Some(b) = resource_barrier_transition(
            r,
            d.color.state,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        ) {
            barriers_before.push(b);
        }
    }

    // Barrier debug tile resources
    let mut tile_resources: Vec<Option<ID3D12Resource>> = Vec::new();
    for tile in left_tiles.iter().chain(right_tiles.iter()) {
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

    if let Some(b) = resource_barrier_transition(
        &output_res,
        d.output.state,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    ) {
        barriers_before.push(b);
    }

    if !barriers_before.is_empty() {
        cmd_list.ResourceBarrier(&barriers_before);
    }

    // --- Create RTV for output ---
    gpu.device.CreateRenderTargetView(
        &output_res,
        None,
        gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart(),
    );

    let rtv_handle = gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart();
    cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

    // --- Clear output to black ---
    cmd_list.ClearRenderTargetView(rtv_handle, &[0.0_f32, 0.0, 0.0, 1.0], None);

    // --- Set shared pipeline state ---
    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);
    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Grid layout: side_w = output_w / 5, center_w = output_w * 3 / 5
    let side_w = output_w / 5;
    let tile_h = output_h / 3;
    let center_x = side_w;
    let center_w = output_w - 2 * side_w;

    // --- Draw center tile (upscaled color) ---
    if let Some(color_r) = &color_res {
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
        let uv_scale_x = render_w as f32 / color_tex_w as f32;
        let uv_scale_y = render_h as f32 / color_tex_h as f32;

        create_typed_srv(gpu, color_r, d.color.description.format, 0);

        cmd_list.SetPipelineState(&gpu.pso_bilinear);
        cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_x.to_bits(), 0);
        cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_y.to_bits(), 1);
        cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_w as f32).to_bits(), 2);
        cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_h as f32).to_bits(), 3);
        cmd_list.SetGraphicsRootDescriptorTable(1, gpu_pipeline::get_srv_gpu_handle(gpu, 0));

        let viewport = D3D12_VIEWPORT {
            TopLeftX: center_x as f32,
            TopLeftY: 0.0,
            Width: center_w as f32,
            Height: output_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: center_x as i32,
            top: 0,
            right: (center_x + center_w) as i32,
            bottom: output_h as i32,
        };
        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);
        cmd_list.DrawInstanced(3, 1, 0, 0);
        gpu_pipeline::log_device_removed_reason(&gpu.device);
    }

    // --- Draw debug tiles ---
    let all_tiles: Vec<_> = left_tiles.iter().chain(right_tiles.iter()).collect();

    for (i, tile) in all_tiles.iter().enumerate() {
        let res = &tile_resources[i];
        let res = match res {
            Some(r) => r,
            None => continue, // null resource, leave black
        };

        // Determine tile position
        let (tile_x, tile_y) = if i < 3 {
            // Left column
            (0u32, (i as u32) * tile_h)
        } else {
            // Right column
            (center_x + center_w, ((i - 3) as u32) * tile_h)
        };

        // Create SRV at the tile's slot (explicit typed format for typeless resources)
        if !create_typed_srv(gpu, res, tile.ffx_format, tile.srv_slot) {
            continue; // skip tile — invalid format would poison GPU command stream
        }

        cmd_list.SetPipelineState(&gpu.pso_debug);
        // uvScale = 1.0, 1.0 for debug tiles (show full texture)
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

    info!(
        output = format_args!("{}x{}", output_w, output_h),
        "dispatch_debug_view rendered"
    );

    // --- Render imgui overlay ---
    // Reset viewport to full output for overlay
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

    // --- Restore barriers ---
    let mut barriers_after = Vec::new();

    if let Some(r) = &color_res {
        if let Some(b) = resource_barrier_transition_d3d12(
            r,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ffx_state_to_d3d12(d.color.state),
        ) {
            barriers_after.push(b);
        }
    }

    let all_ffx_tiles_data = [
        (d.motion_vectors.state, 0usize),
        (d.transparency_and_composition.state, 1),
        (d.dilated_depth.state, 2),
        (d.reconstructed_prev_nearest_depth.state, 3),
        (d.reactive.state, 4),
        (d.depth.state, 5),
    ];
    for (ffx_state, idx) in &all_ffx_tiles_data {
        if let Some(r) = &tile_resources[*idx] {
            if let Some(b) = resource_barrier_transition_d3d12(
                r,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ffx_state_to_d3d12(*ffx_state),
            ) {
                barriers_after.push(b);
            }
        }
    }

    if let Some(b) = resource_barrier_transition_d3d12(
        &output_res,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        ffx_state_to_d3d12(d.output.state),
    ) {
        barriers_after.push(b);
    }

    if !barriers_after.is_empty() {
        cmd_list.ResourceBarrier(&barriers_after);
    }

    0 // FFX_OK
}

/// Borrow a COM resource from a raw FFX pointer, returning None if null.
unsafe fn borrow_resource(raw: *mut core::ffi::c_void) -> Option<ID3D12Resource> {
    if raw.is_null() {
        return None;
    }
    <ID3D12Resource as windows::core::Interface>::from_raw_borrowed(&raw).map(|b| b.clone())
}

/// Lightweight color → output copy for AA mode (render_size == output_size).
/// No GPU pipeline, no shaders — just barrier + CopyResource + barrier.
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

    // Barriers: restore original states
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
fn ffx_state_to_d3d12(state: u32) -> D3D12_RESOURCE_STATES {
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
fn resource_barrier_transition(
    resource: &ID3D12Resource,
    state_before_ffx: u32,
    state_after: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_BARRIER> {
    let state_before = ffx_state_to_d3d12(state_before_ffx);
    resource_barrier_transition_d3d12(resource, state_before, state_after)
}

/// Build a transition barrier between two D3D12 states, returning None if they're equal.
fn resource_barrier_transition_d3d12(
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
