use crate::gpu_pipeline;
use crate::post_processing::PostContext;
use crate::upscaler_type;
use crate::upscalers::{borrow_resource, create_typed_srv};
use tracing::info;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;

use crate::dispatch::{
    ffx_state_to_d3d12, resource_barrier_transition, resource_barrier_transition_d3d12,
};

pub fn is_enabled() -> bool {
    upscaler_type::debug_view_get()
}

/// Draw 6 debug tiles (FSR3 internal textures) on top of the current render target.
/// Layout: left column (3 tiles) | center (untouched) | right column (3 tiles).
/// Each side column is 1/5 output width.
///
/// Assumes: output is already in RENDER_TARGET state with RTV bound at slot 0.
pub unsafe fn apply(ctx: &PostContext) {
    let gpu = ctx.gpu;
    let cmd_list = ctx.cmd_list;
    let d = ctx.d;
    let output_w = ctx.output_w;
    let output_h = ctx.output_h;

    struct DebugTile {
        resource_ptr: *mut core::ffi::c_void,
        ffx_state: u32,
        ffx_format: u32,
        viz_mode: u32,
        srv_slot: u32,
    }

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
            barriers_before.push(resource_barrier_transition(
                r,
                tile.ffx_state,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ));
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

    // Restore tile resource barriers.
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
            barriers_after.push(resource_barrier_transition_d3d12(
                r,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ffx_state_to_d3d12(*ffx_state),
            ));
        }
    }
    if !barriers_after.is_empty() {
        cmd_list.ResourceBarrier(&barriers_after);
    }

    info!("debug tiles rendered");
}
