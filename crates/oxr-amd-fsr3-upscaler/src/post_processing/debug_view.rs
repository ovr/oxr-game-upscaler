use crate::gpu_pipeline;
use crate::post_processing::PostContext;
use crate::upscaler_type;
use crate::upscalers::{borrow_resource, create_native_srv, create_typed_srv};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;

use crate::dispatch::{
    ffx_state_to_d3d12, resource_barrier_transition, resource_barrier_transition_d3d12,
};

pub fn is_enabled() -> bool {
    upscaler_type::debug_view_get()
}

/// Draw debug tiles (FSR3 internal textures) on top of the current render target.
/// Layout: left column | center (untouched) | right column.
/// Each side column is 1/5 output width. NULL resources are skipped.
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
        /// 0=RGB, 1=depth gray, 2=motion vectors, 3=mask, 4=depth colorized, 5=integer (point sampled)
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
            viz_mode: 5, // R8_UINT — needs point sampler
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
            viz_mode: 4,
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

    // Compact: only tiles with non-null resources get rendered
    let mut visible: Vec<(usize, &DebugTile)> = Vec::new();
    for (i, tile) in tiles.iter().enumerate() {
        if tile_resources[i].is_some() {
            visible.push((i, tile));
        }
    }

    let n = visible.len() as u32;
    if n == 0 {
        return;
    }

    // Layout: two columns (left/right), max 75% screen height, centered vertically
    let left_count = (n + 1) / 2; // ceil(n/2)
    let right_count = n - left_count;
    let margin = 15u32;
    let gap = 2u32;
    let side_w = output_w / 5;
    let max_col_h = output_h * 60 / 100;
    let left_gaps = if left_count > 1 {
        (left_count - 1) * gap
    } else {
        0
    };
    let right_gaps = if right_count > 1 {
        (right_count - 1) * gap
    } else {
        0
    };
    let left_tile_h = (max_col_h - left_gaps) / left_count;
    let right_tile_h = if right_count > 0 {
        (max_col_h - right_gaps) / right_count
    } else {
        0
    };
    let left_total_h = left_tile_h * left_count + left_gaps;
    let right_total_h = right_tile_h * right_count + right_gaps;
    let left_y0 = (output_h - left_total_h) / 2;
    let right_y0 = if right_count > 0 {
        (output_h - right_total_h) / 2
    } else {
        0
    };
    let center_w = output_w - 2 * side_w;

    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);
    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    let rtv_output = gpu_pipeline::get_rtv_cpu_handle(gpu, 0);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_output), false, None);

    for (vi, &(res_idx, tile)) in visible.iter().enumerate() {
        let res = tile_resources[res_idx].as_ref().unwrap();
        let vi = vi as u32;

        let (tile_x, tile_y, tile_w, tile_h) = if vi < left_count {
            (
                margin,
                left_y0 + vi * (left_tile_h + gap),
                side_w,
                left_tile_h,
            )
        } else {
            let ri = vi - left_count;
            (
                output_w - side_w - margin,
                right_y0 + ri * (right_tile_h + gap),
                side_w,
                right_tile_h,
            )
        };

        // Create SRV: use native format for integer textures, typed for others
        let srv_ok = if tile.viz_mode == 5 {
            create_native_srv(gpu, res, tile.srv_slot)
        } else {
            create_typed_srv(gpu, res, tile.ffx_format, tile.srv_slot)
        };

        if !srv_ok {
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
            Width: tile_w as f32,
            Height: tile_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: tile_x as i32,
            top: tile_y as i32,
            right: (tile_x + tile_w) as i32,
            bottom: (tile_y + tile_h) as i32,
        };
        cmd_list.RSSetViewports(&[viewport]);
        cmd_list.RSSetScissorRects(&[scissor]);
        cmd_list.DrawInstanced(3, 1, 0, 0);
        gpu_pipeline::log_device_removed_reason(&gpu.device);
    }

    // Restore tile resource barriers.
    for (i, tile) in tiles.iter().enumerate() {
        if let Some(r) = &tile_resources[i] {
            let orig_state = ffx_state_to_d3d12(tile.ffx_state);
            let barrier = resource_barrier_transition_d3d12(
                r,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                orig_state,
            );
            cmd_list.ResourceBarrier(&[barrier]);
        }
    }
}
