use crate::gpu_pipeline;
use crate::upscaler_type;
use crate::upscalers::{create_typed_srv, DispatchContext};
use tracing::info;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;

/// SRV slot for blit color input.
const SRV_COLOR: u32 = 0;

/// Single-pass spatial upscaler dispatch (Bilinear, Lanczos, SGSR).
/// Output must already be in RENDER_TARGET state with color in PIXEL_SHADER_RESOURCE.
/// Returns FFX_OK (0) on success.
pub unsafe fn dispatch(ctx: &DispatchContext) -> u32 {
    let gpu = ctx.gpu;
    let cmd_list = ctx.cmd_list;
    let d = ctx.d;

    let color_tex_w = d.color.description.width;
    let color_tex_h = d.color.description.height;
    let uv_scale_x = ctx.render_w as f32 / color_tex_w as f32;
    let uv_scale_y = ctx.render_h as f32 / color_tex_h as f32;

    // Create SRV for color texture
    create_typed_srv(gpu, ctx.color_res, d.color.description.format, SRV_COLOR);

    // Create RTV for output texture (slot 0)
    gpu.device
        .CreateRenderTargetView(ctx.output_res, None, ctx.rtv_cpu(0));

    // Select PSO based on current upscaler type
    let current_upscaler = upscaler_type::get();
    let pso = match current_upscaler {
        upscaler_type::UpscalerType::Bilinear => &gpu.pso_bilinear,
        upscaler_type::UpscalerType::Lanczos => &gpu.pso_lanczos,
        upscaler_type::UpscalerType::SGSR => &gpu.pso_sgsr,
        _ => unreachable!("simple::dispatch called for non-simple upscaler"),
    };

    // Viewport + scissor
    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: ctx.output_w as f32,
        Height: ctx.output_h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: ctx.output_w as i32,
        bottom: ctx.output_h as i32,
    };

    // Draw fullscreen triangle
    cmd_list.SetGraphicsRootSignature(&gpu.root_signature);
    cmd_list.SetPipelineState(pso);
    cmd_list.SetDescriptorHeaps(&[Some(gpu.srv_heap.clone())]);

    // Root constants: uvScale + inputSize
    cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_x.to_bits(), 0);
    cmd_list.SetGraphicsRoot32BitConstant(0, uv_scale_y.to_bits(), 1);
    cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_w as f32).to_bits(), 2);
    cmd_list.SetGraphicsRoot32BitConstant(0, (color_tex_h as f32).to_bits(), 3);

    cmd_list.SetGraphicsRootDescriptorTable(1, gpu.srv_heap.GetGPUDescriptorHandleForHeapStart());

    cmd_list.RSSetViewports(&[viewport]);
    cmd_list.RSSetScissorRects(&[scissor]);

    let rtv_handle = ctx.rtv_cpu(0);
    cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);

    cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd_list.DrawInstanced(3, 1, 0, 0);
    gpu_pipeline::log_device_removed_reason(&gpu.device);

    info!(
        render = format_args!("{}x{}", ctx.render_w, ctx.render_h),
        output = format_args!("{}x{}", ctx.output_w, ctx.output_h),
        uv_scale = format_args!("({:.3}, {:.3})", uv_scale_x, uv_scale_y),
        "DrawInstanced upscale blit"
    );

    0 // FFX_OK
}
