use std::sync::Mutex;

use tracing::info;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub struct Sgsr2ThreePassState {
    pub ycocg_color: ID3D12Resource,
    pub motion_depth_alpha: ID3D12Resource,
    pub motion_depth_clip_alpha: ID3D12Resource,
    pub luma_history: [ID3D12Resource; 2],
    pub history: [ID3D12Resource; 2],
    pub frame_idx: u32,
    pub initialized: bool,
    pub render_w: u32,
    pub render_h: u32,
    pub output_w: u32,
    pub output_h: u32,
}

static STATE: Mutex<Option<Sgsr2ThreePassState>> = Mutex::new(None);

/// Get or create the SGSRv2 3-pass persistent state. Recreates textures if dimensions change.
pub unsafe fn get_or_create(
    device: &ID3D12Device,
    render_w: u32,
    render_h: u32,
    output_w: u32,
    output_h: u32,
    output_format: DXGI_FORMAT,
) -> Result<std::sync::MutexGuard<'static, Option<Sgsr2ThreePassState>>, String> {
    let mut guard = STATE
        .lock()
        .map_err(|_| "sgsr2 3-pass state mutex poisoned")?;

    let needs_recreate = match guard.as_ref() {
        Some(s) => {
            s.render_w != render_w
                || s.render_h != render_h
                || s.output_w != output_w
                || s.output_h != output_h
        }
        None => true,
    };

    if needs_recreate {
        // Render-resolution textures
        let ycocg_color = create_texture(device, render_w, render_h, DXGI_FORMAT_R32_UINT)?;
        let motion_depth_alpha =
            create_texture(device, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let motion_depth_clip_alpha =
            create_texture(device, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let luma_history_0 = create_texture(device, render_w, render_h, DXGI_FORMAT_R32_UINT)?;
        let luma_history_1 = create_texture(device, render_w, render_h, DXGI_FORMAT_R32_UINT)?;

        // Output-resolution history buffers (always R16G16B16A16_FLOAT — stores tonemapped RGB)
        let history0 = create_texture(device, output_w, output_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let history1 = create_texture(device, output_w, output_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;

        info!(
            "sgsr2_3pass: created textures render={}x{} output={}x{} format={:?}",
            render_w, render_h, output_w, output_h, output_format
        );

        *guard = Some(Sgsr2ThreePassState {
            ycocg_color,
            motion_depth_alpha,
            motion_depth_clip_alpha,
            luma_history: [luma_history_0, luma_history_1],
            history: [history0, history1],
            frame_idx: 0,
            initialized: false,
            render_w,
            render_h,
            output_w,
            output_h,
        });
    }

    Ok(guard)
}

unsafe fn create_texture(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource, String> {
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
        .map_err(|e| format!("sgsr2_3pass CreateCommittedResource failed: {}", e))?;

    resource.ok_or_else(|| "sgsr2_3pass CreateCommittedResource returned null".to_string())
}
