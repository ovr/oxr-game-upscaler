use std::sync::Mutex;

use tracing::info;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub struct Sgsr2State {
    pub motion_depth_clip: ID3D12Resource,
    pub history: [ID3D12Resource; 2],
    pub frame_idx: u32,
    pub initialized: bool,
    pub render_w: u32,
    pub render_h: u32,
    pub output_w: u32,
    pub output_h: u32,
}

static STATE: Mutex<Option<Sgsr2State>> = Mutex::new(None);

/// Get or create the SGSRv2 persistent state. Recreates textures if dimensions change.
pub unsafe fn get_or_create(
    device: &ID3D12Device,
    render_w: u32,
    render_h: u32,
    output_w: u32,
    output_h: u32,
    output_format: DXGI_FORMAT,
) -> Result<std::sync::MutexGuard<'static, Option<Sgsr2State>>, String> {
    let mut guard = STATE.lock().map_err(|_| "sgsr2 state mutex poisoned")?;

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
        let motion_depth_clip =
            create_texture(device, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let history0 = create_texture(device, output_w, output_h, output_format)?;
        let history1 = create_texture(device, output_w, output_h, output_format)?;

        info!(
            "sgsr2: created textures render={}x{} output={}x{} format={:?}",
            render_w, render_h, output_w, output_h, output_format
        );

        *guard = Some(Sgsr2State {
            motion_depth_clip,
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
        .map_err(|e| format!("sgsr2 CreateCommittedResource failed: {}", e))?;

    resource.ok_or_else(|| "sgsr2 CreateCommittedResource returned null".to_string())
}
