/// In-game imgui overlay for runtime upscaler switching.
///
/// **Home** — toggle overlay visible/hidden
/// **1** — select Bilinear
/// **2** — select Lanczos
use std::sync::Mutex;

use imgui::{Condition, Context};
use tracing::{error, info};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

use crate::gpu_pipeline::{self, get_srv_cpu_handle, get_srv_gpu_handle, GpuState};
use crate::imgui_renderer::ImguiDx12Renderer;
use crate::upscaler_type::{self, UpscalerType};

const VK_HOME: i32 = 0x24;
const VK_END: i32 = 0x23;
const VK_1: i32 = 0x31;
const VK_2: i32 = 0x32;
const VK_3: i32 = 0x33;

struct OverlayState {
    ctx: Context,
    renderer: ImguiDx12Renderer,
    visible: bool,
    frame_idx: usize,
    prev_home: bool,
    prev_1: bool,
    prev_2: bool,
    prev_3: bool,
    fonts_ready: bool,
}

// Safety: OverlayState is only accessed under the OVERLAY mutex,
// ensuring single-threaded access. imgui::Context is !Send because
// Dear ImGui uses a global context pointer, but we guarantee
// exclusive access via the Mutex.
unsafe impl Send for OverlayState {}

static OVERLAY: Mutex<Option<OverlayState>> = Mutex::new(None);

fn key_down(vk: i32) -> bool {
    (unsafe { GetAsyncKeyState(vk) } as u16 & 0x8000) != 0
}

/// Call once per dispatch, after `DrawInstanced`, while the output RTV is still bound
/// and the output resource is in RENDER_TARGET state.
pub unsafe fn render_frame(
    cmd_list: &ID3D12GraphicsCommandList,
    gpu: &GpuState,
    output_w: u32,
    output_h: u32,
) {
    let Ok(mut guard) = OVERLAY.lock() else {
        return; // poisoned
    };

    // --- Lazy init ---
    if guard.is_none() {
        match init_overlay(gpu) {
            Ok(state) => {
                *guard = Some(state);
                info!("overlay: initialized");
            }
            Err(e) => {
                error!("overlay: init failed: {}", e);
                return;
            }
        }
    }

    let state = guard.as_mut().unwrap();

    // --- Keyboard input (rising edge) ---
    let home = key_down(VK_HOME);
    if home && !state.prev_home {
        state.visible = !state.visible;
        info!("overlay: visible={}", state.visible);
    }
    state.prev_home = home;

    // End+1 = Bilinear, End+2 = Lanczos
    let end = key_down(VK_END);
    let k1 = key_down(VK_1);
    let k2 = key_down(VK_2);
    let k3 = key_down(VK_3);

    if end && k1 && !state.prev_1 {
        upscaler_type::set(UpscalerType::Bilinear);
        info!("overlay: switched to Bilinear (End+1)");
    }
    if end && k2 && !state.prev_2 {
        upscaler_type::set(UpscalerType::Lanczos);
        info!("overlay: switched to Lanczos (End+2)");
    }
    if end && k3 && !state.prev_3 {
        upscaler_type::set(UpscalerType::DebugView);
        info!("overlay: switched to DebugView (End+3)");
    }

    state.prev_1 = k1;
    state.prev_2 = k2;
    state.prev_3 = k3;

    if !state.visible {
        return;
    }

    // --- Ensure font atlas is uploaded (lazy, on first visible frame) ---
    if !state.fonts_ready {
        let font_cpu = get_srv_cpu_handle(gpu, 1);
        let font_gpu = get_srv_gpu_handle(gpu, 1);
        if let Err(e) = state
            .renderer
            .ensure_fonts(&mut state.ctx, cmd_list, font_cpu, font_gpu)
        {
            error!("overlay: ensure_fonts failed: {}", e);
            return;
        }
        state.fonts_ready = true;
    }

    // --- Feed imgui IO (no mouse — keyboard-only control) ---
    {
        let io = state.ctx.io_mut();
        io.display_size = [output_w as f32, output_h as f32];
        io.delta_time = 1.0 / 60.0;
    }

    // --- Build UI ---
    {
        let ui = state.ctx.frame();

        let bg_tok = ui.push_style_color(imgui::StyleColor::WindowBg, [0.1, 0.1, 0.1, 1.0]);
        let title_tok = ui.push_style_color(imgui::StyleColor::TitleBg, [0.2, 0.2, 0.2, 1.0]);
        let title_active_tok =
            ui.push_style_color(imgui::StyleColor::TitleBgActive, [0.3, 0.3, 0.3, 1.0]);

        ui.window("Upscaler")
            .size([500.0, 200.0], Condition::Always)
            .position([60.0, 400.0], Condition::Always)
            .no_inputs()
            .build(|| {
                let active = upscaler_type::get();
                let sel = |t: UpscalerType| if active == t { ">>>" } else { "   " };

                ui.text(format!("{} [End+1] Bilinear", sel(UpscalerType::Bilinear)));
                ui.text(format!("{} [End+2] Lanczos", sel(UpscalerType::Lanczos)));
                ui.text(format!(
                    "{} [End+3] Debug View",
                    sel(UpscalerType::DebugView)
                ));
                ui.spacing();
                ui.text(format!("Active: {:?}", active));
                ui.text("[Home] toggle overlay");
            });

        title_active_tok.pop();
        title_tok.pop();
        bg_tok.pop();
    }

    let draw_data = state.ctx.render();

    // --- Render ---
    let frame_idx = state.frame_idx;
    let rtv_handle = gpu.rtv_heap.GetCPUDescriptorHandleForHeapStart();
    if let Err(e) = state
        .renderer
        .render(draw_data, cmd_list, frame_idx, &gpu.srv_heap, rtv_handle)
    {
        error!("overlay: render failed: {}", e);
        gpu_pipeline::log_device_removed_reason(&gpu.device);
    }

    state.frame_idx = (state.frame_idx + 1) % 2;
}

unsafe fn init_overlay(gpu: &GpuState) -> Result<OverlayState, String> {
    let mut ctx = Context::create();
    ctx.set_ini_filename(None);

    ctx.fonts().add_font(&[imgui::FontSource::DefaultFontData {
        config: Some(imgui::FontConfig {
            size_pixels: 32.0,
            oversample_h: 1,
            oversample_v: 1,
            ..Default::default()
        }),
    }]);

    let renderer = ImguiDx12Renderer::new(&gpu.device, gpu_pipeline::get_rt_format(gpu))?;

    Ok(OverlayState {
        ctx,
        renderer,
        visible: true,
        frame_idx: 0,
        prev_home: false,
        prev_1: false,
        prev_2: false,
        prev_3: false,
        fonts_ready: false,
    })
}
