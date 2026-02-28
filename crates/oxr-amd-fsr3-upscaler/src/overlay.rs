/// In-game imgui overlay for runtime upscaler switching.
///
/// **Home** — toggle overlay visible/hidden
/// **End + Up/Down/Enter/Space** — navigate and activate widgets (imgui keyboard nav)
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
const VK_UP: i32 = 0x26;
const VK_DOWN: i32 = 0x28;
const VK_LEFT: i32 = 0x25;
const VK_RIGHT: i32 = 0x27;
const VK_RETURN: i32 = 0x0D;
const VK_SPACE: i32 = 0x20;

struct OverlayState {
    ctx: Context,
    renderer: ImguiDx12Renderer,
    visible: bool,
    frame_idx: usize,
    prev_home: bool,
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

    // --- Keyboard input (rising edge for Home toggle) ---
    let home = key_down(VK_HOME);
    if home && !state.prev_home {
        state.visible = !state.visible;
        info!("overlay: visible={}", state.visible);
    }
    state.prev_home = home;

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

    // --- Feed imgui IO (keyboard nav via End modifier) ---
    {
        let io = state.ctx.io_mut();
        io.display_size = [output_w as f32, output_h as f32];
        io.delta_time = 1.0 / 60.0;

        // Feed nav keys to imgui only while End is held (avoid stealing game input)
        let end = key_down(VK_END);
        io.add_key_event(imgui::Key::UpArrow, end && key_down(VK_UP));
        io.add_key_event(imgui::Key::DownArrow, end && key_down(VK_DOWN));
        io.add_key_event(imgui::Key::LeftArrow, end && key_down(VK_LEFT));
        io.add_key_event(imgui::Key::RightArrow, end && key_down(VK_RIGHT));
        io.add_key_event(imgui::Key::Enter, end && key_down(VK_RETURN));
        io.add_key_event(imgui::Key::Space, end && key_down(VK_SPACE));
    }

    // --- Build UI ---
    {
        let ui = state.ctx.frame();

        let bg_tok = ui.push_style_color(imgui::StyleColor::WindowBg, [0.1, 0.1, 0.1, 1.0]);
        let title_tok = ui.push_style_color(imgui::StyleColor::TitleBg, [0.2, 0.2, 0.2, 1.0]);
        let title_active_tok =
            ui.push_style_color(imgui::StyleColor::TitleBgActive, [0.3, 0.3, 0.3, 1.0]);

        ui.window("Upscaler")
            .size([450.0, 0.0], Condition::Always)
            .position([60.0, 400.0], Condition::Always)
            .flags(imgui::WindowFlags::NO_MOUSE_INPUTS)
            .build(|| {
                // Upscaler radio buttons
                let mut active = upscaler_type::get();
                ui.text("Upscaler");
                if ui.radio_button("Bilinear", &mut active, UpscalerType::Bilinear) {
                    upscaler_type::set(active);
                    info!("overlay: switched to {:?}", active);
                }
                ui.same_line();
                if ui.radio_button("Lanczos", &mut active, UpscalerType::Lanczos) {
                    upscaler_type::set(active);
                    info!("overlay: switched to {:?}", active);
                }

                ui.separator();

                // Debug View checkbox
                let mut debug_on = upscaler_type::debug_view_get();
                if ui.checkbox("Debug View", &mut debug_on) {
                    upscaler_type::debug_view_set(debug_on);
                    info!("overlay: debug_view={}", debug_on);
                }
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
    ctx.io_mut().config_flags |= imgui::ConfigFlags::NAV_ENABLE_KEYBOARD;

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
        fonts_ready: false,
    })
}
