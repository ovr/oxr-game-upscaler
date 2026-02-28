/// In-game imgui overlay for runtime upscaler switching.
///
/// **Home** — toggle overlay visible/hidden
/// **End + Up/Down** — navigate focus between UI elements
/// **End + Left/Right** — change value of focused element
use std::sync::Mutex;

use imgui::{Condition, Context};
use tracing::{error, info};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

use crate::gpu_pipeline::{self, get_srv_cpu_handle, get_srv_gpu_handle, GpuState};
use crate::imgui_renderer::ImguiDx12Renderer;
use crate::upscaler_type;

const VK_HOME: i32 = 0x24;
const VK_END: i32 = 0x23;
const VK_UP: i32 = 0x26;
const VK_DOWN: i32 = 0x28;
const VK_LEFT: i32 = 0x25;
const VK_RIGHT: i32 = 0x27;

struct OverlayState {
    ctx: Context,
    renderer: ImguiDx12Renderer,
    visible: bool,
    frame_idx: usize,
    prev_home: bool,
    prev_up: bool,
    prev_down: bool,
    prev_left: bool,
    prev_right: bool,
    focus_index: usize,
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

    // Manual key handling — all arrows with rising edge, End as modifier
    let end = key_down(VK_END);

    let up = end && key_down(VK_UP);
    let down = end && key_down(VK_DOWN);
    let left = end && key_down(VK_LEFT);
    let right = end && key_down(VK_RIGHT);

    let up_pressed = up && !state.prev_up;
    let down_pressed = down && !state.prev_down;
    let left_pressed = left && !state.prev_left;
    let right_pressed = right && !state.prev_right;

    const NUM_ITEMS: usize = 1; // only upscaler for now
    if up_pressed && state.focus_index > 0 {
        state.focus_index -= 1;
    }
    if down_pressed && state.focus_index < NUM_ITEMS - 1 {
        state.focus_index += 1;
    }

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
                let focused = state.focus_index == 0;
                let label = format!(
                    "{}Upscaler  < {:?} >",
                    if focused { ">> " } else { "   " },
                    active
                );

                if focused {
                    let tok = ui.push_style_color(imgui::StyleColor::Text, [1.0, 1.0, 0.0, 1.0]);
                    ui.text(&label);
                    tok.pop();
                } else {
                    ui.text(&label);
                }

                if focused {
                    if left_pressed {
                        let new = active.prev();
                        upscaler_type::set(new);
                        info!("overlay: switched to {:?}", new);
                    }
                    if right_pressed {
                        let new = active.next();
                        upscaler_type::set(new);
                        info!("overlay: switched to {:?}", new);
                    }
                }

                ui.spacing();
                ui.text("[Home] toggle  [End+Arrows] navigate & change");
            });

        title_active_tok.pop();
        title_tok.pop();
        bg_tok.pop();
    }

    state.prev_up = up;
    state.prev_down = down;
    state.prev_left = left;
    state.prev_right = right;

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
        prev_up: false,
        prev_down: false,
        prev_left: false,
        prev_right: false,
        focus_index: 0,
        fonts_ready: false,
    })
}
