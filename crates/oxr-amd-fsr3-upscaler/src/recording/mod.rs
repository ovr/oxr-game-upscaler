mod readback;
mod writer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use tracing::{error, info, warn};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList2,
};
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

use crate::dispatch;
use crate::fsr3_types::FfxFsr3UpscalerDispatchDescription;
use crate::logging;

use readback::{is_depth_stencil_format, ReadbackPool, Slot};
use writer::{FrameMetadata, FramePacket, TextureData, WriterMessage};

const VK_F10: i32 = 0x79;

/// Max queued frames before we start dropping.
const MAX_QUEUE_DEPTH: u64 = 30;

static RECORDING_ACTIVE: AtomicBool = AtomicBool::new(false);
static QUEUED_FRAMES: AtomicU64 = AtomicU64::new(0);

struct RecorderState {
    pool: ReadbackPool,
    sender: std::sync::mpsc::Sender<WriterMessage>,
    _session_dir: PathBuf,
    frame_number: u64,
    parity: usize,
    warmup_frames: u8,
    prev_f9: bool,
    /// Monotonic counter written by GPU via WriteBufferImmediate.
    write_counter: u32,
    /// Expected marker value per parity slot (set when GPU write is enqueued).
    expected_marker: [u32; 3],
}

static RECORDER: Mutex<Option<RecorderState>> = Mutex::new(None);

/// Called before dispatch. Checks hotkey, maps previous frame's readback, sends to writer.
pub unsafe fn pre_dispatch(d: &FfxFsr3UpscalerDispatchDescription) {
    let f9 = (GetAsyncKeyState(VK_F10) as u16 & 0x8000) != 0;

    let mut guard = match RECORDER.lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    // Detect rising edge of F9
    let prev_f9 = guard.as_ref().map(|s| s.prev_f9).unwrap_or(false);
    let toggled = f9 && !prev_f9;

    if let Some(state) = guard.as_mut() {
        state.prev_f9 = f9;
    }

    if toggled {
        if RECORDING_ACTIVE.load(Ordering::Relaxed) {
            // Stop recording
            info!("recording: stopping");
            if let Some(state) = guard.take() {
                let _ = state.sender.send(WriterMessage::Shutdown);
            }
            RECORDING_ACTIVE.store(false, Ordering::Relaxed);
            QUEUED_FRAMES.store(0, Ordering::Relaxed);
            info!("recording: stopped");
            return;
        } else {
            // Start recording
            let dll_dir = logging::dll_directory().unwrap_or_else(|| PathBuf::from("."));
            let recordings_dir = dll_dir.join("recordings");
            if let Err(e) = std::fs::create_dir_all(&recordings_dir) {
                error!("recording: failed to create recordings dir: {}", e);
                return;
            }

            let timestamp = {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                // Simple timestamp: just use seconds since epoch
                format!("session_{}", now)
            };
            let session_dir = recordings_dir.join(&timestamp);
            if let Err(e) = std::fs::create_dir_all(&session_dir) {
                error!("recording: failed to create session dir: {}", e);
                return;
            }

            let sender = writer::spawn_writer(session_dir.clone());

            *guard = Some(RecorderState {
                pool: ReadbackPool::new(),
                sender,
                _session_dir: session_dir.clone(),
                frame_number: 0,
                parity: 0,
                warmup_frames: 2,
                prev_f9: f9,
                write_counter: 1,
                expected_marker: [0; 3],
            });
            RECORDING_ACTIVE.store(true, Ordering::Relaxed);
            info!("recording: started → {}", session_dir.display());
            return;
        }
    }

    // If recording, map previous frame's readback and send to writer
    if !RECORDING_ACTIVE.load(Ordering::Relaxed) {
        return;
    }

    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    if state.warmup_frames > 0 {
        // Need 2 frames of GPU latency before readback is safe
        state.warmup_frames -= 1;
        return;
    }

    // Backpressure check
    if QUEUED_FRAMES.load(Ordering::Relaxed) > MAX_QUEUE_DEPTH {
        warn!(
            "recording: backpressure, skipping frame {} (queue > {})",
            state.frame_number, MAX_QUEUE_DEPTH
        );
        state.frame_number += 1;
        return;
    }

    let prev_parity = (state.parity + 1) % 3;

    // Check GPU completion marker before reading back
    let expected = state.expected_marker[prev_parity];
    if expected != 0 {
        match state.pool.read_marker(prev_parity) {
            Some(marker) if marker == expected => {}
            Some(marker) => {
                warn!(
                    "recording: marker not ready for parity {} (got={}, expected={}), skipping frame {}",
                    prev_parity, marker, expected, state.frame_number
                );
                state.frame_number += 1;
                return;
            }
            None => {
                warn!(
                    "recording: failed to read marker for parity {}, skipping frame {}",
                    prev_parity, state.frame_number
                );
                state.frame_number += 1;
                return;
            }
        }
    }

    // Extract data from previous frame's readback buffers
    let color = state
        .pool
        .map_and_extract(Slot::Color, prev_parity)
        .map(|(info, data)| TextureData { data, info });
    let depth = state
        .pool
        .map_and_extract(Slot::Depth, prev_parity)
        .map(|(info, data)| TextureData { data, info });
    let motion_vectors = state
        .pool
        .map_and_extract(Slot::MotionVectors, prev_parity)
        .map(|(info, data)| TextureData { data, info });

    let metadata = FrameMetadata {
        jitter_x: d.jitter_offset.x,
        jitter_y: d.jitter_offset.y,
        camera_near: d.camera_near,
        camera_far: d.camera_far,
        camera_fov: d.camera_fov_angle_vertical,
        frame_time_delta: d.frame_time_delta,
        render_width: d.render_size.width,
        render_height: d.render_size.height,
        output_width: d.output.description.width,
        output_height: d.output.description.height,
        motion_vector_scale_x: d.motion_vector_scale.x,
        motion_vector_scale_y: d.motion_vector_scale.y,
        pre_exposure: d.pre_exposure,
        view_space_to_meters_factor: d.view_space_to_meters_factor,
        reset: d.reset,
    };

    let packet = FramePacket {
        frame_number: state.frame_number,
        color,
        depth,
        motion_vectors,
        metadata,
    };

    match state.sender.send(WriterMessage::Frame(packet)) {
        Ok(()) => {
            QUEUED_FRAMES.fetch_add(1, Ordering::Relaxed);
        }
        Err(_) => {
            error!("recording: writer channel closed, disabling recording");
            RECORDING_ACTIVE.store(false, Ordering::Relaxed);
            *guard = None;
            return;
        }
    }

    state.frame_number += 1;
}

/// Called after dispatch. Enqueues GPU copies from source textures to readback buffers.
pub unsafe fn post_dispatch(d: &FfxFsr3UpscalerDispatchDescription) {
    if !RECORDING_ACTIVE.load(Ordering::Relaxed) {
        return;
    }

    let mut guard = match RECORDER.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    let cmd_list_raw = d.command_list;
    if cmd_list_raw.is_null() {
        return;
    }
    let cmd_list: ID3D12GraphicsCommandList =
        match <ID3D12GraphicsCommandList as windows::core::Interface>::from_raw_borrowed(
            &cmd_list_raw,
        ) {
            Some(borrowed) => borrowed.clone(),
            None => return,
        };

    // Get device
    let mut device: Option<ID3D12Device> = None;
    if cmd_list.GetDevice(&mut device).is_err() {
        return;
    }
    let device = match device {
        Some(d) => d,
        None => return,
    };

    let parity = state.parity;

    // Helper: borrow resource, get its real D3D12 dims+format, ensure readback buffer, enqueue copy
    let mut copy_slot =
        |slot: Slot, raw: *mut core::ffi::c_void, ffx_state: u32, ffx_w: u32, ffx_h: u32| {
            if raw.is_null() || ffx_w == 0 || ffx_h == 0 {
                return;
            }
            if let Some(res) = dispatch::borrow_resource(raw) {
                let desc = res.GetDesc();
                let actual_w = desc.Width as u32;
                let actual_h = desc.Height;
                info!(
                    "recording: {:?} ffx={}x{} d3d12={}x{} fmt={:?} ffx_state=0x{:x}",
                    slot, ffx_w, ffx_h, actual_w, actual_h, desc.Format, ffx_state
                );
                // Skip depth-stencil multi-plane formats — barrier with ALL_SUBRESOURCES is
                // unsafe when planes are in different states, causing GPU TDR.
                if is_depth_stencil_format(desc.Format) {
                    info!(
                        "recording: {:?} skipping depth-stencil format {:?}",
                        slot, desc.Format
                    );
                    return;
                }
                // Use actual D3D12 resource dimensions — FFX descriptor may report different
                // sizes (e.g. render region vs texture size). Copy box must not exceed the
                // actual resource or we get a GPU crash (TDR).
                let copy_w = ffx_w.min(actual_w);
                let copy_h = ffx_h.min(actual_h);
                if state
                    .pool
                    .ensure_buffer(&device, slot, parity, copy_w, copy_h, desc.Format)
                {
                    info!("recording: enqueue_copy {:?}", slot);
                    state
                        .pool
                        .enqueue_copy(&cmd_list, slot, parity, &res, ffx_state);
                    info!("recording: enqueue_copy {:?} done", slot);
                }
            }
        };

    // Color (use render_size, not texture size — render region may be smaller)
    let color_w = if d.render_size.width > 0 {
        d.render_size.width
    } else {
        d.color.description.width
    };
    let color_h = if d.render_size.height > 0 {
        d.render_size.height
    } else {
        d.color.description.height
    };
    copy_slot(
        Slot::Color,
        d.color.resource,
        d.color.state,
        color_w,
        color_h,
    );

    // Depth
    copy_slot(
        Slot::Depth,
        d.depth.resource,
        d.depth.state,
        d.depth.description.width,
        d.depth.description.height,
    );

    // Motion vectors
    copy_slot(
        Slot::MotionVectors,
        d.motion_vectors.resource,
        d.motion_vectors.state,
        d.motion_vectors.description.width,
        d.motion_vectors.description.height,
    );

    // Write GPU completion marker after all copies
    if state.pool.ensure_marker_buffer(&device) {
        match cmd_list.cast::<ID3D12GraphicsCommandList2>() {
            Ok(cmd_list2) => {
                state
                    .pool
                    .write_marker(&cmd_list2, parity, state.write_counter);
                state.expected_marker[parity] = state.write_counter;
                state.write_counter = state.write_counter.wrapping_add(1);
                if state.write_counter == 0 {
                    state.write_counter = 1; // skip 0 (sentinel for "not yet written")
                }
            }
            Err(e) => {
                error!(
                    "recording: QI to ID3D12GraphicsCommandList2 failed: {} — no marker protection",
                    e
                );
            }
        }
    }

    // Advance parity for next frame
    state.parity = (state.parity + 1) % 3;
}
