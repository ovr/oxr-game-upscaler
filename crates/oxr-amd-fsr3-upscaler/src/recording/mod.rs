mod readback;
pub mod stride;
mod writer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use tracing::{error, info};
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

/// 40 GiB buffer limit — when queued data exceeds this, recording stops automatically.
pub(crate) const MAX_BUFFER_BYTES: u64 = 40 * 1024 * 1024 * 1024;

pub(crate) static RECORDING_ACTIVE: AtomicBool = AtomicBool::new(false);
pub(crate) static QUEUED_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static QUEUED_FRAMES: AtomicU64 = AtomicU64::new(0);
/// Persists across start/stop so we don't lose rising-edge state when RecorderState is dropped.
static PREV_F10: AtomicBool = AtomicBool::new(false);

struct RecorderState {
    pool: ReadbackPool,
    sender: std::sync::mpsc::Sender<WriterMessage>,
    _session_dir: PathBuf,
    frame_number: u64,
    parity: usize,
    warmup_frames: u8,
    /// When true, GPU hasn't caught up — skip post_dispatch copies until marker is ready.
    stalled: bool,
    /// Monotonic counter written by GPU via WriteBufferImmediate.
    write_counter: u32,
    /// Expected marker value per parity slot (set when GPU write is enqueued).
    expected_marker: [u32; 4],
    /// Frame counter for stride logic (increments every dispatch while recording).
    stride_counter: u64,
    /// Set by pre_dispatch when the current frame should be skipped per stride setting.
    skip_this_frame: bool,
    /// Frames remaining before we drop the pool (lets in-flight GPU copies finish).
    drain_frames: u8,
}

static RECORDER: Mutex<Option<RecorderState>> = Mutex::new(None);

/// Called before dispatch. Checks hotkey, maps previous frame's readback, sends to writer.
pub unsafe fn pre_dispatch(d: &FfxFsr3UpscalerDispatchDescription) {
    let f10_down = (GetAsyncKeyState(VK_F10) as u16 & 0x8000) != 0;
    let prev = PREV_F10.swap(f10_down, Ordering::Relaxed);
    let toggled = f10_down && !prev;

    let mut guard = match RECORDER.lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    if toggled {
        if RECORDING_ACTIVE.load(Ordering::Relaxed) {
            // Stop recording — defer pool drop to let in-flight GPU copies finish
            info!("recording: stopping (draining 4 frames)");
            RECORDING_ACTIVE.store(false, Ordering::Relaxed);
            if let Some(state) = guard.as_mut() {
                let _ = state.sender.send(WriterMessage::Shutdown);
                state.drain_frames = 4;
            }
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
                warmup_frames: 3,
                stalled: false,
                write_counter: 1,
                expected_marker: [0; 4],
                stride_counter: 0,
                skip_this_frame: false,
                drain_frames: 0,
            });
            QUEUED_BYTES.store(0, Ordering::Relaxed);
            QUEUED_FRAMES.store(0, Ordering::Relaxed);
            RECORDING_ACTIVE.store(true, Ordering::Relaxed);
            info!("recording: started → {}", session_dir.display());
            return;
        }
    }

    // If recording, map previous frame's readback and send to writer
    if !RECORDING_ACTIVE.load(Ordering::Relaxed) {
        // Drain: keep pool alive for a few frames so in-flight GPU copies complete
        if let Some(state) = guard.as_mut() {
            if state.drain_frames > 0 {
                state.drain_frames -= 1;
                if state.drain_frames == 0 {
                    info!("recording: drain complete, leaking pool (GPU safety)");
                    if let Some(state) = guard.take() {
                        std::mem::forget(state);
                    }
                }
            }
        }
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

    // Stride: skip frames based on configured stride
    match stride::get() {
        stride::Stride::EverySecond if state.stride_counter % 2 == 1 => {
            state.skip_this_frame = true;
            state.stride_counter += 1;
            return;
        }
        stride::Stride::Burst8 if stride::should_skip_burst(state.stride_counter) => {
            state.skip_this_frame = true;
            state.stride_counter += 1;
            return;
        }
        _ => {
            state.skip_this_frame = false;
            state.stride_counter += 1;
        }
    }

    // Buffer limit check — stop recording when exceeded
    let queued = QUEUED_BYTES.load(Ordering::Relaxed);
    if queued >= MAX_BUFFER_BYTES {
        info!(
            "recording: buffer limit reached ({:.1} GiB), stopping (draining 4 frames)",
            queued as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        RECORDING_ACTIVE.store(false, Ordering::Relaxed);
        if let Some(state) = guard.as_mut() {
            let _ = state.sender.send(WriterMessage::Shutdown);
            state.drain_frames = 4;
        }
        return;
    }

    let prev_parity = (state.parity + 1) % 4;

    // Check GPU completion marker before reading back
    let expected = state.expected_marker[prev_parity];
    if expected != 0 {
        match state.pool.read_marker(prev_parity) {
            Some(marker) if marker == expected => {
                state.stalled = false;
            }
            Some(_) | None => {
                // GPU hasn't finished yet — stall: don't read, don't enqueue new copies.
                // Next dispatch will retry the same slot.
                state.stalled = true;
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

    let packet_bytes = color.as_ref().map_or(0, |t| t.data.len() as u64)
        + depth.as_ref().map_or(0, |t| t.data.len() as u64)
        + motion_vectors.as_ref().map_or(0, |t| t.data.len() as u64);

    let packet = FramePacket {
        frame_number: state.frame_number,
        packet_bytes,
        color,
        depth,
        motion_vectors,
        metadata,
    };

    match state.sender.send(WriterMessage::Frame(packet)) {
        Ok(()) => {
            QUEUED_BYTES.fetch_add(packet_bytes, Ordering::Relaxed);
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

    // Don't enqueue new copies while stalled — previous slot hasn't been read yet.
    if state.stalled {
        return;
    }

    // Skip GPU copies when stride says to skip this frame.
    if state.skip_this_frame {
        return;
    }

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
    state.parity = (state.parity + 1) % 4;
}
