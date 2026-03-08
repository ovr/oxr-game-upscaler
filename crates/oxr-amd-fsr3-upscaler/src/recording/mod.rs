mod extractor;
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
use extractor::{estimate_slot_bytes, DeferredFramePacket, DeferredTextureData, ExtractorMessage};
use readback::{is_depth_stencil_format, ReadbackPool, Slot};
use writer::{FrameMetadata, FramePacket, TextureData, WriterMessage};

const VK_F10: i32 = 0x79;
const VK_F11: i32 = 0x7A;

/// 40 GiB buffer limit — when queued data exceeds this, recording stops automatically.
pub(crate) const MAX_BUFFER_BYTES: u64 = 40 * 1024 * 1024 * 1024;

pub(crate) static RECORDING_ACTIVE: AtomicBool = AtomicBool::new(false);
pub(crate) static QUEUED_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static QUEUED_FRAMES: AtomicU64 = AtomicU64::new(0);
/// Persists across start/stop so we don't lose rising-edge state when RecorderState is dropped.
static PREV_F10: AtomicBool = AtomicBool::new(false);
static PREV_F11: AtomicBool = AtomicBool::new(false);

struct RecorderState {
    pool: ReadbackPool,
    sender: std::sync::mpsc::Sender<WriterMessage>,
    extractor_sender: std::sync::mpsc::SyncSender<ExtractorMessage>,
    _session_dir: PathBuf,
    frame_number: u64,
    parity: usize,
    warmup_frames: u8,
    /// When true, GPU hasn't caught up — skip post_dispatch copies until marker is ready.
    stalled: bool,
    /// Monotonic counter written by GPU via WriteBufferImmediate.
    write_counter: u32,
    /// Expected marker value per parity slot (set when GPU write is enqueued).
    expected_marker: [u32; 8],
    /// Frame counter for stride logic (increments every dispatch while recording).
    stride_counter: u64,
    /// Set by pre_dispatch when the current frame should be skipped per stride setting.
    skip_this_frame: bool,
    /// Frames remaining before we drop the pool (lets in-flight GPU copies finish).
    drain_frames: u8,
    // --- Burst8 state ---
    /// Monotonic burst group number (incremented at each new burst cycle).
    burst_number: u64,
    /// How many frames have been GPU-captured in the current burst (0..=8).
    burst_captured: u8,
    /// Next burst frame index to drain via CPU readback (0..burst_captured).
    burst_drain_idx: u8,
    /// Saved per-frame metadata for deferred drain.
    burst_metadata: [Option<FrameMetadata>; 8],
    /// Bitmask: bit i set once staging→readback has been enqueued for parity i.
    readback_queued: u8,
    /// Expected readback-done marker value per parity (set when staging→readback is enqueued).
    readback_expected_marker: [u32; 8],
    /// One-shot burst: F11 fires a single burst of 8 frames then auto-stops.
    one_shot: bool,
    /// Timestamp label for one-shot burst filenames (e.g. "20260308_211643").
    one_shot_label: Option<String>,
}

impl RecorderState {
    /// Burst label for filenames: timestamp for one-shot, "burst_NNN" for continuous.
    fn burst_label(&self) -> String {
        if let Some(label) = &self.one_shot_label {
            label.clone()
        } else {
            format!("burst_{:03}", self.burst_number)
        }
    }
}

static RECORDER: Mutex<Option<RecorderState>> = Mutex::new(None);

/// Called before dispatch. Checks hotkey, maps previous frame's readback, sends to writer.
pub unsafe fn pre_dispatch(d: &FfxFsr3UpscalerDispatchDescription) {
    let f10_down = (GetAsyncKeyState(VK_F10) as u16 & 0x8000) != 0;
    let prev = PREV_F10.swap(f10_down, Ordering::Relaxed);
    let toggled = f10_down && !prev;

    let f11_down = (GetAsyncKeyState(VK_F11) as u16 & 0x8000) != 0;
    let prev_f11 = PREV_F11.swap(f11_down, Ordering::Relaxed);
    let f11_toggled = f11_down && !prev_f11;

    let mut guard = match RECORDER.lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    if toggled {
        if RECORDING_ACTIVE.load(Ordering::Relaxed) {
            // Stop recording — defer pool drop to let in-flight GPU copies finish
            RECORDING_ACTIVE.store(false, Ordering::Relaxed);
            if let Some(state) = guard.as_mut() {
                // If mid-burst, drain remaining captured frames before shutdown
                let pending_burst = state.burst_captured.saturating_sub(state.burst_drain_idx);
                if pending_burst > 0 {
                    // Drain up to 2 per dispatch, so ceil(pending/2) + safety margin
                    let drain_dispatches = (pending_burst as u8 + 1) / 2 + 2;
                    info!(
                        "recording: stopping mid-burst ({} captured, {} drained), draining {} frames",
                        state.burst_captured, state.burst_drain_idx, drain_dispatches
                    );
                    state.drain_frames = drain_dispatches;
                } else {
                    info!("recording: stopping (draining 4 frames)");
                    let _ = state.sender.send(WriterMessage::Shutdown);
                    state.drain_frames = 4;
                }
            }
            return;
        } else {
            // Start recording
            let recordings_dir = crate::settings::get().recording_path.clone();
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
            let extractor_sender = extractor::spawn_extractor(sender.clone());

            *guard = Some(RecorderState {
                pool: ReadbackPool::new(),
                sender,
                extractor_sender,
                _session_dir: session_dir.clone(),
                frame_number: 0,
                parity: 0,
                warmup_frames: 3,
                stalled: false,
                write_counter: 1,
                expected_marker: [0; 8],
                stride_counter: 0,
                skip_this_frame: false,
                drain_frames: 0,
                burst_number: 0,
                burst_captured: 0,
                burst_drain_idx: 0,
                burst_metadata: Default::default(),
                readback_queued: 0,
                readback_expected_marker: [0; 8],
                one_shot: false,
                one_shot_label: None,
            });
            QUEUED_BYTES.store(0, Ordering::Relaxed);
            QUEUED_FRAMES.store(0, Ordering::Relaxed);
            RECORDING_ACTIVE.store(true, Ordering::Relaxed);
            info!("recording: started → {}", session_dir.display());
            return;
        }
    }

    // F11: one-shot burst recording (8 frames → auto-stop)
    if f11_toggled && !RECORDING_ACTIVE.load(Ordering::Relaxed) {
        let recordings_dir = crate::settings::get().recording_path.clone();
        let burst_dir = recordings_dir.join("burst8");
        if let Err(e) = std::fs::create_dir_all(&burst_dir) {
            error!("recording: failed to create burst8 dir: {}", e);
            return;
        }

        let sender = writer::spawn_writer(burst_dir.clone());
        let extractor_sender = extractor::spawn_extractor(sender.clone());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Format as YYYYMMDD_HHMMSS (UTC)
        let secs_per_day = 86400u64;
        let secs_per_hour = 3600u64;
        let secs_per_min = 60u64;
        let days = now / secs_per_day;
        let time_of_day = now % secs_per_day;
        let hh = time_of_day / secs_per_hour;
        let mm = (time_of_day % secs_per_hour) / secs_per_min;
        let ss = time_of_day % secs_per_min;
        // Days since epoch → year/month/day
        let (y, m_val, d_val) = epoch_days_to_ymd(days as i64);
        let label = format!(
            "burst_{:04}{:02}{:02}_{:02}{:02}{:02}",
            y, m_val, d_val, hh, mm, ss
        );

        *guard = Some(RecorderState {
            pool: ReadbackPool::new(),
            sender,
            extractor_sender,
            _session_dir: burst_dir.clone(),
            frame_number: 0,
            parity: 0,
            warmup_frames: 3,
            stalled: false,
            write_counter: 1,
            expected_marker: [0; 8],
            stride_counter: 0,
            skip_this_frame: false,
            drain_frames: 0,
            burst_number: 0,
            burst_captured: 0,
            burst_drain_idx: 0,
            burst_metadata: Default::default(),
            readback_queued: 0,
            readback_expected_marker: [0; 8],
            one_shot: true,
            one_shot_label: Some(label),
        });
        QUEUED_BYTES.store(0, Ordering::Relaxed);
        QUEUED_FRAMES.store(0, Ordering::Relaxed);
        RECORDING_ACTIVE.store(true, Ordering::Relaxed);
        info!(
            "recording: one-shot burst started → {}",
            burst_dir.display()
        );
        return;
    }

    // If recording, map previous frame's readback and send to writer
    if !RECORDING_ACTIVE.load(Ordering::Relaxed) {
        // Drain: keep pool alive for a few frames so in-flight GPU copies complete
        if let Some(state) = guard.as_mut() {
            if state.drain_frames > 0 {
                // Drain remaining burst frames before final shutdown
                if state.burst_drain_idx < state.burst_captured {
                    let mut drained = 0u8;
                    while state.burst_drain_idx < state.burst_captured && drained < 2 {
                        let idx = state.burst_drain_idx as usize;
                        let rb_queued = (state.readback_queued & (1u8 << idx)) != 0;
                        if !rb_queued {
                            break; // staging→readback not yet enqueued by post_dispatch
                        }
                        let rb_expected = state.readback_expected_marker[idx];
                        if rb_expected != 0 {
                            match state.pool.read_readback_marker(idx) {
                                Some(marker) if marker == rb_expected => {}
                                _ => break,
                            }
                        }

                        if let Some(metadata) = state.burst_metadata[idx].take() {
                            let color = state
                                .pool
                                .get_deferred_readback(Slot::Color, idx)
                                .map(|rb| DeferredTextureData { readback: rb });
                            let depth = state
                                .pool
                                .get_deferred_readback(Slot::Depth, idx)
                                .map(|rb| DeferredTextureData { readback: rb });
                            let motion_vectors = state
                                .pool
                                .get_deferred_readback(Slot::MotionVectors, idx)
                                .map(|rb| DeferredTextureData { readback: rb });

                            let estimated_bytes = color
                                .as_ref()
                                .map_or(0, |d| estimate_slot_bytes(&d.readback.info))
                                + depth
                                    .as_ref()
                                    .map_or(0, |d| estimate_slot_bytes(&d.readback.info))
                                + motion_vectors
                                    .as_ref()
                                    .map_or(0, |d| estimate_slot_bytes(&d.readback.info));

                            let packet = DeferredFramePacket {
                                frame_number: state.frame_number,
                                estimated_bytes,
                                burst_number: Some(state.burst_label()),
                                color,
                                depth,
                                motion_vectors,
                                metadata,
                            };

                            info!(
                                "recording: stop-drain burst idx={} frame={} burst={} (deferred)",
                                idx, state.frame_number, state.burst_number
                            );

                            if let Ok(()) = state
                                .extractor_sender
                                .send(ExtractorMessage::Extract(packet))
                            {
                                QUEUED_BYTES.fetch_add(estimated_bytes, Ordering::Relaxed);
                                QUEUED_FRAMES.fetch_add(1, Ordering::Relaxed);
                            }
                            state.frame_number += 1;
                        }
                        state.burst_drain_idx += 1;
                        drained += 1;
                    }

                    // If all burst frames drained, send shutdown via extractor
                    if state.burst_drain_idx >= state.burst_captured {
                        info!("recording: burst drain complete, sending shutdown");
                        let _ = state.extractor_sender.send(ExtractorMessage::Shutdown);
                    }
                }

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

    // Buffer limit check — stop recording when exceeded
    let queued = QUEUED_BYTES.load(Ordering::Relaxed);
    if queued >= MAX_BUFFER_BYTES {
        info!(
            "recording: buffer limit reached ({:.1} GiB), stopping",
            queued as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        RECORDING_ACTIVE.store(false, Ordering::Relaxed);
        if let Some(state) = guard.as_mut() {
            let _ = state.sender.send(WriterMessage::Shutdown);
            state.drain_frames = 4;
        }
        return;
    }

    let current_stride = stride::get();

    // === Burst8 mode: deferred readback ===
    if current_stride == stride::Stride::Burst8 || state.one_shot {
        let pos = stride::burst_position(state.stride_counter);

        // Reset burst state at cycle start
        if pos == 0 {
            if state.burst_captured > 0 {
                state.burst_number += 1;
            }
            state.burst_captured = 0;
            state.burst_drain_idx = 0;
            state.readback_queued = 0;
            for m in state.burst_metadata.iter_mut() {
                *m = None;
            }
        }

        if pos < stride::BURST_RECORD {
            // Phase A — Capture: save metadata, let post_dispatch enqueue GPU copy
            state.burst_metadata[pos as usize] = Some(FrameMetadata {
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
            });
            state.skip_this_frame = false;
            state.stride_counter += 1;
            return; // No CPU readback during capture
        }

        // Phase B — Drain: send deferred readbacks to extractor (up to 2 per dispatch)
        state.skip_this_frame = true;
        let mut drained = 0u8;
        while state.burst_drain_idx < state.burst_captured && drained < 2 {
            let idx = state.burst_drain_idx as usize;

            // Check GPU readback-done marker: staging→readback must be complete before Map.
            let rb_queued = (state.readback_queued & (1u8 << idx)) != 0;
            if !rb_queued {
                // Staging→readback not yet enqueued (post_dispatch handles that); skip for now.
                break;
            }
            let rb_expected = state.readback_expected_marker[idx];
            if rb_expected != 0 {
                match state.pool.read_readback_marker(idx) {
                    Some(marker) if marker == rb_expected => {}
                    _ => {
                        break; // PCIe copy not yet complete, retry next dispatch
                    }
                }
            }

            let metadata = match state.burst_metadata[idx].take() {
                Some(m) => m,
                None => {
                    state.burst_drain_idx += 1;
                    continue;
                }
            };

            let color = state
                .pool
                .get_deferred_readback(Slot::Color, idx)
                .map(|rb| DeferredTextureData { readback: rb });
            let depth = state
                .pool
                .get_deferred_readback(Slot::Depth, idx)
                .map(|rb| DeferredTextureData { readback: rb });
            let motion_vectors = state
                .pool
                .get_deferred_readback(Slot::MotionVectors, idx)
                .map(|rb| DeferredTextureData { readback: rb });

            let estimated_bytes = color
                .as_ref()
                .map_or(0, |d| estimate_slot_bytes(&d.readback.info))
                + depth
                    .as_ref()
                    .map_or(0, |d| estimate_slot_bytes(&d.readback.info))
                + motion_vectors
                    .as_ref()
                    .map_or(0, |d| estimate_slot_bytes(&d.readback.info));

            let packet = DeferredFramePacket {
                frame_number: state.frame_number,
                estimated_bytes,
                burst_number: Some(state.burst_label()),
                color,
                depth,
                motion_vectors,
                metadata,
            };

            info!(
                "recording: burst drain idx={} frame={} burst={} (deferred)",
                idx, state.frame_number, state.burst_number
            );

            match state
                .extractor_sender
                .send(ExtractorMessage::Extract(packet))
            {
                Ok(()) => {
                    QUEUED_BYTES.fetch_add(estimated_bytes, Ordering::Relaxed);
                    QUEUED_FRAMES.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    error!("recording: extractor channel closed, disabling recording");
                    RECORDING_ACTIVE.store(false, Ordering::Relaxed);
                    *guard = None;
                    return;
                }
            }

            state.frame_number += 1;
            state.burst_drain_idx += 1;
            drained += 1;
        }

        // One-shot: auto-stop once all 8 frames have been drained
        if state.one_shot
            && state.burst_drain_idx >= state.burst_captured
            && state.burst_captured == 8
        {
            info!(
                "recording: one-shot burst complete ({} frames), stopping",
                state.burst_captured
            );
            RECORDING_ACTIVE.store(false, Ordering::Relaxed);
            let _ = state.extractor_sender.send(ExtractorMessage::Shutdown);
            state.drain_frames = 4;
            state.stride_counter += 1;
            return;
        }

        state.stride_counter += 1;
        return;
    }

    // === Non-burst modes (Disabled / EverySecond) — original readback-per-frame logic ===
    match current_stride {
        stride::Stride::EverySecond if state.stride_counter % 2 == 1 => {
            state.skip_this_frame = true;
            state.stride_counter += 1;
            return;
        }
        _ => {
            state.skip_this_frame = false;
            state.stride_counter += 1;
        }
    }

    let prev_parity = (state.parity + 4 - 1) % 4;

    // Check GPU completion marker before reading back
    let expected = state.expected_marker[prev_parity];
    if expected != 0 {
        match state.pool.read_marker(prev_parity) {
            Some(marker) if marker == expected => {
                state.stalled = false;
            }
            Some(_) | None => {
                state.stalled = true;
                return;
            }
        }
    }

    // Extract data from previous frame's readback buffers (CPU Map/memcpy — on render thread!)
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
        burst_number: None,
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

    // During drain phase: check if any parities have staging copy done but readback not yet queued.
    if state.skip_this_frame {
        let has_pending = (0..state.burst_captured as usize).any(|idx| {
            let bit = 1u8 << idx;
            let not_queued = (state.readback_queued & bit) == 0;
            let staging_done = state.expected_marker[idx] != 0
                && state.pool.read_marker(idx) == Some(state.expected_marker[idx]);
            not_queued && staging_done
        });
        if !has_pending {
            return;
        }
        // Fall through to acquire cmd_list and flush staging→readback.
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

    // During drain phase: enqueue staging→readback for all parities whose staging copy is done.
    if state.skip_this_frame {
        for idx in 0..state.burst_captured as usize {
            let bit = 1u8 << idx;
            if (state.readback_queued & bit) != 0 {
                continue; // already queued
            }
            let expected = state.expected_marker[idx];
            if expected == 0 || state.pool.read_marker(idx) != Some(expected) {
                continue; // staging copy not yet done
            }
            // Enqueue VRAM→READBACK for all 3 slots
            for slot in [Slot::Color, Slot::Depth, Slot::MotionVectors] {
                state.pool.enqueue_staging_to_readback(&cmd_list, slot, idx);
            }
            // Write readback-done marker (signals CPU that PCIe copy is complete)
            if let Ok(cmd_list2) = cmd_list.cast::<ID3D12GraphicsCommandList2>() {
                state
                    .pool
                    .write_readback_marker(&cmd_list2, idx, state.write_counter);
                state.readback_expected_marker[idx] = state.write_counter;
                state.write_counter = state.write_counter.wrapping_add(1);
                if state.write_counter == 0 {
                    state.write_counter = 1;
                }
            }
            state.readback_queued |= bit;
        }
        return;
    }

    let is_burst = stride::get() == stride::Stride::Burst8 || state.one_shot;
    if is_burst && state.burst_captured >= stride::BURST_RECORD as u8 {
        return; // All burst slots filled — wait for drain
    }
    let parity = if is_burst {
        state.burst_captured as usize
    } else {
        state.parity
    };

    // Helper: borrow resource, get its real D3D12 dims+format, ensure readback buffer, enqueue copy
    let mut copy_slot = |slot: Slot,
                         raw: *mut core::ffi::c_void,
                         ffx_state: u32,
                         ffx_w: u32,
                         ffx_h: u32| {
        if raw.is_null() {
            return;
        }
        if let Some(res) = dispatch::borrow_resource(raw) {
            let desc = res.GetDesc();
            let actual_w = desc.Width as u32;
            let actual_h = desc.Height;
            let eff_w = if ffx_w > 0 { ffx_w } else { actual_w };
            let eff_h = if ffx_h > 0 { ffx_h } else { actual_h };
            if eff_w == 0 || eff_h == 0 {
                return;
            }
            let copy_w = eff_w.min(actual_w);
            let copy_h = eff_h.min(actual_h);
            if is_depth_stencil_format(desc.Format) {
                let readback_fmt = readback::depth_plane_format(desc.Format);
                if state
                    .pool
                    .ensure_buffer(&device, slot, parity, copy_w, copy_h, readback_fmt)
                {
                    state.pool.enqueue_copy_depth_plane(
                        &cmd_list,
                        slot,
                        parity,
                        &res,
                        ffx_state,
                        desc.Format,
                    );
                }
            } else if state
                .pool
                .ensure_buffer(&device, slot, parity, copy_w, copy_h, desc.Format)
            {
                state
                    .pool
                    .enqueue_copy(&cmd_list, slot, parity, &res, ffx_state);
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

    // Advance parity / burst counter
    if is_burst {
        state.burst_captured += 1;
    } else {
        state.parity = (state.parity + 1) % 4;
    }
}

/// Convert days since Unix epoch to (year, month, day). Civil calendar, no leap-second fuss.
fn epoch_days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from Howard Hinnant (public domain).
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
