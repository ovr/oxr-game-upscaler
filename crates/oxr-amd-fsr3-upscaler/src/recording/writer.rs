use smallvec::smallvec;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;
use tracing::{error, info, warn};
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows::Win32::System::Threading::{
    GetCurrentThread, SetThreadAffinityMask, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
};

use super::readback::BufferInfo;

/// Pre-allocated buffer sizes for EXR compression (avoids reallocs for typical frames).
const EXR_COLOR_BUF_CAPACITY: usize = 24 * 1024 * 1024;
const EXR_MV_BUF_CAPACITY: usize = 6 * 1024 * 1024;
const EXR_DEPTH_BUF_CAPACITY: usize = 6 * 1024 * 1024;

/// Allocate a `Vec<T>` of the given length without zeroing memory.
///
/// # Safety
/// Caller must write every element before reading.
#[inline]
unsafe fn vec_uninit<T>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    unsafe { v.set_len(len) };
    v
}

/// Pixel data extracted from a readback buffer.
pub struct TextureData {
    pub data: Vec<u8>,
    pub info: BufferInfo,
}

/// Per-frame metadata from the dispatch descriptor.
pub struct FrameMetadata {
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov: f32,
    pub frame_time_delta: f32,
    pub render_width: u32,
    pub render_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub motion_vector_scale_x: f32,
    pub motion_vector_scale_y: f32,
    pub pre_exposure: f32,
    pub view_space_to_meters_factor: f32,
    pub reset: bool,
}

/// A frame packet sent to the writer thread.
pub struct FramePacket {
    pub frame_number: u64,
    /// Total raw bytes of texture data in this packet (for buffer accounting).
    pub packet_bytes: u64,
    pub color: Option<TextureData>,
    pub depth: Option<TextureData>,
    pub motion_vectors: Option<TextureData>,
    pub metadata: FrameMetadata,
}

pub enum WriterMessage {
    Frame(FramePacket),
    Shutdown,
}

/// Pin the writer thread to the last logical core and lower its priority
/// so it doesn't compete with game render threads.
fn set_thread_affinity() {
    unsafe {
        let mut sys_info = SYSTEM_INFO::default();
        GetSystemInfo(&mut sys_info);
        let num_cpus = sys_info.dwNumberOfProcessors as usize;
        if num_cpus == 0 {
            warn!("writer: GetSystemInfo returned 0 CPUs, skipping affinity");
            return;
        }

        let mask = 1usize << (num_cpus - 1);
        let thread = GetCurrentThread();

        let prev = SetThreadAffinityMask(thread, mask);
        if prev == 0 {
            warn!("writer: SetThreadAffinityMask failed, continuing without affinity");
        } else {
            info!(
                "writer: pinned to core {} (mask=0x{:x})",
                num_cpus - 1,
                mask
            );
        }

        if let Err(e) = SetThreadPriority(thread, THREAD_PRIORITY_BELOW_NORMAL) {
            warn!("writer: SetThreadPriority failed: {}", e);
        } else {
            info!("writer: priority set to BELOW_NORMAL");
        }
    }
}

/// Spawn the background writer thread. Returns the sender channel.
pub fn spawn_writer(session_dir: PathBuf) -> mpsc::Sender<WriterMessage> {
    let (tx, rx) = mpsc::channel::<WriterMessage>();

    std::thread::Builder::new()
        .name("recording-writer".into())
        .spawn(move || {
            set_thread_affinity();
            info!("writer: thread started, dir={}", session_dir.display());
            writer_loop(&rx, &session_dir);
            info!("writer: thread exiting");
        })
        .expect("failed to spawn writer thread");

    tx
}

fn writer_loop(rx: &mpsc::Receiver<WriterMessage>, session_dir: &PathBuf) {
    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => {
                info!("writer: channel closed");
                break;
            }
        };

        match msg {
            WriterMessage::Shutdown => {
                info!("writer: shutdown received");
                break;
            }
            WriterMessage::Frame(packet) => {
                let bytes = packet.packet_bytes;
                if let Err(e) = write_frame(session_dir, &packet) {
                    error!("writer: frame {} failed: {}", packet.frame_number, e);
                }
                drop(packet);
                super::QUEUED_BYTES.fetch_sub(bytes, Ordering::Relaxed);
                super::QUEUED_FRAMES.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

fn write_frame(session_dir: &PathBuf, packet: &FramePacket) -> Result<(), String> {
    let frame_start = Instant::now();
    let n = packet.frame_number;

    // Determine recorded size from color texture (used for metadata)
    let recorded_size = packet.color.as_ref().and_then(|tex| {
        let w = tex.info.width as usize;
        let h = tex.info.height as usize;
        if should_downscale(w, h) {
            Some((w / 2, h / 2))
        } else {
            None
        }
    });

    // Write color EXR
    if let Some(tex) = &packet.color {
        let path = session_dir.join(format!("frame_{:06}_color.exr", n));
        write_color_exr(&path, tex)?;
    }

    // Write depth EXR
    if let Some(tex) = &packet.depth {
        let path = session_dir.join(format!("frame_{:06}_depth.exr", n));
        write_depth_exr(&path, tex)?;
    }

    // Write motion vectors EXR
    if let Some(tex) = &packet.motion_vectors {
        let path = session_dir.join(format!("frame_{:06}_mv.exr", n));
        write_mv_exr(&path, tex)?;
    }

    // Write metadata JSON
    let path = session_dir.join(format!("frame_{:06}_meta.json", n));
    write_metadata_json(&path, &packet.metadata, recorded_size)?;

    info!(
        "writer: frame {} total={:.1}ms",
        n,
        frame_start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(())
}

/// Convert raw texture data to f16 RGBA pixels based on DXGI format.
/// For R16G16B16A16_FLOAT (the common case), this is a direct copy — no conversion needed.
fn convert_to_rgba_f16(tex: &TextureData) -> Vec<half::f16> {
    let w = tex.info.width as usize;
    let h = tex.info.height as usize;
    let pixel_count = w * h;

    match tex.info.dxgi_format {
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => {
            // 8 bytes per pixel, 4 × f16 — direct copy
            // SAFETY: every element is written in the loop below before being read.
            let mut out = unsafe { vec_uninit::<half::f16>(pixel_count * 4) };
            for i in 0..pixel_count {
                let offset = i * 8;
                for c in 0..4 {
                    let bits = u16::from_le_bytes([
                        tex.data[offset + c * 2],
                        tex.data[offset + c * 2 + 1],
                    ]);
                    out[i * 4 + c] = half::f16::from_bits(bits);
                }
            }
            out
        }
        // All other formats: convert via f32 intermediary
        _ => {
            let f32_data = convert_to_rgba_f32(tex);
            f32_data.into_iter().map(half::f16::from_f32).collect()
        }
    }
}

/// 2x box-filter downscale operating on f16 data.
/// Converts each 2×2 block to f32 for averaging, then back to f16.
fn downscale_2x_f16(
    data: &[half::f16],
    width: usize,
    height: usize,
    channels: usize,
) -> (usize, usize, Vec<half::f16>) {
    let nw = width / 2;
    let nh = height / 2;
    // SAFETY: every element is written in the loop below before being read.
    let mut out = unsafe { vec_uninit::<half::f16>(nw * nh * channels) };

    for y in 0..nh {
        for x in 0..nw {
            let sx = x * 2;
            let sy = y * 2;
            let dst = (y * nw + x) * channels;
            for c in 0..channels {
                let p00 = data[((sy) * width + sx) * channels + c].to_f32();
                let p10 = data[((sy) * width + sx + 1) * channels + c].to_f32();
                let p01 = data[((sy + 1) * width + sx) * channels + c].to_f32();
                let p11 = data[((sy + 1) * width + sx + 1) * channels + c].to_f32();
                out[dst + c] = half::f16::from_f32((p00 + p10 + p01 + p11) * 0.25);
            }
        }
    }

    (nw, nh, out)
}

/// Convert raw texture data to f32 RGBA pixels based on DXGI format.
fn convert_to_rgba_f32(tex: &TextureData) -> Vec<f32> {
    let w = tex.info.width as usize;
    let h = tex.info.height as usize;
    let pixel_count = w * h;
    let mut out = vec![0.0f32; pixel_count * 4];

    match tex.info.dxgi_format {
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => {
            // 8 bytes per pixel, 4 × f16
            for i in 0..pixel_count {
                let offset = i * 8;
                for c in 0..4 {
                    let bits = u16::from_le_bytes([
                        tex.data[offset + c * 2],
                        tex.data[offset + c * 2 + 1],
                    ]);
                    out[i * 4 + c] = half::f16::from_bits(bits).to_f32();
                }
            }
        }
        DXGI_FORMAT_R8G8B8A8_UNORM
        | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        | DXGI_FORMAT_R8G8B8A8_TYPELESS => {
            // 4 bytes per pixel
            for i in 0..pixel_count {
                let offset = i * 4;
                for c in 0..4 {
                    out[i * 4 + c] = tex.data[offset + c] as f32 / 255.0;
                }
            }
        }
        DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        | DXGI_FORMAT_B8G8R8A8_TYPELESS => {
            // 4 bytes per pixel, BGRA → RGBA
            for i in 0..pixel_count {
                let offset = i * 4;
                out[i * 4] = tex.data[offset + 2] as f32 / 255.0; // R
                out[i * 4 + 1] = tex.data[offset + 1] as f32 / 255.0; // G
                out[i * 4 + 2] = tex.data[offset] as f32 / 255.0; // B
                out[i * 4 + 3] = tex.data[offset + 3] as f32 / 255.0; // A
            }
        }
        DXGI_FORMAT_R11G11B10_FLOAT => {
            // 4 bytes per pixel, packed
            for i in 0..pixel_count {
                let offset = i * 4;
                let bits = u32::from_le_bytes([
                    tex.data[offset],
                    tex.data[offset + 1],
                    tex.data[offset + 2],
                    tex.data[offset + 3],
                ]);
                out[i * 4] = unpack_r11(bits & 0x7FF);
                out[i * 4 + 1] = unpack_r11((bits >> 11) & 0x7FF);
                out[i * 4 + 2] = unpack_r10((bits >> 22) & 0x3FF);
                out[i * 4 + 3] = 1.0;
            }
        }
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => {
            for i in 0..pixel_count {
                let offset = i * 4;
                let bits = u32::from_le_bytes([
                    tex.data[offset],
                    tex.data[offset + 1],
                    tex.data[offset + 2],
                    tex.data[offset + 3],
                ]);
                out[i * 4] = (bits & 0x3FF) as f32 / 1023.0;
                out[i * 4 + 1] = ((bits >> 10) & 0x3FF) as f32 / 1023.0;
                out[i * 4 + 2] = ((bits >> 20) & 0x3FF) as f32 / 1023.0;
                out[i * 4 + 3] = ((bits >> 30) & 0x3) as f32 / 3.0;
            }
        }
        DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R32G32B32A32_TYPELESS => {
            // 16 bytes per pixel, 4 × f32
            for i in 0..pixel_count {
                let offset = i * 16;
                for c in 0..4 {
                    out[i * 4 + c] = f32::from_le_bytes([
                        tex.data[offset + c * 4],
                        tex.data[offset + c * 4 + 1],
                        tex.data[offset + c * 4 + 2],
                        tex.data[offset + c * 4 + 3],
                    ]);
                }
            }
        }
        _ => {
            error!(
                "writer: unsupported color format {:?}, filling zeros",
                tex.info.dxgi_format
            );
        }
    }
    out
}

/// Convert raw texture data to single-channel f32 (depth).
fn convert_to_r_f32(tex: &TextureData) -> Vec<f32> {
    use half::slice::HalfFloatSliceExt;

    let w = tex.info.width as usize;
    let h = tex.info.height as usize;
    let pixel_count = w * h;
    let mut out = unsafe { vec_uninit::<f32>(pixel_count) };

    match tex.info.dxgi_format {
        DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32_TYPELESS | DXGI_FORMAT_D32_FLOAT => {
            // x86 LE: raw bytes are already f32 layout — single memcpy
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tex.data.as_ptr(),
                    out.as_mut_ptr() as *mut u8,
                    pixel_count * 4,
                );
            }
        }
        // 24-bit unorm depth + 8-bit stencil packed in 32 bits
        DXGI_FORMAT_R24G8_TYPELESS
        | DXGI_FORMAT_D24_UNORM_S8_UINT
        | DXGI_FORMAT_R24_UNORM_X8_TYPELESS => {
            for i in 0..pixel_count {
                let offset = i * 4;
                let bits = u32::from_le_bytes([
                    tex.data[offset],
                    tex.data[offset + 1],
                    tex.data[offset + 2],
                    tex.data[offset + 3],
                ]);
                // Lower 24 bits are depth (unorm)
                let depth_bits = bits & 0x00FF_FFFF;
                out[i] = depth_bits as f32 / 16_777_215.0; // 2^24 - 1
            }
        }
        DXGI_FORMAT_R16_FLOAT | DXGI_FORMAT_R16_TYPELESS => {
            // Reinterpret &[u8] as &[f16], then batch-convert with SIMD (F16C)
            let src = unsafe {
                std::slice::from_raw_parts(
                    tex.data.as_ptr() as *const half::f16,
                    pixel_count,
                )
            };
            src.convert_to_f32_slice(&mut out);
        }
        DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_D16_UNORM => {
            for i in 0..pixel_count {
                let offset = i * 2;
                let bits = u16::from_le_bytes([tex.data[offset], tex.data[offset + 1]]);
                out[i] = bits as f32 / 65535.0;
            }
        }
        DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R32G32_TYPELESS => {
            // 8 bytes per pixel (2 × f32), take first R32 as depth (strided read)
            for i in 0..pixel_count {
                let offset = i * 8;
                out[i] = f32::from_le_bytes([
                    tex.data[offset],
                    tex.data[offset + 1],
                    tex.data[offset + 2],
                    tex.data[offset + 3],
                ]);
            }
        }
        _ => {
            error!(
                "writer: unsupported depth format {:?}, filling zeros",
                tex.info.dxgi_format
            );
            out.fill(0.0);
        }
    }
    out
}

/// Convert raw texture data to 2-channel f32 (motion vectors).
fn convert_to_rg_f32(tex: &TextureData) -> Vec<f32> {
    use half::slice::HalfFloatSliceExt;

    let w = tex.info.width as usize;
    let h = tex.info.height as usize;
    let pixel_count = w * h;
    let mut out = unsafe { vec_uninit::<f32>(pixel_count * 2) };

    match tex.info.dxgi_format {
        DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_TYPELESS => {
            // Interleaved RG16F → interleaved RG32F: batch f16→f32 with SIMD (F16C)
            let src = unsafe {
                std::slice::from_raw_parts(
                    tex.data.as_ptr() as *const half::f16,
                    pixel_count * 2,
                )
            };
            src.convert_to_f32_slice(&mut out);
        }
        DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R32G32_TYPELESS => {
            // x86 LE: interleaved RG32F = identical layout — single memcpy
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tex.data.as_ptr(),
                    out.as_mut_ptr() as *mut u8,
                    pixel_count * 8,
                );
            }
        }
        _ => {
            error!(
                "writer: unsupported mv format {:?}, filling zeros",
                tex.info.dxgi_format
            );
            out.fill(0.0);
        }
    }
    out
}

fn write_color_exr(path: &PathBuf, tex: &TextureData) -> Result<(), String> {
    let mut w = tex.info.width as usize;
    let mut h = tex.info.height as usize;

    let t0 = Instant::now();
    let mut rgba = convert_to_rgba_f16(tex);
    let convert_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut downscale_ms = 0.0;
    if should_downscale(w, h) {
        let t1 = Instant::now();
        let (nw, nh, scaled) = downscale_2x_f16(&rgba, w, h, 4);
        downscale_ms = t1.elapsed().as_secs_f64() * 1000.0;
        w = nw;
        h = nh;
        rgba = scaled;
    }

    // Split into separate channels
    // SAFETY: every element is written in the loop below before being read.
    let mut r = unsafe { vec_uninit::<half::f16>(w * h) };
    let mut g = unsafe { vec_uninit::<half::f16>(w * h) };
    let mut b = unsafe { vec_uninit::<half::f16>(w * h) };
    let mut a = unsafe { vec_uninit::<half::f16>(w * h) };
    for i in 0..w * h {
        r[i] = rgba[i * 4];
        g[i] = rgba[i * 4 + 1];
        b[i] = rgba[i * 4 + 2];
        a[i] = rgba[i * 4 + 3];
    }

    use exr::prelude::*;

    let channels = AnyChannels::sort(smallvec![
        AnyChannel::new("R", FlatSamples::F16(r)),
        AnyChannel::new("G", FlatSamples::F16(g)),
        AnyChannel::new("B", FlatSamples::F16(b)),
        AnyChannel::new("A", FlatSamples::F16(a)),
    ]);
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("color"),
        Encoding {
            compression: Compression::ZIP16,
            ..Default::default()
        },
        channels,
    );
    let image = Image::from_layer(layer);

    let t2 = Instant::now();
    let mut buf = Vec::with_capacity(EXR_COLOR_BUF_CAPACITY);
    image
        .write()
        .to_buffered(std::io::Cursor::new(&mut buf))
        .map_err(|e| format!("write color EXR: {}", e))?;
    let compress_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    std::fs::write(path, &buf).map_err(|e| format!("write color EXR: {}", e))?;
    let io_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let size_mb = buf.len() as f64 / (1024.0 * 1024.0);
    info!(
        "writer: color convert={:.1}ms downscale={:.1}ms compress={:.1}ms io={:.1}ms ({:.1}MB)",
        convert_ms, downscale_ms, compress_ms, io_ms, size_mb,
    );
    Ok(())
}

fn write_depth_exr(path: &PathBuf, tex: &TextureData) -> Result<(), String> {
    let mut w = tex.info.width as usize;
    let mut h = tex.info.height as usize;

    let t0 = Instant::now();
    let mut depth = convert_to_r_f32(tex);
    let convert_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut downscale_ms = 0.0;
    if should_downscale(w, h) {
        let t1 = Instant::now();
        let (nw, nh, scaled) = downscale_2x(&depth, w, h, 1);
        downscale_ms = t1.elapsed().as_secs_f64() * 1000.0;
        w = nw;
        h = nh;
        depth = scaled;
    }

    // Write single-channel EXR using the low-level API
    use exr::prelude::*;

    let channel = AnyChannel::new("Y", FlatSamples::F32(depth));
    let channels = AnyChannels::sort(smallvec![channel]);
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("depth"),
        Encoding {
            compression: Compression::ZIP16,
            ..Default::default()
        },
        channels,
    );
    let image = Image::from_layer(layer);

    let t2 = Instant::now();
    let mut buf = Vec::with_capacity(EXR_DEPTH_BUF_CAPACITY);
    image
        .write()
        .to_buffered(std::io::Cursor::new(&mut buf))
        .map_err(|e| format!("write depth EXR: {}", e))?;
    let compress_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    std::fs::write(path, &buf).map_err(|e| format!("write depth EXR: {}", e))?;
    let io_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let size_mb = buf.len() as f64 / (1024.0 * 1024.0);
    info!(
        "writer: depth convert={:.1}ms downscale={:.1}ms compress={:.1}ms io={:.1}ms ({:.1}MB)",
        convert_ms, downscale_ms, compress_ms, io_ms, size_mb,
    );
    Ok(())
}

fn write_mv_exr(path: &PathBuf, tex: &TextureData) -> Result<(), String> {
    let mut w = tex.info.width as usize;
    let mut h = tex.info.height as usize;

    let t0 = Instant::now();
    let mut mv = convert_to_rg_f32(tex);
    let convert_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut downscale_ms = 0.0;
    if should_downscale(w, h) {
        // MV values are NOT rescaled — consumer uses render_size + recorded_size from metadata
        let t1 = Instant::now();
        let (nw, nh, scaled) = downscale_2x(&mv, w, h, 2);
        downscale_ms = t1.elapsed().as_secs_f64() * 1000.0;
        w = nw;
        h = nh;
        mv = scaled;
    }

    let mut mv_x = vec![0.0f32; w * h];
    let mut mv_y = vec![0.0f32; w * h];
    for i in 0..w * h {
        mv_x[i] = mv[i * 2];
        mv_y[i] = mv[i * 2 + 1];
    }

    use exr::prelude::*;

    let channels = AnyChannels::sort(smallvec![
        AnyChannel::new("X", FlatSamples::F32(mv_x)),
        AnyChannel::new("Y", FlatSamples::F32(mv_y)),
    ]);
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("motion_vectors"),
        Encoding {
            compression: Compression::ZIP16,
            ..Default::default()
        },
        channels,
    );
    let image = Image::from_layer(layer);

    let t2 = Instant::now();
    let mut buf = Vec::with_capacity(EXR_MV_BUF_CAPACITY);
    image
        .write()
        .to_buffered(std::io::Cursor::new(&mut buf))
        .map_err(|e| format!("write mv EXR: {}", e))?;
    let compress_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    std::fs::write(path, &buf).map_err(|e| format!("write mv EXR: {}", e))?;
    let io_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let size_mb = buf.len() as f64 / (1024.0 * 1024.0);
    info!(
        "writer: mv convert={:.1}ms downscale={:.1}ms compress={:.1}ms io={:.1}ms ({:.1}MB)",
        convert_ms, downscale_ms, compress_ms, io_ms, size_mb,
    );
    Ok(())
}

fn write_metadata_json(
    path: &PathBuf,
    meta: &FrameMetadata,
    recorded_size: Option<(usize, usize)>,
) -> Result<(), String> {
    let downscale_fields = if let Some((rw, rh)) = recorded_size {
        format!(
            ",\n  \"downscaled\": true,\n  \"recorded_size\": [{}, {}]",
            rw, rh
        )
    } else {
        ",\n  \"downscaled\": false".to_string()
    };

    let json = format!(
        r#"{{
  "jitter": [{}, {}],
  "camera_near": {},
  "camera_far": {},
  "camera_fov": {},
  "frame_time_delta": {},
  "render_size": [{}, {}],
  "output_size": [{}, {}],
  "motion_vector_scale": [{}, {}],
  "pre_exposure": {},
  "view_space_to_meters_factor": {},
  "reset": {}{}
}}"#,
        meta.jitter_x,
        meta.jitter_y,
        meta.camera_near,
        meta.camera_far,
        meta.camera_fov,
        meta.frame_time_delta,
        meta.render_width,
        meta.render_height,
        meta.output_width,
        meta.output_height,
        meta.motion_vector_scale_x,
        meta.motion_vector_scale_y,
        meta.pre_exposure,
        meta.view_space_to_meters_factor,
        meta.reset,
        downscale_fields,
    );

    std::fs::write(path, json).map_err(|e| format!("write metadata JSON: {}", e))
}

/// Returns `true` if the texture is larger than 4K and should be downscaled.
fn should_downscale(width: usize, height: usize) -> bool {
    width > 3840 || height > 2160
}

/// 2x box-filter downscale: averages each 2x2 block of pixels.
/// Works for any channel count (1, 2, 4). Odd dimensions use floor division.
fn downscale_2x(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> (usize, usize, Vec<f32>) {
    let nw = width / 2;
    let nh = height / 2;
    let mut out = vec![0.0f32; nw * nh * channels];

    for y in 0..nh {
        for x in 0..nw {
            let sx = x * 2;
            let sy = y * 2;
            let dst = (y * nw + x) * channels;
            for c in 0..channels {
                let p00 = data[((sy) * width + sx) * channels + c];
                let p10 = data[((sy) * width + sx + 1) * channels + c];
                let p01 = data[((sy + 1) * width + sx) * channels + c];
                let p11 = data[((sy + 1) * width + sx + 1) * channels + c];
                out[dst + c] = (p00 + p10 + p01 + p11) * 0.25;
            }
        }
    }

    (nw, nh, out)
}

/// Unpack an 11-bit float (R11G11B10_FLOAT R or G channel) to f32.
/// Format: 5-bit exponent, 6-bit mantissa, no sign bit.
fn unpack_r11(bits: u32) -> f32 {
    let mantissa = bits & 0x3F;
    let exponent = (bits >> 6) & 0x1F;

    if exponent == 0 {
        if mantissa == 0 {
            return 0.0;
        }
        // Denormalized
        return (mantissa as f32 / 64.0) * 2.0f32.powi(-14);
    }
    if exponent == 31 {
        return if mantissa == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        };
    }

    2.0f32.powi(exponent as i32 - 15) * (1.0 + mantissa as f32 / 64.0)
}

/// Unpack a 10-bit float (R11G11B10_FLOAT B channel) to f32.
/// Format: 5-bit exponent, 5-bit mantissa, no sign bit.
fn unpack_r10(bits: u32) -> f32 {
    let mantissa = bits & 0x1F;
    let exponent = (bits >> 5) & 0x1F;

    if exponent == 0 {
        if mantissa == 0 {
            return 0.0;
        }
        return (mantissa as f32 / 32.0) * 2.0f32.powi(-14);
    }
    if exponent == 31 {
        return if mantissa == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        };
    }

    2.0f32.powi(exponent as i32 - 15) * (1.0 + mantissa as f32 / 32.0)
}
