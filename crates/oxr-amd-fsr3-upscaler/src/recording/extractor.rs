use std::sync::atomic::Ordering;
use std::sync::mpsc;
use tracing::{error, info};

use super::readback::{self, BufferInfo, DeferredReadback};
use super::writer::{FrameMetadata, FramePacket, TextureData, WriterMessage};

/// A single texture slot deferred for readback on the extractor thread.
pub struct DeferredTextureData {
    pub readback: DeferredReadback,
}

/// A frame packet where CPU readback is deferred to the extractor thread.
pub struct DeferredFramePacket {
    pub frame_number: u64,
    /// Estimated raw bytes (from BufferInfo: width * height * bpp/8 per slot).
    pub estimated_bytes: u64,
    /// Burst number for Burst8 mode (None for non-burst modes).
    pub burst_number: Option<String>,
    pub color: Option<DeferredTextureData>,
    pub depth: Option<DeferredTextureData>,
    pub motion_vectors: Option<DeferredTextureData>,
    pub metadata: FrameMetadata,
}

pub enum ExtractorMessage {
    Extract(DeferredFramePacket),
    Shutdown,
}

/// Estimate stripped (no padding) byte size for a BufferInfo.
pub fn estimate_slot_bytes(info: &BufferInfo) -> u64 {
    (info.width as u64) * (info.height as u64) * (info.bpp as u64) / 8
}

/// Spawn the extractor thread. Returns sender with capacity 8.
/// The extractor does Map/memcpy/Unmap and forwards FramePackets to writer_tx.
pub fn spawn_extractor(
    writer_tx: mpsc::Sender<WriterMessage>,
) -> mpsc::SyncSender<ExtractorMessage> {
    let (tx, rx) = mpsc::sync_channel::<ExtractorMessage>(8);

    std::thread::Builder::new()
        .name("recording-extractor".into())
        .spawn(move || {
            info!("extractor: thread started");
            extractor_loop(&rx, &writer_tx);
            info!("extractor: thread exiting");
        })
        .expect("failed to spawn extractor thread");

    tx
}

fn extractor_loop(rx: &mpsc::Receiver<ExtractorMessage>, writer_tx: &mpsc::Sender<WriterMessage>) {
    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => {
                info!("extractor: channel closed");
                break;
            }
        };

        match msg {
            ExtractorMessage::Shutdown => {
                info!("extractor: shutdown received, forwarding to writer");
                let _ = writer_tx.send(WriterMessage::Shutdown);
                break;
            }
            ExtractorMessage::Extract(deferred) => {
                let estimated = deferred.estimated_bytes;
                let frame_number = deferred.frame_number;

                let color = extract_slot(deferred.color, "color");
                let depth = extract_slot(deferred.depth, "depth");
                let motion_vectors = extract_slot(deferred.motion_vectors, "mv");

                let packet_bytes = color.as_ref().map_or(0, |t| t.data.len() as u64)
                    + depth.as_ref().map_or(0, |t| t.data.len() as u64)
                    + motion_vectors.as_ref().map_or(0, |t| t.data.len() as u64);

                // Correct the estimate → actual difference in QUEUED_BYTES
                if packet_bytes > estimated {
                    super::QUEUED_BYTES.fetch_add(packet_bytes - estimated, Ordering::Relaxed);
                } else if estimated > packet_bytes {
                    super::QUEUED_BYTES.fetch_sub(estimated - packet_bytes, Ordering::Relaxed);
                }

                let packet = FramePacket {
                    frame_number,
                    packet_bytes,
                    burst_number: deferred.burst_number,
                    color,
                    depth,
                    motion_vectors,
                    metadata: deferred.metadata,
                };

                if let Err(_) = writer_tx.send(WriterMessage::Frame(packet)) {
                    error!("extractor: writer channel closed");
                    break;
                }
            }
        }
    }
}

fn extract_slot(slot: Option<DeferredTextureData>, label: &str) -> Option<TextureData> {
    let deferred = slot?;
    let info = deferred.readback.info.clone();
    match unsafe { readback::extract_from_resource(&deferred.readback) } {
        Some(data) => Some(TextureData { data, info }),
        None => {
            error!("extractor: {} extract failed", label);
            None
        }
    }
}
