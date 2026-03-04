use std::mem;
use tracing::{error, info};
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList2, ID3D12Resource, D3D12_BOX,
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_READBACK,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_SUBRESOURCE_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN,
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER,
};
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::dispatch;
use crate::gpu_pipeline;

/// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT = 256
const PITCH_ALIGNMENT: u32 = 256;

/// Texture slots we readback.
#[derive(Clone, Copy, Debug)]
pub enum Slot {
    Color = 0,
    Depth = 1,
    MotionVectors = 2,
}

const NUM_SLOTS: usize = 3;
const NUM_PARITY: usize = 8;

/// Info about a readback buffer's layout.
#[derive(Clone, Debug)]
pub struct BufferInfo {
    pub width: u32,
    pub height: u32,
    pub dxgi_format: DXGI_FORMAT,
    pub row_pitch: u32,
    pub bpp: u32,
}

/// Readback pool: 3 slots × 8 parities (4 for normal, 8 for Burst8 mode).
pub struct ReadbackPool {
    buffers: [[Option<ID3D12Resource>; NUM_PARITY]; NUM_SLOTS],
    infos: [[Option<BufferInfo>; NUM_PARITY]; NUM_SLOTS],
    /// Small readback buffer holding one u32 per parity slot, used as a GPU completion marker.
    marker_buffer: Option<ID3D12Resource>,
    marker_gpu_va: u64,
}

impl ReadbackPool {
    pub fn new() -> Self {
        Self {
            buffers: Default::default(),
            infos: Default::default(),
            marker_buffer: None,
            marker_gpu_va: 0,
        }
    }

    /// Create the marker buffer if it doesn't exist yet. Returns true on success.
    pub unsafe fn ensure_marker_buffer(&mut self, device: &ID3D12Device) -> bool {
        if self.marker_buffer.is_some() {
            return true;
        }

        // 3 × u32 = 12 bytes, but D3D12 buffers need 256-byte alignment minimum
        let size: u64 = 256;

        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let mut resource: Option<ID3D12Resource> = None;
        if let Err(e) = device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &buf_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut resource,
        ) {
            error!(
                "readback: marker buffer CreateCommittedResource failed: {}",
                e
            );
            return false;
        }

        let res = match resource {
            Some(r) => r,
            None => return false,
        };

        self.marker_gpu_va = res.GetGPUVirtualAddress();
        info!(
            "readback: created marker buffer, gpu_va=0x{:x}",
            self.marker_gpu_va
        );
        self.marker_buffer = Some(res);
        true
    }

    /// Record a GPU-side write of `value` to `marker[parity]` using WriteBufferImmediate
    /// with MARKER_IN mode (waits for all preceding GPU work to complete).
    pub unsafe fn write_marker(
        &self,
        cmd_list: &ID3D12GraphicsCommandList2,
        parity: usize,
        value: u32,
    ) {
        if self.marker_buffer.is_none() || self.marker_gpu_va == 0 {
            return;
        }

        let param = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: self.marker_gpu_va + (parity as u64) * 4,
            Value: value,
        };
        let mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN;
        cmd_list.WriteBufferImmediate(1, &param, Some(&mode));
    }

    /// Map the marker buffer and read the u32 at `parity` offset. Returns None on failure.
    pub unsafe fn read_marker(&self, parity: usize) -> Option<u32> {
        let buf = self.marker_buffer.as_ref()?;

        let mut mapped: *mut u8 = std::ptr::null_mut();
        if let Err(e) = buf.Map(0, None, Some(&mut mapped as *mut *mut u8 as *mut *mut _)) {
            error!("readback: marker Map failed: {}", e);
            return None;
        }

        let value = std::ptr::read_unaligned(mapped.add(parity * 4) as *const u32);
        buf.Unmap(0, None);
        Some(value)
    }

    /// Ensure a readback buffer exists for the given slot+parity with matching dimensions.
    /// Creates or recreates as needed. Returns false on failure.
    pub unsafe fn ensure_buffer(
        &mut self,
        device: &ID3D12Device,
        slot: Slot,
        parity: usize,
        width: u32,
        height: u32,
        dxgi_format: DXGI_FORMAT,
    ) -> bool {
        let s = slot as usize;
        let p = parity;

        // Check if existing buffer matches
        if let Some(info) = &self.infos[s][p] {
            if info.width == width && info.height == height && info.dxgi_format == dxgi_format {
                return true;
            }
        }

        let bpp = dxgi_format_bpp(dxgi_format);
        if bpp == 0 {
            error!(
                "readback: unsupported format {:?} for slot {:?}",
                dxgi_format, slot
            );
            return false;
        }

        let row_pitch = align_up(width * bpp / 8, PITCH_ALIGNMENT);
        let total_size = row_pitch as u64 * height as u64;

        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: total_size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let mut resource: Option<ID3D12Resource> = None;
        match device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &buf_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut resource,
        ) {
            Ok(()) => {}
            Err(e) => {
                error!(
                    "readback: CreateCommittedResource failed for {:?}[{}]: {}",
                    slot, parity, e
                );
                return false;
            }
        }

        info!(
            "readback: created buffer {:?}[{}] {}x{} format={:?} pitch={} size={}",
            slot, parity, width, height, dxgi_format, row_pitch, total_size
        );

        self.buffers[s][p] = resource;
        self.infos[s][p] = Some(BufferInfo {
            width,
            height,
            dxgi_format,
            row_pitch,
            bpp,
        });
        true
    }

    /// Enqueue CopyTextureRegion from a GPU texture to the readback buffer for slot+parity.
    /// Transitions the source texture to COPY_SOURCE, copies, then restores.
    pub unsafe fn enqueue_copy(
        &self,
        cmd_list: &ID3D12GraphicsCommandList,
        slot: Slot,
        parity: usize,
        source: &ID3D12Resource,
        ffx_state: u32,
    ) {
        let s = slot as usize;
        let p = parity;

        let readback = match &self.buffers[s][p] {
            Some(r) => r,
            None => return,
        };
        let info = match &self.infos[s][p] {
            Some(i) => i,
            None => return,
        };

        // Barrier: source → COPY_SOURCE
        let barrier_before = dispatch::resource_barrier_transition(
            source,
            ffx_state,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        if let Some(b) = &barrier_before {
            cmd_list.ResourceBarrier(&[b.clone()]);
        }

        // CopyTextureRegion: texture → readback buffer (placed footprint)
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::transmute_copy(readback),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: gpu_pipeline::dxgi_typeless_to_typed(info.dxgi_format),
                        Width: info.width,
                        Height: info.height,
                        Depth: 1,
                        RowPitch: info.row_pitch,
                    },
                },
            },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::transmute_copy(source),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };

        let copy_box = D3D12_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: info.width,
            bottom: info.height,
            back: 1,
        };
        cmd_list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, Some(&copy_box));

        // Barrier: restore original state
        let barrier_after = dispatch::resource_barrier_transition_d3d12(
            source,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            dispatch::ffx_state_to_d3d12(ffx_state),
        );
        if let Some(b) = &barrier_after {
            cmd_list.ResourceBarrier(&[b.clone()]);
        }
    }

    /// Map the readback buffer for slot+parity, copy data out row-by-row (stripping pitch
    /// padding), and return the raw bytes plus buffer info. Returns None on failure.
    pub unsafe fn map_and_extract(
        &self,
        slot: Slot,
        parity: usize,
    ) -> Option<(BufferInfo, Vec<u8>)> {
        let s = slot as usize;
        let p = parity;

        let readback = self.buffers[s][p].as_ref()?;
        let info = self.infos[s][p].as_ref()?;

        let row_bytes = (info.width * info.bpp / 8) as usize;
        let total_bytes = row_bytes * info.height as usize;

        let mut mapped: *mut u8 = std::ptr::null_mut();
        if let Err(e) = readback.Map(0, None, Some(&mut mapped as *mut *mut u8 as *mut *mut _)) {
            error!("readback: Map failed for {:?}[{}]: {}", slot, parity, e);
            return None;
        }

        let mut data = Vec::with_capacity(total_bytes);
        if info.row_pitch as usize == row_bytes {
            // No padding — single bulk copy
            std::ptr::copy_nonoverlapping(mapped, data.as_mut_ptr(), total_bytes);
        } else {
            // Strip pitch padding row-by-row
            for row in 0..info.height as usize {
                let src = mapped.add(row * info.row_pitch as usize);
                let dst = data.as_mut_ptr().add(row * row_bytes);
                std::ptr::copy_nonoverlapping(src, dst, row_bytes);
            }
        }
        data.set_len(total_bytes);

        readback.Unmap(0, None);

        Some((info.clone(), data))
    }
}

/// Bytes per pixel for common DXGI formats used by FSR3.
pub fn dxgi_format_bpp(format: DXGI_FORMAT) -> u32 {
    match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => 64,
        DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R32G32B32A32_TYPELESS => 128,
        DXGI_FORMAT_R8G8B8A8_UNORM
        | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        | DXGI_FORMAT_R8G8B8A8_TYPELESS
        | DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        | DXGI_FORMAT_B8G8R8A8_TYPELESS
        | DXGI_FORMAT_R10G10B10A2_UNORM
        | DXGI_FORMAT_R10G10B10A2_TYPELESS
        | DXGI_FORMAT_R11G11B10_FLOAT => 32,
        DXGI_FORMAT_R32_FLOAT
        | DXGI_FORMAT_R32_TYPELESS
        | DXGI_FORMAT_D32_FLOAT
        | DXGI_FORMAT_R32_UINT => 32,
        // Depth-stencil formats (32 bits total per texel)
        DXGI_FORMAT_R24G8_TYPELESS
        | DXGI_FORMAT_D24_UNORM_S8_UINT
        | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
        | DXGI_FORMAT_X24_TYPELESS_G8_UINT => 32,
        // 64-bit depth-stencil
        DXGI_FORMAT_R32G8X24_TYPELESS
        | DXGI_FORMAT_D32_FLOAT_S8X24_UINT
        | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS => 64,
        DXGI_FORMAT_R16G16_FLOAT
        | DXGI_FORMAT_R16G16_TYPELESS
        | DXGI_FORMAT_R16G16_UINT
        | DXGI_FORMAT_R16G16_SINT => 32,
        DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R32G32_TYPELESS => 64,
        DXGI_FORMAT_R16_FLOAT
        | DXGI_FORMAT_R16_TYPELESS
        | DXGI_FORMAT_R16_UNORM
        | DXGI_FORMAT_D16_UNORM => 16,
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_TYPELESS => 8,
        _ => 0,
    }
}

/// Returns true for multi-plane depth-stencil formats where `D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES`
/// is unsafe (planes may be in different states).
pub fn is_depth_stencil_format(format: DXGI_FORMAT) -> bool {
    matches!(
        format,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT
            | DXGI_FORMAT_R32G8X24_TYPELESS
            | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
            | DXGI_FORMAT_D24_UNORM_S8_UINT
            | DXGI_FORMAT_R24G8_TYPELESS
            | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
            | DXGI_FORMAT_X24_TYPELESS_G8_UINT
    )
}

/// COM-refcounted handle for deferred readback on the writer thread.
/// Clones (AddRef's) the ID3D12Resource so the pool can be dropped independently.
pub struct DeferredReadback {
    pub resource: ID3D12Resource,
    pub info: BufferInfo,
}

// ID3D12Resource is Send (free-threaded COM object).
unsafe impl Send for DeferredReadback {}

/// Standalone Map/memcpy/Unmap — same logic as `ReadbackPool::map_and_extract`
/// but operates on a lone COM resource with no pool reference needed.
///
/// # Safety
/// The resource must be a READBACK heap buffer that the GPU has finished writing to.
pub unsafe fn extract_from_resource(rb: &DeferredReadback) -> Option<Vec<u8>> {
    let info = &rb.info;
    let row_bytes = (info.width * info.bpp / 8) as usize;
    let total_bytes = row_bytes * info.height as usize;

    let mut mapped: *mut u8 = std::ptr::null_mut();
    if let Err(e) = rb
        .resource
        .Map(0, None, Some(&mut mapped as *mut *mut u8 as *mut *mut _))
    {
        error!("deferred readback: Map failed: {}", e);
        return None;
    }

    let mut data = Vec::with_capacity(total_bytes);
    if info.row_pitch as usize == row_bytes {
        std::ptr::copy_nonoverlapping(mapped, data.as_mut_ptr(), total_bytes);
    } else {
        for row in 0..info.height as usize {
            let src = mapped.add(row * info.row_pitch as usize);
            let dst = data.as_mut_ptr().add(row * row_bytes);
            std::ptr::copy_nonoverlapping(src, dst, row_bytes);
        }
    }
    data.set_len(total_bytes);

    rb.resource.Unmap(0, None);
    Some(data)
}

impl ReadbackPool {
    /// Clone (AddRef) the COM resource + BufferInfo for deferred readback on the writer thread.
    /// No Map call — the actual CPU readback happens later on the writer thread.
    pub fn get_deferred_readback(&self, slot: Slot, parity: usize) -> Option<DeferredReadback> {
        let s = slot as usize;
        let p = parity;
        let resource = self.buffers[s][p].as_ref()?.clone(); // AddRef
        let info = self.infos[s][p].as_ref()?.clone();
        Some(DeferredReadback { resource, info })
    }
}

fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}
