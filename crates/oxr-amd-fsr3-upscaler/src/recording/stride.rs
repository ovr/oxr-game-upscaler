use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stride {
    Disabled = 0,
    EverySecond = 1,
    Burst8 = 2,
}

/// Burst8: record 8 frames, skip for ~3 seconds.
/// At 60fps, ~3.3s ≈ 200 frames. Total cycle = 8 + 200 = 208.
pub const BURST_RECORD: u64 = 8;
const BURST_SKIP: u64 = 200;
const BURST_CYCLE: u64 = BURST_RECORD + BURST_SKIP;

/// Position within the current burst cycle (0..BURST_CYCLE-1).
pub fn burst_position(stride_counter: u64) -> u64 {
    stride_counter % BURST_CYCLE
}

static STRIDE: AtomicU8 = AtomicU8::new(Stride::Burst8 as u8);

pub fn get() -> Stride {
    match STRIDE.load(Ordering::Relaxed) {
        1 => Stride::EverySecond,
        2 => Stride::Burst8,
        _ => Stride::Disabled,
    }
}

pub fn set(s: Stride) {
    STRIDE.store(s as u8, Ordering::Relaxed);
}
