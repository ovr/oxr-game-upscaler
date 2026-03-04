use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stride {
    Disabled = 0,
    EverySecond = 1,
    Burst8 = 2,
}

/// Burst8: record 8 frames, skip for ~3 seconds.
/// At 60fps, 3s ≈ 180 frames. Total cycle = 8 + 180 = 188.
const BURST_RECORD: u64 = 8;
const BURST_SKIP: u64 = 180;
const BURST_CYCLE: u64 = BURST_RECORD + BURST_SKIP;

pub fn should_skip_burst(stride_counter: u64) -> bool {
    (stride_counter % BURST_CYCLE) >= BURST_RECORD
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
