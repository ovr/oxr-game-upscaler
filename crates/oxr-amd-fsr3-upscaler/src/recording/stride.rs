use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stride {
    Disabled = 0,
    EverySecond = 1,
}

static STRIDE: AtomicU8 = AtomicU8::new(Stride::EverySecond as u8);

pub fn get() -> Stride {
    match STRIDE.load(Ordering::Relaxed) {
        1 => Stride::EverySecond,
        _ => Stride::Disabled,
    }
}

pub fn set(s: Stride) {
    STRIDE.store(s as u8, Ordering::Relaxed);
}
