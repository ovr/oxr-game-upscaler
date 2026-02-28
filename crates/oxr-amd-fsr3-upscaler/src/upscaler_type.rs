use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalerType {
    Bilinear = 0,
    Lanczos = 1,
}

static ACTIVE: AtomicU8 = AtomicU8::new(UpscalerType::Lanczos as u8);

pub fn get() -> UpscalerType {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => UpscalerType::Bilinear,
        _ => UpscalerType::Lanczos,
    }
}

pub fn set(t: UpscalerType) {
    ACTIVE.store(t as u8, Ordering::Relaxed);
}

impl UpscalerType {
    pub fn next(self) -> Self {
        match self {
            Self::Bilinear => Self::Lanczos,
            Self::Lanczos => Self::Bilinear,
        }
    }
    pub fn prev(self) -> Self {
        match self {
            Self::Bilinear => Self::Lanczos,
            Self::Lanczos => Self::Bilinear,
        }
    }
}

static DEBUG_VIEW: AtomicBool = AtomicBool::new(false);

pub fn debug_view_get() -> bool {
    DEBUG_VIEW.load(Ordering::Relaxed)
}

pub fn debug_view_set(on: bool) {
    DEBUG_VIEW.store(on, Ordering::Relaxed);
}
