use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalerType {
    Bilinear = 0,
    Lanczos = 1,
    DebugView = 2,
}

static ACTIVE: AtomicU8 = AtomicU8::new(UpscalerType::Lanczos as u8);

pub fn get() -> UpscalerType {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => UpscalerType::Bilinear,
        2 => UpscalerType::DebugView,
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
            Self::Lanczos => Self::DebugView,
            Self::DebugView => Self::Bilinear,
        }
    }
    pub fn prev(self) -> Self {
        match self {
            Self::Bilinear => Self::DebugView,
            Self::Lanczos => Self::Bilinear,
            Self::DebugView => Self::Lanczos,
        }
    }
}
