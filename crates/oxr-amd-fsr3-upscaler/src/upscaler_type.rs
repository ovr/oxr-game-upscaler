use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalerType {
    Bilinear = 0,
    Lanczos = 1,
    SGSR = 2,
    SGSRv2TwoPass = 3,
    SGSRv2 = 4,
}

static ACTIVE: AtomicU8 = AtomicU8::new(UpscalerType::SGSRv2 as u8);

pub fn get() -> UpscalerType {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => UpscalerType::Bilinear,
        2 => UpscalerType::SGSR,
        3 => UpscalerType::SGSRv2TwoPass,
        4 => UpscalerType::SGSRv2,
        _ => UpscalerType::Lanczos,
    }
}

pub fn set(t: UpscalerType) {
    ACTIVE.store(t as u8, Ordering::Relaxed);
}

static RCAS_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn rcas_get() -> bool {
    RCAS_ENABLED.load(Ordering::Relaxed)
}

pub fn rcas_set(on: bool) {
    RCAS_ENABLED.store(on, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiAliasingType {
    None = 0,
    ImbaV0 = 1,
}

static AA_TYPE: AtomicU8 = AtomicU8::new(AntiAliasingType::ImbaV0 as u8);

pub fn aa_get() -> AntiAliasingType {
    match AA_TYPE.load(Ordering::Relaxed) {
        1 => AntiAliasingType::ImbaV0,
        _ => AntiAliasingType::None,
    }
}

pub fn aa_set(t: AntiAliasingType) {
    AA_TYPE.store(t as u8, Ordering::Relaxed);
}

static DEBUG_VIEW: AtomicBool = AtomicBool::new(false);

pub fn debug_view_get() -> bool {
    DEBUG_VIEW.load(Ordering::Relaxed)
}

pub fn debug_view_set(on: bool) {
    DEBUG_VIEW.store(on, Ordering::Relaxed);
}
