#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalerType {
    Bilinear,
    Lanczos,
}

pub const ACTIVE: UpscalerType = UpscalerType::Lanczos;
