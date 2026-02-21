use fsr_sys::*;
use tracing::{info, warn};

pub unsafe fn handle_query(
    _context: *mut ffxContext,
    desc: *mut ffxQueryDescHeader,
) -> ffxReturnCode_t {
    if desc.is_null() {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    let type_ = (*desc).type_;

    match type_ {
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE => {
            query_upscale_ratio(desc)
        }
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE => {
            query_render_resolution(desc)
        }
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT => query_jitter_phase_count(desc),
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET => query_jitter_offset(desc),
        _ => {
            warn!(type_ = type_, "ffxQuery: unknown descriptor type");
            FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE
        }
    }
}

/// Standard FSR upscale ratios per quality mode.
fn upscale_ratio_for_mode(quality_mode: u32) -> f32 {
    match quality_mode {
        FFX_UPSCALE_QUALITY_MODE_NATIVEAA => 1.0,
        FFX_UPSCALE_QUALITY_MODE_QUALITY => 1.5,
        FFX_UPSCALE_QUALITY_MODE_BALANCED => 1.7,
        FFX_UPSCALE_QUALITY_MODE_PERFORMANCE => 2.0,
        FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE => 3.0,
        _ => 1.0,
    }
}

unsafe fn query_upscale_ratio(desc: *mut ffxQueryDescHeader) -> ffxReturnCode_t {
    let d = &*(desc as *const ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode);
    let ratio = upscale_ratio_for_mode(d.quality_mode);

    info!(
        quality_mode = d.quality_mode,
        ratio = ratio,
        "ffxQuery: GetUpscaleRatioFromQualityMode"
    );

    if !d.p_out_upscale_ratio.is_null() {
        *d.p_out_upscale_ratio = ratio;
    }

    FFX_API_RETURN_OK
}

unsafe fn query_render_resolution(desc: *mut ffxQueryDescHeader) -> ffxReturnCode_t {
    let d = &*(desc as *const ffxQueryDescUpscaleGetRenderResolutionFromQualityMode);
    let ratio = upscale_ratio_for_mode(d.quality_mode);

    let render_w = (d.display_width as f32 / ratio).round() as u32;
    let render_h = (d.display_height as f32 / ratio).round() as u32;

    info!(
        display = format_args!("{}x{}", d.display_width, d.display_height),
        quality_mode = d.quality_mode,
        render = format_args!("{}x{}", render_w, render_h),
        "ffxQuery: GetRenderResolutionFromQualityMode"
    );

    if !d.p_out_render_width.is_null() {
        *d.p_out_render_width = render_w;
    }
    if !d.p_out_render_height.is_null() {
        *d.p_out_render_height = render_h;
    }

    FFX_API_RETURN_OK
}

unsafe fn query_jitter_phase_count(desc: *mut ffxQueryDescHeader) -> ffxReturnCode_t {
    let d = &*(desc as *const ffxQueryDescUpscaleGetJitterPhaseCount);

    // Standard FSR formula: ceil(8 * ratio^2) where ratio = displayWidth / renderWidth
    let ratio = if d.render_width > 0 {
        d.display_width as f32 / d.render_width as f32
    } else {
        1.0
    };
    let phase_count = (8.0 * ratio * ratio).ceil() as i32;

    info!(
        render_width = d.render_width,
        display_width = d.display_width,
        phase_count = phase_count,
        "ffxQuery: GetJitterPhaseCount"
    );

    if !d.p_out_phase_count.is_null() {
        *d.p_out_phase_count = phase_count;
    }

    FFX_API_RETURN_OK
}

unsafe fn query_jitter_offset(desc: *mut ffxQueryDescHeader) -> ffxReturnCode_t {
    let d = &*(desc as *const ffxQueryDescUpscaleGetJitterOffset);

    let (x, y) = halton_jitter(d.index, d.phase_count);

    info!(
        index = d.index,
        phase_count = d.phase_count,
        x = x,
        y = y,
        "ffxQuery: GetJitterOffset"
    );

    if !d.p_out_x.is_null() {
        *d.p_out_x = x;
    }
    if !d.p_out_y.is_null() {
        *d.p_out_y = y;
    }

    FFX_API_RETURN_OK
}

/// Halton sequence jitter (bases 2 and 3), centered around 0.
fn halton_jitter(index: i32, phase_count: i32) -> (f32, f32) {
    if phase_count <= 0 {
        return (0.0, 0.0);
    }
    let i = ((index % phase_count) + phase_count) as u32;
    let x = halton(i + 1, 2) - 0.5;
    let y = halton(i + 1, 3) - 0.5;
    (x, y)
}

/// Halton sequence value for a given index and base.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let inv_base = 1.0 / base as f32;

    while index > 0 {
        f *= inv_base;
        r += f * (index % base) as f32;
        index /= base;
    }

    r
}
