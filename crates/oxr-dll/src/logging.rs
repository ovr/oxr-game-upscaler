use std::path::PathBuf;
use std::sync::Once;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

static INIT: Once = Once::new();

/// Leaked guard kept alive for the process lifetime.
static mut GUARD: Option<WorkerGuard> = None;

/// Initialize file-based logging. Log file is placed next to the DLL.
///
/// # Safety
/// Must be called exactly once, from DllMain DLL_PROCESS_ATTACH (single-threaded).
pub unsafe fn init() {
    INIT.call_once(|| {
        let log_dir = dll_directory().unwrap_or_else(|| PathBuf::from("."));
        let file_appender = tracing_appender::rolling::never(&log_dir, "oxr_upscaler.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .with_writer(non_blocking)
                    .with_ansi(false)
                    .with_target(false),
            )
            .init();

        // Leak the guard so the writer stays alive for the process lifetime.
        GUARD = Some(guard);
    });
}

/// Returns the directory containing the loaded DLL.
fn dll_directory() -> Option<PathBuf> {
    use windows::Win32::System::LibraryLoader::GetModuleFileNameW;

    let mut buf = vec![0u16; 512];
    let len = unsafe { GetModuleFileNameW(None, &mut buf) } as usize;
    if len == 0 {
        return None;
    }
    let path = String::from_utf16_lossy(&buf[..len]);
    let path = PathBuf::from(path);
    path.parent().map(|p| p.to_path_buf())
}
