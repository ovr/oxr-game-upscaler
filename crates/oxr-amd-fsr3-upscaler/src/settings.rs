use std::path::PathBuf;
use std::sync::OnceLock;

use ini::Ini;
use tracing::info;

use crate::logging;

pub struct Settings {
    pub recording_path: PathBuf,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

pub fn init() {
    SETTINGS.get_or_init(load);
}

pub fn get() -> &'static Settings {
    SETTINGS.get_or_init(load)
}

fn load() -> Settings {
    let dll_dir = logging::dll_directory().unwrap_or_else(|| PathBuf::from("."));
    let ini_path = dll_dir.join("oxr.ini");
    let default_recording = dll_dir.join("recordings");

    let recording_path = match Ini::load_from_file_opt(
        &ini_path,
        ini::ParseOption {
            enabled_escape: false,
            ..Default::default()
        },
    ) {
        Ok(ini) => ini
            .section(Some("recording"))
            .and_then(|s| s.get("path"))
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .unwrap_or(default_recording),
        Err(_) => {
            info!("settings: oxr.ini not found, using defaults");
            default_recording
        }
    };

    info!("settings: recording_path = {:?}", recording_path);
    Settings { recording_path }
}
