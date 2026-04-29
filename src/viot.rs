//! VIoT is intentionally disabled for now.
//! Backup of the previous implementation is kept at `src/backup/viot.rs.bak`.

use std::sync::Mutex;
use std::time::Duration;

pub struct ViotCommand {
    pub label: &'static str,
    pub phrases: &'static [&'static str],
    pub anchors: &'static [&'static str],
    pub endpoint: &'static str,
}

pub static VIOT_COMMANDS: &[ViotCommand] = &[];

pub struct ViotMatch {
    pub label: &'static str,
    pub endpoint: &'static str,
    pub score: f32,
    pub matched_input: String,
}

pub struct ViotController {
    esp_base_url: Mutex<String>,
    pub threshold: f32,
    pub cooldown: Duration,
}

impl ViotController {
    pub fn new(esp_base_url: &str) -> Self {
        Self {
            esp_base_url: Mutex::new(esp_base_url.trim_end_matches('/').to_string()),
            threshold: 0.58,
            cooldown: Duration::from_millis(600),
        }
    }

    pub fn set_esp_base_url(&self, url: &str) {
        if let Ok(mut lock) = self.esp_base_url.lock() {
            *lock = url.trim_end_matches('/').to_string();
        }
    }

    pub fn esp_base_url(&self) -> String {
        self.esp_base_url
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "http://192.168.71.1".to_string())
    }

    /// VIoT matching is disabled in this backup mode.
    pub fn check(&self, _transcript: &str) -> Option<ViotMatch> {
        None
    }

    /// VIoT execution is disabled in this backup mode.
    pub fn execute(&self, _m: &ViotMatch) {}
}

/// Public helper kept for FFI compatibility.
/// Returns an error while VIoT is disabled.
pub fn http_get_pub(_url: &str) -> Result<u16, String> {
    Err("viot is disabled (backup mode)".to_string())
}
