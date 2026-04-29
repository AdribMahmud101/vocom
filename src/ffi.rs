use flutter_rust_bridge::frb;
use crate::frb_generated::RustOpaque;

use crate::config::EngineConfig;
use crate::engine::{BargeInTelemetry, BargeInTransitionEntry};
pub use crate::engine_handle::{EngineCommand, EngineEvent, VocomEngineHandle};
use crate::realtime_pipeline::TranscriptEvent;
pub use crate::viot::ViotController;

#[derive(Clone, Debug)]
pub struct FfiBargeInTransitionEntry {
    pub at_ms: u64,
    pub from: String,
    pub to: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct FfiBargeInTelemetry {
    pub requested: u64,
    pub rejected_low_rms: u64,
    pub rejected_render_ratio: u64,
    pub rejected_persistence: u64,
    pub rejected_confidence: u64,
    pub ducked: u64,
    pub stopped: u64,
    pub latency_samples: usize,
    pub latency_p50_ms: Option<u64>,
    pub latency_p95_ms: Option<u64>,
    pub rolling_requested: u64,
    pub rolling_rejected_low_rms: u64,
    pub rolling_rejected_render_ratio: u64,
    pub rolling_rejected_persistence: u64,
    pub rolling_rejected_confidence: u64,
    pub rolling_ducked: u64,
    pub rolling_stopped: u64,
    pub rolling_latency_samples: usize,
    pub rolling_latency_p50_ms: Option<u64>,
    pub rolling_latency_p95_ms: Option<u64>,
    pub state: String,
    pub recent_transitions: Vec<FfiBargeInTransitionEntry>,
}

#[derive(Clone, Debug)]
pub enum FfiEngineEvent {
    BootStateChanged(String),
    TranscriptReady(TranscriptEvent),
    BargeInTransition(String),
    TelemetrySnapshot(FfiBargeInTelemetry),
    DuplexModeChanged(String),
    Error(String),
    Shutdown,
}

#[frb]
pub fn engine_start(config_json: String) -> Result<RustOpaque<VocomEngineHandle>, String> {
    let config: EngineConfig = serde_json::from_str(&config_json)
        .map_err(|err| format!("config parse failed: {err}"))?;
    VocomEngineHandle::start(config)
        .map(RustOpaque::new)
        .map_err(|err| err.to_string())
}

#[frb]
pub fn engine_poll_event(handle: &RustOpaque<VocomEngineHandle>) -> Option<FfiEngineEvent> {
    handle.poll_event().map(map_engine_event)
}

#[frb]
pub fn engine_send_command(
    handle: &RustOpaque<VocomEngineHandle>,
    cmd: EngineCommand,
) -> Result<(), String> {
    handle.send_command(cmd).map_err(|err| err.to_string())
}

#[frb]
pub fn engine_is_running(handle: &RustOpaque<VocomEngineHandle>) -> bool {
    handle.is_running()
}

#[frb]
pub fn engine_shutdown(handle: &RustOpaque<VocomEngineHandle>) -> Result<(), String> {
    handle
        .request_shutdown_blocking(std::time::Duration::from_secs(8))
        .map_err(|err| err.to_string())
}

fn map_engine_event(event: EngineEvent) -> FfiEngineEvent {
    match event {
        EngineEvent::BootStateChanged(value) => FfiEngineEvent::BootStateChanged(value),
        EngineEvent::TranscriptReady(value) => FfiEngineEvent::TranscriptReady(value),
        EngineEvent::BargeInTransition(value) => FfiEngineEvent::BargeInTransition(value),
        EngineEvent::TelemetrySnapshot(value) => {
            FfiEngineEvent::TelemetrySnapshot(map_barge_in_telemetry(value))
        }
        EngineEvent::DuplexModeChanged(value) => FfiEngineEvent::DuplexModeChanged(value),
        EngineEvent::Error(value) => FfiEngineEvent::Error(value),
        EngineEvent::Shutdown => FfiEngineEvent::Shutdown,
    }
}

fn map_barge_in_telemetry(value: BargeInTelemetry) -> FfiBargeInTelemetry {
    FfiBargeInTelemetry {
        requested: value.requested,
        rejected_low_rms: value.rejected_low_rms,
        rejected_render_ratio: value.rejected_render_ratio,
        rejected_persistence: value.rejected_persistence,
        rejected_confidence: value.rejected_confidence,
        ducked: value.ducked,
        stopped: value.stopped,
        latency_samples: value.latency_samples,
        latency_p50_ms: value.latency_p50_ms,
        latency_p95_ms: value.latency_p95_ms,
        rolling_requested: value.rolling_requested,
        rolling_rejected_low_rms: value.rolling_rejected_low_rms,
        rolling_rejected_render_ratio: value.rolling_rejected_render_ratio,
        rolling_rejected_persistence: value.rolling_rejected_persistence,
        rolling_rejected_confidence: value.rolling_rejected_confidence,
        rolling_ducked: value.rolling_ducked,
        rolling_stopped: value.rolling_stopped,
        rolling_latency_samples: value.rolling_latency_samples,
        rolling_latency_p50_ms: value.rolling_latency_p50_ms,
        rolling_latency_p95_ms: value.rolling_latency_p95_ms,
        state: value.state.to_string(),
        recent_transitions: value
            .recent_transitions
            .into_iter()
            .map(map_transition)
            .collect(),
    }
}

fn map_transition(entry: BargeInTransitionEntry) -> FfiBargeInTransitionEntry {
    FfiBargeInTransitionEntry {
        at_ms: entry.at_ms,
        from: entry.from.to_string(),
        to: entry.to.to_string(),
        reason: entry.reason.to_string(),
    }
}

// ── VIoT FFI ─────────────────────────────────────────────────────────────────

/// Result returned to Flutter when a VIoT command is matched.
#[derive(Clone, Debug)]
pub struct FfiViotMatch {
    /// Human-readable label, e.g. "Room Light ON".
    pub label: String,
    /// ESP32 HTTP path, e.g. "/api/light1/on".
    pub endpoint: String,
    /// Blended confidence score [0.0, 1.0].
    pub score: f32,
    /// The raw transcript text that triggered the match.
    pub matched_input: String,
}

// Global singleton — initialised once via `viot_init`, then shared immutably.
// ViotController uses interior Mutex for debounce state so it is Sync.
static VIOT: std::sync::OnceLock<ViotController> = std::sync::OnceLock::new();

/// Initialise the global VIoT controller.
///
/// Call this once at app startup with your ESP32 base URL
/// (e.g. `"http://192.168.71.1"`). Subsequent calls are ignored.
///
/// Returns `true` on first call (controller was created),
/// `false` if already initialised.
#[frb]
pub fn viot_init(esp_base_url: String) -> bool {
    let result = VIOT.set(ViotController::new(&esp_base_url)).is_ok();
    if !result {
        if let Some(ctrl) = VIOT.get() {
            ctrl.set_esp_base_url(&esp_base_url);
        }
    }
    true
}

/// Check a transcript against all registered VIoT commands.
///
/// Returns `Some(FfiViotMatch)` when a command matches above threshold,
/// or `None` if no command matched, cooldown is active, or `viot_init`
/// was never called.
#[frb]
pub fn viot_check(transcript: String) -> Option<FfiViotMatch> {
    let ctrl = VIOT.get()?;
    ctrl.check(&transcript).map(|m| FfiViotMatch {
        label: m.label.to_string(),
        endpoint: m.endpoint.to_string(),
        score: m.score,
        matched_input: m.matched_input,
    })
}

/// Execute a matched VIoT command — fires an HTTP GET to the ESP32.
///
/// The HTTP call runs on a background thread and never blocks the UI or
/// audio loop. Safe to call from the Flutter event loop directly.
#[frb]
pub fn viot_execute(matched: FfiViotMatch) {
    let base_url = VIOT
        .get()
        .map(|c| c.esp_base_url().to_string())
        .unwrap_or_else(|| "http://192.168.71.1".to_string());

    let url = format!("{}{}", base_url, matched.endpoint);
    let label = matched.label.clone();
    std::thread::Builder::new()
        .name(format!("viot-ffi-{label}"))
        .spawn(move || {
            use crate::viot::http_get_pub;
            match http_get_pub(&url) {
                Ok(status) => println!("[viot-ffi] ESP32 HTTP {status} for {url}"),
                Err(e) => eprintln!("[viot-ffi] ESP32 request failed for {url}: {e}"),
            }
        })
        .ok();
}
