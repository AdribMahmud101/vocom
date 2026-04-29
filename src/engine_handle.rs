/// Non-blocking engine handle for Flutter SDK integration.
/// 
/// The engine runs on a dedicated background thread and communicates with the
/// main/UI thread via fast message channels. All heavy work (ASR, TTS, audio processing)
/// stays off the UI thread, ensuring smooth Flutter interaction.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{fs::OpenOptions, io::Write, path::PathBuf};

use crossbeam_channel::{
    bounded, unbounded, Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError,
};

use crate::config::EngineConfig;
use crate::config::DuplexMode;
use crate::engine::{VocomEngine, BargeInTelemetry};
use crate::errors::{ErrorClass, VocomError};
use crate::realtime_pipeline::{RealtimePressureSnapshot, TranscriptEvent};
use crate::wakeword::MariaWakewordSpotter;

#[cfg(target_os = "android")]
const ENGINE_STARTUP_TIMEOUT_SECS: u64 = 90;

#[cfg(not(target_os = "android"))]
const ENGINE_STARTUP_TIMEOUT_SECS: u64 = 30;
const MOCK_REPLY_MIN_INTERVAL_MS: u64 = 900;
const ENGINE_RESTART_MAX_ATTEMPTS: u32 = 3;
const ENGINE_RESTART_WINDOW_SECS: u64 = 60;
const LATENCY_ROLLING_WINDOW_SECS: u64 = 60;
const LATENCY_MAX_SAMPLES: usize = 2048;
const PERF_SNAPSHOT_INTERVAL_SECS: u64 = 10;
const PERF_SNAPSHOT_ERROR_LOG_INTERVAL_SECS: u64 = 30;
const SUSTAINED_PRESSURE_ALERT_SECS: u64 = 5;
const PRESSURE_AUDIO_FILL_ALERT_PCT: u8 = 70;
const PRESSURE_ASR_FILL_ALERT_PCT: u8 = 60;

/// Commands sent from Flutter UI to the background engine thread.
#[derive(Clone, Debug)]
pub enum EngineCommand {
    /// Start heavy engine initialization immediately while in wakeword-entrypoint mode.
    StartEngineNow,
    /// Feed transcript text to wakeword gate while engine is not yet initialized.
    SubmitWakewordTranscript { text: String },
    /// Request telemetry snapshot (non-blocking).
    GetTelemetry,
    /// Request a single transcript event poll (non-blocking, returns None if none available).
    PollTranscript,
    /// Ask the engine to synthesize a mock TTS reply for the latest transcript.
    SpeakMockReply,
    /// Ask the engine to synthesize explicit text through native TTS.
    SpeakText { text: String },
    /// Set duplex mode at runtime.
    SetDuplexMode { half_duplex_mute_mic: bool },
    /// Query current duplex mode.
    GetDuplexMode,
    /// Explicitly stop the background engine thread.
    Shutdown,
}

/// Events emitted from the background engine to the Flutter UI.
#[derive(Clone, Debug)]
pub enum EngineEvent {
    /// Engine boot/lifecycle state update.
    BootStateChanged(String),
    /// A new transcript is available.
    TranscriptReady(TranscriptEvent),
    /// Barge-in FSM state changed with action description.
    BargeInTransition(String),
    /// Telemetry snapshot requested via GetTelemetry command.
    TelemetrySnapshot(BargeInTelemetry),
    /// Duplex mode changed or queried.
    DuplexModeChanged(String),
    /// Engine encountered an error.
    Error(String),
    /// Engine shut down gracefully.
    Shutdown,
}

/// Non-blocking interface to the voice communication engine.
/// 
/// All heavy operations (audio capture, ASR inference, TTS synthesis) run on a
/// dedicated background thread. Flutter UI sends commands and receives events
/// without blocking.
pub struct VocomEngineHandle {
    command_tx: Sender<EngineCommand>,
    event_rx: Receiver<EngineEvent>,
    critical_event_rx: Receiver<EngineEvent>,
    worker: Option<JoinHandle<()>>,
    is_running: Arc<AtomicBool>,
    lifecycle: Arc<AtomicU8>,
    shutdown_requested: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum EngineLifecycleState {
    Starting = 0,
    Running = 1,
    Stopping = 2,
    Stopped = 3,
}

impl EngineLifecycleState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Starting,
            1 => Self::Running,
            2 => Self::Stopping,
            3 => Self::Stopped,
            _ => Self::Stopped,
        }
    }
}

#[derive(Debug)]
struct RestartSupervisor {
    attempts_in_window: u32,
    window_started: Instant,
}

impl RestartSupervisor {
    fn new() -> Self {
        Self {
            attempts_in_window: 0,
            window_started: Instant::now(),
        }
    }

    fn allow_attempt(&mut self) -> Option<u32> {
        if self.window_started.elapsed() >= Duration::from_secs(ENGINE_RESTART_WINDOW_SECS) {
            self.window_started = Instant::now();
            self.attempts_in_window = 0;
        }

        if self.attempts_in_window >= ENGINE_RESTART_MAX_ATTEMPTS {
            return None;
        }

        self.attempts_in_window = self.attempts_in_window.saturating_add(1);
        Some(self.attempts_in_window)
    }

    fn reset(&mut self) {
        self.window_started = Instant::now();
        self.attempts_in_window = 0;
    }
}

#[derive(Debug)]
struct RollingLatencyWindow {
    samples: std::collections::VecDeque<(Instant, u64)>,
    max_age: Duration,
    max_samples: usize,
}

impl RollingLatencyWindow {
    fn new(max_age: Duration, max_samples: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(max_samples.min(256)),
            max_age,
            max_samples: max_samples.max(1),
        }
    }

    fn push(&mut self, now: Instant, value_ms: u64) {
        self.samples.push_back((now, value_ms));
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
        self.prune(now);
    }

    fn percentile_pair(&mut self, now: Instant) -> (Option<u64>, Option<u64>) {
        self.prune(now);
        if self.samples.is_empty() {
            return (None, None);
        }
        let mut values: Vec<u64> = self.samples.iter().map(|(_, v)| *v).collect();
        values.sort_unstable();
        (
            percentile_sorted(&values, 0.50),
            percentile_sorted(&values, 0.95),
        )
    }

    fn prune(&mut self, now: Instant) {
        while let Some((at, _)) = self.samples.front() {
            if now.duration_since(*at) > self.max_age {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PressureAlertSnapshot {
    sustained_windows: u64,
    audio_windows: u64,
    asr_windows: u64,
    cpu_windows: u64,
    active: bool,
}

#[derive(Debug)]
struct PressureAlertTracker {
    threshold: Duration,
    audio_since: Option<Instant>,
    asr_since: Option<Instant>,
    cpu_since: Option<Instant>,
    sustained_since: Option<Instant>,
    audio_alerted: bool,
    asr_alerted: bool,
    cpu_alerted: bool,
    sustained_alerted: bool,
    audio_windows: u64,
    asr_windows: u64,
    cpu_windows: u64,
    sustained_windows: u64,
    last_cpu_starvation_events: u64,
}

#[derive(Debug)]
struct PerfSnapshotLogger {
    path: PathBuf,
    interval: Duration,
    next_due: Instant,
    last_error_log_at: Option<Instant>,
}

impl PerfSnapshotLogger {
    fn new() -> Self {
        let path = std::env::var("VOCOM_PERF_SNAPSHOT_FILE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/vocom_perf_snapshots.jsonl"));

        Self {
            path,
            interval: Duration::from_secs(PERF_SNAPSHOT_INTERVAL_SECS),
            next_due: Instant::now() + Duration::from_secs(PERF_SNAPSHOT_INTERVAL_SECS),
            last_error_log_at: None,
        }
    }

    #[cfg(test)]
    fn with_path_and_interval(path: PathBuf, interval: Duration) -> Self {
        Self {
            path,
            interval,
            next_due: Instant::now(),
            last_error_log_at: None,
        }
    }

    fn should_write(&self, now: Instant) -> bool {
        now >= self.next_due
    }

    fn record_written(&mut self, now: Instant) {
        self.next_due = now + self.interval;
    }

    fn maybe_write_line(&mut self, now: Instant, line: &str) {
        if !self.should_write(now) {
            return;
        }
        self.record_written(now);

        if let Err(err) = self.append_line(line) {
            let should_log = self
                .last_error_log_at
                .map(|t| t.elapsed() >= Duration::from_secs(PERF_SNAPSHOT_ERROR_LOG_INTERVAL_SECS))
                .unwrap_or(true);
            if should_log {
                eprintln!(
                    "perf snapshot logger write failed for '{}': {err}",
                    self.path.display()
                );
                self.last_error_log_at = Some(Instant::now());
            }
        }
    }

    fn append_line(&self, line: &str) -> Result<(), VocomError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    VocomError::Stream(format!(
                        "failed to create perf snapshot directory '{}': {e}",
                        parent.display()
                    ))
                })?;
            }
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| {
                VocomError::Stream(format!(
                    "failed to open perf snapshot file '{}': {e}",
                    self.path.display()
                ))
            })?;

        file.write_all(line.as_bytes()).map_err(|e| {
            VocomError::Stream(format!(
                "failed to append perf snapshot file '{}': {e}",
                self.path.display()
            ))
        })?;
        file.write_all(b"\n").map_err(|e| {
            VocomError::Stream(format!(
                "failed to append newline to perf snapshot file '{}': {e}",
                self.path.display()
            ))
        })?;
        Ok(())
    }
}

impl PressureAlertTracker {
    fn new() -> Self {
        Self::with_threshold(Duration::from_secs(SUSTAINED_PRESSURE_ALERT_SECS))
    }

    fn with_threshold(threshold: Duration) -> Self {
        Self {
            threshold,
            audio_since: None,
            asr_since: None,
            cpu_since: None,
            sustained_since: None,
            audio_alerted: false,
            asr_alerted: false,
            cpu_alerted: false,
            sustained_alerted: false,
            audio_windows: 0,
            asr_windows: 0,
            cpu_windows: 0,
            sustained_windows: 0,
            last_cpu_starvation_events: 0,
        }
    }

    fn observe(&mut self, now: Instant, pressure: &RealtimePressureSnapshot) {
        let audio_high = pressure.audio_queue_fill_pct >= PRESSURE_AUDIO_FILL_ALERT_PCT;
        let asr_fill_pct = ((pressure.asr_task_queue_len.saturating_mul(100))
            / pressure.audio_queue_capacity.max(1))
            .min(100) as u8;
        let asr_high = asr_fill_pct >= PRESSURE_ASR_FILL_ALERT_PCT;

        let cpu_high = pressure.cpu_starvation_events > self.last_cpu_starvation_events;
        self.last_cpu_starvation_events = pressure.cpu_starvation_events;

        Self::update_window(
            now,
            self.threshold,
            audio_high,
            &mut self.audio_since,
            &mut self.audio_alerted,
            &mut self.audio_windows,
        );
        Self::update_window(
            now,
            self.threshold,
            asr_high,
            &mut self.asr_since,
            &mut self.asr_alerted,
            &mut self.asr_windows,
        );
        Self::update_window(
            now,
            self.threshold,
            cpu_high,
            &mut self.cpu_since,
            &mut self.cpu_alerted,
            &mut self.cpu_windows,
        );

        let sustained_high = audio_high || asr_high || cpu_high;
        Self::update_window(
            now,
            self.threshold,
            sustained_high,
            &mut self.sustained_since,
            &mut self.sustained_alerted,
            &mut self.sustained_windows,
        );
    }

    fn snapshot(&self) -> PressureAlertSnapshot {
        PressureAlertSnapshot {
            sustained_windows: self.sustained_windows,
            audio_windows: self.audio_windows,
            asr_windows: self.asr_windows,
            cpu_windows: self.cpu_windows,
            active: self.sustained_since.is_some(),
        }
    }

    fn update_window(
        now: Instant,
        threshold: Duration,
        condition: bool,
        since: &mut Option<Instant>,
        alerted: &mut bool,
        windows: &mut u64,
    ) {
        if !condition {
            *since = None;
            *alerted = false;
            return;
        }

        let start = match *since {
            Some(start) => start,
            None => {
                *since = Some(now);
                now
            }
        };

        if !*alerted && now.duration_since(start) >= threshold {
            *windows = windows.saturating_add(1);
            *alerted = true;
        }
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn perf_snapshot_json_line(
    pressure: &RealtimePressureSnapshot,
    alerts: PressureAlertSnapshot,
    loop_tick_ewma_us: u32,
    loop_tick_max_us: u32,
    loop_starved_ticks: u64,
    ingest_p50: Option<u64>,
    ingest_p95: Option<u64>,
    speak_p50: Option<u64>,
    speak_p95: Option<u64>,
) -> String {
    serde_json::json!({
        "ts_ms": now_epoch_ms(),
        "audio_q_len": pressure.audio_queue_len,
        "audio_q_capacity": pressure.audio_queue_capacity,
        "audio_q_fill_pct": pressure.audio_queue_fill_pct,
        "asr_q_len": pressure.asr_task_queue_len,
        "deferred_online_samples": pressure.deferred_online_chunk_samples,
        "denoiser_bypass": pressure.denoiser_bypass_active,
        "shed_silent_chunks": pressure.shed_silent_chunks,
        "coalesced_online_chunks": pressure.coalesced_online_chunks,
        "dropped_offline_segments": pressure.dropped_offline_segments,
        "dropped_input_chunks": pressure.dropped_input_chunks,
        "dropped_input_samples": pressure.dropped_input_samples,
        "overflow_bursts": pressure.overflow_bursts,
        "cpu_total_ewma_us": pressure.cpu_total_ewma_us,
        "cpu_resample_ewma_us": pressure.cpu_resample_ewma_us,
        "cpu_aec_ewma_us": pressure.cpu_aec_ewma_us,
        "cpu_denoise_ewma_us": pressure.cpu_denoise_ewma_us,
        "cpu_vad_ewma_us": pressure.cpu_vad_ewma_us,
        "cpu_over_budget_chunks": pressure.cpu_over_budget_chunks,
        "cpu_starvation_events": pressure.cpu_starvation_events,
        "alerts_sustained_windows": alerts.sustained_windows,
        "alerts_audio_windows": alerts.audio_windows,
        "alerts_asr_windows": alerts.asr_windows,
        "alerts_cpu_windows": alerts.cpu_windows,
        "alerts_active": alerts.active,
        "loop_tick_ewma_us": loop_tick_ewma_us,
        "loop_tick_max_us": loop_tick_max_us,
        "loop_starved_ticks": loop_starved_ticks,
        "latency_ingest_p50_ms": ingest_p50,
        "latency_ingest_p95_ms": ingest_p95,
        "latency_speak_p50_ms": speak_p50,
        "latency_speak_p95_ms": speak_p95
    })
    .to_string()
}

impl VocomEngineHandle {
    #[inline]
    fn emit_event(event_tx: &Sender<EngineEvent>, event: EngineEvent) {
        // Never block engine loop on a saturated UI/event queue.
        let _ = event_tx.try_send(event);
    }

    #[inline]
    fn is_reliable_data_event(event: &EngineEvent) -> bool {
        matches!(
            event,
            EngineEvent::TranscriptReady(_) | EngineEvent::TelemetrySnapshot(_)
        )
    }

    #[inline]
    fn emit_event_reliable_data(
        event_tx: &Sender<EngineEvent>,
        critical_event_tx: &Sender<EngineEvent>,
        event: EngineEvent,
    ) {
        // Prefer the bounded UI queue for normal operation to keep critical
        // lifecycle traffic separate.
        match event_tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(event)) | Err(TrySendError::Disconnected(event)) => {
                // When saturated/disconnected, preserve high-value data-plane
                // events on the unbounded critical channel so transcripts and
                // telemetry snapshots are not silently dropped.
                if Self::is_reliable_data_event(&event) {
                    let _ = critical_event_tx.send(event);
                }
            }
        }
    }

    #[inline]
    fn emit_critical_event(critical_event_tx: &Sender<EngineEvent>, event: EngineEvent) {
        // Critical lifecycle/error events must not be dropped.
        let _ = critical_event_tx.send(event);
    }

    #[inline]
    fn lifecycle_state(lifecycle: &AtomicU8) -> EngineLifecycleState {
        EngineLifecycleState::from_u8(lifecycle.load(Ordering::Acquire))
    }

    #[inline]
    fn set_lifecycle_state(lifecycle: &AtomicU8, state: EngineLifecycleState) {
        lifecycle.store(state as u8, Ordering::Release);
    }

    /// Start the background engine with the given configuration.
    /// 
    /// This function waits for startup confirmation from the background thread
    /// and returns an error immediately if audio/model initialization fails.
    pub fn start(config: EngineConfig) -> Result<Self, VocomError> {
        let (command_tx, command_rx) = bounded::<EngineCommand>(64);
        let (event_tx, event_rx) = bounded::<EngineEvent>(256);
        let (critical_event_tx, critical_event_rx) = unbounded::<EngineEvent>();
        let critical_event_tx_for_start = critical_event_tx.clone();
        let (startup_tx, startup_rx) = bounded::<Result<(), String>>(1);
        let is_running = Arc::new(AtomicBool::new(true));
        let is_running_worker = Arc::clone(&is_running);
        let shutdown_requested = Arc::new(AtomicBool::new(false));
        let shutdown_requested_worker = Arc::clone(&shutdown_requested);
        let lifecycle = Arc::new(AtomicU8::new(EngineLifecycleState::Starting as u8));
        let lifecycle_worker = Arc::clone(&lifecycle);

        let worker = thread::spawn(move || {
            if config.wakeword.entrypoint_enabled {
                let wakeword_cfg = config.wakeword.clone();
                let mut spotter = MariaWakewordSpotter::new_with_targets(
                    Duration::from_millis(wakeword_cfg.cooldown_ms),
                    wakeword_cfg.keyword,
                    wakeword_cfg.variants,
                );

                let mut bootstrap_config = config.clone();
                bootstrap_config.wakeword.entrypoint_enabled = false;
                bootstrap_config.tts.enabled = false;

                let bootstrap_engine = match VocomEngine::start(bootstrap_config) {
                    Ok(engine) => {
                        engine.keep_alive();
                        Some(engine)
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err.to_string()));
                        eprintln!("failed to start bootstrap wakeword listener: {err}");
                        Self::emit_critical_event(
                            &critical_event_tx,
                            EngineEvent::Error(err.to_string()),
                        );
                        return;
                    }
                };

                Self::set_lifecycle_state(&lifecycle_worker, EngineLifecycleState::Running);
                let _ = startup_tx.send(Ok(()));
                Self::emit_critical_event(
                    &critical_event_tx,
                    EngineEvent::BootStateChanged("waiting_for_wakeword".to_string()),
                );

                Self::run_engine_loop(
                    config,
                    None,
                    bootstrap_engine,
                    Some(&mut spotter),
                    command_rx,
                    event_tx.clone(),
                    critical_event_tx.clone(),
                    Arc::clone(&is_running_worker),
                    Arc::clone(&lifecycle_worker),
                    Arc::clone(&shutdown_requested_worker),
                );
            } else {
                match VocomEngine::start(config.clone()) {
                    Ok(engine) => {
                        Self::set_lifecycle_state(&lifecycle_worker, EngineLifecycleState::Running);
                        let _ = startup_tx.send(Ok(()));
                        engine.keep_alive();
                        Self::run_engine_loop(
                            config,
                            Some(engine),
                            None,
                            None,
                            command_rx,
                            event_tx.clone(),
                            critical_event_tx.clone(),
                            Arc::clone(&is_running_worker),
                            Arc::clone(&lifecycle_worker),
                            Arc::clone(&shutdown_requested_worker),
                        );
                    }
                    Err(err) => {
                        Self::set_lifecycle_state(&lifecycle_worker, EngineLifecycleState::Stopped);
                        let _ = startup_tx.send(Err(err.to_string()));
                        eprintln!("failed to start engine: {err}");
                        Self::emit_critical_event(
                            &critical_event_tx,
                            EngineEvent::Error(err.to_string()),
                        );
                    }
                }
            }

            is_running_worker.store(false, Ordering::Release);
            Self::set_lifecycle_state(&lifecycle_worker, EngineLifecycleState::Stopped);
        });

        match startup_rx.recv_timeout(Duration::from_secs(ENGINE_STARTUP_TIMEOUT_SECS)) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                Self::set_lifecycle_state(&lifecycle, EngineLifecycleState::Stopped);
                is_running.store(false, Ordering::Release);
                let _ = worker.join();
                return Err(VocomError::Stream(format!("failed to start engine: {err}")));
            }
            Err(RecvTimeoutError::Timeout) => {
                // Slow model/audio initialization should not be treated as fatal.
                // Keep the worker alive and let startup continue in background.
                let _ = critical_event_tx_for_start.send(EngineEvent::BootStateChanged(
                    format!(
                        "initializing_engine_slow startup_wait_exceeded_{}s",
                        ENGINE_STARTUP_TIMEOUT_SECS
                    ),
                ));
                eprintln!(
                    "engine startup exceeded {}s; continuing initialization in background",
                    ENGINE_STARTUP_TIMEOUT_SECS
                );
            }
            Err(RecvTimeoutError::Disconnected) => {
                Self::set_lifecycle_state(&lifecycle, EngineLifecycleState::Stopped);
                is_running.store(false, Ordering::Release);
                let _ = worker.join();
                return Err(VocomError::Stream(
                    "engine startup failed before reporting status (worker terminated)".to_string(),
                ));
            }
        }

        Ok(Self {
            command_tx,
            event_rx,
            critical_event_rx,
            worker: Some(worker),
            is_running,
            lifecycle,
            shutdown_requested,
        })
    }

    /// Poll for the next engine event without blocking.
    /// 
    /// Returns immediately if an event is available, or None if the queue is empty.
    pub fn poll_event(&self) -> Option<EngineEvent> {
        if let Ok(event) = self.critical_event_rx.try_recv() {
            return Some(event);
        }
        self.event_rx.try_recv().ok()
    }

    /// Poll for events with a timeout.
    pub fn poll_event_timeout(&self, timeout: Duration) -> Option<EngineEvent> {
        if let Ok(event) = self.critical_event_rx.try_recv() {
            return Some(event);
        }
        if let Ok(event) = self.event_rx.try_recv() {
            return Some(event);
        }
        crossbeam_channel::select! {
            recv(self.critical_event_rx) -> msg => msg.ok(),
            recv(self.event_rx) -> msg => msg.ok(),
            default(timeout) => None,
        }
    }

    /// Send a command to the background engine (non-blocking).
    pub fn send_command(&self, cmd: EngineCommand) -> Result<(), VocomError> {
        if matches!(cmd, EngineCommand::Shutdown) {
            self.shutdown_requested.store(true, Ordering::Release);
            Self::set_lifecycle_state(&self.lifecycle, EngineLifecycleState::Stopping);
        }
        match self.command_tx.try_send(cmd) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(VocomError::Stream(
                "engine command queue is full".to_string(),
            )),
            Err(TrySendError::Disconnected(_)) => Err(VocomError::ChannelDisconnected),
        }
    }

    /// Check if the engine is still running.
    pub fn is_running(&self) -> bool {
        Self::lifecycle_state(&self.lifecycle) != EngineLifecycleState::Stopped
    }

    /// Request shutdown from a shared handle without requiring unique ownership.
    pub fn request_shutdown(&self) -> Result<(), VocomError> {
        self.shutdown_requested.store(true, Ordering::Release);
        let current = Self::lifecycle_state(&self.lifecycle);
        if current == EngineLifecycleState::Stopped {
            return Ok(());
        }
        Self::set_lifecycle_state(&self.lifecycle, EngineLifecycleState::Stopping);
        for _ in 0..32 {
            match self.command_tx.try_send(EngineCommand::Shutdown) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(_)) => thread::sleep(Duration::from_millis(5)),
                Err(TrySendError::Disconnected(_)) => return Ok(()),
            }
        }
        Err(VocomError::Stream(
            "failed to enqueue shutdown command (queue remained full)".to_string(),
        ))
    }

    /// Request shutdown and wait until the worker reports it has stopped.
    ///
    /// This path is intended for FFI callers where the handle may be shared
    /// across concurrent Dart/Rust references.
    pub fn request_shutdown_blocking(&self, timeout: Duration) -> Result<(), VocomError> {
        if Self::lifecycle_state(&self.lifecycle) == EngineLifecycleState::Stopped {
            return Ok(());
        }

        self.request_shutdown()?;
        let started = Instant::now();
        while Self::lifecycle_state(&self.lifecycle) != EngineLifecycleState::Stopped {
            if started.elapsed() >= timeout {
                return Err(VocomError::Stream(format!(
                    "engine shutdown timed out after {}ms",
                    timeout.as_millis()
                )));
            }
            thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }

    /// Stop the background engine and request shutdown without blocking caller.
    pub fn shutdown(&mut self) -> Result<(), VocomError> {
        self.shutdown_requested.store(true, Ordering::Release);
        Self::set_lifecycle_state(&self.lifecycle, EngineLifecycleState::Stopping);
        self.is_running.store(false, Ordering::Release);
        let _ = self.command_tx.try_send(EngineCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = thread::Builder::new()
                .name("vocom-engine-shutdown-join".to_string())
                .spawn(move || {
                    if worker.join().is_err() {
                        eprintln!("failed to join engine worker thread during shutdown");
                    }
                });
        }
        Ok(())
    }

    /// Stop the background engine and wait for shutdown (consumes self).
    pub fn shutdown_blocking(mut self) -> Result<(), VocomError> {
        self.shutdown_requested.store(true, Ordering::Release);
        Self::set_lifecycle_state(&self.lifecycle, EngineLifecycleState::Stopping);
        self.is_running.store(false, Ordering::Release);
        let _ = self.command_tx.try_send(EngineCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| {
                VocomError::Stream(
                    "failed to join engine worker thread during shutdown".to_string(),
                )
            })?;
        }
        Ok(())
    }

    // ── Private engine loop ───────────────────────────────────────────────────

    fn run_engine_loop(
        config: EngineConfig,
        mut engine: Option<VocomEngine>,
        mut bootstrap_engine: Option<VocomEngine>,
        mut wakeword_spotter: Option<&mut MariaWakewordSpotter>,
        command_rx: Receiver<EngineCommand>,
        event_tx: Sender<EngineEvent>,
        critical_event_tx: Sender<EngineEvent>,
        is_running: Arc<AtomicBool>,
        lifecycle: Arc<AtomicU8>,
        shutdown_requested: Arc<AtomicBool>,
    ) {
        let mut latest_transcript_text: Option<String> = None;
        let mut last_mock_reply_at: Option<Instant> = None;
        let mut engine_restart_supervisor = RestartSupervisor::new();
        let mut bootstrap_restart_supervisor = RestartSupervisor::new();
        let mut deferred_non_control_commands = std::collections::VecDeque::<EngineCommand>::new();
        let mut loop_tick_last = Instant::now();
        let mut loop_tick_ewma_us: u32 = 0;
        let mut loop_tick_max_us: u32 = 0;
        let mut loop_starved_ticks: u64 = 0;
        let mut shutdown_ack_emitted = false;
        let mut ingest_to_transcript_latency = RollingLatencyWindow::new(
            Duration::from_secs(LATENCY_ROLLING_WINDOW_SECS),
            LATENCY_MAX_SAMPLES,
        );
        let mut transcript_to_speak_latency = RollingLatencyWindow::new(
            Duration::from_secs(LATENCY_ROLLING_WINDOW_SECS),
            LATENCY_MAX_SAMPLES,
        );
        let mut last_transcript_at: Option<Instant> = None;
        let mut next_turn_id: u64 = 1;
        let mut last_turn_id: Option<u64> = None;
        let mut pressure_alerts = PressureAlertTracker::new();
        let mut perf_snapshot_logger = PerfSnapshotLogger::new();

        // Primary loop: listen for transcripts and handle commands.
        loop {
            let now = Instant::now();
            let tick_us = now.duration_since(loop_tick_last).as_micros() as u64;
            loop_tick_last = now;
            loop_tick_ewma_us = ewma_us(loop_tick_ewma_us, tick_us);
            let tick_clamped = tick_us.min(u32::MAX as u64) as u32;
            loop_tick_max_us = loop_tick_max_us.max(tick_clamped);
            if tick_us > 40_000 {
                loop_starved_ticks = loop_starved_ticks.saturating_add(1);
            }

            if let Some(running) = engine.as_ref() {
                let pressure = running.runtime_pressure_snapshot();
                pressure_alerts.observe(now, &pressure);
                let alert_snapshot = pressure_alerts.snapshot();
                let (ingest_p50, ingest_p95) = ingest_to_transcript_latency.percentile_pair(now);
                let (speak_p50, speak_p95) = transcript_to_speak_latency.percentile_pair(now);
                let line = perf_snapshot_json_line(
                    &pressure,
                    alert_snapshot,
                    loop_tick_ewma_us,
                    loop_tick_max_us,
                    loop_starved_ticks,
                    ingest_p50,
                    ingest_p95,
                    speak_p50,
                    speak_p95,
                );
                perf_snapshot_logger.maybe_write_line(now, &line);
            }

            // Check for shutdown flag periodically without blocking indefinitely.
            if !is_running.load(Ordering::Acquire)
                || shutdown_requested.load(Ordering::Acquire)
            {
                Self::set_lifecycle_state(&lifecycle, EngineLifecycleState::Stopping);
                if !shutdown_ack_emitted {
                    Self::emit_critical_event(
                        &critical_event_tx,
                        EngineEvent::BootStateChanged("shutdown_ack".to_string()),
                    );
                    shutdown_ack_emitted = true;
                }
                Self::emit_critical_event(
                    &critical_event_tx,
                    EngineEvent::BootStateChanged("stopping_engine".to_string()),
                );
                Self::emit_critical_event(&critical_event_tx, EngineEvent::Shutdown);
                break;
            }

            // Prefer control-plane commands (shutdown/mode) over data-plane work
            // so lifecycle changes preempt queued background traffic.
            let mut command_disconnected = false;
            let mut prioritized_cmd: Option<EngineCommand> = None;
            for _ in 0..8 {
                match command_rx.try_recv() {
                    Ok(cmd) => {
                        if Self::is_control_command(&cmd) {
                            prioritized_cmd = Some(cmd);
                            break;
                        }
                        if deferred_non_control_commands.len() >= 128 {
                            deferred_non_control_commands.pop_front();
                        }
                        deferred_non_control_commands.push_back(cmd);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        command_disconnected = true;
                        break;
                    }
                }
            }

            if command_disconnected
                && prioritized_cmd.is_none()
                && deferred_non_control_commands.is_empty()
            {
                break;
            }

            let recv_result = if let Some(cmd) = prioritized_cmd {
                Ok(cmd)
            } else if let Some(cmd) = deferred_non_control_commands.pop_front() {
                Ok(cmd)
            } else {
                // Attempt to receive a command with a short timeout so we can check
                // shutdown flag regularly and also poll transcripts.
                command_rx.recv_timeout(Duration::from_millis(20))
            };

            match recv_result {
                Ok(cmd) => {
                    match cmd {
                        EngineCommand::StartEngineNow => {
                            if engine.is_some() {
                                Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                    "engine_already_initialized".to_string(),
                                ));
                                continue;
                            }
                            if shutdown_requested.load(Ordering::Acquire) {
                                continue;
                            }
                            bootstrap_engine = None;
                            Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                "initializing_engine".to_string(),
                            ));
                            let mut full_config = config.clone();
                            full_config.wakeword.entrypoint_enabled = false;
                            match VocomEngine::start(full_config) {
                                Ok(ready_engine) => {
                                    ready_engine.keep_alive();
                                    engine = Some(ready_engine);
                                    wakeword_spotter = None;
                                    Self::emit_critical_event(
                                        &critical_event_tx,
                                        EngineEvent::BootStateChanged(
                                            "engine_initialized".to_string(),
                                        ),
                                    );
                                }
                                Err(err) => {
                                    Self::emit_critical_event(
                                        &critical_event_tx,
                                        EngineEvent::Error(format!(
                                            "engine initialization failed: {err}"
                                        )),
                                    );
                                    Self::emit_critical_event(
                                        &critical_event_tx,
                                        EngineEvent::BootStateChanged(
                                            "waiting_for_wakeword".to_string(),
                                        ),
                                    );
                                }
                            }
                        }
                        EngineCommand::SubmitWakewordTranscript { text } => {
                            if engine.is_some() {
                                continue;
                            }
                            let Some(spotter) = wakeword_spotter.as_deref_mut() else {
                                continue;
                            };

                            if let Some(hit) = spotter.detect(&text) {
                                Self::emit_event(&event_tx, EngineEvent::BootStateChanged(format!(
                                    "wakeword_detected keyword={} confidence={:.2}",
                                    hit.keyword,
                                    hit.confidence
                                )));

                                // Promote the already-warm bootstrap engine instead of
                                // cold-starting a new one. Only TTS needs to be attached.
                                if let Some(mut warm_engine) = bootstrap_engine.take() {
                                    Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                        "initializing_engine".to_string(),
                                    ));
                                    // Hot-attach TTS (the only thing missing from bootstrap).
                                    if let Err(e) = warm_engine.attach_tts(&config) {
                                        eprintln!("[engine] TTS attach warning (non-fatal): {e}");
                                    }
                                    engine = Some(warm_engine);
                                    wakeword_spotter = None;
                                    Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                        "engine_initialized".to_string(),
                                    ));
                                } else {
                                    // Fallback: no bootstrap available, cold-start.
                                    if shutdown_requested.load(Ordering::Acquire) {
                                        continue;
                                    }
                                    Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                        "initializing_engine".to_string(),
                                    ));
                                    let mut full_config = config.clone();
                                    full_config.wakeword.entrypoint_enabled = false;
                                    match VocomEngine::start(full_config) {
                                        Ok(ready_engine) => {
                                            ready_engine.keep_alive();
                                            engine = Some(ready_engine);
                                            wakeword_spotter = None;
                                            Self::emit_critical_event(
                                                &critical_event_tx,
                                                EngineEvent::BootStateChanged(
                                                    "engine_initialized".to_string(),
                                                ),
                                            );
                                        }
                                        Err(err) => {
                                            Self::emit_critical_event(
                                                &critical_event_tx,
                                                EngineEvent::Error(format!(
                                                    "engine initialization failed after wakeword: {err}"
                                                )),
                                            );
                                            Self::emit_critical_event(
                                                &critical_event_tx,
                                                EngineEvent::BootStateChanged(
                                                    "waiting_for_wakeword".to_string(),
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        EngineCommand::GetTelemetry => {
                            let Some(engine) = engine.as_mut() else {
                                continue;
                            };
                            let telemetry = engine.barge_in_telemetry();
                            Self::emit_event_reliable_data(
                                &event_tx,
                                &critical_event_tx,
                                EngineEvent::TelemetrySnapshot(telemetry),
                            );
                            let pressure = engine.runtime_pressure_snapshot();
                            let now = Instant::now();
                            pressure_alerts.observe(now, &pressure);
                            let alert_snapshot = pressure_alerts.snapshot();
                            let (ingest_p50, ingest_p95) =
                                ingest_to_transcript_latency.percentile_pair(now);
                            let (speak_p50, speak_p95) =
                                transcript_to_speak_latency.percentile_pair(now);
                            Self::emit_event(
                                &event_tx,
                                EngineEvent::BargeInTransition(format!(
                                    "pressure audio_q={}/{}({}%) asr_q={} deferred_online_samples={} denoiser_bypass={} shed_silent={} coalesced_online={} dropped_offline={} dropped_chunks={} bursts={} asr_epoch={} half_duplex_gate_active={} half_duplex_gate_engaged={} stale_asr_tasks_dropped={} stale_asr_events_dropped={} cpu_us[total={} resample={} aec={} denoise={} vad={}] cpu_over_budget={} cpu_starved={} alerts[sustained={} audio={} asr={} cpu={} active={}] loop_us[ewma={} max={}] loop_starved={} latency_ms[ingest_p50={} ingest_p95={} speak_p50={} speak_p95={}]",
                                    pressure.audio_queue_len,
                                    pressure.audio_queue_capacity,
                                    pressure.audio_queue_fill_pct,
                                    pressure.asr_task_queue_len,
                                    pressure.deferred_online_chunk_samples,
                                    pressure.denoiser_bypass_active,
                                    pressure.shed_silent_chunks,
                                    pressure.coalesced_online_chunks,
                                    pressure.dropped_offline_segments,
                                    pressure.dropped_input_chunks,
                                    pressure.overflow_bursts,
                                    pressure.asr_epoch,
                                    pressure.half_duplex_mute_gate_active,
                                    pressure.half_duplex_mute_gate_engagements,
                                    pressure.dropped_stale_asr_tasks,
                                    pressure.dropped_stale_asr_events,
                                    pressure.cpu_total_ewma_us,
                                    pressure.cpu_resample_ewma_us,
                                    pressure.cpu_aec_ewma_us,
                                    pressure.cpu_denoise_ewma_us,
                                    pressure.cpu_vad_ewma_us,
                                    pressure.cpu_over_budget_chunks,
                                    pressure.cpu_starvation_events,
                                    alert_snapshot.sustained_windows,
                                    alert_snapshot.audio_windows,
                                    alert_snapshot.asr_windows,
                                    alert_snapshot.cpu_windows,
                                    alert_snapshot.active,
                                    loop_tick_ewma_us,
                                    loop_tick_max_us,
                                    loop_starved_ticks,
                                    fmt_opt_ms(ingest_p50),
                                    fmt_opt_ms(ingest_p95),
                                    fmt_opt_ms(speak_p50),
                                    fmt_opt_ms(speak_p95),
                                )),
                            );
                        }
                        EngineCommand::PollTranscript => {
                            let Some(running) = engine.as_mut() else {
                                continue;
                            };
                            // Non-blocking transcript poll: try to get one event if available.
                            match Self::try_recv_transcript(running) {
                                Ok(Some(event)) => {
                                    let now = Instant::now();
                                    let turn_id = next_turn_id;
                                    next_turn_id = next_turn_id.saturating_add(1);
                                    ingest_to_transcript_latency
                                        .push(now, event.decode_latency_ms as u64);
                                    let audio_ms = if event.sample_rate > 0 {
                                        ((event.sample_count as f64 / event.sample_rate as f64)
                                            * 1000.0)
                                            .round() as u64
                                    } else {
                                        0
                                    };
                                    eprintln!(
                                        "[latency] turn={turn_id} stage=transcript decode_ms={} samples={} sr={} audio_ms={}",
                                        event.decode_latency_ms,
                                        event.sample_count,
                                        event.sample_rate,
                                        audio_ms
                                    );
                                    last_transcript_at = Some(now);
                                    last_turn_id = Some(turn_id);
                                    latest_transcript_text = Some(event.text.clone());
                                    Self::emit_event_reliable_data(
                                        &event_tx,
                                        &critical_event_tx,
                                        EngineEvent::TranscriptReady(event),
                                    );
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    let duplex_mode = running.duplex_mode();
                                    if Self::is_recoverable_runtime_error(&err) {
                                        Self::emit_event(
                                            &event_tx,
                                            EngineEvent::Error(format!(
                                                "transcriber runtime error ({:?}): {err}",
                                                ErrorClass::Transient
                                            )),
                                        );
                                        engine = Self::restart_full_engine(
                                            &config,
                                            &event_tx,
                                            &mut engine_restart_supervisor,
                                            "poll_transcript_recoverable_error",
                                            Some(duplex_mode),
                                            &shutdown_requested,
                                        );
                                    } else {
                                        Self::emit_event(
                                            &event_tx,
                                            EngineEvent::Error(format!(
                                                "transcriber runtime error ({:?}): {err}",
                                                ErrorClass::Fatal
                                            )),
                                        );
                                    }
                                }
                            }
                        }
                        EngineCommand::SpeakMockReply => {
                            let Some(engine) = engine.as_ref() else {
                                let _ = event_tx.send(EngineEvent::Error(
                                    "engine is not initialized yet".to_string(),
                                ));
                                continue;
                            };
                            let Some(text) = latest_transcript_text.as_deref() else {
                                let _ = event_tx.send(EngineEvent::Error(
                                    "no transcript available yet for TTS reply".to_string(),
                                ));
                                continue;
                            };

                            if let Some(last) = last_mock_reply_at {
                                if last.elapsed() < Duration::from_millis(MOCK_REPLY_MIN_INTERVAL_MS) {
                                    continue;
                                }
                            }

                            match engine.synthesize_mock_reply(text) {
                                Ok(true) => {
                                    if let Some(at) = last_transcript_at {
                                        let elapsed_ms =
                                            Instant::now().duration_since(at).as_millis() as u64;
                                        transcript_to_speak_latency.push(Instant::now(), elapsed_ms);
                                        if let Some(turn_id) = last_turn_id {
                                            eprintln!(
                                                "[latency] turn={turn_id} stage=tts_start transcript_to_speak_ms={elapsed_ms} mode=mock"
                                            );
                                        }
                                    }
                                    last_mock_reply_at = Some(Instant::now());
                                }
                                Ok(false) => {
                                    Self::emit_event(&event_tx, EngineEvent::Error(
                                        "TTS is disabled or unavailable in current configuration"
                                            .to_string(),
                                    ));
                                }
                                Err(err) => {
                                Self::emit_event(&event_tx, EngineEvent::Error(format!(
                                    "mock TTS synthesis failed: {err}"
                                )));
                                }
                            }
                        }
                        EngineCommand::SpeakText { text } => {
                            let Some(engine) = engine.as_ref() else {
                                let _ = event_tx.send(EngineEvent::Error(
                                    "engine is not initialized yet".to_string(),
                                ));
                                continue;
                            };
                            match engine.synthesize_text(&text) {
                                Ok(true) => {
                                    if let Some(at) = last_transcript_at {
                                        let elapsed_ms =
                                            Instant::now().duration_since(at).as_millis() as u64;
                                        transcript_to_speak_latency.push(Instant::now(), elapsed_ms);
                                        if let Some(turn_id) = last_turn_id {
                                            eprintln!(
                                                "[latency] turn={turn_id} stage=tts_start transcript_to_speak_ms={elapsed_ms} mode=text"
                                            );
                                        }
                                    }
                                }
                                Ok(false) => {
                                    Self::emit_event(&event_tx, EngineEvent::Error(
                                        "TTS is disabled or unavailable in current configuration"
                                            .to_string(),
                                    ));
                                }
                                Err(err) => {
                                    Self::emit_event(&event_tx, EngineEvent::Error(format!(
                                        "TTS speak_text failed: {err}"
                                    )));
                                }
                            }
                        }
                        EngineCommand::SetDuplexMode { half_duplex_mute_mic } => {
                            let Some(engine) = engine.as_mut() else {
                                let _ = event_tx.send(EngineEvent::Error(
                                    "engine is not initialized yet".to_string(),
                                ));
                                continue;
                            };
                            let mode = if half_duplex_mute_mic {
                                DuplexMode::HalfDuplexMuteMic
                            } else {
                                DuplexMode::FullDuplex
                            };
                            engine.set_duplex_mode(mode);
                            Self::emit_critical_event(
                                &critical_event_tx,
                                EngineEvent::BootStateChanged(format!(
                                    "duplex_applied {}",
                                    duplex_mode_name(mode),
                                )),
                            );
                            Self::emit_event(&event_tx, EngineEvent::DuplexModeChanged(
                                duplex_mode_name(mode).to_string(),
                            ));
                        }
                        EngineCommand::GetDuplexMode => {
                            let mode_name = engine
                                .as_ref()
                                .map(|running| duplex_mode_name(running.duplex_mode()).to_string())
                                .unwrap_or_else(|| "engine_not_initialized".to_string());
                            Self::emit_event(&event_tx, EngineEvent::DuplexModeChanged(mode_name));
                        }
                        EngineCommand::Shutdown => {
                            Self::set_lifecycle_state(&lifecycle, EngineLifecycleState::Stopping);
                            if !shutdown_ack_emitted {
                                Self::emit_critical_event(
                                    &critical_event_tx,
                                    EngineEvent::BootStateChanged("shutdown_ack".to_string()),
                                );
                                shutdown_ack_emitted = true;
                            }
                            Self::emit_critical_event(
                                &critical_event_tx,
                                EngineEvent::BootStateChanged("stopping_engine".to_string()),
                            );
                            Self::emit_critical_event(&critical_event_tx, EngineEvent::Shutdown);
                            break;
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if engine.is_none() {
                        if let (Some(bootstrap), Some(spotter)) = (
                            bootstrap_engine.as_mut(),
                            wakeword_spotter.as_deref_mut(),
                        ) {
                            match Self::try_recv_transcript(bootstrap) {
                                Ok(Some(event)) => {
                                    if let Some(hit) = spotter.detect(&event.text) {
                                        Self::emit_event(&event_tx, EngineEvent::BootStateChanged(format!(
                                            "wakeword_detected keyword={} confidence={:.2}",
                                            hit.keyword,
                                            hit.confidence
                                        )));
                                        if let Some(mut warm_engine) = bootstrap_engine.take() {
                                            Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                                "initializing_engine".to_string(),
                                            ));
                                            if let Err(e) = warm_engine.attach_tts(&config) {
                                                eprintln!("[engine] TTS attach warning (non-fatal): {e}");
                                            }
                                            engine = Some(warm_engine);
                                            wakeword_spotter = None;
                                            engine_restart_supervisor.reset();
                                            Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                                "engine_initialized".to_string(),
                                            ));
                                        } else {
                                            if shutdown_requested.load(Ordering::Acquire) {
                                                continue;
                                            }
                                            Self::emit_critical_event(&critical_event_tx, EngineEvent::BootStateChanged(
                                                "initializing_engine".to_string(),
                                            ));
                                            let mut full_config = config.clone();
                                            full_config.wakeword.entrypoint_enabled = false;
                                            match VocomEngine::start(full_config) {
                                                Ok(ready_engine) => {
                                                    ready_engine.keep_alive();
                                                    engine = Some(ready_engine);
                                                    wakeword_spotter = None;
                                                    engine_restart_supervisor.reset();
                                                    Self::emit_critical_event(
                                                        &critical_event_tx,
                                                        EngineEvent::BootStateChanged(
                                                            "engine_initialized".to_string(),
                                                        ),
                                                    );
                                                }
                                                Err(err) => {
                                                    Self::emit_critical_event(
                                                        &critical_event_tx,
                                                        EngineEvent::Error(format!(
                                                            "engine initialization failed after wakeword: {err}"
                                                        )),
                                                    );
                                                    Self::emit_critical_event(
                                                        &critical_event_tx,
                                                        EngineEvent::BootStateChanged(
                                                            "waiting_for_wakeword".to_string(),
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    if Self::is_recoverable_runtime_error(&err) {
                                        Self::emit_event(
                                            &event_tx,
                                            EngineEvent::Error(format!(
                                                "bootstrap listener runtime error ({:?}): {err}",
                                                ErrorClass::Transient
                                            )),
                                        );
                                        bootstrap_engine = Self::restart_bootstrap_engine(
                                            &config,
                                            &event_tx,
                                            &mut bootstrap_restart_supervisor,
                                            "bootstrap_recoverable_error",
                                            &shutdown_requested,
                                        );
                                        if bootstrap_engine.is_some() {
                                            Self::emit_event(
                                                &event_tx,
                                                EngineEvent::BootStateChanged(
                                                    "waiting_for_wakeword".to_string(),
                                                ),
                                            );
                                        }
                                    } else {
                                        Self::emit_event(
                                            &event_tx,
                                            EngineEvent::Error(format!(
                                                "bootstrap listener runtime error ({:?}): {err}",
                                                ErrorClass::Fatal
                                            )),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    let Some(running) = engine.as_mut() else {
                        continue;
                    };
                    // No command arrived within 50ms. Check for transcripts.
                    match Self::try_recv_transcript(running) {
                        Ok(Some(event)) => {
                            let now = Instant::now();
                            ingest_to_transcript_latency.push(now, event.decode_latency_ms as u64);
                            last_transcript_at = Some(now);
                            latest_transcript_text = Some(event.text.clone());
                            Self::emit_event_reliable_data(
                                &event_tx,
                                &critical_event_tx,
                                EngineEvent::TranscriptReady(event),
                            );
                        }
                        Ok(None) => {}
                        Err(err) => {
                            let duplex_mode = running.duplex_mode();
                            if Self::is_recoverable_runtime_error(&err) {
                                Self::emit_event(
                                    &event_tx,
                                    EngineEvent::Error(format!(
                                        "transcriber runtime error ({:?}): {err}",
                                        ErrorClass::Transient
                                    )),
                                );
                                engine = Self::restart_full_engine(
                                    &config,
                                    &event_tx,
                                    &mut engine_restart_supervisor,
                                    "runtime_poll_recoverable_error",
                                    Some(duplex_mode),
                                    &shutdown_requested,
                                );
                                continue;
                            }
                            Self::emit_event(
                                &event_tx,
                                EngineEvent::Error(format!(
                                    "transcriber runtime error ({:?}): {err}",
                                    ErrorClass::Fatal
                                )),
                            );
                        }
                    }

                    // Also check for barge-in actions (non-blocking).
                    if let Some(transition) = running.handle_barge_in() {
                        Self::emit_event(&event_tx, EngineEvent::BargeInTransition(transition));
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        Self::set_lifecycle_state(&lifecycle, EngineLifecycleState::Stopped);
    }

    /// Try to receive a transcript without blocking.
    fn try_recv_transcript(engine: &mut VocomEngine) -> Result<Option<TranscriptEvent>, VocomError> {
        match engine.recv_timeout(Duration::from_millis(0)) {
            Ok(Some(event)) => Ok(Some(event)),
            Ok(None) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn is_recoverable_runtime_error(err: &VocomError) -> bool {
        err.is_transient()
    }

    fn is_control_command(cmd: &EngineCommand) -> bool {
        matches!(
            cmd,
            EngineCommand::Shutdown
                | EngineCommand::SetDuplexMode { .. }
                | EngineCommand::GetDuplexMode
        )
    }

    fn restart_full_engine(
        config: &EngineConfig,
        event_tx: &Sender<EngineEvent>,
        supervisor: &mut RestartSupervisor,
        reason: &str,
        duplex_mode: Option<DuplexMode>,
        shutdown_requested: &AtomicBool,
    ) -> Option<VocomEngine> {
        if shutdown_requested.load(Ordering::Acquire) {
            return None;
        }
        let Some(attempt) = supervisor.allow_attempt() else {
            Self::emit_event(
                event_tx,
                EngineEvent::Error(format!(
                    "engine recovery budget exhausted: max={} within {}s (reason={reason})",
                    ENGINE_RESTART_MAX_ATTEMPTS,
                    ENGINE_RESTART_WINDOW_SECS
                )),
            );
            return None;
        };

        Self::emit_event(
            event_tx,
            EngineEvent::BootStateChanged(format!(
                "recovering_engine attempt={attempt} reason={reason}"
            )),
        );
        let mut full_config = config.clone();
        full_config.wakeword.entrypoint_enabled = false;
        if shutdown_requested.load(Ordering::Acquire) {
            return None;
        }
        match VocomEngine::start(full_config) {
            Ok(mut restarted) => {
                if let Some(mode) = duplex_mode {
                    restarted.set_duplex_mode(mode);
                }
                restarted.keep_alive();
                supervisor.reset();
                Self::emit_event(
                    event_tx,
                    EngineEvent::BootStateChanged("engine_recovered".to_string()),
                );
                Some(restarted)
            }
            Err(err) => {
                Self::emit_event(
                    event_tx,
                    EngineEvent::Error(format!("engine recovery attempt {attempt} failed: {err}")),
                );
                None
            }
        }
    }

    fn restart_bootstrap_engine(
        config: &EngineConfig,
        event_tx: &Sender<EngineEvent>,
        supervisor: &mut RestartSupervisor,
        reason: &str,
        shutdown_requested: &AtomicBool,
    ) -> Option<VocomEngine> {
        if shutdown_requested.load(Ordering::Acquire) {
            return None;
        }
        let Some(attempt) = supervisor.allow_attempt() else {
            Self::emit_event(
                event_tx,
                EngineEvent::Error(format!(
                    "bootstrap recovery budget exhausted: max={} within {}s (reason={reason})",
                    ENGINE_RESTART_MAX_ATTEMPTS,
                    ENGINE_RESTART_WINDOW_SECS
                )),
            );
            return None;
        };

        Self::emit_event(
            event_tx,
            EngineEvent::BootStateChanged(format!(
                "recovering_bootstrap attempt={attempt} reason={reason}"
            )),
        );
        let mut bootstrap_config = config.clone();
        bootstrap_config.wakeword.entrypoint_enabled = false;
        bootstrap_config.tts.enabled = false;
        if shutdown_requested.load(Ordering::Acquire) {
            return None;
        }
        match VocomEngine::start(bootstrap_config) {
            Ok(restarted) => {
                restarted.keep_alive();
                supervisor.reset();
                Self::emit_event(
                    event_tx,
                    EngineEvent::BootStateChanged("bootstrap_recovered".to_string()),
                );
                Some(restarted)
            }
            Err(err) => {
                Self::emit_event(
                    event_tx,
                    EngineEvent::Error(format!(
                        "bootstrap recovery attempt {attempt} failed: {err}"
                    )),
                );
                None
            }
        }
    }
}

fn ewma_us(current: u32, sample: u64) -> u32 {
    let clamped = sample.min(u32::MAX as u64) as u32;
    if current == 0 {
        return clamped;
    }
    (((current as u64).saturating_mul(4)).saturating_add(clamped as u64) / 5) as u32
}

fn percentile_sorted(sorted_samples: &[u64], p: f64) -> Option<u64> {
    if sorted_samples.is_empty() {
        return None;
    }
    let idx = ((sorted_samples.len() - 1) as f64 * p).round() as usize;
    sorted_samples.get(idx).copied()
}

fn fmt_opt_ms(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "na".to_string())
}

fn duplex_mode_name(mode: DuplexMode) -> &'static str {
    match mode {
        DuplexMode::FullDuplex => "full_duplex",
        DuplexMode::HalfDuplexMuteMic => "half_duplex_mute_mic",
    }
}

impl Drop for VocomEngineHandle {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EngineEvent, PerfSnapshotLogger, PressureAlertSnapshot, PressureAlertTracker,
        RestartSupervisor, RollingLatencyWindow, TranscriptEvent, VocomEngineHandle, ewma_us,
        perf_snapshot_json_line,
    };
    use crate::realtime_pipeline::RealtimePressureSnapshot;
    use crossbeam_channel::{TryRecvError, bounded, unbounded};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[test]
    fn restart_supervisor_limits_attempts_within_window() {
        let mut sup = RestartSupervisor::new();
        assert_eq!(sup.allow_attempt(), Some(1));
        assert_eq!(sup.allow_attempt(), Some(2));
        assert_eq!(sup.allow_attempt(), Some(3));
        assert_eq!(sup.allow_attempt(), None);
    }

    #[test]
    fn restart_supervisor_reset_restores_budget() {
        let mut sup = RestartSupervisor::new();
        let _ = sup.allow_attempt();
        let _ = sup.allow_attempt();
        let _ = sup.allow_attempt();
        assert_eq!(sup.allow_attempt(), None);
        sup.reset();
        assert_eq!(sup.allow_attempt(), Some(1));
    }

    #[test]
    fn loop_ewma_tracks_recent_tick() {
        assert_eq!(ewma_us(0, 50_000), 50_000);
        assert_eq!(ewma_us(40_000, 60_000), 44_000);
    }

    #[test]
    fn rolling_latency_window_percentiles() {
        let mut w = RollingLatencyWindow::new(Duration::from_secs(60), 16);
        let now = Instant::now();
        for v in [10, 20, 30, 40, 50] {
            w.push(now, v);
        }
        let (p50, p95) = w.percentile_pair(now);
        assert_eq!(p50, Some(30));
        assert_eq!(p95, Some(50));
    }

    #[test]
    fn reliable_data_event_prefers_normal_event_queue_when_available() {
        let (event_tx, event_rx) = bounded::<EngineEvent>(1);
        let (critical_tx, critical_rx) = unbounded::<EngineEvent>();

        VocomEngineHandle::emit_event_reliable_data(
            &event_tx,
            &critical_tx,
            EngineEvent::TranscriptReady(TranscriptEvent {
                text: "hello".to_string(),
                sample_rate: 16_000,
                sample_count: 320,
                decode_latency_ms: 12,
            }),
        );

        match event_rx.try_recv() {
            Ok(EngineEvent::TranscriptReady(ev)) => assert_eq!(ev.text, "hello"),
            other => panic!("expected transcript on normal event queue, got {other:?}"),
        }
        assert!(matches!(critical_rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn reliable_data_event_falls_back_to_critical_queue_when_event_queue_full() {
        let (event_tx, _event_rx) = bounded::<EngineEvent>(1);
        let (critical_tx, critical_rx) = unbounded::<EngineEvent>();

        // Saturate the normal queue first.
        let _ = event_tx.try_send(EngineEvent::BootStateChanged("busy".to_string()));

        VocomEngineHandle::emit_event_reliable_data(
            &event_tx,
            &critical_tx,
            EngineEvent::TranscriptReady(TranscriptEvent {
                text: "fallback".to_string(),
                sample_rate: 16_000,
                sample_count: 160,
                decode_latency_ms: 9,
            }),
        );

        match critical_rx.try_recv() {
            Ok(EngineEvent::TranscriptReady(ev)) => assert_eq!(ev.text, "fallback"),
            other => panic!("expected transcript fallback on critical queue, got {other:?}"),
        }
    }

    #[test]
    fn non_reliable_event_is_not_forwarded_to_critical_queue_when_full() {
        let (event_tx, _event_rx) = bounded::<EngineEvent>(1);
        let (critical_tx, critical_rx) = unbounded::<EngineEvent>();

        // Saturate the normal queue first.
        let _ = event_tx.try_send(EngineEvent::BootStateChanged("busy".to_string()));

        VocomEngineHandle::emit_event_reliable_data(
            &event_tx,
            &critical_tx,
            EngineEvent::BargeInTransition("state:a->b".to_string()),
        );

        assert!(matches!(critical_rx.try_recv(), Err(TryRecvError::Empty)));
    }

    fn pressure(audio_fill_pct: u8, asr_q: usize, cpu_starvation_events: u64) -> RealtimePressureSnapshot {
        RealtimePressureSnapshot {
            audio_queue_capacity: 100,
            audio_queue_fill_pct: audio_fill_pct,
            asr_task_queue_len: asr_q,
            cpu_starvation_events,
            ..RealtimePressureSnapshot::default()
        }
    }

    #[test]
    fn pressure_alert_tracker_counts_once_per_continuous_audio_window() {
        let mut tracker = PressureAlertTracker::with_threshold(Duration::from_secs(5));
        let base = Instant::now();

        for sec in 0..=7 {
            tracker.observe(base + Duration::from_secs(sec), &pressure(80, 10, 0));
        }
        let first = tracker.snapshot();
        assert_eq!(first.audio_windows, 1);
        assert_eq!(first.sustained_windows, 1);

        // Recover below threshold resets active window.
        tracker.observe(base + Duration::from_secs(8), &pressure(20, 10, 0));
        let mid = tracker.snapshot();
        assert!(!mid.active);

        // A new sustained burst should increment counters again.
        for sec in 9..=15 {
            tracker.observe(base + Duration::from_secs(sec), &pressure(75, 10, 0));
        }
        let second = tracker.snapshot();
        assert_eq!(second.audio_windows, 2);
        assert_eq!(second.sustained_windows, 2);
    }

    #[test]
    fn pressure_alert_tracker_cpu_window_requires_continuous_starvation_progress() {
        let mut tracker = PressureAlertTracker::with_threshold(Duration::from_secs(5));
        let base = Instant::now();

        // Static starvation counter should not create a sustained CPU window.
        for sec in 0..=6 {
            tracker.observe(base + Duration::from_secs(sec), &pressure(20, 10, 0));
        }
        let none = tracker.snapshot();
        assert_eq!(none.cpu_windows, 0);
        assert_eq!(none.sustained_windows, 0);

        // Repeated starvation growth across consecutive observations should count.
        for sec in 7..=13 {
            let starved = (sec - 6) as u64;
            tracker.observe(base + Duration::from_secs(sec), &pressure(20, 10, starved));
        }
        let cpu = tracker.snapshot();
        assert_eq!(cpu.cpu_windows, 1);
        assert_eq!(cpu.sustained_windows, 1);
    }

    #[test]
    fn perf_snapshot_json_line_contains_expected_keys() {
        let pressure = RealtimePressureSnapshot {
            audio_queue_len: 42,
            audio_queue_capacity: 100,
            audio_queue_fill_pct: 42,
            asr_task_queue_len: 9,
            cpu_starvation_events: 3,
            ..RealtimePressureSnapshot::default()
        };
        let alerts = PressureAlertSnapshot {
            sustained_windows: 2,
            audio_windows: 1,
            asr_windows: 1,
            cpu_windows: 0,
            active: true,
        };

        let line = perf_snapshot_json_line(
            &pressure,
            alerts,
            1200,
            6500,
            7,
            Some(120),
            Some(260),
            Some(55),
            Some(98),
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("snapshot json must parse");

        assert_eq!(parsed["audio_q_len"], 42);
        assert_eq!(parsed["alerts_sustained_windows"], 2);
        assert_eq!(parsed["alerts_active"], true);
        assert_eq!(parsed["latency_ingest_p95_ms"], 260);
        assert_eq!(parsed["latency_speak_p50_ms"], 55);
    }

    #[test]
    fn perf_snapshot_logger_appends_line_after_interval() {
        let unique = format!(
            "vocom-perf-test-{}-{}.jsonl",
            std::process::id(),
            now_nanos()
        );
        let path: PathBuf = std::env::temp_dir().join(unique);
        let mut logger =
            PerfSnapshotLogger::with_path_and_interval(path.clone(), Duration::from_millis(1));
        let now = Instant::now();

        logger.maybe_write_line(now, "{\"k\":1}");
        // Immediate second call should be throttled by interval.
        logger.maybe_write_line(now, "{\"k\":2}");
        logger.maybe_write_line(now + Duration::from_millis(2), "{\"k\":3}");

        let raw = fs::read_to_string(&path).expect("perf snapshot file should exist");
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"k\":1}");
        assert_eq!(lines[1], "{\"k\":3}");

        let _ = fs::remove_file(path);
    }

    fn now_nanos() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }
}
