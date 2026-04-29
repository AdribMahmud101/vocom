use std::collections::VecDeque;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::asr_manager::{ASRModelBuilder, ASRVariant};
use crate::config::{AsrModeConfig, AsrVariantConfig, DuplexMode, EngineConfig};
use crate::duplex_audio::{BargeInMetricsSnapshot, DuplexPlaybackGate, render_reference_bus};
use crate::errors::VocomError;
use crate::realtime_pipeline::{
    AsrBackend, RealtimePressureSnapshot, RealtimeTranscriber, TranscriptEvent,
    start_realtime_transcriber,
};
use crate::tts_manager::TtsManager;
use crate::vad_manager::VadBuilder;

// ── Barge-in FSM ─────────────────────────────────────────────────────────────

/// All tunable parameters for the barge-in FSM, separate from the FSM state.
#[derive(Clone, Debug)]
pub struct BargeInParams {
    /// Target gain applied to TTS output when ducked (0.0 = mute, 1.0 = full).
    pub duck_level: f32,
    /// How long to hold the `Ducked` state before stopping TTS entirely (ms).
    pub stop_after_ms: u64,
    /// How long a barge-in candidate must persist before transitioning to `Ducked` (ms).
    pub suspect_hold_ms: u64,
    /// How long a barge-in request may be absent before dropping back to `Speaking`.
    /// Must be > the pipeline's `BARGE_IN_MIN_INTERVAL_MS` (250ms) and the main-loop
    /// poll timeout (300ms). Default 500ms.
    pub suspect_drop_grace_ms: u64,
    /// Post-stop cooldown before the FSM may transition back to `Speaking` (ms).
    pub recover_ms: u64,
}

impl Default for BargeInParams {
    fn default() -> Self {
        Self {
            duck_level: 0.55,
            stop_after_ms: 350,
            suspect_hold_ms: 90,
            suspect_drop_grace_ms: 500,
            recover_ms: 350,
        }
    }
}

/// Every state the barge-in FSM can be in.
#[derive(Clone, Copy, Debug)]
enum BargeInState {
    /// No TTS is playing; the engine is just listening.
    Idle,
    /// TTS is active; no barge-in candidate seen yet.
    Speaking,
    /// A barge-in candidate was detected; waiting for sustained confirmation.
    SuspectInterrupt {
        since_ms: u64,
        /// Timestamp of the most recent barge-in request seen in this suspect window.
        last_request_ms: u64,
    },
    /// Confirmed interrupt; TTS volume has been ducked.
    Ducked { since_ms: u64 },
    /// TTS has been stopped; waiting for the acoustic environment to settle.
    Recovering { until_ms: u64 },
}

impl BargeInState {
    fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Speaking => "speaking",
            Self::SuspectInterrupt { .. } => "suspect_interrupt",
            Self::Ducked { .. } => "ducked",
            Self::Recovering { .. } => "recovering",
        }
    }
}

/// Action the engine should take based on the FSM step output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BargeInAction {
    None,
    Duck,
    Stop,
    RestoreVolume,
}

/// A single recorded state transition with a human-readable reason code.
#[derive(Clone, Debug)]
pub struct BargeInTransitionEntry {
    pub at_ms: u64,
    pub from: &'static str,
    pub to: &'static str,
    pub reason: &'static str,
}

const MAX_TRANSITION_LOG: usize = 64;
const MAX_LATENCY_SAMPLES: usize = 256;
const MAX_ROLLING_LATENCY_SAMPLES: usize = 2048;
const ROLLING_WINDOW_MS: u64 = 60_000;
const MAX_COUNTER_HISTORY: usize = 512;

#[derive(Clone, Copy, Debug)]
struct CounterHistoryEntry {
    at_ms: u64,
    metrics: BargeInMetricsSnapshot,
}

/// Input snapshot passed to `BargeInFsm::step()` each poll cycle.
pub struct StepInput {
    pub is_tts_active: bool,
    /// Timestamp (ms since epoch) of the most recent barge-in request, if any.
    pub barge_in_request_ts: Option<u64>,
    pub now_ms: u64,
}

/// Self-contained barge-in state machine. All state lives here; no borrow from
/// the outer engine is needed once constructed.
pub struct BargeInFsm {
    state: BargeInState,
    params: BargeInParams,
    transition_log: VecDeque<BargeInTransitionEntry>,
    latency_samples_ms: VecDeque<u64>,
    rolling_latency_samples_ms: VecDeque<(u64, u64)>,
}

impl BargeInFsm {
    pub fn new(params: BargeInParams) -> Self {
        Self {
            state: BargeInState::Idle,
            params,
            transition_log: VecDeque::with_capacity(MAX_TRANSITION_LOG),
            latency_samples_ms: VecDeque::with_capacity(MAX_LATENCY_SAMPLES),
            rolling_latency_samples_ms: VecDeque::with_capacity(MAX_ROLLING_LATENCY_SAMPLES),
        }
    }

    /// Advance the FSM by one poll cycle. Returns the action the engine should
    /// execute against the TTS player. All state transitions are recorded in
    /// the internal log for telemetry.
    pub fn step(&mut self, input: &StepInput) -> BargeInAction {
        let now = input.now_ms;

        // ── Global rule: TTS ended → return to Idle immediately ──────────────
        if !input.is_tts_active {
            match self.state {
                BargeInState::Idle => {}
                _ => {
                    self.transition(BargeInState::Idle, "tts_ended", now);
                    return BargeInAction::RestoreVolume;
                }
            }
            return BargeInAction::None;
        }

        // ── TTS is active; evaluate per-state transitions ────────────────────
        match self.state {
            // ── Idle → Speaking ───────────────────────────────────────────────
            BargeInState::Idle => {
                self.transition(BargeInState::Speaking, "tts_started", now);
                BargeInAction::None
            }

            // ── Speaking ──────────────────────────────────────────────────────
            BargeInState::Speaking => {
                if let Some(_request_ts) = input.barge_in_request_ts {
                    self.transition(
                        BargeInState::SuspectInterrupt {
                            since_ms: now,
                            last_request_ms: now,
                        },
                        "barge_in_candidate",
                        now,
                    );
                }
                BargeInAction::None
            }

            // ── SuspectInterrupt ──────────────────────────────────────────────
            //
            // A barge-in request is a one-shot atomic flag consumed each poll.
            // The pipeline throttles requests to ≥250ms apart, and the main loop
            // polls every ≈300ms, so consecutive polls will often see no request
            // even for genuine speech.
            //
            // Policy:
            //   • Refresh `last_request_ms` whenever a request arrives.
            //   • Duck when (elapsed ≥ hold_ms) AND (request was recent enough).
            //   • Drop to Speaking only when no request for `suspect_drop_grace_ms`.
            BargeInState::SuspectInterrupt { since_ms, last_request_ms } => {
                // Refresh last-seen timestamp if a new request came in.
                let last_ms = if input.barge_in_request_ts.is_some() {
                    now
                } else {
                    last_request_ms
                };

                let elapsed_since_entry = now.saturating_sub(since_ms);
                let request_age = now.saturating_sub(last_ms);
                let request_stale = request_age >= self.params.suspect_drop_grace_ms;

                if request_stale {
                    // No barge-in signal for the full grace window → give up.
                    self.transition(BargeInState::Speaking, "suspect_dropped", now);
                    return BargeInAction::None;
                }

                if elapsed_since_entry >= self.params.suspect_hold_ms {
                    // Sustained enough: duck.
                    // Record latency as time from first-suspect to actual duck action —
                    // this is the true user-perceptible interrupt detection latency.
                    self.record_latency(elapsed_since_entry, now);
                    self.transition(
                        BargeInState::Ducked { since_ms: now },
                        "suspect_confirmed",
                        now,
                    );
                    return BargeInAction::Duck;
                }

                // Still within hold window — refresh last_request_ms in-place.
                self.state = BargeInState::SuspectInterrupt {
                    since_ms,
                    last_request_ms: last_ms,
                };
                BargeInAction::None
            }

            // ── Ducked ────────────────────────────────────────────────────────
            BargeInState::Ducked { since_ms } => {
                let elapsed = now.saturating_sub(since_ms);
                if elapsed >= self.params.stop_after_ms {
                    let until_ms = now.saturating_add(self.params.recover_ms);
                    self.transition(BargeInState::Recovering { until_ms }, "duck_timeout", now);
                    return BargeInAction::Stop;
                }
                BargeInAction::None
            }

            // ── Recovering ───────────────────────────────────────────────────
            BargeInState::Recovering { until_ms } => {
                if now >= until_ms {
                    // Recovery period expired while TTS is still active — go back to Speaking.
                    self.transition(BargeInState::Speaking, "recovery_expired", now);
                }
                BargeInAction::None
            }
        }
    }

    pub fn state_name(&self) -> &'static str {
        self.state.name()
    }

    pub fn latency_percentiles(&self) -> (Option<u64>, Option<u64>) {
        let mut samples: Vec<u64> = self.latency_samples_ms.iter().copied().collect();
        samples.sort_unstable();
        (percentile(&samples, 0.50), percentile(&samples, 0.95))
    }

    pub fn latency_sample_count(&self) -> usize {
        self.latency_samples_ms.len()
    }

    pub fn rolling_latency_stats(&self, now_ms: u64) -> (usize, Option<u64>, Option<u64>) {
        let samples: Vec<u64> = self
            .rolling_latency_samples_ms
            .iter()
            .filter_map(|(at_ms, latency)| {
                if now_ms.saturating_sub(*at_ms) <= ROLLING_WINDOW_MS {
                    Some(*latency)
                } else {
                    None
                }
            })
            .collect();

        if samples.is_empty() {
            return (0, None, None);
        }

        let mut sorted = samples;
        sorted.sort_unstable();
        (
            sorted.len(),
            percentile(&sorted, 0.50),
            percentile(&sorted, 0.95),
        )
    }

    /// Most-recent N transition entries for telemetry output.
    pub fn recent_transitions(&self, n: usize) -> Vec<BargeInTransitionEntry> {
        self.transition_log
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn transition(&mut self, next: BargeInState, reason: &'static str, now: u64) {
        let entry = BargeInTransitionEntry {
            at_ms: now,
            from: self.state.name(),
            to: next.name(),
            reason,
        };
        if self.transition_log.len() == MAX_TRANSITION_LOG {
            self.transition_log.pop_front();
        }
        self.transition_log.push_back(entry);
        self.state = next;
    }

    fn record_latency(&mut self, latency_ms: u64, now_ms: u64) {
        if self.latency_samples_ms.len() == MAX_LATENCY_SAMPLES {
            self.latency_samples_ms.pop_front();
        }
        self.latency_samples_ms.push_back(latency_ms);

        if self.rolling_latency_samples_ms.len() == MAX_ROLLING_LATENCY_SAMPLES {
            self.rolling_latency_samples_ms.pop_front();
        }
        self.rolling_latency_samples_ms
            .push_back((now_ms, latency_ms));

        while let Some((at_ms, _)) = self.rolling_latency_samples_ms.front() {
            if now_ms.saturating_sub(*at_ms) > ROLLING_WINDOW_MS {
                self.rolling_latency_samples_ms.pop_front();
            } else {
                break;
            }
        }
    }
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BargeInTelemetry {
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
    pub state: &'static str,
    pub recent_transitions: Vec<BargeInTransitionEntry>,
}

// ── VocomEngine ───────────────────────────────────────────────────────────────

pub struct VocomEngine {
    transcriber: RealtimeTranscriber,
    tts: Option<TtsManager>,
    duplex_gate: DuplexPlaybackGate,
    barge_in_fsm: BargeInFsm,
    metrics_history: VecDeque<CounterHistoryEntry>,
}

impl VocomEngine {
    pub fn start(config: EngineConfig) -> Result<Self, VocomError> {
        let mut config = config;
        config.resolve_paths();
        config.validate()?;

        let variant = match config.asr.variant {
            AsrVariantConfig::Whisper => ASRVariant::Whisper,
            AsrVariantConfig::MoonshineV2 => ASRVariant::Moonshinev2,
            AsrVariantConfig::StreamingZipformer => ASRVariant::StreamingZipformer,
            AsrVariantConfig::NemotronStreaming => ASRVariant::NemotronStreaming,
        };

        let mut asr_builder = ASRModelBuilder::new(variant)
            .encoder(&config.asr.encoder_path)
            .decoder(&config.asr.decoder_path)
            .tokens(&config.asr.tokens_path)
            .provider(&config.asr.provider)
            .num_threads(config.asr.num_threads)
            .whisper_language(&config.asr.whisper_language)
            .whisper_task(&config.asr.whisper_task)
            .whisper_tail_paddings(config.asr.whisper_tail_paddings)
            .whisper_enable_token_timestamps(config.asr.whisper_enable_token_timestamps)
            .whisper_enable_segment_timestamps(config.asr.whisper_enable_segment_timestamps)
            .online_decoding_method(&config.asr.online_decoding_method)
            .online_enable_endpoint(config.asr.online_enable_endpoint)
            .online_rule1_min_trailing_silence(config.asr.online_rule1_min_trailing_silence)
            .online_rule2_min_trailing_silence(config.asr.online_rule2_min_trailing_silence)
            .online_rule3_min_utterance_length(config.asr.online_rule3_min_utterance_length);

        if let Some(joiner_path) = config.asr.joiner_path.as_deref() {
            asr_builder = asr_builder.joiner(joiner_path);
        }

        asr_builder = asr_builder.hotwords_path(config.asr.hotwords_path.clone());
        asr_builder = asr_builder.hotwords_score(config.asr.hotwords_score);

        let asr_backend = match config.asr.mode {
            AsrModeConfig::Offline => AsrBackend::Offline(asr_builder.build()?),
            AsrModeConfig::Online => AsrBackend::Online(asr_builder.build_online()?),
        };

        let vad = VadBuilder::new()
            .model(&config.vad.model_path)
            .threshold(config.vad.threshold)
            .min_silence_duration(config.vad.min_silence_duration)
            .min_speech_duration(config.vad.min_speech_duration)
            .max_speech_duration(config.vad.max_speech_duration)
            .window_size(config.vad.window_size)
            .sample_rate(config.vad.sample_rate)
            .num_threads(config.vad.num_threads)
            .provider(&config.vad.provider)
            .debug(config.vad.debug)
            .build()?;

        let barge_in_vad = VadBuilder::new()
            .model(&config.barge_in_vad.model_path)
            .threshold(config.barge_in_vad.threshold)
            .min_silence_duration(config.barge_in_vad.min_silence_duration)
            .min_speech_duration(config.barge_in_vad.min_speech_duration)
            .max_speech_duration(config.barge_in_vad.max_speech_duration)
            .window_size(config.barge_in_vad.window_size)
            .sample_rate(config.barge_in_vad.sample_rate)
            .num_threads(config.barge_in_vad.num_threads)
            .provider(&config.barge_in_vad.provider)
            .debug(config.barge_in_vad.debug)
            .build()?;

        let (render_pub, render_consumer) = render_reference_bus(
            config.realtime.render_reference_capacity,
            config.aec.sample_rate,
        );
        let duplex_gate = DuplexPlaybackGate::new();

        let transcriber = start_realtime_transcriber(
            asr_backend,
            vad,
            barge_in_vad,
            config.realtime.to_runtime(),
            config.aec.clone(),
            config.denoiser.clone(),
            render_consumer,
            duplex_gate.clone(),
            config.vad.dynamic_gate_enabled,
            config.vad.noise_smoothing,
            config.vad.noise_gate_multiplier,
            config.vad.noise_gate_min_rms,
            config.vad.noise_gate_max_rms,
        )?;

        let tts = if config.tts.enabled {
            Some(TtsManager::from_config(
                &config.tts,
                Some(render_pub),
                Some(duplex_gate.clone()),
            )?)
        } else {
            None
        };

        let fsm_params = BargeInParams {
            duck_level: config.realtime.tts_barge_in_duck_level,
            stop_after_ms: config.realtime.tts_barge_in_stop_after_ms,
            suspect_hold_ms: config.realtime.tts_barge_in_suspect_hold_ms,
            suspect_drop_grace_ms: config.realtime.tts_barge_in_suspect_drop_grace_ms,
            recover_ms: config.realtime.tts_barge_in_recover_ms,
        };

        let mut engine = Self {
            transcriber,
            tts,
            duplex_gate,
            barge_in_fsm: BargeInFsm::new(fsm_params),
            metrics_history: VecDeque::with_capacity(MAX_COUNTER_HISTORY),
        };

        // Seed baseline snapshot so rolling deltas are meaningful immediately.
        engine.metrics_history.push_back(CounterHistoryEntry {
            at_ms: now_ms(),
            metrics: BargeInMetricsSnapshot::default(),
        });

        Ok(engine)
    }

    /// Hot-attach TTS to a running engine that was started without TTS.
    /// This avoids the multi-second cold-start of rebuilding ASR/VAD/denoiser
    /// when promoting a bootstrap (wakeword-listener) engine to a full engine.
    pub fn attach_tts(&mut self, config: &EngineConfig) -> Result<(), VocomError> {
        if self.tts.is_some() {
            return Ok(()); // TTS already attached
        }
        let mut config = config.clone();
        config.resolve_paths();

        let (render_pub, _render_consumer) = render_reference_bus(
            config.realtime.render_reference_capacity,
            config.aec.sample_rate,
        );

        let tts = TtsManager::from_config(
            &config.tts,
            Some(render_pub),
            Some(self.duplex_gate.clone()),
        )?;
        self.tts = Some(tts);
        Ok(())
    }

    pub fn recv(&mut self) -> Result<TranscriptEvent, VocomError> {
        self.transcriber.recv()
    }

    pub fn recv_timeout(&mut self, timeout: Duration) -> Result<Option<TranscriptEvent>, VocomError> {
        self.transcriber.recv_timeout(timeout)
    }

    pub fn keep_alive(&self) {
        self.transcriber.keep_alive();
    }

    pub fn set_duplex_mode(&mut self, mode: crate::config::DuplexMode) {
        if mode == DuplexMode::HalfDuplexMuteMic {
            if let Some(tts) = self.tts.as_ref() {
                tts.restore_volume();
            }
        }
        self.transcriber.set_duplex_mode(mode);
        self.barge_in_fsm = BargeInFsm::new(self.barge_in_fsm.params.clone());
    }

    pub fn duplex_mode(&self) -> crate::config::DuplexMode {
        self.transcriber.duplex_mode()
    }

    pub fn synthesize_mock_reply(&self, transcript_text: &str) -> Result<bool, VocomError> {
        let Some(tts) = self.tts.as_ref() else {
            return Ok(false);
        };

        let reply = TtsManager::mock_response(transcript_text);
        tts.speak_text(&reply)?;
        Ok(true)
    }

    pub fn synthesize_text(&self, text: &str) -> Result<bool, VocomError> {
        let Some(tts) = self.tts.as_ref() else {
            return Ok(false);
        };

        tts.speak_text(text)?;
        Ok(true)
    }

    /// Advance the barge-in FSM and execute the resulting action against the TTS
    /// player. Returns a human-readable description of any transition that
    /// occurred, suitable for logging.
    pub fn handle_barge_in(&mut self) -> Option<String> {
        if self.transcriber.duplex_mode() == DuplexMode::HalfDuplexMuteMic {
            return None;
        }

        let is_tts_active = self.duplex_gate.is_tts_active();
        let barge_in_request_ts = self.duplex_gate.take_barge_in_request_with_timestamp();

        let input = StepInput {
            is_tts_active,
            barge_in_request_ts,
            now_ms: now_ms(),
        };

        let prev_state = self.barge_in_fsm.state_name();
        let action = self.barge_in_fsm.step(&input);
        let next_state = self.barge_in_fsm.state_name();

        // Execute the action against TTS.
        match action {
            BargeInAction::Duck => {
                if let Some(tts) = self.tts.as_ref() {
                    tts.duck_current(self.barge_in_fsm.params.duck_level);
                }
                self.duplex_gate.note_ducked();
            }
            BargeInAction::Stop => {
                if let Some(tts) = self.tts.as_ref() {
                    tts.stop_current();
                    tts.restore_volume();
                }
                self.duplex_gate.note_stopped();
            }
            BargeInAction::RestoreVolume => {
                if let Some(tts) = self.tts.as_ref() {
                    tts.restore_volume();
                }
            }
            BargeInAction::None => {}
        }

        // Report transition only when the state actually changed.
        if prev_state != next_state {
            Some(format!(
                "state:{prev_state}->{next_state} action:{action:?}"
            ))
        } else {
            None
        }
    }

    pub fn barge_in_telemetry(&mut self) -> BargeInTelemetry {
        let now = now_ms();
        let metrics = self.duplex_gate.barge_in_metrics_snapshot();
        let (p50, p95) = self.barge_in_fsm.latency_percentiles();
        self.record_metrics_snapshot(now, metrics);
        let rolling = self.rolling_counter_delta(metrics);
        let (rolling_latency_samples, rolling_latency_p50_ms, rolling_latency_p95_ms) =
            self.barge_in_fsm.rolling_latency_stats(now);

        BargeInTelemetry {
            requested: metrics.requested,
            rejected_low_rms: metrics.rejected_low_rms,
            rejected_render_ratio: metrics.rejected_render_ratio,
            rejected_persistence: metrics.rejected_persistence,
            rejected_confidence: metrics.rejected_confidence,
            ducked: metrics.ducked,
            stopped: metrics.stopped,
            latency_samples: self.barge_in_fsm.latency_sample_count(),
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            rolling_requested: rolling.requested,
            rolling_rejected_low_rms: rolling.rejected_low_rms,
            rolling_rejected_render_ratio: rolling.rejected_render_ratio,
            rolling_rejected_persistence: rolling.rejected_persistence,
            rolling_rejected_confidence: rolling.rejected_confidence,
            rolling_ducked: rolling.ducked,
            rolling_stopped: rolling.stopped,
            rolling_latency_samples,
            rolling_latency_p50_ms,
            rolling_latency_p95_ms,
            state: self.barge_in_fsm.state_name(),
            recent_transitions: self.barge_in_fsm.recent_transitions(8),
        }
    }

    pub fn runtime_pressure_snapshot(&self) -> RealtimePressureSnapshot {
        self.transcriber.pressure_snapshot()
    }

    fn record_metrics_snapshot(&mut self, now_ms: u64, metrics: BargeInMetricsSnapshot) {
        self.metrics_history.push_back(CounterHistoryEntry {
            at_ms: now_ms,
            metrics,
        });

        while self.metrics_history.len() > MAX_COUNTER_HISTORY {
            self.metrics_history.pop_front();
        }

        while let Some(entry) = self.metrics_history.front() {
            if now_ms.saturating_sub(entry.at_ms) > ROLLING_WINDOW_MS {
                self.metrics_history.pop_front();
            } else {
                break;
            }
        }
    }

    fn rolling_counter_delta(&self, latest: BargeInMetricsSnapshot) -> BargeInMetricsSnapshot {
        let baseline = self
            .metrics_history
            .front()
            .map(|entry| entry.metrics)
            .unwrap_or_default();

        BargeInMetricsSnapshot {
            requested: latest.requested.saturating_sub(baseline.requested),
            rejected_low_rms: latest
                .rejected_low_rms
                .saturating_sub(baseline.rejected_low_rms),
            rejected_render_ratio: latest
                .rejected_render_ratio
                .saturating_sub(baseline.rejected_render_ratio),
            rejected_persistence: latest
                .rejected_persistence
                .saturating_sub(baseline.rejected_persistence),
            rejected_confidence: latest
                .rejected_confidence
                .saturating_sub(baseline.rejected_confidence),
            ducked: latest.ducked.saturating_sub(baseline.ducked),
            stopped: latest.stopped.saturating_sub(baseline.stopped),
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn percentile(sorted_samples: &[u64], p: f64) -> Option<u64> {
    if sorted_samples.is_empty() {
        return None;
    }
    let idx = ((sorted_samples.len() - 1) as f64 * p).round() as usize;
    sorted_samples.get(idx).copied()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
