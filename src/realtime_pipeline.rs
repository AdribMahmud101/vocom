use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use sherpa_onnx::{LinearResampler, OnlineRecognizer, OnlineStream, VoiceActivityDetector};

use crate::aec_manager::AecProcessor;
use crate::asr_manager::{ASRModelBuilder, OfflineAsrBackend};
use crate::config::{AecConfig, DenoiserConfig, DuplexMode};
use crate::denoiser_manager::DenoiserProcessor;
use crate::duplex_audio::{DuplexPlaybackGate, RenderReferenceConsumer};
use crate::errors::VocomError;

const HISTORY_SECONDS: usize = 30;

#[derive(Clone, Default)]
struct InputOverflowStats {
    dropped_chunks: Arc<AtomicU64>,
    dropped_samples: Arc<AtomicU64>,
    overflow_bursts: Arc<AtomicU64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct InputOverflowSnapshot {
    dropped_chunks: u64,
    dropped_samples: u64,
    overflow_bursts: u64,
}

impl InputOverflowStats {
    fn snapshot(&self) -> InputOverflowSnapshot {
        InputOverflowSnapshot {
            dropped_chunks: self.dropped_chunks.load(Ordering::Acquire),
            dropped_samples: self.dropped_samples.load(Ordering::Acquire),
            overflow_bursts: self.overflow_bursts.load(Ordering::Acquire),
        }
    }
}

struct InputOverflowState {
    active: bool,
    cooldown_until: Option<Instant>,
    shed_ms: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AdaptiveLeakObservation {
    tts_suppressed: bool,
    barge_in_vad_detected: bool,
    requested_emitted: bool,
    rejected_low_rms: bool,
    rejected_render_ratio: bool,
    rejected_persistence: bool,
    rejected_confidence: bool,
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveLeakParams {
    rms_threshold: f32,
    render_ratio_min: f32,
    leak_suppress_ratio_min: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct AdaptiveLeakWindowMetrics {
    req_rate: f32,
    reject_ratio_rate: f32,
    reject_conf_rate: f32,
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveLeakAppliedState {
    reason: &'static str,
    baseline_metrics: AdaptiveLeakWindowMetrics,
    previous_params: AdaptiveLeakParams,
    eval_windows: u32,
    worsen_streak: u32,
}

#[derive(Debug)]
struct AdaptiveLeakTuner {
    enabled: bool,
    observe_only: bool,
    log_interval_ms: u64,
    last_log: Instant,
    last_apply: Option<Instant>,
    last_reason: &'static str,
    reason_streak: u32,
    applied_state: Option<AdaptiveLeakAppliedState>,
    window_chunks: u64,
    window_tts_suppressed: u64,
    window_candidates: u64,
    window_requested: u64,
    window_rejected_low_rms: u64,
    window_rejected_render_ratio: u64,
    window_rejected_persistence: u64,
    window_rejected_confidence: u64,
}

impl AdaptiveLeakTuner {
    fn new(enabled: bool, observe_only: bool, log_interval_ms: u64) -> Self {
        Self {
            enabled,
            observe_only,
            log_interval_ms,
            last_log: Instant::now(),
            last_apply: None,
            last_reason: "stable",
            reason_streak: 0,
            applied_state: None,
            window_chunks: 0,
            window_tts_suppressed: 0,
            window_candidates: 0,
            window_requested: 0,
            window_rejected_low_rms: 0,
            window_rejected_render_ratio: 0,
            window_rejected_persistence: 0,
            window_rejected_confidence: 0,
        }
    }

    fn observe(&mut self, obs: AdaptiveLeakObservation, params: &mut AdaptiveLeakParams) {
        if !self.enabled {
            return;
        }

        self.window_chunks = self.window_chunks.saturating_add(1);
        if obs.tts_suppressed {
            self.window_tts_suppressed = self.window_tts_suppressed.saturating_add(1);
        }
        if obs.barge_in_vad_detected {
            self.window_candidates = self.window_candidates.saturating_add(1);
        }
        if obs.requested_emitted {
            self.window_requested = self.window_requested.saturating_add(1);
        }
        if obs.rejected_low_rms {
            self.window_rejected_low_rms = self.window_rejected_low_rms.saturating_add(1);
        }
        if obs.rejected_render_ratio {
            self.window_rejected_render_ratio = self.window_rejected_render_ratio.saturating_add(1);
        }
        if obs.rejected_persistence {
            self.window_rejected_persistence = self.window_rejected_persistence.saturating_add(1);
        }
        if obs.rejected_confidence {
            self.window_rejected_confidence = self.window_rejected_confidence.saturating_add(1);
        }

        if self.last_log.elapsed() < Duration::from_millis(self.log_interval_ms) {
            return;
        }

        let tts_den = self.window_tts_suppressed.max(1) as f32;
        let cand_den = self.window_candidates.max(1) as f32;
        let metrics = AdaptiveLeakWindowMetrics {
            req_rate: self.window_requested as f32 / tts_den,
            reject_ratio_rate: self.window_rejected_render_ratio as f32 / cand_den,
            reject_conf_rate: self.window_rejected_confidence as f32 / cand_den,
        };

        let mut proposed = *params;
        let mut reason = "stable";

        if self.window_tts_suppressed >= 50 {
            if metrics.req_rate > 0.20 && metrics.reject_ratio_rate < 0.25 {
                proposed.rms_threshold = (proposed.rms_threshold + 0.002).clamp(0.010, 0.200);
                proposed.render_ratio_min = (proposed.render_ratio_min + 0.10).clamp(1.20, 8.00);
                proposed.leak_suppress_ratio_min =
                    (proposed.leak_suppress_ratio_min + 0.08).clamp(1.20, 4.50);
                reason = "tighten_on_high_request";
            } else if metrics.req_rate < 0.02
                && self.window_candidates >= 20
                && (metrics.reject_ratio_rate > 0.80 || metrics.reject_conf_rate > 0.80)
            {
                proposed.rms_threshold = (proposed.rms_threshold - 0.0015).clamp(0.010, 0.200);
                proposed.render_ratio_min = (proposed.render_ratio_min - 0.08).clamp(1.20, 8.00);
                proposed.leak_suppress_ratio_min =
                    (proposed.leak_suppress_ratio_min - 0.06).clamp(1.20, 4.50);
                reason = "loosen_on_over_reject";
            }
        }

        if self.observe_only {
            eprintln!(
                "adaptive-leak-tuner observe-only: reason={} req_rate={:.3} reject_ratio_rate={:.3} reject_conf_rate={:.3} current[rms={:.4} ratio={:.3} leak_ratio={:.3}] proposed[rms={:.4} ratio={:.3} leak_ratio={:.3}]",
                reason,
                metrics.req_rate,
                metrics.reject_ratio_rate,
                metrics.reject_conf_rate,
                params.rms_threshold,
                params.render_ratio_min,
                params.leak_suppress_ratio_min,
                proposed.rms_threshold,
                proposed.render_ratio_min,
                proposed.leak_suppress_ratio_min,
            );
        } else {
            self.evaluate_or_rollback(params, metrics);

            if reason == self.last_reason && reason != "stable" {
                self.reason_streak = self.reason_streak.saturating_add(1);
            } else if reason != "stable" {
                self.reason_streak = 1;
            } else {
                self.reason_streak = 0;
            }
            self.last_reason = reason;

            let can_apply = self
                .last_apply
                .map(|t| t.elapsed() >= Duration::from_millis(self.log_interval_ms.saturating_mul(2)))
                .unwrap_or(true);

            if reason != "stable" && self.reason_streak >= 2 && can_apply {
                let previous_params = *params;
                *params = proposed;
                self.last_apply = Some(Instant::now());
                self.applied_state = Some(AdaptiveLeakAppliedState {
                    reason,
                    baseline_metrics: metrics,
                    previous_params,
                    eval_windows: 0,
                    worsen_streak: 0,
                });

                eprintln!(
                    "adaptive-leak-tuner applied: reason={} req_rate={:.3} reject_ratio_rate={:.3} reject_conf_rate={:.3} new[rms={:.4} ratio={:.3} leak_ratio={:.3}]",
                    reason,
                    metrics.req_rate,
                    metrics.reject_ratio_rate,
                    metrics.reject_conf_rate,
                    params.rms_threshold,
                    params.render_ratio_min,
                    params.leak_suppress_ratio_min,
                );
            } else {
                eprintln!(
                    "adaptive-leak-tuner active-await: reason={} streak={} can_apply={} req_rate={:.3} reject_ratio_rate={:.3} reject_conf_rate={:.3}",
                    reason,
                    self.reason_streak,
                    can_apply,
                    metrics.req_rate,
                    metrics.reject_ratio_rate,
                    metrics.reject_conf_rate,
                );
            }
        }

        self.window_chunks = 0;
        self.window_tts_suppressed = 0;
        self.window_candidates = 0;
        self.window_requested = 0;
        self.window_rejected_low_rms = 0;
        self.window_rejected_render_ratio = 0;
        self.window_rejected_persistence = 0;
        self.window_rejected_confidence = 0;
        self.last_log = Instant::now();
    }

    fn evaluate_or_rollback(&mut self, params: &mut AdaptiveLeakParams, current: AdaptiveLeakWindowMetrics) {
        let Some(mut applied) = self.applied_state else {
            return;
        };

        applied.eval_windows = applied.eval_windows.saturating_add(1);

        let worsened = match applied.reason {
            "tighten_on_high_request" => {
                current.req_rate > applied.baseline_metrics.req_rate * 1.10
                    && current.reject_conf_rate > applied.baseline_metrics.reject_conf_rate + 0.15
            }
            "loosen_on_over_reject" => {
                current.req_rate > applied.baseline_metrics.req_rate + 0.15
                    && (current.reject_ratio_rate >= applied.baseline_metrics.reject_ratio_rate
                        || current.reject_conf_rate >= applied.baseline_metrics.reject_conf_rate)
            }
            _ => false,
        };

        if worsened {
            applied.worsen_streak = applied.worsen_streak.saturating_add(1);
        } else {
            applied.worsen_streak = 0;
        }

        if applied.eval_windows >= 2 && applied.worsen_streak >= 2 {
            *params = applied.previous_params;
            eprintln!(
                "adaptive-leak-tuner rollback: reason={} eval_windows={} req_rate={:.3} reject_ratio_rate={:.3} reject_conf_rate={:.3} restored[rms={:.4} ratio={:.3} leak_ratio={:.3}]",
                applied.reason,
                applied.eval_windows,
                current.req_rate,
                current.reject_ratio_rate,
                current.reject_conf_rate,
                params.rms_threshold,
                params.render_ratio_min,
                params.leak_suppress_ratio_min,
            );
            self.applied_state = None;
            return;
        }

        if applied.eval_windows >= 6 {
            eprintln!(
                "adaptive-leak-tuner commit-window: reason={} eval_windows={} req_rate={:.3} reject_ratio_rate={:.3} reject_conf_rate={:.3}",
                applied.reason,
                applied.eval_windows,
                current.req_rate,
                current.reject_ratio_rate,
                current.reject_conf_rate,
            );
            self.applied_state = None;
            return;
        }

        self.applied_state = Some(applied);
    }
}

impl InputOverflowState {
    fn new(shed_ms: u64) -> Self {
        Self {
            active: false,
            cooldown_until: None,
            shed_ms,
        }
    }

    fn in_cooldown(&mut self) -> bool {
        if let Some(until) = self.cooldown_until {
            if Instant::now() < until {
                return true;
            }
            self.cooldown_until = None;
        }
        false
    }

    fn on_overflow(&mut self) {
        if self.shed_ms > 0 {
            self.cooldown_until = Some(Instant::now() + Duration::from_millis(self.shed_ms));
        }
    }
}

struct PendingSegment {
    start_index: usize,
    end_index: usize,
    samples: Vec<f32>,
}

enum AsrTask {
    OfflineSegment { epoch: u64, samples: Vec<f32> },
    OnlineChunk { epoch: u64, samples: Vec<f32> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AsrQueueSaturationOutcome {
    CoalescedOnline,
    DroppedOffline,
}

pub enum AsrBackend {
    Offline(OfflineAsrBackend),
    Online(OnlineRecognizer),
}

enum AsrRuntime {
    Offline(OfflineAsrBackend),
    Online {
        recognizer: OnlineRecognizer,
        stream: OnlineStream,
        utterance_samples: usize,
    },
}

struct AsrRuntimeWrapper(AsrRuntime);

impl AsrRuntimeWrapper {
    fn into_inner(self) -> AsrRuntime {
        self.0
    }
}

// It is safe to send AsrRuntime to another thread because we give ownership to the worker thread
// and do not use it concurrently from multiple threads.
unsafe impl Send for AsrRuntimeWrapper {}

impl AsrRuntime {
    fn decode_segment(&mut self, samples: &[f32], target_sample_rate: i32) -> Result<String, VocomError> {
        match self {
            AsrRuntime::Offline(recognizer) => {
                ASRModelBuilder::transcribe_samples(recognizer, target_sample_rate, samples)
            }
            AsrRuntime::Online { recognizer, stream, .. } => ASRModelBuilder::transcribe_samples_online(
                recognizer,
                stream,
                target_sample_rate,
                samples,
            ),
        }
    }

    fn decode_online_continuous(&mut self, samples: &[f32], target_sample_rate: i32) -> Option<Result<TranscriptEvent, VocomError>> {
        let AsrRuntime::Online {
            recognizer,
            stream,
            utterance_samples,
        } = self
        else {
            return None;
        };

        if samples.is_empty() {
            return None;
        }

        stream.accept_waveform(target_sample_rate, samples);
        *utterance_samples += samples.len();

        let started = Instant::now();
        while recognizer.is_ready(stream) {
            recognizer.decode(stream);
        }

        let is_endpoint = recognizer.is_endpoint(stream);
        if !is_endpoint {
            return None;
        }

        let result = match recognizer.get_result(stream) {
            Some(r) => r,
            None => {
                recognizer.reset(stream);
                *utterance_samples = 0;
                return Some(Err(VocomError::AsrConfig(
                    "failed to get online transcription result".to_string(),
                )));
            }
        };

        let text = result.text.trim().to_string();
        let sample_count = *utterance_samples;
        recognizer.reset(stream);
        *utterance_samples = 0;

        if text.is_empty() {
            return None;
        }

        Some(Ok(TranscriptEvent {
            text,
            sample_rate: target_sample_rate,
            sample_count,
            decode_latency_ms: started.elapsed().as_millis(),
        }))
    }

    fn reset_online_state(&mut self) {
        if let AsrRuntime::Online {
            recognizer,
            stream,
            utterance_samples,
        } = self
        {
            recognizer.reset(stream);
            *utterance_samples = 0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    pub target_sample_rate: i32,
    pub chunk_ms: u32,
    pub audio_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub duplex_mode: DuplexMode,
    pub input_overflow_log_interval_ms: u64,
    pub input_overflow_shed_ms: u64,
    pub post_roll_ms: u32,
    pub short_silence_merge_ms: u32,
    pub input_normalize_enabled: bool,
    pub input_normalize_target_peak: f32,
    pub input_normalize_max_gain: f32,
    pub input_clip_guard_enabled: bool,
    pub input_clip_threshold: f32,
    pub pre_speech_buffer_ms: u32,
    pub tts_suppression_cooldown_ms: u64,
    pub tts_barge_in_rms_threshold: f32,
    pub tts_barge_in_render_ratio_min: f32,
    pub tts_barge_in_render_ratio_boost_enabled: bool,
    pub tts_barge_in_render_ratio_boost: f32,
    pub tts_barge_in_render_ratio_boost_start_rms: f32,
    pub tts_barge_in_render_ratio_boost_end_rms: f32,
    pub tts_barge_in_render_rms_suppress_threshold: f32,
    pub tts_barge_in_render_rms_suppress_ratio_min: f32,
    pub tts_asr_leak_suppress_enabled: bool,
    pub tts_asr_leak_suppress_ratio_min: f32,
    pub tts_asr_leak_suppress_render_rms_min: f32,
    pub tts_barge_in_render_rms_max_age_ms: u64,
    pub tts_barge_in_persistence_ms: u64,
    pub tts_barge_in_confidence_threshold: f32,
    pub tts_barge_in_confidence_smoothing: f32,
    pub tts_barge_in_suspect_hold_ms: u64,
    pub tts_barge_in_recover_ms: u64,
    /// Minimum interval between successive barge-in requests emitted to the FSM (ms).
    pub barge_in_min_interval_ms: u64,
    pub adaptive_leak_tuner_enabled: bool,
    pub adaptive_leak_tuner_observe_only: bool,
    pub adaptive_leak_tuner_log_interval_ms: u64,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            chunk_ms: 20,
            audio_queue_capacity: 64,
            event_queue_capacity: 64,
            duplex_mode: DuplexMode::FullDuplex,
            input_overflow_log_interval_ms: 5_000,
            input_overflow_shed_ms: 120,
            post_roll_ms: 120,
            short_silence_merge_ms: 150,
            input_normalize_enabled: true,
            input_normalize_target_peak: 0.90,
            input_normalize_max_gain: 3.0,
            input_clip_guard_enabled: true,
            input_clip_threshold: 0.98,
            pre_speech_buffer_ms: 300,
            tts_suppression_cooldown_ms: 700,
            tts_barge_in_rms_threshold: 0.02,
            tts_barge_in_render_ratio_min: 1.35,
            tts_barge_in_render_ratio_boost_enabled: true,
            tts_barge_in_render_ratio_boost: 0.75,
            tts_barge_in_render_ratio_boost_start_rms: 0.02,
            tts_barge_in_render_ratio_boost_end_rms: 0.08,
            tts_barge_in_render_rms_suppress_threshold: 0.12,
            tts_barge_in_render_rms_suppress_ratio_min: 2.6,
            tts_asr_leak_suppress_enabled: true,
            tts_asr_leak_suppress_ratio_min: 1.6,
            tts_asr_leak_suppress_render_rms_min: 0.02,
            tts_barge_in_render_rms_max_age_ms: 200,
            tts_barge_in_persistence_ms: 60,
            tts_barge_in_confidence_threshold: 0.68,
            tts_barge_in_confidence_smoothing: 0.3,
            tts_barge_in_suspect_hold_ms: 90,
            tts_barge_in_recover_ms: 350,
            barge_in_min_interval_ms: 150,
            adaptive_leak_tuner_enabled: false,
            adaptive_leak_tuner_observe_only: true,
            adaptive_leak_tuner_log_interval_ms: 5_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TranscriptEvent {
    pub text: String,
    pub sample_rate: i32,
    pub sample_count: usize,
    pub decode_latency_ms: u128,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RealtimePressureSnapshot {
    pub audio_queue_len: usize,
    pub audio_queue_capacity: usize,
    pub audio_queue_fill_pct: u8,
    pub asr_task_queue_len: usize,
    pub deferred_online_chunk_samples: usize,
    pub denoiser_bypass_active: bool,
    pub shed_silent_chunks: u64,
    pub coalesced_online_chunks: u64,
    pub dropped_offline_segments: u64,
    pub dropped_input_chunks: u64,
    pub dropped_input_samples: u64,
    pub overflow_bursts: u64,
    pub cpu_total_ewma_us: u32,
    pub cpu_resample_ewma_us: u32,
    pub cpu_aec_ewma_us: u32,
    pub cpu_denoise_ewma_us: u32,
    pub cpu_vad_ewma_us: u32,
    pub cpu_over_budget_chunks: u64,
    pub cpu_starvation_events: u64,
    pub asr_epoch: u64,
    pub half_duplex_mute_gate_active: bool,
    pub half_duplex_mute_gate_engagements: u64,
    pub dropped_stale_asr_tasks: u64,
    pub dropped_stale_asr_events: u64,
}

pub struct RealtimeTranscriber {
    stream: Stream,
    audio_rx: Receiver<Vec<f32>>,
    asr_task_tx: Sender<AsrTask>,
    asr_event_rx: Receiver<Result<TranscriptEvent, VocomError>>,
    asr_event_tx_keepalive: Option<Sender<Result<TranscriptEvent, VocomError>>>,
    offline_recognizer: Option<OfflineAsrBackend>,
    is_online_asr: bool,
    vad: VoiceActivityDetector,
    barge_in_vad: VoiceActivityDetector,
    aec: AecProcessor,
    duplex_gate: DuplexPlaybackGate,
    resampler: Option<LinearResampler>,
    target_sample_rate: i32,
    pre_speech_buffer_samples: usize,
    history_samples: VecDeque<f32>,
    history_start_index: usize,
    total_samples_seen: usize,
    history_capacity_samples: usize,
    tts_suppression_cooldown_ms: u64,
    tts_barge_in_rms_threshold: f32,
    tts_barge_in_render_ratio_min: f32,
    tts_barge_in_render_ratio_boost_enabled: bool,
    tts_barge_in_render_ratio_boost: f32,
    tts_barge_in_render_ratio_boost_start_rms: f32,
    tts_barge_in_render_ratio_boost_end_rms: f32,
    tts_barge_in_render_rms_suppress_threshold: f32,
    tts_barge_in_render_rms_suppress_ratio_min: f32,
    tts_asr_leak_suppress_enabled: bool,
    tts_asr_leak_suppress_ratio_min: f32,
    tts_asr_leak_suppress_render_rms_min: f32,
    tts_barge_in_render_rms_max_age_ms: u64,
    tts_barge_in_persistence_ms: u64,
    barge_in_confident_ms: u64,
    barge_in_confidence: f32,
    tts_barge_in_confidence_threshold: f32,
    tts_barge_in_confidence_smoothing: f32,
    barge_in_min_interval_ms: u64,
    online_replay_buffer: VecDeque<f32>,
    online_replay_capacity_samples: usize,
    pending_segments: VecDeque<PendingSegment>,
    deferred_online_chunk: Option<(u64, Vec<f32>)>,
    asr_epoch: Arc<AtomicU64>,
    asr_stale_tasks_dropped: Arc<AtomicU64>,
    post_roll_samples: usize,
    short_silence_merge_samples: usize,
    input_normalize_enabled: bool,
    input_normalize_target_peak: f32,
    input_normalize_max_gain: f32,
    input_clip_guard_enabled: bool,
    input_clip_threshold: f32,
    duplex_mode: DuplexMode,
    vad_dynamic_gate_enabled: bool,
    vad_noise_floor_rms: f32,
    vad_noise_smoothing: f32,
    vad_noise_gate_multiplier: f32,
    vad_noise_gate_min_rms: f32,
    vad_noise_gate_max_rms: f32,
    denoiser: Option<DenoiserProcessor>,
    denoiser_backpressure_bypass: bool,
    last_denoiser_bypass_log: Instant,
    silent_shed_active: bool,
    shed_silent_chunks: u64,
    coalesced_online_chunks: u64,
    dropped_offline_segments: u64,
    pending_events: VecDeque<Result<TranscriptEvent, VocomError>>,
    input_overflow_stats: InputOverflowStats,
    input_overflow_log_interval_ms: u64,
    input_overflow_last_log: Instant,
    input_overflow_last_dropped_chunks: u64,
    input_overflow_last_bursts: u64,
    input_chunk_ms: u64,
    audio_queue_capacity: usize,
    last_backpressure_log: Instant,
    cpu_stage_samples: u64,
    cpu_total_ewma_us: u32,
    cpu_resample_ewma_us: u32,
    cpu_aec_ewma_us: u32,
    cpu_denoise_ewma_us: u32,
    cpu_vad_ewma_us: u32,
    cpu_over_budget_chunks: u64,
    cpu_starvation_events: u64,
    half_duplex_tts_gate_active: bool,
    half_duplex_mute_gate_engagements: u64,
    dropped_stale_asr_events: u64,
    adaptive_leak_tuner: AdaptiveLeakTuner,
}

impl RealtimeTranscriber {
    pub fn recv(&mut self) -> Result<TranscriptEvent, VocomError> {
        loop {
            self.flush_deferred_asr_task();
            self.poll_asr_events();
            if let Some(event) = self.pending_events.pop_front() {
                return event;
            }

            let budget = self.dynamic_drain_budget(64);
            self.drain_audio_nonblocking(budget);
            if let Some(event) = self.pending_events.pop_front() {
                return event;
            }

            crossbeam_channel::select! {
                recv(self.audio_rx) -> msg => {
                    let chunk = msg.map_err(|_| VocomError::ChannelDisconnected)?;
                    self.process_chunk(chunk);
                }
                recv(self.asr_event_rx) -> msg => {
                    let event = msg.map_err(|_| VocomError::ChannelDisconnected)?;
                    self.pending_events.push_back(event);
                }
            }
        }
    }

    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<TranscriptEvent>, VocomError> {
        self.flush_deferred_asr_task();
        self.poll_asr_events();
        if let Some(event) = self.pending_events.pop_front() {
            return event.map(Some);
        }

        let budget = self.dynamic_drain_budget(64);
        self.drain_audio_nonblocking(budget);
        if let Some(event) = self.pending_events.pop_front() {
            return event.map(Some);
        }

        let started = Instant::now();
        let mut remaining = timeout;

        loop {
            crossbeam_channel::select! {
                recv(self.audio_rx) -> msg => {
                    match msg {
                        Ok(chunk) => self.process_chunk(chunk),
                        Err(_) => return Err(VocomError::ChannelDisconnected),
                    }
                }
                recv(self.asr_event_rx) -> msg => {
                    match msg {
                        Ok(event) => self.pending_events.push_back(event),
                        Err(_) => return Err(VocomError::ChannelDisconnected),
                    }
                }
                default(remaining) => return Ok(None),
            }

            self.poll_asr_events();
            self.flush_deferred_asr_task();
            if let Some(event) = self.pending_events.pop_front() {
                return event.map(Some);
            }
            
            let elapsed = started.elapsed();
            if elapsed >= timeout {
                return Ok(None);
            }
            remaining = timeout - elapsed;
        }
    }

    fn poll_asr_events(&mut self) {
        while let Ok(event) = self.asr_event_rx.try_recv() {
            self.pending_events.push_back(event);
        }
    }

    pub fn keep_alive(&self) {
        let _ = &self.stream;
    }

    pub fn set_duplex_mode(&mut self, mode: DuplexMode) {
        self.duplex_mode = mode;
        self.barge_in_confident_ms = 0;
        self.barge_in_confidence = 0.0;
        self.half_duplex_tts_gate_active = false;
        if mode == DuplexMode::HalfDuplexMuteMic && self.duplex_gate.is_tts_active() {
            self.engage_half_duplex_tts_mute_gate();
            self.half_duplex_tts_gate_active = true;
        }
    }

    pub fn duplex_mode(&self) -> DuplexMode {
        self.duplex_mode
    }

    pub fn pressure_snapshot(&self) -> RealtimePressureSnapshot {
        let overflow = self.input_overflow_stats.snapshot();
        let fill_pct = ((self.audio_rx.len().saturating_mul(100))
            / self.audio_queue_capacity.max(1))
            .min(100) as u8;
        RealtimePressureSnapshot {
            audio_queue_len: self.audio_rx.len(),
            audio_queue_capacity: self.audio_queue_capacity,
            audio_queue_fill_pct: fill_pct,
            asr_task_queue_len: self.asr_task_tx.len(),
            deferred_online_chunk_samples: self
                .deferred_online_chunk
                .as_ref()
                .map(|(_, v)| v.len())
                .unwrap_or(0),
            denoiser_bypass_active: self.denoiser_backpressure_bypass,
            shed_silent_chunks: self.shed_silent_chunks,
            coalesced_online_chunks: self.coalesced_online_chunks,
            dropped_offline_segments: self.dropped_offline_segments,
            dropped_input_chunks: overflow.dropped_chunks,
            dropped_input_samples: overflow.dropped_samples,
            overflow_bursts: overflow.overflow_bursts,
            cpu_total_ewma_us: self.cpu_total_ewma_us,
            cpu_resample_ewma_us: self.cpu_resample_ewma_us,
            cpu_aec_ewma_us: self.cpu_aec_ewma_us,
            cpu_denoise_ewma_us: self.cpu_denoise_ewma_us,
            cpu_vad_ewma_us: self.cpu_vad_ewma_us,
            cpu_over_budget_chunks: self.cpu_over_budget_chunks,
            cpu_starvation_events: self.cpu_starvation_events,
            asr_epoch: self.current_asr_epoch(),
            half_duplex_mute_gate_active: self.half_duplex_tts_gate_active,
            half_duplex_mute_gate_engagements: self.half_duplex_mute_gate_engagements,
            dropped_stale_asr_tasks: self.asr_stale_tasks_dropped.load(Ordering::Acquire),
            dropped_stale_asr_events: self.dropped_stale_asr_events,
        }
    }

    fn process_chunk(&mut self, chunk: Vec<f32>) {
        self.report_input_overflow_if_needed();
        let mut resample_us = 0u64;
        let mut aec_us = 0u64;
        let mut denoise_us = 0u64;
        let vad_us;

        // In half-duplex mode we intentionally mute user mic while TTS is active.
        // Skip all heavy processing immediately to reduce callback pressure.
        if self.duplex_mode == DuplexMode::HalfDuplexMuteMic && self.duplex_gate.is_tts_active() {
            if !self.half_duplex_tts_gate_active {
                self.engage_half_duplex_tts_mute_gate();
                self.half_duplex_tts_gate_active = true;
            }
            return;
        }
        self.half_duplex_tts_gate_active = false;

        let audio = if let Some(ref r) = self.resampler {
            let started = Instant::now();
            let out = r.resample(&chunk, false);
            resample_us = started.elapsed().as_micros() as u64;
            out
        } else {
            chunk
        };

        if audio.is_empty() {
            return;
        }

        let half_duplex_mode = self.duplex_mode == DuplexMode::HalfDuplexMuteMic;
        let mut processed = if half_duplex_mode {
            // Half-duplex already mutes mic while TTS is active and disables barge-in.
            // Skip AEC/denoise to reduce unnecessary CPU usage.
            audio
        } else {
            let aec_started = Instant::now();
            let aec_audio = match self.aec.process_capture_chunk(audio) {
                Ok(v) => v,
                Err(err) => {
                    self.pending_events.push_back(Err(err));
                    return;
                }
            };
            aec_us = aec_started.elapsed().as_micros() as u64;

            if aec_audio.is_empty() {
                return;
            }

            let mut out = aec_audio;
            self.update_denoiser_bypass_state();
            if let Some(denoiser) = &mut self.denoiser {
                if !self.denoiser_backpressure_bypass {
                    let denoise_started = Instant::now();
                    match denoiser.process_chunk(out, self.target_sample_rate) {
                        Ok(denoised) => {
                            denoise_us = denoise_started.elapsed().as_micros() as u64;
                            out = denoised
                        }
                        Err(err) => {
                            self.pending_events.push_back(Err(err));
                            return;
                        }
                    }
                }
            }
            out
        };

        if processed.is_empty() {
            return;
        }

        self.apply_input_gain(&mut processed);
        self.apply_clip_guard(&mut processed);

        self.append_history(&processed);

        let chunk_duration_ms = ((processed.len() as u64) * 1000)
            .saturating_div(self.target_sample_rate.max(1) as u64)
            .max(1);
        let rms = compute_rms(&processed);
        let render_rms = self
            .duplex_gate
            .render_rms_recent(self.tts_barge_in_render_rms_max_age_ms);
        self.silent_shed_active = compute_silent_shed_state(
            self.silent_shed_active,
            self.audio_rx.len(),
            self.audio_queue_capacity,
            rms,
            self.vad_noise_gate_min_rms.max(0.006),
        );
        if self.silent_shed_active {
            self.shed_silent_chunks = self.shed_silent_chunks.saturating_add(1);
            self.barge_in_confident_ms = 0;
            self.barge_in_confidence = 0.0;
            return;
        }
        let suppress_asr_for_tts = self.should_suppress_asr_for_tts(render_rms, rms);

        let vad_started = Instant::now();
        if !suppress_asr_for_tts {
            self.vad.accept_waveform(&processed);
        }
        if !half_duplex_mode {
            self.barge_in_vad.accept_waveform(&processed);
        }
        let vad_detected = if suppress_asr_for_tts {
            false
        } else {
            self.vad.detected()
        };
        let barge_in_vad_detected = if half_duplex_mode {
            false
        } else {
            self.barge_in_vad.detected()
        };
        if self.vad_dynamic_gate_enabled && !suppress_asr_for_tts {
            if !vad_detected && !barge_in_vad_detected {
                self.update_noise_floor(rms);
            }
        }
        let vad_gate = if self.vad_dynamic_gate_enabled {
            self.vad_dynamic_gate()
        } else {
            0.0
        };
        vad_us = vad_started.elapsed().as_micros() as u64;
        self.note_cpu_stage_timings(resample_us, aec_us, denoise_us, vad_us);

        if !half_duplex_mode
            && self
            .duplex_gate
            .should_suppress_asr(self.tts_suppression_cooldown_ms)
        {
            let mut obs = AdaptiveLeakObservation {
                tts_suppressed: true,
                barge_in_vad_detected,
                ..Default::default()
            };

            if rms < self.tts_barge_in_rms_threshold {
                self.duplex_gate.note_rejected_low_rms();
                obs.rejected_low_rms = true;
                self.observe_adaptive_leak_tuner(obs);
                self.barge_in_confident_ms = 0;
                self.barge_in_confidence = 0.0;
                return;
            }

            if !barge_in_vad_detected {
                // Not speech according to VAD.
                self.observe_adaptive_leak_tuner(obs);
                self.barge_in_confident_ms = 0;
                self.barge_in_confidence = 0.0;
                return;
            }

            if self.vad_dynamic_gate_enabled && rms < vad_gate {
                obs.rejected_low_rms = true;
                self.duplex_gate.note_rejected_low_rms();
            }

            if let Some(r) = render_rms {
                let ratio_min = self.dynamic_render_ratio_min(r);
                let min_near_end_rms = if r >= self.tts_barge_in_render_rms_suppress_threshold {
                    r * self.tts_barge_in_render_rms_suppress_ratio_min
                } else {
                    r * ratio_min
                };
                if rms < min_near_end_rms {
                    self.duplex_gate.note_rejected_render_ratio();
                    obs.rejected_render_ratio = true;
                }
            }

            self.barge_in_confident_ms = self
                .barge_in_confident_ms
                .saturating_add(chunk_duration_ms)
                .min(self.tts_barge_in_persistence_ms);

            if self.barge_in_confident_ms < self.tts_barge_in_persistence_ms {
                self.duplex_gate.note_rejected_persistence();
                obs.rejected_persistence = true;
            }

            let confidence = compute_barge_in_confidence(
                rms,
                render_rms,
                self.tts_barge_in_rms_threshold,
                self.dynamic_render_ratio_min(render_rms.unwrap_or(0.0)),
                self.barge_in_confident_ms,
                self.tts_barge_in_persistence_ms,
            );

            self.barge_in_confidence = smooth_confidence(
                self.barge_in_confidence,
                confidence,
                self.tts_barge_in_confidence_smoothing,
            );

            if self.barge_in_confidence < self.tts_barge_in_confidence_threshold {
                self.buffer_online_replay_if_enabled(&processed);
                self.duplex_gate.note_rejected_confidence();
                obs.rejected_confidence = true;
            }

            obs.requested_emitted = self.duplex_gate.request_barge_in_if_active(
                rms,
                self.tts_barge_in_rms_threshold,
                self.barge_in_min_interval_ms,
            );
            self.observe_adaptive_leak_tuner(obs);
        } else {
            self.barge_in_confident_ms = 0;
            self.barge_in_confidence = 0.0;
            if !half_duplex_mode {
                self.observe_adaptive_leak_tuner(AdaptiveLeakObservation::default());
            }
        }

        if suppress_asr_for_tts {
            return;
        }

        if !self.is_online_asr {
                while let Some(segment) = self.vad.front() {
                    let samples = segment.samples().to_vec();
                    let segment_start = segment.start().max(0) as usize;
                    self.vad.pop();

                    if samples.is_empty() {
                        continue;
                    }

                    if self.vad_dynamic_gate_enabled {
                        let segment_rms = compute_rms(&samples);
                        if segment_rms < vad_gate {
                            continue;
                        }
                    }

                    self.enqueue_offline_segment(segment_start, samples);
                }

                self.flush_pending_segments();
        } else {
                let mut combined = self.take_online_replay();
                combined.extend_from_slice(&processed);
                if !combined.is_empty() {
                    self.enqueue_asr_task(AsrTask::OnlineChunk {
                        epoch: self.current_asr_epoch(),
                        samples: combined,
                    });
                }
        }
    }

    fn drain_audio_nonblocking(&mut self, max_chunks: usize) {
        for _ in 0..max_chunks {
            let Ok(chunk) = self.audio_rx.try_recv() else {
                break;
            };
            self.process_chunk(chunk);
        }
    }

    fn dynamic_drain_budget(&mut self, base: usize) -> usize {
        let queue_len = self.audio_rx.len();
        let budget = compute_drain_budget(queue_len, self.audio_queue_capacity, base);

        if queue_len > self.audio_queue_capacity.saturating_mul(3) / 4
            && self.last_backpressure_log.elapsed() >= Duration::from_secs(2)
        {
            eprintln!(
                "audio backpressure: queue_len={} capacity={} dynamic_drain_budget={}",
                queue_len, self.audio_queue_capacity, budget
            );
            self.last_backpressure_log = Instant::now();
        }

        budget
    }

    fn update_denoiser_bypass_state(&mut self) {
        if self.denoiser.is_none() {
            self.denoiser_backpressure_bypass = false;
            return;
        }

        let queue_len = self.audio_rx.len();
        let next = compute_denoiser_bypass_state(
            self.denoiser_backpressure_bypass,
            queue_len,
            self.audio_queue_capacity,
        );
        if next != self.denoiser_backpressure_bypass
            && self.last_denoiser_bypass_log.elapsed() >= Duration::from_secs(1)
        {
            if next {
                eprintln!(
                    "backpressure degrade: bypassing denoiser (audio_q={}/{})",
                    queue_len, self.audio_queue_capacity
                );
            } else {
                eprintln!(
                    "backpressure recover: denoiser restored (audio_q={}/{})",
                    queue_len, self.audio_queue_capacity
                );
            }
            self.last_denoiser_bypass_log = Instant::now();
        }
        self.denoiser_backpressure_bypass = next;
    }

    fn enqueue_asr_task(&mut self, task: AsrTask) {
        if let Err(err) = self.asr_task_tx.try_send(task) {
            match err {
                TrySendError::Full(task) => {
                    let cap = (self.target_sample_rate.max(1) as usize) * 2;
                    match handle_asr_queue_full(
                        task,
                        &mut self.deferred_online_chunk,
                        cap,
                        &mut self.coalesced_online_chunks,
                        &mut self.dropped_offline_segments,
                    ) {
                        AsrQueueSaturationOutcome::CoalescedOnline => {
                            eprintln!("ASR task queue full, coalescing online chunk");
                        }
                        AsrQueueSaturationOutcome::DroppedOffline => {
                            eprintln!("ASR task queue full, dropping offline segment");
                        }
                    }
                }
                TrySendError::Disconnected(_) => {}
            }
        }
    }

    fn flush_deferred_asr_task(&mut self) {
        let Some((epoch, samples)) = self.deferred_online_chunk.take() else {
            return;
        };
        match self
            .asr_task_tx
            .try_send(AsrTask::OnlineChunk { epoch, samples })
        {
            Ok(()) => {}
            Err(TrySendError::Full(AsrTask::OnlineChunk { epoch, samples })) => {
                self.deferred_online_chunk = Some((epoch, samples));
            }
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                self.deferred_online_chunk = None;
            }
        }
    }

    fn buffer_online_replay_if_enabled(&mut self, samples: &[f32]) {
        if !self.is_online_asr || samples.is_empty() {
            return;
        }

        self.online_replay_buffer.extend(samples.iter().copied());
        while self.online_replay_buffer.len() > self.online_replay_capacity_samples {
            self.online_replay_buffer.pop_front();
        }
    }

    fn take_online_replay(&mut self) -> Vec<f32> {
        if !self.is_online_asr || self.online_replay_buffer.is_empty() {
            return Vec::new();
        }

        self.online_replay_buffer.drain(..).collect()
    }

    fn update_noise_floor(&mut self, rms: f32) {
        let s = self.vad_noise_smoothing.clamp(0.0, 1.0);
        let next = (1.0 - s) * self.vad_noise_floor_rms + s * rms;
        self.vad_noise_floor_rms = next.max(self.vad_noise_gate_min_rms);
    }

    fn vad_dynamic_gate(&self) -> f32 {
        let gate = self.vad_noise_floor_rms * self.vad_noise_gate_multiplier.max(1.0);
        gate.clamp(self.vad_noise_gate_min_rms, self.vad_noise_gate_max_rms)
    }

    fn dynamic_render_ratio_min(&self, render_rms: f32) -> f32 {
        if !self.tts_barge_in_render_ratio_boost_enabled {
            return self.tts_barge_in_render_ratio_min;
        }

        let start = self.tts_barge_in_render_ratio_boost_start_rms;
        let end = self.tts_barge_in_render_ratio_boost_end_rms.max(start + 1e-6);
        let t = ((render_rms - start) / (end - start)).clamp(0.0, 1.0);
        let boosted = self.tts_barge_in_render_ratio_min
            + self.tts_barge_in_render_ratio_boost * t;
        boosted.max(self.tts_barge_in_render_ratio_min)
    }

    fn should_suppress_asr_for_tts(&self, render_rms: Option<f32>, rms: f32) -> bool {
        if !self.tts_asr_leak_suppress_enabled || !self.duplex_gate.is_tts_active() {
            return false;
        }
        let Some(render_rms) = render_rms else {
            return false;
        };
        if render_rms < self.tts_asr_leak_suppress_render_rms_min {
            return false;
        }
        rms < render_rms * self.tts_asr_leak_suppress_ratio_min
    }

    fn append_history(&mut self, samples: &[f32]) {
        self.history_samples.extend(samples.iter().copied());
        self.total_samples_seen = self.total_samples_seen.saturating_add(samples.len());

        while self.history_samples.len() > self.history_capacity_samples {
            self.history_samples.pop_front();
            self.history_start_index = self.history_start_index.saturating_add(1);
        }
    }

    fn history_range(&self, start: usize, end: usize) -> Vec<f32> {
        if start >= end {
            return Vec::new();
        }

        let available_start = start.max(self.history_start_index);
        let available_end = end.min(self.total_samples_seen);
        if available_start >= available_end {
            return Vec::new();
        }

        let from = available_start - self.history_start_index;
        let to = available_end - self.history_start_index;
        self.history_samples
            .iter()
            .skip(from)
            .take(to - from)
            .copied()
            .collect()
    }

    fn apply_input_gain(&self, samples: &mut [f32]) {
        if !self.input_normalize_enabled || samples.is_empty() {
            return;
        }

        let peak = samples
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        if peak <= 0.0 {
            return;
        }

        let gain = (self.input_normalize_target_peak / peak)
            .min(self.input_normalize_max_gain)
            .max(0.0);
        if !gain.is_finite() || (gain - 1.0).abs() < 0.01 {
            return;
        }

        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }

    fn apply_clip_guard(&self, samples: &mut [f32]) {
        if !self.input_clip_guard_enabled || samples.is_empty() {
            return;
        }

        let threshold = self.input_clip_threshold.clamp(0.0, 1.0);
        for sample in samples.iter_mut() {
            if *sample > threshold {
                *sample = threshold;
            } else if *sample < -threshold {
                *sample = -threshold;
            }
        }
    }

    fn enqueue_offline_segment(&mut self, segment_start: usize, samples: Vec<f32>) {
        let segment_end = segment_start.saturating_add(samples.len());

        let mut merge_from = None;
        if let Some(last) = self.pending_segments.back() {
            let gap = segment_start.saturating_sub(last.end_index);
            if gap <= self.short_silence_merge_samples {
                merge_from = Some(last.end_index);
            }
        }

        if let Some(last_end) = merge_from {
            let bridge = self.history_range(last_end, segment_start);
            if let Some(last) = self.pending_segments.back_mut() {
                if !bridge.is_empty() {
                    last.samples.extend_from_slice(&bridge);
                }
                last.samples.extend_from_slice(&samples);
                last.end_index = segment_end;
            }
            return;
        }

        self.pending_segments.push_back(PendingSegment {
            start_index: segment_start,
            end_index: segment_end,
            samples,
        });
    }

    fn flush_pending_segments(&mut self) {
        loop {
            let Some(front) = self.pending_segments.front() else {
                return;
            };

            let post_roll_end = front.end_index.saturating_add(self.post_roll_samples);
            if self.total_samples_seen < post_roll_end {
                return;
            }

            let pending = self.pending_segments.pop_front().expect("pending segment");
            let pre_start = pending
                .start_index
                .saturating_sub(self.pre_speech_buffer_samples);
            let mut samples_with_roll =
                self.history_range(pre_start, pending.start_index);
            samples_with_roll.extend_from_slice(&pending.samples);
            if self.post_roll_samples > 0 {
                let post_roll =
                    self.history_range(pending.end_index, pending.end_index + self.post_roll_samples);
                samples_with_roll.extend_from_slice(&post_roll);
            }

            if let Some(recognizer) = self.offline_recognizer.as_ref() {
                let started = Instant::now();
                let result = ASRModelBuilder::transcribe_samples(
                    recognizer,
                    self.target_sample_rate,
                    &samples_with_roll,
                )
                .map(|text| TranscriptEvent {
                    text,
                    sample_rate: self.target_sample_rate,
                    sample_count: samples_with_roll.len(),
                    decode_latency_ms: started.elapsed().as_millis(),
                });
                self.pending_events.push_back(result);
            } else {
                self.enqueue_asr_task(AsrTask::OfflineSegment {
                    epoch: self.current_asr_epoch(),
                    samples: samples_with_roll,
                });
            }
        }
    }

    fn current_asr_epoch(&self) -> u64 {
        self.asr_epoch.load(Ordering::Acquire)
    }

    fn bump_asr_epoch(&self) -> u64 {
        self.asr_epoch.fetch_add(1, Ordering::AcqRel).saturating_add(1)
    }

    fn clear_capture_buffers(&mut self) {
        self.barge_in_confident_ms = 0;
        self.barge_in_confidence = 0.0;
        self.online_replay_buffer.clear();
        self.pending_segments.clear();
        self.deferred_online_chunk = None;
        self.pending_events.clear();
        self.history_samples.clear();
        // Keep timeline monotonic. VAD segment start/end indices are based on
        // continuous accepted waveform time. Resetting these counters to zero
        // while VAD state is still live can stall post-roll flushing and make
        // subsequent speech appear "missing" after TTS windows.
        self.history_start_index = self.total_samples_seen;
    }

    fn drain_asr_events(&mut self) -> usize {
        let mut drained = 0usize;
        while self.asr_event_rx.try_recv().is_ok() {
            drained = drained.saturating_add(1);
        }
        drained
    }

    fn engage_half_duplex_tts_mute_gate(&mut self) {
        let epoch = self.bump_asr_epoch();
        self.half_duplex_mute_gate_engagements =
            self.half_duplex_mute_gate_engagements.saturating_add(1);
        self.clear_capture_buffers();
        let drained = self.drain_asr_events();
        self.dropped_stale_asr_events = self
            .dropped_stale_asr_events
            .saturating_add(drained as u64);
        if drained > 0 {
            eprintln!(
                "half-duplex mute gate engaged: dropped {drained} stale ASR events at epoch={epoch}"
            );
        }
    }

    fn report_input_overflow_if_needed(&mut self) {
        if self.input_overflow_log_interval_ms == 0 {
            return;
        }

        if self.input_overflow_last_log.elapsed()
            < Duration::from_millis(self.input_overflow_log_interval_ms)
        {
            return;
        }

        let snapshot = self.input_overflow_stats.snapshot();
        let delta_chunks = snapshot
            .dropped_chunks
            .saturating_sub(self.input_overflow_last_dropped_chunks);
        let delta_bursts = snapshot
            .overflow_bursts
            .saturating_sub(self.input_overflow_last_bursts);

        if delta_chunks > 0 || delta_bursts > 0 {
            let dropped_ms = delta_chunks.saturating_mul(self.input_chunk_ms);
            eprintln!(
                "input overflow: dropped {delta_chunks} chunks (~{dropped_ms} ms), bursts {delta_bursts}, total dropped {} chunks ({} samples)",
                snapshot.dropped_chunks,
                snapshot.dropped_samples
            );
            self.input_overflow_last_dropped_chunks = snapshot.dropped_chunks;
            self.input_overflow_last_bursts = snapshot.overflow_bursts;
        }

        self.input_overflow_last_log = Instant::now();
    }

    fn observe_adaptive_leak_tuner(&mut self, obs: AdaptiveLeakObservation) {
        let mut params = AdaptiveLeakParams {
            rms_threshold: self.tts_barge_in_rms_threshold,
            render_ratio_min: self.tts_barge_in_render_ratio_min,
            leak_suppress_ratio_min: self.tts_asr_leak_suppress_ratio_min,
        };

        self.adaptive_leak_tuner.observe(obs, &mut params);

        self.tts_barge_in_rms_threshold = params.rms_threshold;
        self.tts_barge_in_render_ratio_min = params.render_ratio_min;
        self.tts_asr_leak_suppress_ratio_min = params.leak_suppress_ratio_min;
    }

    fn note_cpu_stage_timings(
        &mut self,
        resample_us: u64,
        aec_us: u64,
        denoise_us: u64,
        vad_us: u64,
    ) {
        let total_us = resample_us
            .saturating_add(aec_us)
            .saturating_add(denoise_us)
            .saturating_add(vad_us);

        self.cpu_stage_samples = self.cpu_stage_samples.saturating_add(1);
        self.cpu_total_ewma_us = ewma_us(self.cpu_total_ewma_us, total_us);
        self.cpu_resample_ewma_us = ewma_us(self.cpu_resample_ewma_us, resample_us);
        self.cpu_aec_ewma_us = ewma_us(self.cpu_aec_ewma_us, aec_us);
        self.cpu_denoise_ewma_us = ewma_us(self.cpu_denoise_ewma_us, denoise_us);
        self.cpu_vad_ewma_us = ewma_us(self.cpu_vad_ewma_us, vad_us);

        let budget_us = self.input_chunk_ms.saturating_mul(1000);
        if total_us > budget_us {
            self.cpu_over_budget_chunks = self.cpu_over_budget_chunks.saturating_add(1);
        }
        if total_us > budget_us.saturating_mul(2) {
            self.cpu_starvation_events = self.cpu_starvation_events.saturating_add(1);
        }
    }
}

pub fn start_realtime_transcriber(
    asr_backend: AsrBackend,
    vad: VoiceActivityDetector,
    barge_in_vad: VoiceActivityDetector,
    config: RealtimeConfig,
    aec_config: AecConfig,
    denoiser_config: DenoiserConfig,
    render_reference_consumer: RenderReferenceConsumer,
    duplex_gate: DuplexPlaybackGate,
    vad_dynamic_gate_enabled: bool,
    vad_noise_smoothing: f32,
    vad_noise_gate_multiplier: f32,
    vad_noise_gate_min_rms: f32,
    vad_noise_gate_max_rms: f32,
) -> Result<RealtimeTranscriber, VocomError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(VocomError::NoInputDevice)?;

    let default_config = device.default_input_config()?;
    let input_sample_rate = default_config.sample_rate() as i32;
    let channels = default_config.channels() as usize;
    let stream_config: cpal::StreamConfig = default_config.clone().into();

    let chunk_frames = ((input_sample_rate as u32 * config.chunk_ms) / 1000).max(1) as usize;

    let (audio_tx, audio_rx) = bounded::<Vec<f32>>(config.audio_queue_capacity);
    let input_overflow_stats = InputOverflowStats::default();
    let stream = build_input_stream(
        &device,
        &stream_config,
        default_config.sample_format(),
        channels,
        chunk_frames,
        audio_tx,
        input_overflow_stats.clone(),
        config.input_overflow_shed_ms,
    )?;

    let target_sample_rate = config.target_sample_rate;

    let resampler = if input_sample_rate != target_sample_rate {
        Some(
            LinearResampler::create(input_sample_rate, target_sample_rate).ok_or_else(|| {
                VocomError::Stream(format!(
                    "failed to create resampler: {input_sample_rate} -> {target_sample_rate}"
                ))
            })?,
        )
    } else {
        None
    };

    let aec = AecProcessor::new(&aec_config, target_sample_rate, render_reference_consumer)?;
    let denoiser = DenoiserProcessor::new(&denoiser_config, target_sample_rate)?;
    let pre_speech_buffer_samples =
        (target_sample_rate as usize * config.pre_speech_buffer_ms as usize) / 1000;
    let post_roll_samples = (target_sample_rate as usize * config.post_roll_ms as usize) / 1000;
    let short_silence_merge_samples =
        (target_sample_rate as usize * config.short_silence_merge_ms as usize) / 1000;
    let history_capacity_samples = (target_sample_rate as usize) * HISTORY_SECONDS;
    let online_replay_capacity_samples =
        pre_speech_buffer_samples.max(target_sample_rate as usize / 5);

    stream.play()?;

    let mut offline_recognizer = None;
    let mut online_asr_wrapper = None;
    match asr_backend {
        AsrBackend::Offline(recognizer) => {
            offline_recognizer = Some(recognizer);
        }
        AsrBackend::Online(recognizer) => {
            online_asr_wrapper = Some(AsrRuntimeWrapper(AsrRuntime::Online {
                stream: recognizer.create_stream(),
                recognizer,
                utterance_samples: 0,
            }));
        }
    };
    let is_online_asr = online_asr_wrapper.is_some();

    // Keep queue deep enough for short bursts but avoid long stale backlogs
    // that inflate end-to-end transcript latency under sustained load.
    let asr_task_queue_capacity = config.event_queue_capacity.saturating_mul(2).max(32);
    let (asr_task_tx, asr_task_rx) = bounded::<AsrTask>(asr_task_queue_capacity);
    let (asr_event_tx, asr_event_rx) =
        bounded::<Result<TranscriptEvent, VocomError>>(config.event_queue_capacity);
    let mut asr_event_tx_keepalive = None;
    let asr_epoch = Arc::new(AtomicU64::new(0));
    let asr_stale_tasks_dropped = Arc::new(AtomicU64::new(0));

    if let Some(asr_wrapper) = online_asr_wrapper {
        let asr_epoch_worker = Arc::clone(&asr_epoch);
        let stale_tasks_dropped_worker = Arc::clone(&asr_stale_tasks_dropped);
        std::thread::Builder::new()
            .name("vocom-asr-worker".into())
            .spawn(move || {
                let mut asr = asr_wrapper.into_inner();
                let mut worker_epoch = asr_epoch_worker.load(Ordering::Acquire);
                for task in asr_task_rx {
                    match task {
                        AsrTask::OfflineSegment { epoch, samples } => {
                            let live_epoch = asr_epoch_worker.load(Ordering::Acquire);
                            if epoch != live_epoch {
                                stale_tasks_dropped_worker.fetch_add(1, Ordering::AcqRel);
                                continue;
                            }
                            worker_epoch = epoch;
                            let started = Instant::now();
                            let result = asr.decode_segment(&samples, target_sample_rate).map(|text| {
                                TranscriptEvent {
                                    text,
                                    sample_rate: target_sample_rate,
                                    sample_count: samples.len(),
                                    decode_latency_ms: started.elapsed().as_millis(),
                                }
                            });
                            if asr_event_tx.send(result).is_err() {
                                break;
                            }
                        }
                        AsrTask::OnlineChunk { epoch, samples } => {
                            let live_epoch = asr_epoch_worker.load(Ordering::Acquire);
                            if epoch != live_epoch {
                                stale_tasks_dropped_worker.fetch_add(1, Ordering::AcqRel);
                                continue;
                            }
                            if epoch != worker_epoch {
                                asr.reset_online_state();
                                worker_epoch = epoch;
                            }
                            if let Some(event) = asr.decode_online_continuous(&samples, target_sample_rate) {
                                if asr_event_tx.send(event).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
            })
            .map_err(|e| VocomError::AsrConfig(format!("failed to spawn ASR worker thread: {e}")))?;
    } else {
        // Offline ASR decodes inline on the transcriber thread. Keep one sender
        // alive so `asr_event_rx` doesn't become permanently disconnected.
        asr_event_tx_keepalive = Some(asr_event_tx);
    }

    Ok(RealtimeTranscriber {
        stream,
        audio_rx,
        asr_task_tx,
        asr_event_rx,
        asr_event_tx_keepalive,
        offline_recognizer,
        is_online_asr,
        vad,
        barge_in_vad,
        aec,
        duplex_gate,
        resampler,
        target_sample_rate,
        pre_speech_buffer_samples,
        history_samples: VecDeque::with_capacity(history_capacity_samples),
        history_start_index: 0,
        total_samples_seen: 0,
        history_capacity_samples,
        tts_suppression_cooldown_ms: config.tts_suppression_cooldown_ms,
        tts_barge_in_rms_threshold: config.tts_barge_in_rms_threshold,
        tts_barge_in_render_ratio_min: config.tts_barge_in_render_ratio_min,
        tts_barge_in_render_ratio_boost_enabled: config.tts_barge_in_render_ratio_boost_enabled,
        tts_barge_in_render_ratio_boost: config.tts_barge_in_render_ratio_boost,
        tts_barge_in_render_ratio_boost_start_rms: config.tts_barge_in_render_ratio_boost_start_rms,
        tts_barge_in_render_ratio_boost_end_rms: config.tts_barge_in_render_ratio_boost_end_rms,
        tts_barge_in_render_rms_suppress_threshold: config.tts_barge_in_render_rms_suppress_threshold,
        tts_barge_in_render_rms_suppress_ratio_min: config.tts_barge_in_render_rms_suppress_ratio_min,
        tts_asr_leak_suppress_enabled: config.tts_asr_leak_suppress_enabled,
        tts_asr_leak_suppress_ratio_min: config.tts_asr_leak_suppress_ratio_min,
        tts_asr_leak_suppress_render_rms_min: config.tts_asr_leak_suppress_render_rms_min,
        tts_barge_in_render_rms_max_age_ms: config.tts_barge_in_render_rms_max_age_ms,
        tts_barge_in_persistence_ms: config.tts_barge_in_persistence_ms,
        barge_in_confident_ms: 0,
        barge_in_confidence: 0.0,
        tts_barge_in_confidence_threshold: config.tts_barge_in_confidence_threshold,
        tts_barge_in_confidence_smoothing: config.tts_barge_in_confidence_smoothing,
        barge_in_min_interval_ms: config.barge_in_min_interval_ms,
        online_replay_buffer: VecDeque::with_capacity(online_replay_capacity_samples),
        online_replay_capacity_samples,
        pending_segments: VecDeque::new(),
        deferred_online_chunk: None,
        asr_epoch,
        asr_stale_tasks_dropped,
        post_roll_samples,
        short_silence_merge_samples,
        input_normalize_enabled: config.input_normalize_enabled,
        input_normalize_target_peak: config.input_normalize_target_peak,
        input_normalize_max_gain: config.input_normalize_max_gain,
        input_clip_guard_enabled: config.input_clip_guard_enabled,
        input_clip_threshold: config.input_clip_threshold,
        duplex_mode: config.duplex_mode,
        vad_dynamic_gate_enabled,
        vad_noise_floor_rms: vad_noise_gate_min_rms.max(1e-6),
        vad_noise_smoothing,
        vad_noise_gate_multiplier,
        vad_noise_gate_min_rms,
        vad_noise_gate_max_rms,
        denoiser,
        denoiser_backpressure_bypass: false,
        last_denoiser_bypass_log: Instant::now(),
        silent_shed_active: false,
        shed_silent_chunks: 0,
        coalesced_online_chunks: 0,
        dropped_offline_segments: 0,
        pending_events: VecDeque::with_capacity(config.event_queue_capacity),
        input_overflow_stats,
        input_overflow_log_interval_ms: config.input_overflow_log_interval_ms,
        input_overflow_last_log: Instant::now(),
        input_overflow_last_dropped_chunks: 0,
        input_overflow_last_bursts: 0,
        input_chunk_ms: config.chunk_ms as u64,
        audio_queue_capacity: config.audio_queue_capacity.max(1),
        last_backpressure_log: Instant::now(),
        cpu_stage_samples: 0,
        cpu_total_ewma_us: 0,
        cpu_resample_ewma_us: 0,
        cpu_aec_ewma_us: 0,
        cpu_denoise_ewma_us: 0,
        cpu_vad_ewma_us: 0,
        cpu_over_budget_chunks: 0,
        cpu_starvation_events: 0,
        half_duplex_tts_gate_active: false,
        half_duplex_mute_gate_engagements: 0,
        dropped_stale_asr_events: 0,
        adaptive_leak_tuner: AdaptiveLeakTuner::new(
            config.adaptive_leak_tuner_enabled,
            config.adaptive_leak_tuner_observe_only,
            config.adaptive_leak_tuner_log_interval_ms,
        ),
    })
}

fn smooth_confidence(previous: f32, current: f32, smoothing: f32) -> f32 {
    let s = smoothing.clamp(0.0, 1.0);
    ((1.0 - s) * previous + s * current).clamp(0.0, 1.0)
}

fn compute_barge_in_confidence(
    near_end_rms: f32,
    render_rms: Option<f32>,
    rms_threshold: f32,
    ratio_min: f32,
    confident_ms: u64,
    persistence_ms: u64,
) -> f32 {
    let energy_component = (near_end_rms / rms_threshold.max(1e-6)).clamp(0.0, 1.0);

    let ratio_component = if let Some(render) = render_rms {
        let ratio = near_end_rms / render.max(1e-6);
        (ratio / ratio_min.max(1e-6)).clamp(0.0, 1.0)
    } else {
        0.6
    };

    let persistence_component = (confident_ms as f32 / persistence_ms.max(1) as f32)
        .clamp(0.0, 1.0);

    (0.40 * energy_component + 0.35 * ratio_component + 0.25 * persistence_component)
        .clamp(0.0, 1.0)
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: SampleFormat,
    channels: usize,
    chunk_frames: usize,
    audio_tx: Sender<Vec<f32>>,
    overflow_stats: InputOverflowStats,
    overflow_shed_ms: u64,
) -> Result<Stream, VocomError> {
    let err_fn = |err: cpal::StreamError| {
        eprintln!("audio stream error: {err}");
    };

    match format {
        SampleFormat::F32 => {
            let tx = audio_tx.clone();
            let mut pending = Vec::<f32>::with_capacity(chunk_frames * 2);
            let mut overflow_state = InputOverflowState::new(overflow_shed_ms);
            let stats = overflow_stats;
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    if overflow_state.in_cooldown() {
                        pending.clear();
                        return;
                    }
                    push_mono_frames_f32(
                        data,
                        channels,
                        chunk_frames,
                        &tx,
                        &mut pending,
                        &stats,
                        &mut overflow_state,
                    )
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        SampleFormat::I16 => {
            let tx = audio_tx.clone();
            let mut pending = Vec::<f32>::with_capacity(chunk_frames * 2);
            let mut overflow_state = InputOverflowState::new(overflow_shed_ms);
            let stats = overflow_stats;
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    if overflow_state.in_cooldown() {
                        pending.clear();
                        return;
                    }
                    push_mono_frames_i16(
                        data,
                        channels,
                        chunk_frames,
                        &tx,
                        &mut pending,
                        &stats,
                        &mut overflow_state,
                    )
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        SampleFormat::U16 => {
            let tx = audio_tx;
            let mut pending = Vec::<f32>::with_capacity(chunk_frames * 2);
            let mut overflow_state = InputOverflowState::new(overflow_shed_ms);
            let stats = overflow_stats;
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    if overflow_state.in_cooldown() {
                        pending.clear();
                        return;
                    }
                    push_mono_frames_u16(
                        data,
                        channels,
                        chunk_frames,
                        &tx,
                        &mut pending,
                        &stats,
                        &mut overflow_state,
                    )
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => Err(VocomError::Stream(format!(
            "unsupported input sample format: {other:?}"
        ))),
    }
}

fn push_mono_frames_f32(
    data: &[f32],
    channels: usize,
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
    overflow_stats: &InputOverflowStats,
    overflow_state: &mut InputOverflowState,
) {
    for frame in data.chunks(channels) {
        let mono = frame.iter().copied().sum::<f32>() / channels as f32;
        pending.push(mono);
        flush_chunks(
            chunk_frames,
            tx,
            pending,
            overflow_stats,
            overflow_state,
        );
    }
}

fn push_mono_frames_i16(
    data: &[i16],
    channels: usize,
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
    overflow_stats: &InputOverflowStats,
    overflow_state: &mut InputOverflowState,
) {
    for frame in data.chunks(channels) {
        let mono = frame
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .sum::<f32>()
            / channels as f32;
        pending.push(mono);
        flush_chunks(
            chunk_frames,
            tx,
            pending,
            overflow_stats,
            overflow_state,
        );
    }
}

fn push_mono_frames_u16(
    data: &[u16],
    channels: usize,
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
    overflow_stats: &InputOverflowStats,
    overflow_state: &mut InputOverflowState,
) {
    for frame in data.chunks(channels) {
        let mono = frame
            .iter()
            .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
            .sum::<f32>()
            / channels as f32;
        pending.push(mono);
        flush_chunks(
            chunk_frames,
            tx,
            pending,
            overflow_stats,
            overflow_state,
        );
    }
}

fn flush_chunks(
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
    overflow_stats: &InputOverflowStats,
    overflow_state: &mut InputOverflowState,
) {
    while pending.len() >= chunk_frames {
        let chunk: Vec<f32> = pending.drain(..chunk_frames).collect();
        match tx.try_send(chunk) {
            Ok(_) => {
                overflow_state.active = false;
            }
            Err(TrySendError::Full(_)) => {
                overflow_stats
                    .dropped_chunks
                    .fetch_add(1, Ordering::AcqRel);
                overflow_stats
                    .dropped_samples
                    .fetch_add(chunk_frames as u64, Ordering::AcqRel);
                if !overflow_state.active {
                    overflow_state.active = true;
                    overflow_stats
                        .overflow_bursts
                        .fetch_add(1, Ordering::AcqRel);
                }
                overflow_state.on_overflow();
                pending.clear();
                return;
            }
            Err(TrySendError::Disconnected(_)) => return,
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

fn compute_drain_budget(queue_len: usize, queue_capacity: usize, base: usize) -> usize {
    let cap = queue_capacity.max(1);
    let base = base.max(1);
    let q25 = (cap + 3) / 4;
    let q50 = cap.div_ceil(2);
    let q75 = cap.saturating_mul(3).div_ceil(4);
    let q90 = cap.saturating_mul(9).div_ceil(10);

    if queue_len >= cap {
        base.max(1024)
    } else if queue_len >= q90 {
        base.max(768)
    } else if queue_len >= q75 {
        base.max(512)
    } else if queue_len >= q50 {
        base.max(256)
    } else if queue_len >= q25 {
        base.max(128)
    } else {
        base
    }
}

fn compute_denoiser_bypass_state(current: bool, queue_len: usize, queue_capacity: usize) -> bool {
    let cap = queue_capacity.max(1);
    // Hysteresis to avoid rapid toggling.
    // Engage earlier (60%) so we preserve real-time behavior before hard overflow,
    // then recover only after pressure materially drops (30%).
    let engage = cap.saturating_mul(3).div_ceil(5); // 60%
    let recover = cap.saturating_mul(3).div_ceil(10); // 30%
    if current {
        queue_len > recover
    } else {
        queue_len >= engage
    }
}

fn compute_silent_shed_state(
    current: bool,
    queue_len: usize,
    queue_capacity: usize,
    rms: f32,
    silence_rms_threshold: f32,
) -> bool {
    let cap = queue_capacity.max(1);
    // Hysteresis to avoid rapid toggle near overload threshold.
    // Engage at 60% queue pressure to shed silent frames before hard overflow.
    let engage = cap.saturating_mul(3).div_ceil(5); // 60%
    let recover = cap.saturating_mul(3).div_ceil(10); // 30%
    if current {
        queue_len > recover && rms <= silence_rms_threshold * 1.5
    } else {
        queue_len >= engage && rms <= silence_rms_threshold
    }
}

fn limit_latest_samples(mut samples: Vec<f32>, max_samples: usize) -> Vec<f32> {
    if samples.len() <= max_samples {
        return samples;
    }
    let keep_from = samples.len() - max_samples;
    samples.drain(..keep_from);
    samples
}

fn merge_online_chunks(deferred: &mut Vec<f32>, incoming: Vec<f32>, max_samples: usize) {
    deferred.extend_from_slice(&incoming);
    if deferred.len() > max_samples {
        let drop = deferred.len() - max_samples;
        deferred.drain(..drop);
    }
}

fn handle_asr_queue_full(
    task: AsrTask,
    deferred_online_chunk: &mut Option<(u64, Vec<f32>)>,
    max_online_samples: usize,
    coalesced_online_chunks: &mut u64,
    dropped_offline_segments: &mut u64,
) -> AsrQueueSaturationOutcome {
    match task {
        AsrTask::OnlineChunk { epoch, samples } => {
            if let Some((deferred_epoch, deferred)) = deferred_online_chunk {
                if *deferred_epoch == epoch {
                    merge_online_chunks(deferred, samples, max_online_samples);
                } else {
                    *deferred_epoch = epoch;
                    *deferred = limit_latest_samples(samples, max_online_samples);
                }
            } else {
                *deferred_online_chunk =
                    Some((epoch, limit_latest_samples(samples, max_online_samples)));
            }
            *coalesced_online_chunks = coalesced_online_chunks.saturating_add(1);
            AsrQueueSaturationOutcome::CoalescedOnline
        }
        AsrTask::OfflineSegment { .. } => {
            *dropped_offline_segments = dropped_offline_segments.saturating_add(1);
            AsrQueueSaturationOutcome::DroppedOffline
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_denoiser_bypass_state, compute_drain_budget, flush_chunks, handle_asr_queue_full,
        merge_online_chunks, AsrQueueSaturationOutcome, AsrTask, InputOverflowState,
        InputOverflowStats, compute_silent_shed_state,
        ewma_us,
    };
    use crossbeam_channel::bounded;
    use std::time::{Duration, Instant};

    #[test]
    fn drain_budget_scales_with_pressure() {
        let cap = 256usize;
        assert_eq!(compute_drain_budget(0, cap, 64), 64);
        assert_eq!(compute_drain_budget(63, cap, 64), 64);
        assert_eq!(compute_drain_budget(64, cap, 64), 128);
        assert_eq!(compute_drain_budget(128, cap, 64), 256);
        assert_eq!(compute_drain_budget(192, cap, 64), 512);
        assert_eq!(compute_drain_budget(231, cap, 64), 768);
        assert_eq!(compute_drain_budget(256, cap, 64), 1024);
    }

    #[test]
    fn drain_budget_handles_small_capacity() {
        assert_eq!(compute_drain_budget(0, 0, 0), 1);
        assert_eq!(compute_drain_budget(1, 1, 1), 1024);
    }

    #[test]
    fn online_chunk_coalescing_keeps_latest_audio() {
        let mut deferred = vec![1.0, 2.0, 3.0, 4.0];
        merge_online_chunks(&mut deferred, vec![5.0, 6.0, 7.0], 5);
        assert_eq!(deferred, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn denoiser_bypass_hysteresis() {
        let cap = 100usize;
        assert!(!compute_denoiser_bypass_state(false, 59, cap));
        assert!(compute_denoiser_bypass_state(false, 60, cap));
        assert!(compute_denoiser_bypass_state(true, 31, cap));
        assert!(!compute_denoiser_bypass_state(true, 30, cap));
    }

    #[test]
    fn shed_silent_chunks_only_when_overloaded() {
        assert!(!compute_silent_shed_state(false, 59, 100, 0.001, 0.006));
        assert!(!compute_silent_shed_state(false, 60, 100, 0.020, 0.006));
        assert!(compute_silent_shed_state(false, 60, 100, 0.004, 0.006));
    }

    #[test]
    fn silent_shed_hysteresis_prevents_flapping() {
        // Engage once overloaded and silent.
        let mut active = compute_silent_shed_state(false, 60, 100, 0.004, 0.006);
        assert!(active);

        // Stay active even if queue dips slightly (still above recovery floor).
        active = compute_silent_shed_state(active, 40, 100, 0.004, 0.006);
        assert!(active);

        // Recover once pressure clears.
        active = compute_silent_shed_state(active, 30, 100, 0.004, 0.006);
        assert!(!active);

        // Do not engage on louder input.
        active = compute_silent_shed_state(false, 80, 100, 0.020, 0.006);
        assert!(!active);
    }

    #[test]
    fn audio_ingress_overload_tracks_drops_and_bursts() {
        let (tx, _rx) = bounded::<Vec<f32>>(1);
        let _ = tx.try_send(vec![0.0; 4]);

        let stats = InputOverflowStats::default();
        let mut state = InputOverflowState::new(0);
        let mut pending = vec![0.1; 4];

        flush_chunks(4, &tx, &mut pending, &stats, &mut state);
        let first = stats.snapshot();
        assert_eq!(first.dropped_chunks, 1);
        assert_eq!(first.dropped_samples, 4);
        assert_eq!(first.overflow_bursts, 1);
        assert!(state.active);
        assert!(pending.is_empty());

        pending = vec![0.2; 4];
        flush_chunks(4, &tx, &mut pending, &stats, &mut state);
        let second = stats.snapshot();
        assert_eq!(second.dropped_chunks, 2);
        assert_eq!(second.dropped_samples, 8);
        // Same continuous overload should not count a second burst.
        assert_eq!(second.overflow_bursts, 1);
    }

    #[test]
    fn asr_queue_overload_coalesces_online_and_drops_offline() {
        let mut deferred = None;
        let mut coalesced = 0u64;
        let mut dropped = 0u64;

        let online = handle_asr_queue_full(
            AsrTask::OnlineChunk {
                epoch: 7,
                samples: vec![1.0, 2.0, 3.0],
            },
            &mut deferred,
            2,
            &mut coalesced,
            &mut dropped,
        );
        assert_eq!(online, AsrQueueSaturationOutcome::CoalescedOnline);
        assert_eq!(coalesced, 1);
        assert_eq!(dropped, 0);
        assert_eq!(deferred, Some((7, vec![2.0, 3.0])));

        let offline = handle_asr_queue_full(
            AsrTask::OfflineSegment {
                epoch: 7,
                samples: vec![0.0; 16],
            },
            &mut deferred,
            2,
            &mut coalesced,
            &mut dropped,
        );
        assert_eq!(offline, AsrQueueSaturationOutcome::DroppedOffline);
        assert_eq!(coalesced, 1);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn asr_queue_overload_replaces_deferred_online_chunk_on_epoch_change() {
        let mut deferred = Some((1, vec![1.0, 2.0]));
        let mut coalesced = 0u64;
        let mut dropped = 0u64;

        let outcome = handle_asr_queue_full(
            AsrTask::OnlineChunk {
                epoch: 2,
                samples: vec![3.0, 4.0, 5.0],
            },
            &mut deferred,
            2,
            &mut coalesced,
            &mut dropped,
        );
        assert_eq!(outcome, AsrQueueSaturationOutcome::CoalescedOnline);
        assert_eq!(coalesced, 1);
        assert_eq!(dropped, 0);
        assert_eq!(deferred, Some((2, vec![4.0, 5.0])));
    }

    #[test]
    #[ignore = "long-running soak; run manually with -- --ignored"]
    fn overload_burst_growth_bounded_soak() {
        let (tx, _rx) = bounded::<Vec<f32>>(1);
        let _ = tx.try_send(vec![0.0; 16]);

        let stats = InputOverflowStats::default();
        let mut state = InputOverflowState::new(0);
        let start = Instant::now();
        let soak_for = Duration::from_secs(30 * 60);
        let mut pending = vec![0.0; 16];

        while start.elapsed() < soak_for {
            flush_chunks(16, &tx, &mut pending, &stats, &mut state);
            pending = vec![0.0; 16];
        }

        let snap = stats.snapshot();
        // In sustained overload without recovery, bursts should remain a single active burst.
        assert!(snap.dropped_chunks > 0);
        assert_eq!(snap.overflow_bursts, 1);
    }

    #[test]
    fn ewma_us_moves_toward_sample() {
        assert_eq!(ewma_us(0, 120), 120);
        assert_eq!(ewma_us(100, 200), 120);
    }
}
