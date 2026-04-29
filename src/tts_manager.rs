use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;

use crossbeam_channel::{Sender, TryRecvError, TrySendError, bounded};
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsSupertonicModelConfig, OfflineTtsVitsModelConfig,
};

use crate::config::{TtsBackend, TtsConfig};
use crate::duplex_audio::{DuplexPlaybackGate, RenderReferencePublisher};
use crate::errors::VocomError;
use crate::tts_fx::apply_tts_fx_in_place;
use crate::tts_sink::{default_tts_sink, TtsPlaybackControl, TtsPlaybackRequest};

const TTS_MAX_ATTEMPTS: usize = 3;
#[cfg(target_os = "android")]
const TTS_INTERRUPT_GRACE_MS: u64 = 320;

#[cfg(not(target_os = "android"))]
const TTS_INTERRUPT_GRACE_MS: u64 = 1400;

#[derive(Clone, Debug)]
struct GeneratedAudio {
    samples: Vec<f32>,
    sample_rate: i32,
}

#[derive(Clone, Debug)]
struct AudioStats {
    duration_ms: f32,
    peak: f32,
    rms: f32,
}

enum TtsCommand {
    Speak(String),
    Shutdown,
}

pub struct TtsManager {
    command_tx: Sender<TtsCommand>,
    stop_requested: Arc<AtomicBool>,
    interruptible_after_ms: Arc<AtomicU64>,
    playback_target_gain_bits: Arc<AtomicU32>,
    worker: Option<JoinHandle<()>>,
}

impl TtsManager {
    pub fn from_config(
        config: &TtsConfig,
        render_reference: Option<RenderReferencePublisher>,
        playback_gate: Option<DuplexPlaybackGate>,
    ) -> Result<Self, VocomError> {
        let tts_config = build_tts_config(config)?;

        let tts = OfflineTts::create(&tts_config).ok_or_else(|| {
            VocomError::TtsConfig("failed to create offline tts from configuration".to_string())
        })?;

        eprintln!(
            "tts backend initialized: backend={:?} sample_rate={} speakers={}",
            config.backend,
            tts.sample_rate(),
            tts.num_speakers()
        );

        let generation_config = GenerationConfig {
            speed: config.speed,
            sid: config.speaker_id,
            num_steps: if matches!(config.backend, TtsBackend::Supertonic) {
                config.supertonic.num_steps
            } else {
                GenerationConfig::default().num_steps
            },
            extra: if matches!(config.backend, TtsBackend::Supertonic) {
                let mut extra = HashMap::new();
                extra.insert(
                    "lang".to_string(),
                    serde_json::json!(config.supertonic.lang.clone()),
                );
                Some(extra)
            } else {
                None
            },
            ..Default::default()
        };
        let fx_config = config.fx.clone();

        let (command_tx, command_rx) = bounded::<TtsCommand>(64);
        let stop_requested = Arc::new(AtomicBool::new(false));
        let interruptible_after_ms = Arc::new(AtomicU64::new(0));
        let stop_requested_worker = Arc::clone(&stop_requested);
        let interruptible_after_ms_worker = Arc::clone(&interruptible_after_ms);
        let playback_target_gain_bits = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let playback_target_gain_bits_worker = Arc::clone(&playback_target_gain_bits);
        let playback_current_gain_bits = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let playback_current_gain_bits_worker = Arc::clone(&playback_current_gain_bits);
        let sink = default_tts_sink();
        let sink_worker = Arc::clone(&sink);
        let worker = thread::spawn(move || {
            'worker: while let Ok(cmd) = command_rx.recv() {
                match cmd {
                    TtsCommand::Speak(mut text) => {
                        let mut gate_needs_manual_end = false;
                        if let Some(gate) = playback_gate.as_ref() {
                            // Mark TTS active from generation start, not only playback start.
                            // This allows half-duplex capture fast-path to skip expensive mic
                            // processing while TTS model inference is running.
                            gate.mark_tts_start();
                            gate_needs_manual_end = true;
                        }

                        'speak: loop {
                            // Latest-wins before generation.
                            let mut coalesced = 0usize;
                            loop {
                                match command_rx.try_recv() {
                                    Ok(TtsCommand::Speak(next_text)) => {
                                        text = next_text;
                                        coalesced = coalesced.saturating_add(1);
                                    }
                                    Ok(TtsCommand::Shutdown) => break 'worker,
                                    Err(TryRecvError::Empty) => break,
                                    Err(TryRecvError::Disconnected) => break 'worker,
                                }
                            }
                            if coalesced > 0 {
                                eprintln!(
                                    "tts queue coalesced {coalesced} stale requests; playing latest"
                                );
                            }

                            stop_requested_worker.store(false, Ordering::Release);
                            interruptible_after_ms_worker.store(
                                now_ms().saturating_add(TTS_INTERRUPT_GRACE_MS),
                                Ordering::Release,
                            );
                            playback_target_gain_bits_worker
                                .store(1.0f32.to_bits(), Ordering::Release);
                            playback_current_gain_bits_worker
                                .store(1.0f32.to_bits(), Ordering::Release);

                            let min_expected_ms = min_expected_duration_ms(&text);
                            let mut accepted: Option<(GeneratedAudio, AudioStats, usize)> = None;
                            let mut best_degraded: Option<(GeneratedAudio, AudioStats, usize)> = None;

                            for attempt in 0..TTS_MAX_ATTEMPTS {
                                let candidate_text = if attempt == 0 {
                                    text.clone()
                                } else {
                                    sanitize_text_for_retry(&text)
                                };

                                let attempt_cfg =
                                    generation_config_for_attempt(&generation_config, attempt);

                                let result = tts.generate_with_config::<fn(&[f32], f32) -> bool>(
                                    &candidate_text,
                                    &attempt_cfg,
                                    None,
                                );

                                let Some(audio) = result else {
                                    eprintln!(
                                        "tts generation attempt {} failed: no audio returned",
                                        attempt + 1
                                    );
                                    continue;
                                };

                                let generated = GeneratedAudio {
                                    samples: audio.samples().to_vec(),
                                    sample_rate: audio.sample_rate(),
                                };
                                let stats = audio_stats(&generated.samples, generated.sample_rate);

                                eprintln!(
                                    "tts generated attempt {}: samples={} sr={} duration_ms={:.1} peak={:.5} rms={:.5}",
                                    attempt + 1,
                                    generated.samples.len(),
                                    generated.sample_rate,
                                    stats.duration_ms,
                                    stats.peak,
                                    stats.rms
                                );

                                if passes_quality_gate(&stats, min_expected_ms) {
                                    accepted = Some((generated, stats, attempt));
                                    break;
                                }

                                if is_better_candidate(
                                    &generated,
                                    &stats,
                                    best_degraded.as_ref().map(|(a, s, _)| (a, s)),
                                ) {
                                    best_degraded = Some((generated, stats, attempt));
                                }
                            }

                            let selected = if let Some(ok) = accepted {
                                Some(ok)
                            } else if let Some((audio, stats, attempt)) = best_degraded {
                                let degraded_min_duration = (min_expected_ms * 0.80).max(450.0);
                                let degraded_has_energy =
                                    stats.rms >= 0.0020 || stats.peak >= 0.030;
                                if stats.duration_ms >= degraded_min_duration && degraded_has_energy
                                {
                                    eprintln!(
                                        "tts quality gate degraded accept at attempt {}: duration_ms={:.1} peak={:.5} rms={:.5} (min_expected_ms={:.1})",
                                        attempt + 1,
                                        stats.duration_ms,
                                        stats.peak,
                                        stats.rms,
                                        min_expected_ms,
                                    );
                                    Some((audio, stats, attempt))
                                } else {
                                    let relaxed_min_duration = 220.0;
                                    let relaxed_has_energy =
                                        stats.rms >= 0.0008 || stats.peak >= 0.010;
                                    if stats.duration_ms >= relaxed_min_duration && relaxed_has_energy
                                    {
                                        eprintln!(
                                            "tts quality gate relaxed fallback at attempt {}: duration_ms={:.1} peak={:.5} rms={:.5} (min_expected_ms={:.1})",
                                            attempt + 1,
                                            stats.duration_ms,
                                            stats.peak,
                                            stats.rms,
                                            min_expected_ms,
                                        );
                                        Some((audio, stats, attempt))
                                    } else {
                                        None
                                    }
                                }
                            } else {
                                None
                            };

                            // Latest-wins check again after potentially long generation: if a newer
                            // Speak arrived while generating, drop stale playback and regenerate latest.
                            let mut superseded = false;
                            loop {
                                match command_rx.try_recv() {
                                    Ok(TtsCommand::Speak(next_text)) => {
                                        text = next_text;
                                        superseded = true;
                                    }
                                    Ok(TtsCommand::Shutdown) => break 'worker,
                                    Err(TryRecvError::Empty) => break,
                                    Err(TryRecvError::Disconnected) => break 'worker,
                                }
                            }
                            if superseded {
                                eprintln!("tts generation superseded by newer request; skipping stale playback");
                                continue 'speak;
                            }

                            if let Some((mut audio, stats, attempt)) = selected {
                                eprintln!(
                                    "tts playback selected attempt {}: duration_ms={:.1} peak={:.5} rms={:.5}",
                                    attempt + 1,
                                    stats.duration_ms,
                                    stats.peak,
                                    stats.rms
                                );
                                let fx_rms_before = if fx_config.enabled {
                                    Some(audio_stats(&audio.samples, audio.sample_rate).rms)
                                } else {
                                    None
                                };
                                let fx_probe_before = if fx_config.enabled {
                                    Some(
                                        audio
                                            .samples
                                            .iter()
                                            .take(2048)
                                            .copied()
                                            .collect::<Vec<f32>>(),
                                    )
                                } else {
                                    None
                                };
                                apply_tts_fx_in_place(
                                    &mut audio.samples,
                                    audio.sample_rate,
                                    &fx_config,
                                );
                                if let (Some(before_rms), Some(before_probe)) =
                                    (fx_rms_before, fx_probe_before)
                                {
                                    let after_rms = audio_stats(&audio.samples, audio.sample_rate).rms;
                                    let mut delta_sum = 0.0f32;
                                    let mut n = 0usize;
                                    for (a, b) in before_probe.iter().zip(audio.samples.iter()) {
                                        delta_sum += (a - b).abs();
                                        n = n.saturating_add(1);
                                    }
                                    let mean_abs_delta = if n == 0 {
                                        0.0
                                    } else {
                                        delta_sum / n as f32
                                    };
                                    eprintln!(
                                        "tts fx applied: enabled=true low_cut_hz={:.1} high_cut_hz={:.1} robot_mix={:.2} ring_mod_hz={:.1} distortion_drive={:.2} echo_delay_ms={} echo_feedback={:.2} echo_mix={:.2} output_gain={:.2} rms_before={:.5} rms_after={:.5} mean_abs_delta={:.5}",
                                        fx_config.low_cut_hz,
                                        fx_config.high_cut_hz,
                                        fx_config.robot_mix,
                                        fx_config.ring_mod_hz,
                                        fx_config.distortion_drive,
                                        fx_config.echo_delay_ms,
                                        fx_config.echo_feedback,
                                        fx_config.echo_mix,
                                        fx_config.output_gain,
                                        before_rms,
                                        after_rms,
                                        mean_abs_delta,
                                    );
                                }

                                if let Err(err) = sink_worker.play(
                                    TtsPlaybackRequest {
                                        samples: &audio.samples,
                                        sample_rate: audio.sample_rate,
                                        render_reference: render_reference.as_ref(),
                                        playback_gate: playback_gate.as_ref(),
                                    },
                                    TtsPlaybackControl {
                                        stop_requested: Arc::clone(&stop_requested_worker),
                                        interruptible_after_ms: Arc::clone(
                                            &interruptible_after_ms_worker,
                                        ),
                                        playback_target_gain_bits: Arc::clone(
                                            &playback_target_gain_bits_worker,
                                        ),
                                        playback_current_gain_bits: Arc::clone(
                                            &playback_current_gain_bits_worker,
                                        ),
                                        interrupt_grace_ms: TTS_INTERRUPT_GRACE_MS,
                                    },
                                ) {
                                    eprintln!("tts playback failed: {err}");
                                }
                                // Sink handles mark_tts_end via its playback guard.
                                gate_needs_manual_end = false;
                            } else {
                                eprintln!(
                                    "tts generation failed quality gate for all attempts (min_expected_ms={min_expected_ms:.1})"
                                );
                            }

                            break 'speak;
                        }

                        if gate_needs_manual_end {
                            if let Some(gate) = playback_gate.as_ref() {
                                gate.mark_tts_end();
                            }
                        }
                    }
                    TtsCommand::Shutdown => break,
                }
            }
        });

        Ok(Self {
            command_tx,
            stop_requested,
            interruptible_after_ms,
            playback_target_gain_bits,
            worker: Some(worker),
        })
    }

    pub fn mock_response(input: &str) -> String {
        let cleaned = sanitize_text_for_retry(input)
            .replace(['"', '\'', '`'], "")
            .trim()
            .trim_matches(|c: char| matches!(c, '.' | '!' | '?'))
            .to_string();

        if cleaned.is_empty() {
            "I heard you. This is a local sherpa response.".to_string()
        } else {
            format!("I heard you say {cleaned}. This is a local sherpa response.")
        }
    }

    pub fn speak_text(&self, text: &str) -> Result<(), VocomError> {
        self.stop_requested.store(false, Ordering::Release);
        self.interruptible_after_ms.store(
            now_ms().saturating_add(TTS_INTERRUPT_GRACE_MS),
            Ordering::Release,
        );
        match self.command_tx.try_send(TtsCommand::Speak(text.to_string())) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                // Keep engine loop responsive under heavy transcript bursts.
                // Dropping excess TTS requests is safer than blocking audio processing.
                eprintln!("tts queue full, dropping speak request");
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(VocomError::ChannelDisconnected),
        }
    }

    pub fn stop_current(&self) {
        let now = now_ms();
        let interruptible_after = self.interruptible_after_ms.load(Ordering::Acquire);
        if now < interruptible_after {
            eprintln!(
                "tts stop ignored during grace window: now_ms={} interruptible_after_ms={}",
                now, interruptible_after
            );
            return;
        }
        self.stop_requested.store(true, Ordering::Release);
    }

    pub fn duck_current(&self, level: f32) {
        let gain = level.clamp(0.0, 1.0);
        self.playback_target_gain_bits
            .store(gain.to_bits(), Ordering::Release);
    }

    pub fn restore_volume(&self) {
        self.playback_target_gain_bits
            .store(1.0f32.to_bits(), Ordering::Release);
    }

}

impl Drop for TtsManager {
    fn drop(&mut self) {
        // Avoid blocking shutdown on a busy/full TTS command queue.
        self.stop_requested.store(true, Ordering::Release);
        let _ = self.command_tx.try_send(TtsCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = thread::Builder::new()
                .name("vocom-tts-shutdown-join".to_string())
                .spawn(move || {
                    let _ = worker.join();
                });
        }
    }
}

fn resolve_existing_path_with_aliases(path: &str) -> Result<String, VocomError> {
    if Path::new(path).exists() {
        return Ok(path.to_string());
    }

    for candidate in tts_path_alias_candidates(path) {
        if Path::new(&candidate).exists() {
            eprintln!("tts path fallback: '{path}' -> '{candidate}'");
            return Ok(candidate);
        }
    }

    Err(VocomError::MissingModelPath(path.to_string()))
}

fn resolve_data_dir_with_fallback(path: &str, model_dir: Option<&str>) -> Result<String, VocomError> {
    if Path::new(path).exists() {
        return Ok(path.to_string());
    }

    for candidate in tts_path_alias_candidates(path) {
        if Path::new(&candidate).exists() {
            eprintln!("tts data_dir fallback: '{path}' -> '{candidate}'");
            return Ok(candidate);
        }
    }

    if let Some(model_dir) = model_dir {
        let nested = Path::new(model_dir).join("espeak-ng-data");
        if nested.exists() {
            let val = nested.to_string_lossy().to_string();
            eprintln!("tts data_dir fallback from model dir: '{path}' -> '{val}'");
            return Ok(val);
        }

        if Path::new(model_dir).exists() {
            let val = model_dir.to_string();
            eprintln!("tts data_dir fallback to model dir: '{path}' -> '{val}'");
            return Ok(val);
        }
    }

    Err(VocomError::MissingModelPath(path.to_string()))
}

fn tts_path_alias_candidates(path: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    if path.contains("vits-pipe-en_US_female-medium") {
        candidates.push(path.replace(
            "vits-pipe-en_US_female-medium",
            "vits-piper-en_US-hfc_female-medium",
        ));
    }

    if path.contains("vits-piper-en_US_female-medium") {
        candidates.push(path.replace(
            "vits-piper-en_US_female-medium",
            "vits-piper-en_US-hfc_female-medium",
        ));
    }

    candidates
}

fn validate_espeak_data_dir(data_dir: &str) -> Result<(), VocomError> {
    let required = ["phontab", "phonindex", "phondata", "intonations"];

    for rel in required {
        let p = Path::new(data_dir).join(rel);
        if !p.exists() {
            return Err(VocomError::TtsConfig(format!(
                "missing required espeak data file: {}",
                p.to_string_lossy()
            )));
        }
    }

    let voices_dir = Path::new(data_dir).join("voices");
    if !voices_dir.exists() {
        return Err(VocomError::TtsConfig(format!(
            "missing required espeak data directory: {}",
            voices_dir.to_string_lossy()
        )));
    }

    let mut has_voice_file = false;
    if let Ok(entries) = std::fs::read_dir(&voices_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                has_voice_file = true;
                break;
            }

            if path.is_dir()
                && std::fs::read_dir(&path)
                    .ok()
                    .and_then(|mut it| it.next())
                    .is_some()
            {
                has_voice_file = true;
                break;
            }
        }
    }

    if !has_voice_file {
        return Err(VocomError::TtsConfig(format!(
            "espeak voices directory is empty: {}",
            voices_dir.to_string_lossy()
        )));
    }

    Ok(())
}

fn sanitize_text_for_retry(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_space = false;
    let mut last_was_punct = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
            continue;
        }

        let is_punct = matches!(ch, ',' | ';' | ':' | '!' | '?' | '.');
        if is_punct {
            if !last_was_punct {
                out.push(ch);
            }
            last_was_punct = true;
            last_was_space = false;
            continue;
        }

        out.push(ch);
        last_was_space = false;
        last_was_punct = false;
    }

    let mut cleaned = out.trim().to_string();
    if cleaned.is_empty() {
        cleaned = "Hello. How can I help you today?".to_string();
    }
    cleaned
}

fn generation_config_for_attempt(base: &GenerationConfig, attempt: usize) -> GenerationConfig {
    let mut cfg = base.clone();
    let speed_scale = match attempt {
        0 => 1.0,
        1 => 0.92,
        _ => 0.84,
    };

    cfg.speed = (base.speed * speed_scale).clamp(0.62, 1.20);
    cfg
}

fn build_tts_config(config: &TtsConfig) -> Result<OfflineTtsConfig, VocomError> {
    let model = match config.backend {
        TtsBackend::Vits => {
            eprintln!(
                "tts init (vits) raw paths: model='{}' tokens='{}' data_dir='{}'",
                config.model_path, config.tokens_path, config.data_dir
            );

            let model_path = resolve_existing_path_with_aliases(&config.model_path)?;
            let tokens_path = resolve_existing_path_with_aliases(&config.tokens_path)?;
            let model_dir = Path::new(&model_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string());

            let data_dir = resolve_data_dir_with_fallback(&config.data_dir, model_dir.as_deref())?;
            validate_espeak_data_dir(&data_dir)?;

            eprintln!(
                "tts init (vits) resolved paths: model='{}' tokens='{}' data_dir='{}'",
                model_path, tokens_path, data_dir
            );

            let dict_dir = if let Some(dict_dir) = &config.dict_dir {
                Some(resolve_existing_path_with_aliases(dict_dir)?)
            } else {
                None
            };

            OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(model_path),
                    tokens: Some(tokens_path),
                    data_dir: Some(data_dir),
                    dict_dir,
                    ..Default::default()
                },
                num_threads: config.num_threads,
                provider: Some(config.provider.clone()),
                ..Default::default()
            }
        }
        TtsBackend::Supertonic => {
            let duration_predictor = require_supertonic_path(
                "duration_predictor",
                config.supertonic.duration_predictor.as_deref(),
            )?;
            let text_encoder = require_supertonic_path(
                "text_encoder",
                config.supertonic.text_encoder.as_deref(),
            )?;
            let vector_estimator = require_supertonic_path(
                "vector_estimator",
                config.supertonic.vector_estimator.as_deref(),
            )?;
            let vocoder = require_supertonic_path(
                "vocoder",
                config.supertonic.vocoder.as_deref(),
            )?;
            let tts_json = require_supertonic_path(
                "tts_json",
                config.supertonic.tts_json.as_deref(),
            )?;
            let unicode_indexer = require_supertonic_path(
                "unicode_indexer",
                config.supertonic.unicode_indexer.as_deref(),
            )?;
            let voice_style = require_supertonic_path(
                "voice_style",
                config.supertonic.voice_style.as_deref(),
            )?;

            eprintln!(
                "tts init (supertonic) resolved paths: duration_predictor='{}' text_encoder='{}' vector_estimator='{}' vocoder='{}' tts_json='{}' unicode_indexer='{}' voice_style='{}' lang='{}' num_steps={}",
                duration_predictor,
                text_encoder,
                vector_estimator,
                vocoder,
                tts_json,
                unicode_indexer,
                voice_style,
                config.supertonic.lang,
                config.supertonic.num_steps,
            );

            OfflineTtsModelConfig {
                supertonic: OfflineTtsSupertonicModelConfig {
                    duration_predictor: Some(duration_predictor),
                    text_encoder: Some(text_encoder),
                    vector_estimator: Some(vector_estimator),
                    vocoder: Some(vocoder),
                    tts_json: Some(tts_json),
                    unicode_indexer: Some(unicode_indexer),
                    voice_style: Some(voice_style),
                },
                num_threads: config.num_threads,
                provider: Some(config.provider.clone()),
                ..Default::default()
            }
        }
    };

    Ok(OfflineTtsConfig {
        model,
        ..Default::default()
    })
}

fn require_supertonic_path(name: &str, value: Option<&str>) -> Result<String, VocomError> {
    let raw = value.ok_or_else(|| {
        VocomError::TtsConfig(format!(
            "missing tts.supertonic.{name} in configuration"
        ))
    })?;

    resolve_existing_path_with_aliases(raw)
}

fn min_expected_duration_ms(text: &str) -> f32 {
    let non_ws_chars = text.chars().filter(|c| !c.is_whitespace()).count();
    if non_ws_chars <= 10 {
        return 350.0;
    }
    ((non_ws_chars as f32) * 45.0).clamp(450.0, 6000.0)
}

fn audio_stats(samples: &[f32], sample_rate: i32) -> AudioStats {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms = if samples.is_empty() {
        0.0
    } else {
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    };
    let duration_ms = if sample_rate > 0 {
        (samples.len() as f32 * 1000.0) / sample_rate as f32
    } else {
        0.0
    };

    AudioStats {
        duration_ms,
        peak,
        rms,
    }
}

fn passes_quality_gate(stats: &AudioStats, min_expected_ms: f32) -> bool {
    let min_duration = (min_expected_ms * 0.60).max(320.0);
    let enough_duration = stats.duration_ms >= min_duration;
    let enough_energy = stats.rms >= 0.0025 || stats.peak >= 0.040;
    enough_duration && enough_energy
}

fn is_better_candidate(
    audio: &GeneratedAudio,
    stats: &AudioStats,
    current_best: Option<(&GeneratedAudio, &AudioStats)>,
) -> bool {
    let current_score = stats.duration_ms + (stats.rms * 1000.0) + (stats.peak * 400.0);
    let current_nonempty = !audio.samples.is_empty() && audio.sample_rate > 0;
    if !current_nonempty {
        return false;
    }

    let Some((best_audio, best_stats)) = current_best else {
        return true;
    };

    let best_nonempty = !best_audio.samples.is_empty() && best_audio.sample_rate > 0;
    if !best_nonempty {
        return true;
    }

    let best_score = best_stats.duration_ms + (best_stats.rms * 1000.0) + (best_stats.peak * 400.0);
    current_score > best_score
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
