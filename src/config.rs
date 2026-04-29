use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Deserialize;

use crate::errors::VocomError;
use crate::realtime_pipeline::RealtimeConfig;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrVariantConfig {
    Whisper,
    MoonshineV2,
    StreamingZipformer,
    NemotronStreaming,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrModeConfig {
    Offline,
    Online,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DuplexMode {
    FullDuplex,
    HalfDuplexMuteMic,
}

impl Default for DuplexMode {
    fn default() -> Self {
        Self::FullDuplex
    }
}

impl Default for AsrModeConfig {
    fn default() -> Self {
        Self::Offline
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AsrConfig {
    #[serde(default)]
    pub mode: AsrModeConfig,
    pub variant: AsrVariantConfig,
    pub encoder_path: String,
    pub decoder_path: String,
    #[serde(default)]
    pub joiner_path: Option<String>,
    pub tokens_path: String,
    pub num_threads: i32,
    pub provider: String,
    pub whisper_language: String,
    pub whisper_task: String,
    pub whisper_tail_paddings: i32,
    pub whisper_enable_token_timestamps: bool,
    pub whisper_enable_segment_timestamps: bool,
    #[serde(default = "default_online_decoding_method")]
    pub online_decoding_method: String,
    #[serde(default = "default_online_enable_endpoint")]
    pub online_enable_endpoint: bool,
    #[serde(default = "default_online_rule1_min_trailing_silence")]
    pub online_rule1_min_trailing_silence: f32,
    #[serde(default = "default_online_rule2_min_trailing_silence")]
    pub online_rule2_min_trailing_silence: f32,
    #[serde(default = "default_online_rule3_min_utterance_length")]
    pub online_rule3_min_utterance_length: f32,
    #[serde(default)]
    pub hotwords_path: Option<String>,
    #[serde(default = "default_hotwords_score")]
    pub hotwords_score: f32,
}

fn default_hotwords_score() -> f32 {
    1.5
}

fn default_online_decoding_method() -> String {
    "greedy_search".to_string()
}

fn default_online_enable_endpoint() -> bool {
    true
}

fn default_online_rule1_min_trailing_silence() -> f32 {
    0.35
}

fn default_online_rule2_min_trailing_silence() -> f32 {
    0.8
}

fn default_online_rule3_min_utterance_length() -> f32 {
    8.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct VadConfig {
    pub model_path: String,
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub max_speech_duration: f32,
    pub window_size: i32,
    pub sample_rate: i32,
    pub num_threads: i32,
    pub provider: String,
    pub debug: bool,
    #[serde(default = "default_vad_dynamic_gate_enabled")]
    pub dynamic_gate_enabled: bool,
    #[serde(default = "default_vad_noise_smoothing")]
    pub noise_smoothing: f32,
    #[serde(default = "default_vad_noise_gate_multiplier")]
    pub noise_gate_multiplier: f32,
    #[serde(default = "default_vad_noise_gate_min_rms")]
    pub noise_gate_min_rms: f32,
    #[serde(default = "default_vad_noise_gate_max_rms")]
    pub noise_gate_max_rms: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BargeInVadConfig {
    pub model_path: String,
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub max_speech_duration: f32,
    pub window_size: i32,
    pub sample_rate: i32,
    pub num_threads: i32,
    pub provider: String,
    pub debug: bool,
}

impl Default for BargeInVadConfig {
    fn default() -> Self {
        Self {
            model_path: "models/silero_vad.onnx".to_string(),
            threshold: 0.4,
            min_silence_duration: 0.25,
            min_speech_duration: 0.1,
            max_speech_duration: 8.0,
            window_size: 512,
            sample_rate: 16_000,
            num_threads: 1,
            provider: "cpu".to_string(),
            debug: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenoiserModelFamily {
    Gtcrn,
    Dpdfnet,
}

impl Default for DenoiserModelFamily {
    fn default() -> Self {
        Self::Gtcrn
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct DenoiserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub family: DenoiserModelFamily,
    #[serde(default)]
    pub model_path: String,
    #[serde(default = "default_denoiser_threads")]
    pub num_threads: i32,
    #[serde(default = "default_denoiser_provider")]
    pub provider: String,
    #[serde(default)]
    pub debug: bool,
}

impl Default for DenoiserConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            family: DenoiserModelFamily::default(),
            model_path: "models/denoiser/gtcrn_simple.onnx".to_string(),
            num_threads: default_denoiser_threads(),
            provider: default_denoiser_provider(),
            debug: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RealtimeEngineConfig {
    pub target_sample_rate: i32,
    pub chunk_ms: u32,
    pub audio_queue_capacity: usize,
    pub event_queue_capacity: usize,
    #[serde(default)]
    pub duplex_mode: DuplexMode,
    #[serde(default = "default_input_overflow_log_interval_ms")]
    pub input_overflow_log_interval_ms: u64,
    #[serde(default = "default_input_overflow_shed_ms")]
    pub input_overflow_shed_ms: u64,
    #[serde(default = "default_post_roll_ms")]
    pub post_roll_ms: u32,
    #[serde(default = "default_short_silence_merge_ms")]
    pub short_silence_merge_ms: u32,
    #[serde(default = "default_input_normalize_enabled")]
    pub input_normalize_enabled: bool,
    #[serde(default = "default_input_normalize_target_peak")]
    pub input_normalize_target_peak: f32,
    #[serde(default = "default_input_normalize_max_gain")]
    pub input_normalize_max_gain: f32,
    #[serde(default = "default_input_clip_guard_enabled")]
    pub input_clip_guard_enabled: bool,
    #[serde(default = "default_input_clip_threshold")]
    pub input_clip_threshold: f32,
    #[serde(default = "default_render_reference_capacity")]
    pub render_reference_capacity: usize,
    #[serde(default = "default_pre_speech_buffer_ms")]
    pub pre_speech_buffer_ms: u32,
    #[serde(default = "default_tts_suppression_cooldown_ms")]
    pub tts_suppression_cooldown_ms: u64,
    #[serde(default = "default_tts_barge_in_rms_threshold")]
    pub tts_barge_in_rms_threshold: f32,
    #[serde(default = "default_tts_barge_in_duck_level")]
    pub tts_barge_in_duck_level: f32,
    #[serde(default = "default_tts_barge_in_stop_after_ms")]
    pub tts_barge_in_stop_after_ms: u64,
    #[serde(default = "default_tts_barge_in_render_ratio_min")]
    pub tts_barge_in_render_ratio_min: f32,
    #[serde(default = "default_tts_barge_in_render_ratio_boost_enabled")]
    pub tts_barge_in_render_ratio_boost_enabled: bool,
    #[serde(default = "default_tts_barge_in_render_ratio_boost")]
    pub tts_barge_in_render_ratio_boost: f32,
    #[serde(default = "default_tts_barge_in_render_ratio_boost_start_rms")]
    pub tts_barge_in_render_ratio_boost_start_rms: f32,
    #[serde(default = "default_tts_barge_in_render_ratio_boost_end_rms")]
    pub tts_barge_in_render_ratio_boost_end_rms: f32,
    #[serde(default = "default_tts_barge_in_render_rms_suppress_threshold")]
    pub tts_barge_in_render_rms_suppress_threshold: f32,
    #[serde(default = "default_tts_barge_in_render_rms_suppress_ratio_min")]
    pub tts_barge_in_render_rms_suppress_ratio_min: f32,
    #[serde(default = "default_tts_asr_leak_suppress_enabled")]
    pub tts_asr_leak_suppress_enabled: bool,
    #[serde(default = "default_tts_asr_leak_suppress_ratio_min")]
    pub tts_asr_leak_suppress_ratio_min: f32,
    #[serde(default = "default_tts_asr_leak_suppress_render_rms_min")]
    pub tts_asr_leak_suppress_render_rms_min: f32,
    #[serde(default = "default_tts_barge_in_render_rms_max_age_ms")]
    pub tts_barge_in_render_rms_max_age_ms: u64,
    #[serde(default = "default_tts_barge_in_persistence_ms")]
    pub tts_barge_in_persistence_ms: u64,
    #[serde(default = "default_tts_barge_in_confidence_threshold")]
    pub tts_barge_in_confidence_threshold: f32,
    #[serde(default = "default_tts_barge_in_confidence_smoothing")]
    pub tts_barge_in_confidence_smoothing: f32,
    #[serde(default = "default_tts_barge_in_suspect_hold_ms")]
    pub tts_barge_in_suspect_hold_ms: u64,
    #[serde(default = "default_tts_barge_in_recover_ms")]
    pub tts_barge_in_recover_ms: u64,
    #[serde(default = "default_tts_barge_in_suspect_drop_grace_ms")]
    pub tts_barge_in_suspect_drop_grace_ms: u64,
    #[serde(default = "default_tts_barge_in_min_interval_ms")]
    pub tts_barge_in_min_interval_ms: u64,
    #[serde(default = "default_adaptive_leak_tuner_enabled")]
    pub adaptive_leak_tuner_enabled: bool,
    #[serde(default = "default_adaptive_leak_tuner_observe_only")]
    pub adaptive_leak_tuner_observe_only: bool,
    #[serde(default = "default_adaptive_leak_tuner_log_interval_ms")]
    pub adaptive_leak_tuner_log_interval_ms: u64,
}

fn default_pre_speech_buffer_ms() -> u32 {
    300
}

fn default_tts_suppression_cooldown_ms() -> u64 {
    700
}

fn default_tts_barge_in_rms_threshold() -> f32 {
    0.02
}

fn default_tts_barge_in_duck_level() -> f32 {
    0.55
}

fn default_tts_barge_in_stop_after_ms() -> u64 {
    350
}

fn default_tts_barge_in_render_ratio_min() -> f32 {
    1.35
}

fn default_tts_barge_in_render_ratio_boost_enabled() -> bool {
    true
}

fn default_tts_barge_in_render_ratio_boost() -> f32 {
    0.75
}

fn default_tts_barge_in_render_ratio_boost_start_rms() -> f32 {
    0.02
}

fn default_tts_barge_in_render_ratio_boost_end_rms() -> f32 {
    0.08
}

fn default_tts_barge_in_render_rms_suppress_threshold() -> f32 {
    0.12
}

fn default_tts_barge_in_render_rms_suppress_ratio_min() -> f32 {
    2.6
}

fn default_tts_asr_leak_suppress_enabled() -> bool {
    true
}

fn default_tts_asr_leak_suppress_ratio_min() -> f32 {
    1.6
}

fn default_tts_asr_leak_suppress_render_rms_min() -> f32 {
    0.02
}

fn default_tts_barge_in_render_rms_max_age_ms() -> u64 {
    200
}

fn default_tts_barge_in_persistence_ms() -> u64 {
    60
}

fn default_tts_barge_in_confidence_threshold() -> f32 {
    0.68
}

fn default_tts_barge_in_confidence_smoothing() -> f32 {
    0.3
}

fn default_tts_barge_in_suspect_hold_ms() -> u64 {
    90
}

fn default_tts_barge_in_recover_ms() -> u64 {
    350
}

fn default_tts_barge_in_suspect_drop_grace_ms() -> u64 {
    500
}

fn default_tts_barge_in_min_interval_ms() -> u64 {
    150
}

fn default_adaptive_leak_tuner_enabled() -> bool {
    false
}

fn default_adaptive_leak_tuner_observe_only() -> bool {
    true
}

fn default_adaptive_leak_tuner_log_interval_ms() -> u64 {
    5_000
}

fn default_input_overflow_log_interval_ms() -> u64 {
    5_000
}

fn default_input_overflow_shed_ms() -> u64 {
    120
}

fn default_post_roll_ms() -> u32 {
    120
}

fn default_short_silence_merge_ms() -> u32 {
    150
}

fn default_input_normalize_enabled() -> bool {
    true
}

fn default_input_normalize_target_peak() -> f32 {
    0.90
}

fn default_input_normalize_max_gain() -> f32 {
    3.0
}

fn default_input_clip_guard_enabled() -> bool {
    true
}

fn default_input_clip_threshold() -> f32 {
    0.98
}

fn default_vad_dynamic_gate_enabled() -> bool {
    true
}

fn default_vad_noise_smoothing() -> f32 {
    0.05
}

fn default_vad_noise_gate_multiplier() -> f32 {
    2.0
}

fn default_vad_noise_gate_min_rms() -> f32 {
    0.002
}

fn default_vad_noise_gate_max_rms() -> f32 {
    0.08
}

fn default_denoiser_threads() -> i32 {
    2
}

fn default_denoiser_provider() -> String {
    "cpu".to_string()
}

#[derive(Clone, Debug, Deserialize)]
pub struct AecConfig {
    pub enabled: bool,
    #[serde(default)]
    pub backend: AecBackend,
    pub sample_rate: i32,
    pub stream_delay_ms: Option<i32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AecBackend {
    PureRustAec3,
}

impl Default for AecBackend {
    fn default() -> Self {
        Self::PureRustAec3
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TtsConfig {
    pub enabled: bool,
    #[serde(default)]
    pub backend: TtsBackend,
    pub model_path: String,
    pub tokens_path: String,
    pub data_dir: String,
    pub dict_dir: Option<String>,
    pub num_threads: i32,
    pub provider: String,
    pub speed: f32,
    pub speaker_id: i32,
    #[serde(default)]
    pub fx: TtsFxConfig,
    #[serde(default)]
    pub supertonic: TtsSupertonicConfig,
    pub output_dir: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TtsFxConfig {
    #[serde(default = "default_tts_fx_enabled")]
    pub enabled: bool,
    #[serde(default = "default_tts_fx_low_cut_hz")]
    pub low_cut_hz: f32,
    #[serde(default = "default_tts_fx_high_cut_hz")]
    pub high_cut_hz: f32,
    #[serde(default = "default_tts_fx_robot_mix")]
    pub robot_mix: f32,
    #[serde(default = "default_tts_fx_ring_mod_hz")]
    pub ring_mod_hz: f32,
    #[serde(default = "default_tts_fx_distortion_drive")]
    pub distortion_drive: f32,
    #[serde(default = "default_tts_fx_echo_delay_ms")]
    pub echo_delay_ms: u32,
    #[serde(default = "default_tts_fx_echo_feedback")]
    pub echo_feedback: f32,
    #[serde(default = "default_tts_fx_echo_mix")]
    pub echo_mix: f32,
    #[serde(default = "default_tts_fx_output_gain")]
    pub output_gain: f32,
}

fn default_tts_fx_enabled() -> bool {
    false
}

fn default_tts_fx_low_cut_hz() -> f32 {
    110.0
}

fn default_tts_fx_high_cut_hz() -> f32 {
    4_200.0
}

fn default_tts_fx_robot_mix() -> f32 {
    0.45
}

fn default_tts_fx_ring_mod_hz() -> f32 {
    38.0
}

fn default_tts_fx_distortion_drive() -> f32 {
    1.8
}

fn default_tts_fx_echo_delay_ms() -> u32 {
    68
}

fn default_tts_fx_echo_feedback() -> f32 {
    0.26
}

fn default_tts_fx_echo_mix() -> f32 {
    0.16
}

fn default_tts_fx_output_gain() -> f32 {
    0.94
}

impl Default for TtsFxConfig {
    fn default() -> Self {
        Self {
            enabled: default_tts_fx_enabled(),
            low_cut_hz: default_tts_fx_low_cut_hz(),
            high_cut_hz: default_tts_fx_high_cut_hz(),
            robot_mix: default_tts_fx_robot_mix(),
            ring_mod_hz: default_tts_fx_ring_mod_hz(),
            distortion_drive: default_tts_fx_distortion_drive(),
            echo_delay_ms: default_tts_fx_echo_delay_ms(),
            echo_feedback: default_tts_fx_echo_feedback(),
            echo_mix: default_tts_fx_echo_mix(),
            output_gain: default_tts_fx_output_gain(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsBackend {
    Vits,
    Supertonic,
}

impl Default for TtsBackend {
    fn default() -> Self {
        Self::Vits
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct TtsSupertonicConfig {
    #[serde(default)]
    pub duration_predictor: Option<String>,
    #[serde(default)]
    pub text_encoder: Option<String>,
    #[serde(default)]
    pub vector_estimator: Option<String>,
    #[serde(default)]
    pub vocoder: Option<String>,
    #[serde(default)]
    pub tts_json: Option<String>,
    #[serde(default)]
    pub unicode_indexer: Option<String>,
    #[serde(default)]
    pub voice_style: Option<String>,
    #[serde(default = "default_supertonic_lang")]
    pub lang: String,
    #[serde(default = "default_supertonic_num_steps")]
    pub num_steps: i32,
}

fn default_supertonic_lang() -> String {
    "en".to_string()
}

fn default_supertonic_num_steps() -> i32 {
    5
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: TtsBackend::Vits,
            model_path: "models/vits-piper-en_US-hfc_female-medium/en_US-hfc_female-medium.onnx"
                .to_string(),
            tokens_path: "models/vits-piper-en_US-hfc_female-medium/tokens.txt".to_string(),
            data_dir: "models/vits-piper-en_US-hfc_female-medium/espeak-ng-data".to_string(),
            dict_dir: None,
            num_threads: 2,
            provider: "cpu".to_string(),
            speed: 1.0,
            speaker_id: 0,
            fx: TtsFxConfig::default(),
            supertonic: TtsSupertonicConfig::default(),
            output_dir: "target/tts".to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct WakewordConfig {
    #[serde(default = "default_wakeword_keyword")]
    pub keyword: String,
    #[serde(default = "default_wakeword_variants")]
    pub variants: Vec<String>,
    #[serde(default = "default_wakeword_cooldown_ms")]
    pub cooldown_ms: u64,
    #[serde(default = "default_wakeword_entrypoint_enabled")]
    pub entrypoint_enabled: bool,
}

impl Default for WakewordConfig {
    fn default() -> Self {
        Self {
            keyword: default_wakeword_keyword(),
            variants: default_wakeword_variants(),
            cooldown_ms: default_wakeword_cooldown_ms(),
            entrypoint_enabled: default_wakeword_entrypoint_enabled(),
        }
    }
}

fn default_wakeword_keyword() -> String {
    "maria".to_string()
}

fn default_wakeword_variants() -> Vec<String> {
    vec!["maria".to_string(), "mariam".to_string()]
}

fn default_wakeword_cooldown_ms() -> u64 {
    1_500
}

fn default_wakeword_entrypoint_enabled() -> bool {
    false
}

impl RealtimeEngineConfig {
    pub fn to_runtime(&self) -> RealtimeConfig {
        RealtimeConfig {
            target_sample_rate: self.target_sample_rate,
            chunk_ms: self.chunk_ms,
            audio_queue_capacity: self.audio_queue_capacity,
            event_queue_capacity: self.event_queue_capacity,
            duplex_mode: self.duplex_mode,
            input_overflow_log_interval_ms: self.input_overflow_log_interval_ms,
            input_overflow_shed_ms: self.input_overflow_shed_ms,
            post_roll_ms: self.post_roll_ms,
            short_silence_merge_ms: self.short_silence_merge_ms,
            input_normalize_enabled: self.input_normalize_enabled,
            input_normalize_target_peak: self.input_normalize_target_peak,
            input_normalize_max_gain: self.input_normalize_max_gain,
            input_clip_guard_enabled: self.input_clip_guard_enabled,
            input_clip_threshold: self.input_clip_threshold,
            pre_speech_buffer_ms: self.pre_speech_buffer_ms,
            tts_suppression_cooldown_ms: self.tts_suppression_cooldown_ms,
            tts_barge_in_rms_threshold: self.tts_barge_in_rms_threshold,
            tts_barge_in_render_ratio_min: self.tts_barge_in_render_ratio_min,
            tts_barge_in_render_ratio_boost_enabled: self.tts_barge_in_render_ratio_boost_enabled,
            tts_barge_in_render_ratio_boost: self.tts_barge_in_render_ratio_boost,
            tts_barge_in_render_ratio_boost_start_rms: self.tts_barge_in_render_ratio_boost_start_rms,
            tts_barge_in_render_ratio_boost_end_rms: self.tts_barge_in_render_ratio_boost_end_rms,
            tts_barge_in_render_rms_suppress_threshold: self.tts_barge_in_render_rms_suppress_threshold,
            tts_barge_in_render_rms_suppress_ratio_min: self.tts_barge_in_render_rms_suppress_ratio_min,
            tts_asr_leak_suppress_enabled: self.tts_asr_leak_suppress_enabled,
            tts_asr_leak_suppress_ratio_min: self.tts_asr_leak_suppress_ratio_min,
            tts_asr_leak_suppress_render_rms_min: self.tts_asr_leak_suppress_render_rms_min,
            tts_barge_in_render_rms_max_age_ms: self.tts_barge_in_render_rms_max_age_ms,
            tts_barge_in_persistence_ms: self.tts_barge_in_persistence_ms,
            tts_barge_in_confidence_threshold: self.tts_barge_in_confidence_threshold,
            tts_barge_in_confidence_smoothing: self.tts_barge_in_confidence_smoothing,
            tts_barge_in_suspect_hold_ms: self.tts_barge_in_suspect_hold_ms,
            tts_barge_in_recover_ms: self.tts_barge_in_recover_ms,
            barge_in_min_interval_ms: self.tts_barge_in_min_interval_ms,
            adaptive_leak_tuner_enabled: self.adaptive_leak_tuner_enabled,
            adaptive_leak_tuner_observe_only: self.adaptive_leak_tuner_observe_only,
            adaptive_leak_tuner_log_interval_ms: self.adaptive_leak_tuner_log_interval_ms,
        }
    }
}


fn default_render_reference_capacity() -> usize {
    256
}

#[derive(Clone, Debug, Deserialize)]
pub struct EngineConfig {
    pub asr: AsrConfig,
    pub vad: VadConfig,
    #[serde(default)]
    pub barge_in_vad: BargeInVadConfig,
    #[serde(default)]
    pub denoiser: DenoiserConfig,
    pub realtime: RealtimeEngineConfig,
    #[serde(default)]
    pub tts: TtsConfig,
    #[serde(default)]
    pub aec: AecConfig,
    #[serde(default)]
    pub wakeword: WakewordConfig,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: AecBackend::PureRustAec3,
            sample_rate: 48_000,
            stream_delay_ms: Some(120),
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::balanced_profile()
    }
}

impl EngineConfig {
    pub fn resolve_paths(&mut self) {
        let env_root = env::var("VOCOM_MODELS_DIR").ok().filter(|v| !v.is_empty());
        let models_root = env_root
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(resolve_models_root);
        let base_root = models_root
            .as_ref()
            .and_then(|p| p.parent().map(|parent| parent.to_path_buf()));

        self.asr.encoder_path = resolve_path(&self.asr.encoder_path, &models_root, &base_root, &env_root);
        self.asr.decoder_path = resolve_path(&self.asr.decoder_path, &models_root, &base_root, &env_root);
        if let Some(joiner) = self.asr.joiner_path.as_ref() {
            self.asr.joiner_path = Some(resolve_path(joiner, &models_root, &base_root, &env_root));
        }
        self.asr.tokens_path = resolve_path(&self.asr.tokens_path, &models_root, &base_root, &env_root);
        if let Some(hotwords) = self.asr.hotwords_path.as_ref() {
            self.asr.hotwords_path = Some(resolve_path(hotwords, &models_root, &base_root, &env_root));
        }

        self.vad.model_path = resolve_path(&self.vad.model_path, &models_root, &base_root, &env_root);
        self.barge_in_vad.model_path = resolve_path(
            &self.barge_in_vad.model_path,
            &models_root,
            &base_root,
            &env_root,
        );

        if self.denoiser.enabled {
            self.denoiser.model_path = resolve_path(
                &self.denoiser.model_path,
                &models_root,
                &base_root,
                &env_root,
            );
        }

        if self.tts.enabled {
            match self.tts.backend {
                TtsBackend::Vits => {
                    self.tts.model_path = resolve_path(&self.tts.model_path, &models_root, &base_root, &env_root);
                    self.tts.tokens_path = resolve_path(&self.tts.tokens_path, &models_root, &base_root, &env_root);
                    self.tts.data_dir = resolve_path(&self.tts.data_dir, &models_root, &base_root, &env_root);
                    if let Some(dict_dir) = self.tts.dict_dir.as_ref() {
                        self.tts.dict_dir = Some(resolve_path(dict_dir, &models_root, &base_root, &env_root));
                    }
                }
                TtsBackend::Supertonic => {
                    if let Some(path) = self.tts.supertonic.duration_predictor.as_ref() {
                        self.tts.supertonic.duration_predictor = Some(resolve_path(path, &models_root, &base_root, &env_root));
                    }
                    if let Some(path) = self.tts.supertonic.text_encoder.as_ref() {
                        self.tts.supertonic.text_encoder = Some(resolve_path(path, &models_root, &base_root, &env_root));
                    }
                    if let Some(path) = self.tts.supertonic.vector_estimator.as_ref() {
                        self.tts.supertonic.vector_estimator = Some(resolve_path(path, &models_root, &base_root, &env_root));
                    }
                    if let Some(path) = self.tts.supertonic.vocoder.as_ref() {
                        self.tts.supertonic.vocoder = Some(resolve_path(path, &models_root, &base_root, &env_root));
                    }
                    if let Some(path) = self.tts.supertonic.tts_json.as_ref() {
                        self.tts.supertonic.tts_json = Some(resolve_path(path, &models_root, &base_root, &env_root));
                    }
                    if let Some(path) = self.tts.supertonic.unicode_indexer.as_ref() {
                        self.tts.supertonic.unicode_indexer = Some(resolve_path(path, &models_root, &base_root, &env_root));
                    }
                }
            }
            self.tts.output_dir = resolve_path(&self.tts.output_dir, &models_root, &base_root, &env_root);
        }
    }

    pub fn balanced_profile() -> Self {
        Self {
            asr: AsrConfig {
                mode: AsrModeConfig::Offline,
                variant: AsrVariantConfig::Whisper,
                encoder_path: "models/sherpa-onnx-whisper-base.en/base.en-encoder.onnx".to_string(),
                decoder_path: "models/sherpa-onnx-whisper-base.en/base.en-decoder.onnx".to_string(),
                joiner_path: None,
                tokens_path: "models/sherpa-onnx-whisper-base.en/base.en-tokens.txt".to_string(),
                num_threads: 4,
                provider: "cpu".to_string(),
                whisper_language: "".to_string(),
                whisper_task: "transcribe".to_string(),
                whisper_tail_paddings: 0,
                whisper_enable_token_timestamps: false,
                whisper_enable_segment_timestamps: false,
                online_decoding_method: default_online_decoding_method(),
                online_enable_endpoint: default_online_enable_endpoint(),
                online_rule1_min_trailing_silence: default_online_rule1_min_trailing_silence(),
                online_rule2_min_trailing_silence: default_online_rule2_min_trailing_silence(),
                online_rule3_min_utterance_length: default_online_rule3_min_utterance_length(),
                hotwords_path: None,
                hotwords_score: default_hotwords_score(),
            },
            vad: VadConfig {
                model_path: "models/silero_vad.onnx".to_string(),
                threshold: 0.5,
                min_silence_duration: 0.4,
                min_speech_duration: 0.2,
                max_speech_duration: 15.0,
                window_size: 512,
                sample_rate: 16_000,
                num_threads: 1,
                provider: "cpu".to_string(),
                debug: false,
                dynamic_gate_enabled: default_vad_dynamic_gate_enabled(),
                noise_smoothing: default_vad_noise_smoothing(),
                noise_gate_multiplier: default_vad_noise_gate_multiplier(),
                noise_gate_min_rms: default_vad_noise_gate_min_rms(),
                noise_gate_max_rms: default_vad_noise_gate_max_rms(),
            },
            barge_in_vad: BargeInVadConfig {
                model_path: "models/silero_vad.onnx".to_string(),
                threshold: 0.4,
                min_silence_duration: 0.25,
                min_speech_duration: 0.1,
                max_speech_duration: 8.0,
                window_size: 512,
                sample_rate: 16_000,
                num_threads: 1,
                provider: "cpu".to_string(),
                debug: false,
            },
            denoiser: DenoiserConfig::default(),
            realtime: RealtimeEngineConfig {
                target_sample_rate: 16_000,
                chunk_ms: 20,
                audio_queue_capacity: 64,
                event_queue_capacity: 64,
                duplex_mode: DuplexMode::FullDuplex,
                input_overflow_log_interval_ms: default_input_overflow_log_interval_ms(),
                input_overflow_shed_ms: default_input_overflow_shed_ms(),
                post_roll_ms: default_post_roll_ms(),
                short_silence_merge_ms: default_short_silence_merge_ms(),
                input_normalize_enabled: default_input_normalize_enabled(),
                input_normalize_target_peak: default_input_normalize_target_peak(),
                input_normalize_max_gain: default_input_normalize_max_gain(),
                input_clip_guard_enabled: default_input_clip_guard_enabled(),
                input_clip_threshold: default_input_clip_threshold(),
                render_reference_capacity: 256,
                pre_speech_buffer_ms: default_pre_speech_buffer_ms(),
                tts_suppression_cooldown_ms: default_tts_suppression_cooldown_ms(),
                tts_barge_in_rms_threshold: default_tts_barge_in_rms_threshold(),
                tts_barge_in_duck_level: default_tts_barge_in_duck_level(),
                tts_barge_in_stop_after_ms: default_tts_barge_in_stop_after_ms(),
                tts_barge_in_render_ratio_min: default_tts_barge_in_render_ratio_min(),
                tts_barge_in_render_ratio_boost_enabled: default_tts_barge_in_render_ratio_boost_enabled(),
                tts_barge_in_render_ratio_boost: default_tts_barge_in_render_ratio_boost(),
                tts_barge_in_render_ratio_boost_start_rms: default_tts_barge_in_render_ratio_boost_start_rms(),
                tts_barge_in_render_ratio_boost_end_rms: default_tts_barge_in_render_ratio_boost_end_rms(),
                tts_barge_in_render_rms_suppress_threshold: default_tts_barge_in_render_rms_suppress_threshold(),
                tts_barge_in_render_rms_suppress_ratio_min: default_tts_barge_in_render_rms_suppress_ratio_min(),
                tts_asr_leak_suppress_enabled: default_tts_asr_leak_suppress_enabled(),
                tts_asr_leak_suppress_ratio_min: default_tts_asr_leak_suppress_ratio_min(),
                tts_asr_leak_suppress_render_rms_min: default_tts_asr_leak_suppress_render_rms_min(),
                tts_barge_in_render_rms_max_age_ms: default_tts_barge_in_render_rms_max_age_ms(),
                tts_barge_in_persistence_ms: default_tts_barge_in_persistence_ms(),
                tts_barge_in_confidence_threshold: default_tts_barge_in_confidence_threshold(),
                tts_barge_in_confidence_smoothing: default_tts_barge_in_confidence_smoothing(),
                tts_barge_in_suspect_hold_ms: default_tts_barge_in_suspect_hold_ms(),
                tts_barge_in_recover_ms: default_tts_barge_in_recover_ms(),
                tts_barge_in_suspect_drop_grace_ms: default_tts_barge_in_suspect_drop_grace_ms(),
                tts_barge_in_min_interval_ms: default_tts_barge_in_min_interval_ms(),
                adaptive_leak_tuner_enabled: default_adaptive_leak_tuner_enabled(),
                adaptive_leak_tuner_observe_only: default_adaptive_leak_tuner_observe_only(),
                adaptive_leak_tuner_log_interval_ms: default_adaptive_leak_tuner_log_interval_ms(),
            },
            tts: TtsConfig::default(),
            aec: AecConfig::default(),
            wakeword: WakewordConfig::default(),
        }
    }

    pub fn low_latency_profile() -> Self {
        let mut cfg = Self::balanced_profile();
        cfg.vad.min_silence_duration = 0.25;
        cfg.realtime.chunk_ms = 10;
        cfg.realtime.audio_queue_capacity = 128;
        cfg.realtime.event_queue_capacity = 128;
        cfg
    }

    pub fn noisy_room_profile() -> Self {
        let mut cfg = Self::balanced_profile();
        cfg.vad.threshold = 0.6;
        cfg.vad.min_silence_duration = 0.55;
        cfg.vad.min_speech_duration = 0.3;
        cfg
    }

    pub fn laptop_earbud_profile() -> Self {
        let mut cfg = Self::balanced_profile();
        cfg.realtime.chunk_ms = 10;
        cfg.realtime.audio_queue_capacity = 128;
        cfg.realtime.event_queue_capacity = 128;
        cfg.realtime.tts_barge_in_render_rms_max_age_ms = 180;
        cfg.realtime.tts_barge_in_persistence_ms = 100;
        cfg.realtime.tts_barge_in_confidence_threshold = 0.65;
        cfg.realtime.tts_barge_in_suspect_hold_ms = 70;
        cfg.realtime.tts_barge_in_recover_ms = 280;
        cfg.realtime.tts_barge_in_duck_level = 0.6;
        cfg.realtime.tts_barge_in_stop_after_ms = 220;
        cfg
    }

    pub fn laptop_profile() -> Self {
        let mut cfg = Self::balanced_profile();
        cfg.realtime.chunk_ms = 30;
        cfg.realtime.audio_queue_capacity = 128;
        cfg.realtime.event_queue_capacity = 128;
        cfg.realtime.render_reference_capacity = 512;
        cfg.realtime.input_normalize_enabled = false;
        cfg.realtime.input_clip_guard_enabled = false;
        cfg.realtime.tts_barge_in_rms_threshold = 0.018;
        cfg.realtime.tts_barge_in_render_ratio_min = 1.3;
        cfg.realtime.tts_barge_in_render_ratio_boost = 0.75;
        cfg.realtime.tts_barge_in_render_rms_suppress_threshold = 0.10;
        cfg.realtime.tts_barge_in_render_rms_suppress_ratio_min = 2.2;
        cfg.realtime.tts_barge_in_render_rms_max_age_ms = 180;
        cfg.realtime.tts_barge_in_persistence_ms = 80;
        cfg.realtime.tts_barge_in_confidence_threshold = 0.6;
        cfg.realtime.tts_barge_in_suspect_hold_ms = 70;
        cfg.realtime.tts_barge_in_recover_ms = 280;
        cfg.realtime.tts_barge_in_duck_level = 0.6;
        cfg.realtime.tts_barge_in_stop_after_ms = 220;
        cfg.barge_in_vad.threshold = 0.5;
        cfg
    }

    pub fn close_speaker_edge_profile() -> Self {
        let mut cfg = Self::balanced_profile();
        cfg.realtime.chunk_ms = 10;
        cfg.realtime.audio_queue_capacity = 128;
        cfg.realtime.event_queue_capacity = 128;
        cfg.vad.threshold = 0.62;
        cfg.vad.min_speech_duration = 0.3;
        cfg.realtime.tts_barge_in_rms_threshold = 0.03;
        cfg.realtime.tts_barge_in_render_ratio_min = 2.2;
        cfg.realtime.tts_barge_in_render_rms_max_age_ms = 180;
        cfg.realtime.tts_barge_in_persistence_ms = 180;
        cfg.realtime.tts_barge_in_confidence_threshold = 0.75;
        cfg.realtime.tts_barge_in_suspect_hold_ms = 110;
        cfg.realtime.tts_barge_in_recover_ms = 420;
        cfg.realtime.tts_barge_in_duck_level = 0.45;
        cfg.realtime.tts_barge_in_stop_after_ms = 220;
        cfg.realtime.tts_suppression_cooldown_ms = 850;
        cfg
    }

    pub fn from_env() -> Result<Self, VocomError> {
        Self::from_sources()
    }

    pub fn from_sources() -> Result<Self, VocomError> {
        let profile = std::env::var("VOCOM_PROFILE").unwrap_or_else(|_| "balanced".to_string());

        let mut cfg = match profile.as_str() {
            "low_latency" => Self::low_latency_profile(),
            "noisy_room" => Self::noisy_room_profile(),
            "laptop_earbud" => Self::laptop_earbud_profile(),
            "laptop" => Self::laptop_profile(),
            "close_speaker_edge" => Self::close_speaker_edge_profile(),
            "balanced" => Self::balanced_profile(),
            other => {
                return Err(VocomError::ConfigValidation(format!(
                    "unknown VOCOM_PROFILE: {other}. expected balanced|low_latency|noisy_room|laptop|laptop_earbud|close_speaker_edge"
                )))
            }
        };

        if let Some(path) = Self::config_file_path_from_env() {
            cfg = Self::from_json_file(&path)?;
        }

        Self::apply_env_overrides(&mut cfg)?;
        apply_whisper_fast_accuracy_tuning(&mut cfg);

        Ok(cfg)
    }

    pub fn config_file_path_from_env() -> Option<PathBuf> {
        std::env::var("VOCOM_CONFIG_FILE")
            .ok()
            .map(PathBuf::from)
    }

    pub fn from_json_file(path: &Path) -> Result<Self, VocomError> {
        let raw = fs::read_to_string(path).map_err(|e| {
            VocomError::ConfigIo(format!("failed to read config file {}: {e}", path.display()))
        })?;

        serde_json::from_str::<EngineConfig>(&raw).map_err(|e| {
            VocomError::ConfigParse(format!("failed to parse config file {}: {e}", path.display()))
        })
    }

    fn apply_env_overrides(cfg: &mut Self) -> Result<(), VocomError> {
        if let Ok(v) = std::env::var("VOCOM_ASR_MODE") {
            cfg.asr.mode = parse_asr_mode(&v)?;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_VARIANT") {
            cfg.asr.variant = parse_asr_variant(&v)?;
            apply_asr_variant_default_paths(cfg);
        }

        if let Ok(v) = std::env::var("VOCOM_VAD_MODEL") {
            cfg.vad.model_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_MODEL") {
            cfg.barge_in_vad.model_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_ENCODER") {
            cfg.asr.encoder_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_DECODER") {
            cfg.asr.decoder_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_JOINER") {
            cfg.asr.joiner_path = Some(v);
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_TOKENS") {
            cfg.asr.tokens_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_ONLINE_DECODING_METHOD") {
            cfg.asr.online_decoding_method = v;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_ONLINE_ENABLE_ENDPOINT") {
            cfg.asr.online_enable_endpoint = parse_bool(&v, "VOCOM_ASR_ONLINE_ENABLE_ENDPOINT")?;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_ONLINE_RULE1_MIN_TRAILING_SILENCE") {
            cfg.asr.online_rule1_min_trailing_silence =
                parse_f32(&v, "VOCOM_ASR_ONLINE_RULE1_MIN_TRAILING_SILENCE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_ONLINE_RULE2_MIN_TRAILING_SILENCE") {
            cfg.asr.online_rule2_min_trailing_silence =
                parse_f32(&v, "VOCOM_ASR_ONLINE_RULE2_MIN_TRAILING_SILENCE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_ASR_ONLINE_RULE3_MIN_UTTERANCE_LENGTH") {
            cfg.asr.online_rule3_min_utterance_length =
                parse_f32(&v, "VOCOM_ASR_ONLINE_RULE3_MIN_UTTERANCE_LENGTH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_DUPLEX_MODE") {
            cfg.realtime.duplex_mode = parse_duplex_mode(&v)?;
        }

        if let Ok(v) = std::env::var("VOCOM_ASR_THREADS") {
            cfg.asr.num_threads = parse_i32(&v, "VOCOM_ASR_THREADS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_THRESHOLD") {
            cfg.vad.threshold = parse_f32(&v, "VOCOM_VAD_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_THRESHOLD") {
            cfg.barge_in_vad.threshold = parse_f32(&v, "VOCOM_BARGE_IN_VAD_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_DENOISER_ENABLED") {
            cfg.denoiser.enabled = parse_bool(&v, "VOCOM_DENOISER_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_DENOISER_FAMILY") {
            cfg.denoiser.family = parse_denoiser_family(&v)?;
        }
        if let Ok(v) = std::env::var("VOCOM_DENOISER_MODEL") {
            cfg.denoiser.model_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_DENOISER_THREADS") {
            cfg.denoiser.num_threads = parse_i32(&v, "VOCOM_DENOISER_THREADS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_DENOISER_PROVIDER") {
            cfg.denoiser.provider = v;
        }
        if let Ok(v) = std::env::var("VOCOM_DENOISER_DEBUG") {
            cfg.denoiser.debug = parse_bool(&v, "VOCOM_DENOISER_DEBUG")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_DYNAMIC_GATE_ENABLED") {
            cfg.vad.dynamic_gate_enabled = parse_bool(&v, "VOCOM_VAD_DYNAMIC_GATE_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_NOISE_SMOOTHING") {
            cfg.vad.noise_smoothing = parse_f32(&v, "VOCOM_VAD_NOISE_SMOOTHING")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_NOISE_GATE_MULTIPLIER") {
            cfg.vad.noise_gate_multiplier = parse_f32(&v, "VOCOM_VAD_NOISE_GATE_MULTIPLIER")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_NOISE_GATE_MIN_RMS") {
            cfg.vad.noise_gate_min_rms = parse_f32(&v, "VOCOM_VAD_NOISE_GATE_MIN_RMS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_NOISE_GATE_MAX_RMS") {
            cfg.vad.noise_gate_max_rms = parse_f32(&v, "VOCOM_VAD_NOISE_GATE_MAX_RMS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_MIN_SILENCE") {
            cfg.vad.min_silence_duration = parse_f32(&v, "VOCOM_VAD_MIN_SILENCE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_MIN_SILENCE") {
            cfg.barge_in_vad.min_silence_duration =
                parse_f32(&v, "VOCOM_BARGE_IN_VAD_MIN_SILENCE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_MIN_SPEECH") {
            cfg.vad.min_speech_duration = parse_f32(&v, "VOCOM_VAD_MIN_SPEECH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_MIN_SPEECH") {
            cfg.barge_in_vad.min_speech_duration =
                parse_f32(&v, "VOCOM_BARGE_IN_VAD_MIN_SPEECH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_MAX_SPEECH") {
            cfg.vad.max_speech_duration = parse_f32(&v, "VOCOM_VAD_MAX_SPEECH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_MAX_SPEECH") {
            cfg.barge_in_vad.max_speech_duration =
                parse_f32(&v, "VOCOM_BARGE_IN_VAD_MAX_SPEECH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_WINDOW_SIZE") {
            cfg.barge_in_vad.window_size = parse_i32(&v, "VOCOM_BARGE_IN_VAD_WINDOW_SIZE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_SAMPLE_RATE") {
            cfg.barge_in_vad.sample_rate = parse_i32(&v, "VOCOM_BARGE_IN_VAD_SAMPLE_RATE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_THREADS") {
            cfg.barge_in_vad.num_threads = parse_i32(&v, "VOCOM_BARGE_IN_VAD_THREADS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_PROVIDER") {
            cfg.barge_in_vad.provider = v;
        }
        if let Ok(v) = std::env::var("VOCOM_BARGE_IN_VAD_DEBUG") {
            cfg.barge_in_vad.debug = parse_bool(&v, "VOCOM_BARGE_IN_VAD_DEBUG")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TARGET_SAMPLE_RATE") {
            cfg.realtime.target_sample_rate = parse_i32(&v, "VOCOM_TARGET_SAMPLE_RATE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_CHUNK_MS") {
            cfg.realtime.chunk_ms = parse_u32(&v, "VOCOM_CHUNK_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_AUDIO_QUEUE_CAPACITY") {
            cfg.realtime.audio_queue_capacity = parse_usize(&v, "VOCOM_AUDIO_QUEUE_CAPACITY")?;
        }
        if let Ok(v) = std::env::var("VOCOM_EVENT_QUEUE_CAPACITY") {
            cfg.realtime.event_queue_capacity = parse_usize(&v, "VOCOM_EVENT_QUEUE_CAPACITY")?;
        }
        if let Ok(v) = std::env::var("VOCOM_RENDER_REFERENCE_CAPACITY") {
            cfg.realtime.render_reference_capacity =
                parse_usize(&v, "VOCOM_RENDER_REFERENCE_CAPACITY")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_OVERFLOW_LOG_INTERVAL_MS") {
            cfg.realtime.input_overflow_log_interval_ms =
                parse_u64(&v, "VOCOM_INPUT_OVERFLOW_LOG_INTERVAL_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_OVERFLOW_SHED_MS") {
            cfg.realtime.input_overflow_shed_ms =
                parse_u64(&v, "VOCOM_INPUT_OVERFLOW_SHED_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_POST_ROLL_MS") {
            cfg.realtime.post_roll_ms = parse_u32(&v, "VOCOM_POST_ROLL_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_SHORT_SILENCE_MERGE_MS") {
            cfg.realtime.short_silence_merge_ms =
                parse_u32(&v, "VOCOM_SHORT_SILENCE_MERGE_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_NORMALIZE_ENABLED") {
            cfg.realtime.input_normalize_enabled =
                parse_bool(&v, "VOCOM_INPUT_NORMALIZE_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_NORMALIZE_TARGET_PEAK") {
            cfg.realtime.input_normalize_target_peak =
                parse_f32(&v, "VOCOM_INPUT_NORMALIZE_TARGET_PEAK")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_NORMALIZE_MAX_GAIN") {
            cfg.realtime.input_normalize_max_gain =
                parse_f32(&v, "VOCOM_INPUT_NORMALIZE_MAX_GAIN")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_CLIP_GUARD_ENABLED") {
            cfg.realtime.input_clip_guard_enabled =
                parse_bool(&v, "VOCOM_INPUT_CLIP_GUARD_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_INPUT_CLIP_THRESHOLD") {
            cfg.realtime.input_clip_threshold =
                parse_f32(&v, "VOCOM_INPUT_CLIP_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_PRE_SPEECH_BUFFER_MS") {
            cfg.realtime.pre_speech_buffer_ms = parse_u32(&v, "VOCOM_PRE_SPEECH_BUFFER_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_SUPPRESSION_COOLDOWN_MS") {
            cfg.realtime.tts_suppression_cooldown_ms =
                parse_u64(&v, "VOCOM_TTS_SUPPRESSION_COOLDOWN_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RMS_THRESHOLD") {
            cfg.realtime.tts_barge_in_rms_threshold =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RMS_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_DUCK_LEVEL") {
            cfg.realtime.tts_barge_in_duck_level =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_DUCK_LEVEL")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_STOP_AFTER_MS") {
            cfg.realtime.tts_barge_in_stop_after_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_STOP_AFTER_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RATIO_MIN") {
            cfg.realtime.tts_barge_in_render_ratio_min =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RENDER_RATIO_MIN")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST_ENABLED") {
            cfg.realtime.tts_barge_in_render_ratio_boost_enabled =
                parse_bool(&v, "VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST") {
            cfg.realtime.tts_barge_in_render_ratio_boost =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST_START_RMS") {
            cfg.realtime.tts_barge_in_render_ratio_boost_start_rms =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST_START_RMS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST_END_RMS") {
            cfg.realtime.tts_barge_in_render_ratio_boost_end_rms =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RENDER_RATIO_BOOST_END_RMS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RMS_SUPPRESS_THRESHOLD") {
            cfg.realtime.tts_barge_in_render_rms_suppress_threshold =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RENDER_RMS_SUPPRESS_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RMS_SUPPRESS_RATIO_MIN") {
            cfg.realtime.tts_barge_in_render_rms_suppress_ratio_min =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_RENDER_RMS_SUPPRESS_RATIO_MIN")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_ASR_LEAK_SUPPRESS_ENABLED") {
            cfg.realtime.tts_asr_leak_suppress_enabled =
                parse_bool(&v, "VOCOM_TTS_ASR_LEAK_SUPPRESS_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_ASR_LEAK_SUPPRESS_RATIO_MIN") {
            cfg.realtime.tts_asr_leak_suppress_ratio_min =
                parse_f32(&v, "VOCOM_TTS_ASR_LEAK_SUPPRESS_RATIO_MIN")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_ASR_LEAK_SUPPRESS_RENDER_RMS_MIN") {
            cfg.realtime.tts_asr_leak_suppress_render_rms_min =
                parse_f32(&v, "VOCOM_TTS_ASR_LEAK_SUPPRESS_RENDER_RMS_MIN")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RENDER_RMS_MAX_AGE_MS") {
            cfg.realtime.tts_barge_in_render_rms_max_age_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_RENDER_RMS_MAX_AGE_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_PERSISTENCE_MS") {
            cfg.realtime.tts_barge_in_persistence_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_PERSISTENCE_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_CONFIDENCE_THRESHOLD") {
            cfg.realtime.tts_barge_in_confidence_threshold =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_CONFIDENCE_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_CONFIDENCE_SMOOTHING") {
            cfg.realtime.tts_barge_in_confidence_smoothing =
                parse_f32(&v, "VOCOM_TTS_BARGE_IN_CONFIDENCE_SMOOTHING")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_SUSPECT_HOLD_MS") {
            cfg.realtime.tts_barge_in_suspect_hold_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_SUSPECT_HOLD_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_RECOVER_MS") {
            cfg.realtime.tts_barge_in_recover_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_RECOVER_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_SUSPECT_DROP_GRACE_MS") {
            cfg.realtime.tts_barge_in_suspect_drop_grace_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_SUSPECT_DROP_GRACE_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_BARGE_IN_MIN_INTERVAL_MS") {
            cfg.realtime.tts_barge_in_min_interval_ms =
                parse_u64(&v, "VOCOM_TTS_BARGE_IN_MIN_INTERVAL_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_ADAPTIVE_LEAK_TUNER_ENABLED") {
            cfg.realtime.adaptive_leak_tuner_enabled =
                parse_bool(&v, "VOCOM_ADAPTIVE_LEAK_TUNER_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_ADAPTIVE_LEAK_TUNER_OBSERVE_ONLY") {
            cfg.realtime.adaptive_leak_tuner_observe_only =
                parse_bool(&v, "VOCOM_ADAPTIVE_LEAK_TUNER_OBSERVE_ONLY")?;
        }
        if let Ok(v) = std::env::var("VOCOM_ADAPTIVE_LEAK_TUNER_LOG_INTERVAL_MS") {
            cfg.realtime.adaptive_leak_tuner_log_interval_ms =
                parse_u64(&v, "VOCOM_ADAPTIVE_LEAK_TUNER_LOG_INTERVAL_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_ENABLED") {
            cfg.tts.enabled = parse_bool(&v, "VOCOM_TTS_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_MODEL") {
            cfg.tts.model_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_TOKENS") {
            cfg.tts.tokens_path = v;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_DATA_DIR") {
            cfg.tts.data_dir = v;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_DICT_DIR") {
            cfg.tts.dict_dir = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_PROVIDER") {
            cfg.tts.provider = v;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_THREADS") {
            cfg.tts.num_threads = parse_i32(&v, "VOCOM_TTS_THREADS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_SPEED") {
            cfg.tts.speed = parse_f32(&v, "VOCOM_TTS_SPEED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_SPEAKER") {
            cfg.tts.speaker_id = parse_i32(&v, "VOCOM_TTS_SPEAKER")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_ENABLED") {
            cfg.tts.fx.enabled = parse_bool(&v, "VOCOM_TTS_FX_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_LOW_CUT_HZ") {
            cfg.tts.fx.low_cut_hz = parse_f32(&v, "VOCOM_TTS_FX_LOW_CUT_HZ")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_HIGH_CUT_HZ") {
            cfg.tts.fx.high_cut_hz = parse_f32(&v, "VOCOM_TTS_FX_HIGH_CUT_HZ")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_ROBOT_MIX") {
            cfg.tts.fx.robot_mix = parse_f32(&v, "VOCOM_TTS_FX_ROBOT_MIX")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_RING_MOD_HZ") {
            cfg.tts.fx.ring_mod_hz = parse_f32(&v, "VOCOM_TTS_FX_RING_MOD_HZ")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_DISTORTION_DRIVE") {
            cfg.tts.fx.distortion_drive =
                parse_f32(&v, "VOCOM_TTS_FX_DISTORTION_DRIVE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_ECHO_DELAY_MS") {
            cfg.tts.fx.echo_delay_ms = parse_u32(&v, "VOCOM_TTS_FX_ECHO_DELAY_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_ECHO_FEEDBACK") {
            cfg.tts.fx.echo_feedback = parse_f32(&v, "VOCOM_TTS_FX_ECHO_FEEDBACK")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_ECHO_MIX") {
            cfg.tts.fx.echo_mix = parse_f32(&v, "VOCOM_TTS_FX_ECHO_MIX")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_FX_OUTPUT_GAIN") {
            cfg.tts.fx.output_gain = parse_f32(&v, "VOCOM_TTS_FX_OUTPUT_GAIN")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TTS_OUTPUT_DIR") {
            cfg.tts.output_dir = v;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_ENABLED") {
            cfg.aec.enabled = parse_bool(&v, "VOCOM_AEC_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_SAMPLE_RATE") {
            cfg.aec.sample_rate = parse_i32(&v, "VOCOM_AEC_SAMPLE_RATE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_STREAM_DELAY_MS") {
            cfg.aec.stream_delay_ms = Some(parse_i32(&v, "VOCOM_AEC_STREAM_DELAY_MS")?);
        }
        if let Ok(v) = std::env::var("VOCOM_WAKEWORD") {
            cfg.wakeword.keyword = v;
        }
        if let Ok(v) = std::env::var("VOCOM_WAKEWORD_VARIANTS") {
            cfg.wakeword.variants = parse_csv_list(&v);
        }
        if let Ok(v) = std::env::var("VOCOM_WAKEWORD_COOLDOWN_MS") {
            cfg.wakeword.cooldown_ms = parse_u64(&v, "VOCOM_WAKEWORD_COOLDOWN_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_WAKEWORD_ENTRYPOINT") {
            cfg.wakeword.entrypoint_enabled = parse_bool(&v, "VOCOM_WAKEWORD_ENTRYPOINT")?;
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<(), VocomError> {
        if self.asr.num_threads <= 0 {
            return Err(VocomError::ConfigValidation("asr.num_threads must be > 0".to_string()));
        }
        if self.asr.online_rule1_min_trailing_silence < 0.0 {
            return Err(VocomError::ConfigValidation(
                "asr.online_rule1_min_trailing_silence must be >= 0.0".to_string(),
            ));
        }
        if self.asr.online_rule2_min_trailing_silence < 0.0 {
            return Err(VocomError::ConfigValidation(
                "asr.online_rule2_min_trailing_silence must be >= 0.0".to_string(),
            ));
        }
        if self.asr.online_rule3_min_utterance_length < 0.0 {
            return Err(VocomError::ConfigValidation(
                "asr.online_rule3_min_utterance_length must be >= 0.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.realtime.input_normalize_target_peak) {
            return Err(VocomError::ConfigValidation(
                "realtime.input_normalize_target_peak must be between 0.0 and 1.0"
                    .to_string(),
            ));
        }
        if self.realtime.input_normalize_max_gain < 1.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.input_normalize_max_gain must be >= 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.realtime.input_clip_threshold) {
            return Err(VocomError::ConfigValidation(
                "realtime.input_clip_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.vad.sample_rate <= 0 {
            return Err(VocomError::ConfigValidation("vad.sample_rate must be > 0".to_string()));
        }
        if self.barge_in_vad.sample_rate <= 0 {
            return Err(VocomError::ConfigValidation(
                "barge_in_vad.sample_rate must be > 0".to_string(),
            ));
        }
        if !self
            .wakeword
            .keyword
            .chars()
            .any(|ch| ch.is_ascii_alphabetic())
        {
            return Err(VocomError::ConfigValidation(
                "wakeword.keyword must contain alphabetic characters".to_string(),
            ));
        }
        if self.realtime.target_sample_rate <= 0 {
            return Err(VocomError::ConfigValidation(
                "realtime.target_sample_rate must be > 0".to_string(),
            ));
        }
        if self.realtime.chunk_ms == 0 {
            return Err(VocomError::ConfigValidation("realtime.chunk_ms must be > 0".to_string()));
        }
        if self.realtime.audio_queue_capacity == 0 || self.realtime.event_queue_capacity == 0 {
            return Err(VocomError::ConfigValidation(
                "realtime queue capacities must be > 0".to_string(),
            ));
        }
        if self.realtime.render_reference_capacity == 0 {
            return Err(VocomError::ConfigValidation(
                "realtime.render_reference_capacity must be > 0".to_string(),
            ));
        }
        if self.realtime.tts_barge_in_rms_threshold < 0.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_rms_threshold must be >= 0.0".to_string(),
            ));
        }
        if self.realtime.tts_barge_in_render_ratio_min < 1.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_render_ratio_min must be >= 1.0".to_string(),
            ));
        }
        if self.realtime.tts_barge_in_render_ratio_boost < 0.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_render_ratio_boost must be >= 0.0".to_string(),
            ));
        }
        if self.realtime.tts_barge_in_render_ratio_boost_end_rms
            <= self.realtime.tts_barge_in_render_ratio_boost_start_rms
        {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_render_ratio_boost_end_rms must be > boost_start_rms"
                    .to_string(),
            ));
        }
        if self.realtime.tts_barge_in_render_rms_suppress_threshold < 0.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_render_rms_suppress_threshold must be >= 0.0"
                    .to_string(),
            ));
        }
        if self.realtime.tts_barge_in_render_rms_suppress_ratio_min < 1.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_render_rms_suppress_ratio_min must be >= 1.0"
                    .to_string(),
            ));
        }
        if self.realtime.tts_asr_leak_suppress_ratio_min < 1.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_asr_leak_suppress_ratio_min must be >= 1.0".to_string(),
            ));
        }
        if self.realtime.tts_asr_leak_suppress_render_rms_min < 0.0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_asr_leak_suppress_render_rms_min must be >= 0.0".to_string(),
            ));
        }
        if self.realtime.tts_barge_in_persistence_ms == 0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_persistence_ms must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.realtime.tts_barge_in_confidence_threshold) {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_confidence_threshold must be between 0.0 and 1.0"
                    .to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.realtime.tts_barge_in_confidence_smoothing) {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_confidence_smoothing must be between 0.0 and 1.0"
                    .to_string(),
            ));
        }
        if self.realtime.tts_barge_in_suspect_hold_ms == 0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_suspect_hold_ms must be > 0".to_string(),
            ));
        }
        if self.realtime.tts_barge_in_recover_ms == 0 {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_recover_ms must be > 0".to_string(),
            ));
        }
        if self.realtime.adaptive_leak_tuner_log_interval_ms == 0 {
            return Err(VocomError::ConfigValidation(
                "realtime.adaptive_leak_tuner_log_interval_ms must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.realtime.tts_barge_in_duck_level) {
            return Err(VocomError::ConfigValidation(
                "realtime.tts_barge_in_duck_level must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.vad.threshold) {
            return Err(VocomError::ConfigValidation(
                "vad.threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.barge_in_vad.threshold) {
            return Err(VocomError::ConfigValidation(
                "barge_in_vad.threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.vad.noise_smoothing) {
            return Err(VocomError::ConfigValidation(
                "vad.noise_smoothing must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.vad.noise_gate_multiplier < 1.0 {
            return Err(VocomError::ConfigValidation(
                "vad.noise_gate_multiplier must be >= 1.0".to_string(),
            ));
        }
        if self.vad.noise_gate_min_rms < 0.0 || self.vad.noise_gate_max_rms <= 0.0 {
            return Err(VocomError::ConfigValidation(
                "vad.noise_gate_min_rms must be >= 0.0 and vad.noise_gate_max_rms must be > 0.0"
                    .to_string(),
            ));
        }
        if self.vad.noise_gate_min_rms > self.vad.noise_gate_max_rms {
            return Err(VocomError::ConfigValidation(
                "vad.noise_gate_min_rms must be <= vad.noise_gate_max_rms".to_string(),
            ));
        }

        ensure_exists(&self.asr.encoder_path)?;
        ensure_exists(&self.asr.decoder_path)?;
        ensure_exists(&self.asr.tokens_path)?;
        if matches!(self.asr.mode, AsrModeConfig::Online)
            || matches!(self.asr.variant, AsrVariantConfig::StreamingZipformer)
        {
            let joiner = self.asr.joiner_path.as_ref().ok_or_else(|| {
                VocomError::ConfigValidation(
                    "asr.joiner_path is required for online/streaming zipformer".to_string(),
                )
            })?;
            ensure_exists(joiner)?;
        }
        ensure_exists(&self.vad.model_path)?;

        if self.tts.enabled {
            if self.tts.num_threads <= 0 {
                return Err(VocomError::ConfigValidation(
                    "tts.num_threads must be > 0".to_string(),
                ));
            }
            if self.tts.speed <= 0.0 {
                return Err(VocomError::ConfigValidation(
                    "tts.speed must be > 0.0".to_string(),
                ));
            }
            if self.tts.fx.low_cut_hz < 0.0 {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.low_cut_hz must be >= 0.0".to_string(),
                ));
            }
            if self.tts.fx.high_cut_hz <= 0.0 {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.high_cut_hz must be > 0.0".to_string(),
                ));
            }
            if self.tts.fx.low_cut_hz >= self.tts.fx.high_cut_hz {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.low_cut_hz must be < tts.fx.high_cut_hz".to_string(),
                ));
            }
            if !(0.0..=1.0).contains(&self.tts.fx.robot_mix) {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.robot_mix must be between 0.0 and 1.0".to_string(),
                ));
            }
            if self.tts.fx.ring_mod_hz < 0.0 {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.ring_mod_hz must be >= 0.0".to_string(),
                ));
            }
            if self.tts.fx.distortion_drive <= 0.0 {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.distortion_drive must be > 0.0".to_string(),
                ));
            }
            if self.tts.fx.echo_delay_ms > 1_000 {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.echo_delay_ms must be <= 1000".to_string(),
                ));
            }
            if !(0.0..=0.98).contains(&self.tts.fx.echo_feedback) {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.echo_feedback must be between 0.0 and 0.98".to_string(),
                ));
            }
            if !(0.0..=1.0).contains(&self.tts.fx.echo_mix) {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.echo_mix must be between 0.0 and 1.0".to_string(),
                ));
            }
            if !(0.0..=2.0).contains(&self.tts.fx.output_gain) {
                return Err(VocomError::ConfigValidation(
                    "tts.fx.output_gain must be between 0.0 and 2.0".to_string(),
                ));
            }

            ensure_exists(&self.tts.model_path)?;
            ensure_exists(&self.tts.tokens_path)?;
            ensure_exists(&self.tts.data_dir)?;

            if let Some(ref dict_dir) = self.tts.dict_dir {
                ensure_exists(dict_dir)?;
            }
        }

        if self.aec.enabled {
            if self.aec.sample_rate <= 0 {
                return Err(VocomError::ConfigValidation(
                    "aec.sample_rate must be > 0".to_string(),
                ));
            }
            if self.aec.sample_rate < 16_000 {
                return Err(VocomError::ConfigValidation(
                    "aec.sample_rate must be >= 16000".to_string(),
                ));
            }
            if let Some(delay_ms) = self.aec.stream_delay_ms {
                if delay_ms < 0 {
                    return Err(VocomError::ConfigValidation(
                        "aec.stream_delay_ms must be >= 0".to_string(),
                    ));
                }
            }
        }

        if self.denoiser.enabled {
            if self.denoiser.num_threads <= 0 {
                return Err(VocomError::ConfigValidation(
                    "denoiser.num_threads must be > 0".to_string(),
                ));
            }
            if self.denoiser.model_path.is_empty() {
                return Err(VocomError::ConfigValidation(
                    "denoiser.model_path must be set when denoiser is enabled".to_string(),
                ));
            }
            ensure_exists(&self.denoiser.model_path)?;
        }

        Ok(())
    }
}

fn ensure_exists(path: &str) -> Result<(), VocomError> {
    let normalized = normalize_legacy_tts_path(path);
    let path = normalized.as_str();

    let direct = Path::new(path);
    if direct.exists() {
        return Ok(());
    }

    if let Ok(env_root) = env::var("VOCOM_MODELS_DIR") {
        if !env_root.is_empty() {
            let env_path = PathBuf::from(env_root);
            if env_path.is_dir() && path.starts_with("models/") {
                if let Some(stripped) = path.strip_prefix("models/") {
                    let candidate = env_path.join(stripped);
                    if candidate.exists() {
                        return Ok(());
                    }
                }
            }
        }
    }

    let mut last_candidate: Option<PathBuf> = None;
    if direct.is_relative() {
        if path.starts_with("models/") {
            if let Some(root) = resolve_models_root() {
                if let Some(stripped) = path.strip_prefix("models/") {
                    let candidate = root.join(stripped);
                    if candidate.exists() {
                        return Ok(());
                    }
                    last_candidate = Some(candidate);
                }
            }
        }

        if path.starts_with("target/") {
            if let Some(root) = resolve_models_root().and_then(|p| p.parent().map(|p| p.to_path_buf())) {
                let candidate = root.join(path);
                if candidate.exists() {
                    return Ok(());
                }
                last_candidate = Some(candidate);
            }
        }

        if let Ok(cwd) = env::current_dir() {
            let candidate = cwd.join(path);
            if candidate.exists() {
                return Ok(());
            }
            last_candidate = Some(candidate);
        }
    }

    let reported = last_candidate
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    Err(VocomError::MissingModelPath(reported))
}

fn resolve_path(
    value: &str,
    models_root: &Option<PathBuf>,
    base_root: &Option<PathBuf>,
    env_root: &Option<String>,
) -> String {
    let value = normalize_legacy_tts_path(value);

    if let Some(env_root) = env_root {
        let env_token = "${VOCOM_MODELS_DIR}";
        if value.contains(env_token) {
            return value.replace(env_token, env_root);
        }
    }

    let path = Path::new(&value);
    if path.is_absolute() {
        return value.to_string();
    }

    if value.starts_with("models/") {
        if let Some(root) = models_root {
            if let Some(stripped) = value.strip_prefix("models/") {
                return root.join(stripped).to_string_lossy().to_string();
            }
        }
    }

    if value.starts_with("target/") {
        if let Some(root) = base_root {
            return root.join(value).to_string_lossy().to_string();
        }
    }

    if let Some(root) = base_root {
        return root.join(value).to_string_lossy().to_string();
    }

    if let Ok(cwd) = env::current_dir() {
        return cwd.join(value).to_string_lossy().to_string();
    }

    value.to_string()
}

fn normalize_legacy_tts_path(value: &str) -> String {
    // Backward-compat: earlier Android builds used typoed Piper folder names.
    // Normalize them here so startup does not fail on stale configs.
    if value.contains("vits-pipe-en_US_female-medium") {
        return value.replace(
            "vits-pipe-en_US_female-medium",
            "vits-piper-en_US-hfc_female-medium",
        );
    }

    if value.contains("vits-piper-en_US_female-medium") {
        return value.replace(
            "vits-piper-en_US_female-medium",
            "vits-piper-en_US-hfc_female-medium",
        );
    }

    value.to_string()
}

fn resolve_models_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.to_path_buf());
            if let Some(grand) = parent.parent() {
                candidates.push(grand.to_path_buf());
            }
        }
    }

    for candidate in candidates {
        let mut current = candidate;
        for _ in 0..12 {
            let models_dir = current.join("models");
            if models_dir.is_dir() {
                return Some(models_dir);
            }

            if let Some(parent) = current.parent() {
                let sibling_models = parent.join("vocom").join("models");
                if sibling_models.is_dir() {
                    return Some(sibling_models);
                }
            }
            let Some(parent) = current.parent() else {
                break;
            };
            current = parent.to_path_buf();
        }
    }

    None
}

fn parse_i32(raw: &str, name: &str) -> Result<i32, VocomError> {
    raw.parse::<i32>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid integer for {name}: {raw}")))
}

fn parse_csv_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_u32(raw: &str, name: &str) -> Result<u32, VocomError> {
    raw.parse::<u32>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid unsigned integer for {name}: {raw}")))
}

fn parse_usize(raw: &str, name: &str) -> Result<usize, VocomError> {
    raw.parse::<usize>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid unsigned integer for {name}: {raw}")))
}

fn parse_u64(raw: &str, name: &str) -> Result<u64, VocomError> {
    raw.parse::<u64>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid unsigned integer for {name}: {raw}")))
}

fn parse_f32(raw: &str, name: &str) -> Result<f32, VocomError> {
    raw.parse::<f32>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid float for {name}: {raw}")))
}

fn parse_bool(raw: &str, name: &str) -> Result<bool, VocomError> {
    raw.parse::<bool>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid bool for {name}: {raw}")))
}

fn parse_asr_variant(raw: &str) -> Result<AsrVariantConfig, VocomError> {
    match raw {
        "whisper" => Ok(AsrVariantConfig::Whisper),
        "moonshine_v2" => Ok(AsrVariantConfig::MoonshineV2),
        "streaming_zipformer" => Ok(AsrVariantConfig::StreamingZipformer),
        "nemotron_streaming" => Ok(AsrVariantConfig::NemotronStreaming),
        other => Err(VocomError::ConfigValidation(format!(
            "invalid ASR variant: {other}. expected whisper|moonshine_v2|streaming_zipformer|nemotron_streaming"
        ))),
    }
}

fn parse_asr_mode(raw: &str) -> Result<AsrModeConfig, VocomError> {
    match raw {
        "offline" => Ok(AsrModeConfig::Offline),
        "online" => Ok(AsrModeConfig::Online),
        other => Err(VocomError::ConfigValidation(format!(
            "invalid ASR mode: {other}. expected offline|online"
        ))),
    }
}

fn parse_duplex_mode(raw: &str) -> Result<DuplexMode, VocomError> {
    match raw {
        "full_duplex" => Ok(DuplexMode::FullDuplex),
        "half_duplex_mute_mic" => Ok(DuplexMode::HalfDuplexMuteMic),
        other => Err(VocomError::ConfigValidation(format!(
            "invalid duplex mode: {other}. expected full_duplex|half_duplex_mute_mic"
        ))),
    }
}

fn parse_denoiser_family(raw: &str) -> Result<DenoiserModelFamily, VocomError> {
    match raw {
        "gtcrn" => Ok(DenoiserModelFamily::Gtcrn),
        "dpdfnet" => Ok(DenoiserModelFamily::Dpdfnet),
        other => Err(VocomError::ConfigValidation(format!(
            "invalid denoiser family: {other}. expected gtcrn|dpdfnet"
        ))),
    }
}

fn apply_asr_variant_default_paths(cfg: &mut EngineConfig) {
    match cfg.asr.variant {
        AsrVariantConfig::Whisper => {
            cfg.asr.mode = AsrModeConfig::Offline;
            cfg.asr.encoder_path = "models/sherpa-onnx-whisper-base.en/base.en-encoder.onnx".to_string();
            cfg.asr.decoder_path = "models/sherpa-onnx-whisper-base.en/base.en-decoder.onnx".to_string();
            cfg.asr.joiner_path = None;
            cfg.asr.tokens_path = "models/sherpa-onnx-whisper-base.en/base.en-tokens.txt".to_string();
        }
        AsrVariantConfig::MoonshineV2 => {
            cfg.asr.mode = AsrModeConfig::Offline;
            cfg.asr.encoder_path =
                "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/encoder_model.ort"
                    .to_string();
            cfg.asr.decoder_path = "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/decoder_model_merged.ort".to_string();
            cfg.asr.joiner_path = None;
            cfg.asr.tokens_path = "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/tokens.txt".to_string();
        }
        AsrVariantConfig::StreamingZipformer => {
            cfg.asr.mode = AsrModeConfig::Online;
            cfg.asr.encoder_path = "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/encoder.onnx".to_string();
            cfg.asr.decoder_path = "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/decoder.onnx".to_string();
            cfg.asr.joiner_path = Some(
                "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/joiner.onnx"
                    .to_string(),
            );
            cfg.asr.tokens_path = "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/tokens.txt".to_string();
            cfg.asr.online_decoding_method = "greedy_search".to_string();
            cfg.asr.online_enable_endpoint = true;
            cfg.asr.online_rule1_min_trailing_silence = 0.35;
            cfg.asr.online_rule2_min_trailing_silence = 0.8;
            cfg.asr.online_rule3_min_utterance_length = 8.0;

            // Full voice workflow defaults for streaming zipformer.
            cfg.tts.enabled = true;
            cfg.aec.enabled = true;
        }
        AsrVariantConfig::NemotronStreaming => {
            cfg.asr.mode = AsrModeConfig::Online;
            cfg.asr.encoder_path = "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/encoder.int8.onnx".to_string();
            cfg.asr.decoder_path = "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/decoder.int8.onnx".to_string();
            cfg.asr.joiner_path = Some(
                "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/joiner.int8.onnx"
                    .to_string(),
            );
            cfg.asr.tokens_path = "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/tokens.txt".to_string();
            cfg.asr.online_decoding_method = "greedy_search".to_string();
            cfg.asr.online_enable_endpoint = true;
            cfg.asr.online_rule1_min_trailing_silence = 0.45;
            cfg.asr.online_rule2_min_trailing_silence = 1.0;
            cfg.asr.online_rule3_min_utterance_length = 10.0;
        }
    }
}

fn apply_whisper_fast_accuracy_tuning(cfg: &mut EngineConfig) {
    if !matches!(cfg.asr.variant, AsrVariantConfig::Whisper)
        || !matches!(cfg.asr.mode, AsrModeConfig::Offline)
    {
        return;
    }

    // Keep chunk cadence unchanged for latency, but improve speech recall
    // and reduce truncation risk for low-energy/short utterances.
    cfg.vad.threshold = cfg.vad.threshold.min(0.46);
    cfg.vad.min_speech_duration = cfg.vad.min_speech_duration.min(0.18);
    cfg.vad.min_silence_duration = cfg.vad.min_silence_duration.min(0.35);

    if cfg.vad.dynamic_gate_enabled {
        cfg.vad.noise_gate_multiplier = cfg.vad.noise_gate_multiplier.min(1.7);
        cfg.vad.noise_gate_min_rms = cfg.vad.noise_gate_min_rms.min(0.0015);
        cfg.vad.noise_gate_max_rms = cfg.vad.noise_gate_max_rms.min(0.07);
    }

    cfg.realtime.pre_speech_buffer_ms = cfg.realtime.pre_speech_buffer_ms.max(360);
    cfg.realtime.post_roll_ms = cfg.realtime.post_roll_ms.max(160);
    cfg.realtime.short_silence_merge_ms = cfg.realtime.short_silence_merge_ms.max(180);
}

#[cfg(test)]
mod tests {
    use super::{AsrModeConfig, AsrVariantConfig, EngineConfig, apply_whisper_fast_accuracy_tuning};

    #[test]
    fn whisper_offline_tuning_improves_recall_without_touching_chunk_ms() {
        let mut cfg = EngineConfig::balanced_profile();
        cfg.asr.variant = AsrVariantConfig::Whisper;
        cfg.asr.mode = AsrModeConfig::Offline;
        cfg.vad.threshold = 0.62;
        cfg.vad.min_speech_duration = 0.30;
        cfg.vad.min_silence_duration = 0.45;
        cfg.vad.noise_gate_multiplier = 2.2;
        cfg.vad.noise_gate_min_rms = 0.003;
        cfg.vad.noise_gate_max_rms = 0.08;
        cfg.realtime.chunk_ms = 20;
        cfg.realtime.pre_speech_buffer_ms = 300;
        cfg.realtime.post_roll_ms = 120;
        cfg.realtime.short_silence_merge_ms = 150;

        apply_whisper_fast_accuracy_tuning(&mut cfg);

        assert_eq!(cfg.realtime.chunk_ms, 20);
        assert_eq!(cfg.vad.threshold, 0.46);
        assert_eq!(cfg.vad.min_speech_duration, 0.18);
        assert_eq!(cfg.vad.min_silence_duration, 0.35);
        assert_eq!(cfg.vad.noise_gate_multiplier, 1.7);
        assert_eq!(cfg.vad.noise_gate_min_rms, 0.0015);
        assert_eq!(cfg.vad.noise_gate_max_rms, 0.07);
        assert_eq!(cfg.realtime.pre_speech_buffer_ms, 360);
        assert_eq!(cfg.realtime.post_roll_ms, 160);
        assert_eq!(cfg.realtime.short_silence_merge_ms, 180);
    }

    #[test]
    fn non_whisper_or_online_configs_are_unchanged() {
        let mut cfg = EngineConfig::balanced_profile();
        cfg.asr.variant = AsrVariantConfig::StreamingZipformer;
        cfg.asr.mode = AsrModeConfig::Online;
        let before = (
            cfg.vad.threshold,
            cfg.realtime.pre_speech_buffer_ms,
            cfg.realtime.post_roll_ms,
            cfg.realtime.short_silence_merge_ms,
        );

        apply_whisper_fast_accuracy_tuning(&mut cfg);

        let after = (
            cfg.vad.threshold,
            cfg.realtime.pre_speech_buffer_ms,
            cfg.realtime.post_roll_ms,
            cfg.realtime.short_silence_merge_ms,
        );
        assert_eq!(before, after);
    }
}

pub struct ConfigWatcher {
    path: Option<PathBuf>,
    last_modified: Option<SystemTime>,
}

impl ConfigWatcher {
    pub fn new_from_env() -> Result<Self, VocomError> {
        let path = EngineConfig::config_file_path_from_env();
        let last_modified = if let Some(ref p) = path {
            Some(read_modified(p)?)
        } else {
            None
        };

        Ok(Self {
            path,
            last_modified,
        })
    }

    pub fn has_path(&self) -> bool {
        self.path.is_some()
    }

    pub fn changed(&mut self) -> Result<bool, VocomError> {
        let Some(path) = self.path.as_ref() else {
            return Ok(false);
        };

        let modified = read_modified(path)?;
        let has_changed = match self.last_modified {
            Some(prev) => modified > prev,
            None => true,
        };

        if has_changed {
            self.last_modified = Some(modified);
        }

        Ok(has_changed)
    }
}

fn read_modified(path: &Path) -> Result<SystemTime, VocomError> {
    let metadata = fs::metadata(path).map_err(|e| {
        VocomError::ConfigIo(format!("failed to read metadata for {}: {e}", path.display()))
    })?;

    metadata.modified().map_err(|e| {
        VocomError::ConfigIo(format!(
            "failed to read modification time for {}: {e}",
            path.display()
        ))
    })
}
