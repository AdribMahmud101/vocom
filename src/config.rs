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
}

fn default_online_decoding_method() -> String {
    "greedy_search".to_string()
}

fn default_online_enable_endpoint() -> bool {
    true
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
}

#[derive(Clone, Debug, Deserialize)]
pub struct RealtimeEngineConfig {
    pub target_sample_rate: i32,
    pub chunk_ms: u32,
    pub audio_queue_capacity: usize,
    pub event_queue_capacity: usize,
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
    WebrtcNative,
}

impl Default for AecBackend {
    fn default() -> Self {
        Self::PureRustAec3
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TtsConfig {
    pub enabled: bool,
    pub model_path: String,
    pub tokens_path: String,
    pub data_dir: String,
    pub dict_dir: Option<String>,
    pub num_threads: i32,
    pub provider: String,
    pub speed: f32,
    pub speaker_id: i32,
    pub output_dir: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: "models/vits-piper-en_US-hfc_female-medium/en_US-hfc_female-medium.onnx"
                .to_string(),
            tokens_path: "models/vits-piper-en_US-hfc_female-medium/tokens.txt".to_string(),
            data_dir: "models/vits-piper-en_US-hfc_female-medium/espeak-ng-data".to_string(),
            dict_dir: None,
            num_threads: 2,
            provider: "cpu".to_string(),
            speed: 1.0,
            speaker_id: 0,
            output_dir: "target/tts".to_string(),
        }
    }
}

impl RealtimeEngineConfig {
    pub fn to_runtime(&self) -> RealtimeConfig {
        RealtimeConfig {
            target_sample_rate: self.target_sample_rate,
            chunk_ms: self.chunk_ms,
            audio_queue_capacity: self.audio_queue_capacity,
            event_queue_capacity: self.event_queue_capacity,
            pre_speech_buffer_ms: self.pre_speech_buffer_ms,
            tts_suppression_cooldown_ms: self.tts_suppression_cooldown_ms,
            tts_barge_in_rms_threshold: self.tts_barge_in_rms_threshold,
            tts_barge_in_render_ratio_min: self.tts_barge_in_render_ratio_min,
            tts_barge_in_render_rms_max_age_ms: self.tts_barge_in_render_rms_max_age_ms,
            tts_barge_in_persistence_ms: self.tts_barge_in_persistence_ms,
            tts_barge_in_confidence_threshold: self.tts_barge_in_confidence_threshold,
            tts_barge_in_confidence_smoothing: self.tts_barge_in_confidence_smoothing,
            tts_barge_in_suspect_hold_ms: self.tts_barge_in_suspect_hold_ms,
            tts_barge_in_recover_ms: self.tts_barge_in_recover_ms,
            barge_in_min_interval_ms: self.tts_barge_in_min_interval_ms,
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
    pub realtime: RealtimeEngineConfig,
    #[serde(default)]
    pub tts: TtsConfig,
    #[serde(default)]
    pub aec: AecConfig,
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
                whisper_language: "en".to_string(),
                whisper_task: "transcribe".to_string(),
                whisper_tail_paddings: -1,
                whisper_enable_token_timestamps: false,
                whisper_enable_segment_timestamps: false,
                online_decoding_method: default_online_decoding_method(),
                online_enable_endpoint: default_online_enable_endpoint(),
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
            },
            realtime: RealtimeEngineConfig {
                target_sample_rate: 16_000,
                chunk_ms: 20,
                audio_queue_capacity: 64,
                event_queue_capacity: 64,
                render_reference_capacity: 256,
                pre_speech_buffer_ms: default_pre_speech_buffer_ms(),
                tts_suppression_cooldown_ms: default_tts_suppression_cooldown_ms(),
                tts_barge_in_rms_threshold: default_tts_barge_in_rms_threshold(),
                tts_barge_in_duck_level: default_tts_barge_in_duck_level(),
                tts_barge_in_stop_after_ms: default_tts_barge_in_stop_after_ms(),
                tts_barge_in_render_ratio_min: default_tts_barge_in_render_ratio_min(),
                tts_barge_in_render_rms_max_age_ms: default_tts_barge_in_render_rms_max_age_ms(),
                tts_barge_in_persistence_ms: default_tts_barge_in_persistence_ms(),
                tts_barge_in_confidence_threshold: default_tts_barge_in_confidence_threshold(),
                tts_barge_in_confidence_smoothing: default_tts_barge_in_confidence_smoothing(),
                tts_barge_in_suspect_hold_ms: default_tts_barge_in_suspect_hold_ms(),
                tts_barge_in_recover_ms: default_tts_barge_in_recover_ms(),
                tts_barge_in_suspect_drop_grace_ms: default_tts_barge_in_suspect_drop_grace_ms(),
                tts_barge_in_min_interval_ms: default_tts_barge_in_min_interval_ms(),
            },
            tts: TtsConfig::default(),
            aec: AecConfig::default(),
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
        cfg.realtime.tts_barge_in_rms_threshold = 0.02;
        cfg.realtime.tts_barge_in_render_ratio_min = 1.5;
        cfg.realtime.tts_barge_in_render_rms_max_age_ms = 180;
        cfg.realtime.tts_barge_in_persistence_ms = 100;
        cfg.realtime.tts_barge_in_confidence_threshold = 0.65;
        cfg.realtime.tts_barge_in_suspect_hold_ms = 70;
        cfg.realtime.tts_barge_in_recover_ms = 280;
        cfg.realtime.tts_barge_in_duck_level = 0.6;
        cfg.realtime.tts_barge_in_stop_after_ms = 220;
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
            "close_speaker_edge" => Self::close_speaker_edge_profile(),
            "balanced" => Self::balanced_profile(),
            other => {
                return Err(VocomError::ConfigValidation(format!(
                    "unknown VOCOM_PROFILE: {other}. expected balanced|low_latency|noisy_room|laptop_earbud|close_speaker_edge"
                )))
            }
        };

        if let Some(path) = Self::config_file_path_from_env() {
            cfg = Self::from_json_file(&path)?;
        }

        Self::apply_env_overrides(&mut cfg)?;

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
            apply_asr_variant_default_paths(&mut cfg.asr);
        }

        if let Ok(v) = std::env::var("VOCOM_VAD_MODEL") {
            cfg.vad.model_path = v;
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

        if let Ok(v) = std::env::var("VOCOM_ASR_THREADS") {
            cfg.asr.num_threads = parse_i32(&v, "VOCOM_ASR_THREADS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_THRESHOLD") {
            cfg.vad.threshold = parse_f32(&v, "VOCOM_VAD_THRESHOLD")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_MIN_SILENCE") {
            cfg.vad.min_silence_duration = parse_f32(&v, "VOCOM_VAD_MIN_SILENCE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_MIN_SPEECH") {
            cfg.vad.min_speech_duration = parse_f32(&v, "VOCOM_VAD_MIN_SPEECH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_VAD_MAX_SPEECH") {
            cfg.vad.max_speech_duration = parse_f32(&v, "VOCOM_VAD_MAX_SPEECH")?;
        }
        if let Ok(v) = std::env::var("VOCOM_TARGET_SAMPLE_RATE") {
            cfg.realtime.target_sample_rate = parse_i32(&v, "VOCOM_TARGET_SAMPLE_RATE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_CHUNK_MS") {
            cfg.realtime.chunk_ms = parse_u32(&v, "VOCOM_CHUNK_MS")?;
        }
        if let Ok(v) = std::env::var("VOCOM_RENDER_REFERENCE_CAPACITY") {
            cfg.realtime.render_reference_capacity =
                parse_usize(&v, "VOCOM_RENDER_REFERENCE_CAPACITY")?;
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
        if let Ok(v) = std::env::var("VOCOM_TTS_OUTPUT_DIR") {
            cfg.tts.output_dir = v;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_ENABLED") {
            cfg.aec.enabled = parse_bool(&v, "VOCOM_AEC_ENABLED")?;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_BACKEND") {
            cfg.aec.backend = parse_aec_backend(&v)?;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_SAMPLE_RATE") {
            cfg.aec.sample_rate = parse_i32(&v, "VOCOM_AEC_SAMPLE_RATE")?;
        }
        if let Ok(v) = std::env::var("VOCOM_AEC_STREAM_DELAY_MS") {
            cfg.aec.stream_delay_ms = Some(parse_i32(&v, "VOCOM_AEC_STREAM_DELAY_MS")?);
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<(), VocomError> {
        if self.asr.num_threads <= 0 {
            return Err(VocomError::ConfigValidation("asr.num_threads must be > 0".to_string()));
        }
        if self.vad.sample_rate <= 0 {
            return Err(VocomError::ConfigValidation("vad.sample_rate must be > 0".to_string()));
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

        Ok(())
    }
}

fn ensure_exists(path: &str) -> Result<(), VocomError> {
    if Path::new(path).exists() {
        Ok(())
    } else {
        Err(VocomError::MissingModelPath(path.to_string()))
    }
}

fn parse_i32(raw: &str, name: &str) -> Result<i32, VocomError> {
    raw.parse::<i32>()
        .map_err(|_| VocomError::ConfigValidation(format!("invalid integer for {name}: {raw}")))
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

fn parse_aec_backend(raw: &str) -> Result<AecBackend, VocomError> {
    match raw {
        "pure_rust_aec3" => Ok(AecBackend::PureRustAec3),
        "webrtc_native" => Ok(AecBackend::WebrtcNative),
        other => Err(VocomError::ConfigValidation(format!(
            "invalid AEC backend: {other}. expected pure_rust_aec3|webrtc_native"
        ))),
    }
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

fn apply_asr_variant_default_paths(asr: &mut AsrConfig) {
    match asr.variant {
        AsrVariantConfig::Whisper => {
            asr.mode = AsrModeConfig::Offline;
            asr.encoder_path = "models/sherpa-onnx-whisper-base.en/base.en-encoder.onnx".to_string();
            asr.decoder_path = "models/sherpa-onnx-whisper-base.en/base.en-decoder.onnx".to_string();
            asr.joiner_path = None;
            asr.tokens_path = "models/sherpa-onnx-whisper-base.en/base.en-tokens.txt".to_string();
        }
        AsrVariantConfig::MoonshineV2 => {
            asr.mode = AsrModeConfig::Offline;
            asr.encoder_path =
                "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/encoder_model.ort"
                    .to_string();
            asr.decoder_path = "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/decoder_model_merged.ort".to_string();
            asr.joiner_path = None;
            asr.tokens_path = "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/tokens.txt".to_string();
        }
        AsrVariantConfig::StreamingZipformer => {
            asr.mode = AsrModeConfig::Online;
            asr.encoder_path = "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/encoder.onnx".to_string();
            asr.decoder_path = "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/decoder.onnx".to_string();
            asr.joiner_path = Some(
                "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/joiner.onnx"
                    .to_string(),
            );
            asr.tokens_path = "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/tokens.txt".to_string();
            asr.online_decoding_method = "greedy_search".to_string();
            asr.online_enable_endpoint = true;
        }
        AsrVariantConfig::NemotronStreaming => {
            asr.mode = AsrModeConfig::Online;
            asr.encoder_path = "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/encoder.int8.onnx".to_string();
            asr.decoder_path = "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/decoder.int8.onnx".to_string();
            asr.joiner_path = Some(
                "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/joiner.int8.onnx"
                    .to_string(),
            );
            asr.tokens_path = "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/tokens.txt".to_string();
            asr.online_decoding_method = "greedy_search".to_string();
            asr.online_enable_endpoint = true;
        }
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
