use thiserror::Error;

#[derive(Debug, Error)]
pub enum VocomError {
    #[error("input audio device not found")]
    NoInputDevice,

    #[error("failed to get default input config: {0}")]
    DefaultInputConfig(#[from] cpal::DefaultStreamConfigError),

    #[error("failed to build input stream: {0}")]
    BuildInputStream(#[from] cpal::BuildStreamError),

    #[error("failed to start audio stream: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),

    #[error("audio stream error: {0}")]
    Stream(String),

    #[error("ASR configuration error: {0}")]
    AsrConfig(String),

    #[error("VAD configuration error: {0}")]
    VadConfig(String),

    #[error("TTS configuration error: {0}")]
    TtsConfig(String),

    #[error("TTS generation failed: {0}")]
    TtsGeneration(String),

    #[error("TTS output IO failed: {0}")]
    TtsIo(String),

    #[error("AEC configuration error: {0}")]
    AecConfig(String),

    #[error("AEC processing error: {0}")]
    AecProcessing(String),

    #[error("model path does not exist: {0}")]
    MissingModelPath(String),

    #[error("channel disconnected")]
    ChannelDisconnected,

    #[error("configuration validation error: {0}")]
    ConfigValidation(String),

    #[error("configuration IO error: {0}")]
    ConfigIo(String),

    #[error("configuration parse error: {0}")]
    ConfigParse(String),
}
