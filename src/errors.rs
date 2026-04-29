use thiserror::Error;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ErrorClass {
    Transient,
    Fatal,
}

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

    #[error("denoiser configuration error: {0}")]
    DenoiserConfig(String),

    #[error("denoiser processing error: {0}")]
    DenoiserProcessing(String),

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

impl VocomError {
    pub fn class(&self) -> ErrorClass {
        match self {
            // Runtime transport/stream errors can be transient on mobile/embedded devices.
            Self::ChannelDisconnected | Self::Stream(_) => ErrorClass::Transient,
            // Startup/recoverable device acquisition failures can be transient as well.
            Self::NoInputDevice
            | Self::DefaultInputConfig(_)
            | Self::BuildInputStream(_)
            | Self::PlayStream(_) => ErrorClass::Transient,
            // The rest are deterministic/configuration/model issues and treated as fatal.
            Self::AsrConfig(_)
            | Self::VadConfig(_)
            | Self::TtsConfig(_)
            | Self::TtsGeneration(_)
            | Self::TtsIo(_)
            | Self::AecConfig(_)
            | Self::AecProcessing(_)
            | Self::DenoiserConfig(_)
            | Self::DenoiserProcessing(_)
            | Self::MissingModelPath(_)
            | Self::ConfigValidation(_)
            | Self::ConfigIo(_)
            | Self::ConfigParse(_) => ErrorClass::Fatal,
        }
    }

    pub fn is_transient(&self) -> bool {
        self.class() == ErrorClass::Transient
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorClass, VocomError};

    #[test]
    fn classify_transient_runtime_errors() {
        assert_eq!(VocomError::ChannelDisconnected.class(), ErrorClass::Transient);
        assert_eq!(
            VocomError::Stream("temporary backend glitch".to_string()).class(),
            ErrorClass::Transient
        );
    }

    #[test]
    fn classify_fatal_config_errors() {
        assert_eq!(
            VocomError::ConfigValidation("bad value".to_string()).class(),
            ErrorClass::Fatal
        );
        assert_eq!(
            VocomError::AsrConfig("invalid model".to_string()).class(),
            ErrorClass::Fatal
        );
    }
}
