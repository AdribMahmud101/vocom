
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::path::Path;

use crate::errors::VocomError;


#[allow(unused)]
pub struct VadBuilder{

    // silero configs
    model: Option<String>,
    threshold: f32,
    min_silence_duration: f32,
    min_speech_duration: f32,
    max_speech_duration: f32,

    window_size: i32,
    sample_rate: i32,
    num_threads: i32,
    provider: Option<String>,
    debug: bool,
}

impl VadBuilder {
    pub fn new() -> Self{
        Self{
            model: None,
            threshold: 0.5,
            min_silence_duration: 0.25,
            min_speech_duration: 0.25,
            max_speech_duration: 5.0,

            window_size: 512,
            sample_rate: 16000,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: false,
        }
    }

    pub fn model(mut self, model_path: &str) -> Self {
        self.model = Some(model_path.to_string());
        self
    }

    pub fn threshold(mut self, t: f32) -> Self{
        self.threshold = t;
        self
    }

    pub fn min_silence_duration(mut self, min_silence: f32) -> Self{
        self.min_silence_duration = min_silence;
        self
    }

    pub fn min_speech_duration(mut self, min_speech: f32)-> Self{
        self.min_speech_duration = min_speech;
        self
    }

    pub fn max_speech_duration(mut self, max_speech: f32) -> Self{
        self.max_speech_duration = max_speech;
        self
    }

    pub fn sample_rate(mut self, sample_rate: i32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn window_size(mut self, window_size: i32) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn num_threads(mut self, num_threads: i32) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn provider(mut self, provider: &str) -> Self {
        self.provider = Some(provider.to_string());
        self
    }

    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }



    pub fn build(self)-> Result<VoiceActivityDetector, VocomError> {
        if let Some(ref path) = self.model {
            if !Path::new(path).exists() {
                return Err(VocomError::MissingModelPath(path.clone()));
            }
        }

        let mut config = SileroVadModelConfig::default();

        config.model = self.model;
        config.threshold = self.threshold;
        config.min_silence_duration = self.min_silence_duration;
        config.min_speech_duration = self.min_speech_duration;
        config.max_speech_duration = self.max_speech_duration;
        config.window_size = self.window_size;

        let vad_config = VadModelConfig{
            silero_vad: config,
            ten_vad: Default::default(),
            sample_rate: self.sample_rate,
            num_threads: self.num_threads,
            provider: self.provider,
            debug: self.debug,
        };


        if let Some(vad) = VoiceActivityDetector::create(&vad_config, 30.0) {
            Ok(vad)
        } else {
            Err(VocomError::VadConfig("Failed to build VAD instance".to_string()))
        }

    }

}