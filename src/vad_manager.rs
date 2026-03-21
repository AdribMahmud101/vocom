

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

#[allow(unused)]
pub struct VadBuilder{

    // silero configs
    model: Option<String>,
    threshold: f32,
    min_silence_duration: f32,
    min_speech_duration: f32,
    max_speech_duration: f32,

    // vad configs
    silero_vad: Option<SileroVadModelConfig>,
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

            silero_vad: None,
            sample_rate: 16000,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: false,
        }
    }

    pub fn model(mut self, model_path: &str){
        self.model = Some(model_path.to_string());

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



    pub fn build(self)-> Result<VoiceActivityDetector, String> {

        let mut config = SileroVadModelConfig::default();

        let vad_model = self.model;

        config.model = vad_model;
        config.threshold = self.threshold;
        config.min_silence_duration = self.min_silence_duration;
        config.min_speech_duration = self.min_speech_duration;
        config.max_speech_duration = self.max_speech_duration;

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
            Err("Failed to build VAD instance".to_string())
        }

    }



}