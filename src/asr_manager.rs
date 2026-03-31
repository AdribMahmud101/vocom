
use std::path::Path;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig,
    OnlineStream, Wave,
};

use crate::errors::VocomError;

#[allow(unused)]
pub enum ASRVariant {
    Moonshinev2,
    Whisper,
    StreamingZipformer,
    NemotronStreaming,
    Unknown,
}

#[allow(unused)]
pub struct ASRModelBuilder{
    variant: ASRVariant,
    encoder: Option<String>,
    decoder: Option<String>,
    tokens: Option<String>,
    joiner: Option<String>,
    num_threads: i32,
    provider: Option<String>,

    // whisper specific
    language: Option<String>,
    task: Option<String>,
    tail_paddings: i32,
    enable_token_timestamps: bool,
    enable_segment_timestamps: bool,

    // online specific
    online_decoding_method: Option<String>,
    online_enable_endpoint: bool,

}


impl ASRVariant {

    pub fn get_variant(file_name: &str)-> Self {

        let file = file_name.to_lowercase();

        if file.contains("moonshine"){
            Self::Moonshinev2
        } else if file.contains("nemotron") {
            Self::NemotronStreaming
        } else if file.contains("zipformer") || file.contains("streaming") {
            Self::StreamingZipformer
        } else if file.contains("whisper"){
            Self::Whisper
        } else{
            Self::Unknown
        }
         
    }

}


impl ASRModelBuilder{

    pub fn new(variant: ASRVariant) -> Self{
        Self{
            variant,
            encoder: None,
            decoder: None,
            tokens: None,
            joiner: None,
            num_threads: 2, // defualt to 2 threads
            provider: Some("cpu".to_string()), // default provider
            // whisper specific
            language: Some("en".to_string()),
            task: Some("transcribe".to_string()),
            tail_paddings: -1,
            enable_token_timestamps: false, // false by default
            enable_segment_timestamps: false,
            online_decoding_method: Some("greedy_search".to_string()),
            online_enable_endpoint: true,
        }
    }

    pub fn encoder(mut self, path: &str)-> Self{
        self.encoder = Some(path.to_string());
        self
    }

    pub fn decoder(mut self, path: &str)-> Self{
        self.decoder = Some(path.to_string());
        self
    }

    pub fn tokens(mut self, path: &str)-> Self{
        self.tokens = Some(path.to_string());
        self
    }

    pub fn joiner(mut self, path: &str) -> Self {
        self.joiner = Some(path.to_string());
        self
    }

    pub fn num_threads(mut self, threads: i32)-> Self{
        self.num_threads = threads;
        self
    }

    pub fn provider(mut self, value: &str)-> Self{
        self.provider = Some(value.to_string());
        self
    }

    pub fn whisper_language(mut self, value: &str) -> Self {
        self.language = Some(value.to_string());
        self
    }

    pub fn whisper_task(mut self, value: &str) -> Self {
        self.task = Some(value.to_string());
        self
    }

    pub fn whisper_tail_paddings(mut self, value: i32) -> Self {
        self.tail_paddings = value;
        self
    }

    pub fn whisper_enable_token_timestamps(mut self, value: bool) -> Self {
        self.enable_token_timestamps = value;
        self
    }

    pub fn whisper_enable_segment_timestamps(mut self, value: bool) -> Self {
        self.enable_segment_timestamps = value;
        self
    }

    pub fn online_decoding_method(mut self, value: &str) -> Self {
        self.online_decoding_method = Some(value.to_string());
        self
    }

    pub fn online_enable_endpoint(mut self, value: bool) -> Self {
        self.online_enable_endpoint = value;
        self
    }


    pub fn build(self)-> Result<OfflineRecognizer, VocomError>{
        let encoder = self.encoder;
        let decoder = self.decoder;
        let tokens = self.tokens;

        Self::assert_path_exists(encoder.as_deref())?;
        Self::assert_path_exists(decoder.as_deref())?;
        Self::assert_path_exists(tokens.as_deref())?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.tokens = tokens;
        config.model_config.num_threads = self.num_threads;
        config.model_config.provider = self.provider;
   
        match self.variant {
            ASRVariant::Moonshinev2 => {
                config.model_config.moonshine.encoder = encoder;
                config.model_config.moonshine.merged_decoder = decoder;
                
            },
            ASRVariant::Whisper => {
                config.model_config.whisper.encoder = encoder;
                config.model_config.whisper.decoder = decoder;
                config.model_config.whisper.language = self.language;
                config.model_config.whisper.task = self.task;
                config.model_config.whisper.tail_paddings = self.tail_paddings;
                config.model_config.whisper.enable_token_timestamps = self.enable_token_timestamps;
                config.model_config.whisper.enable_segment_timestamps = self.enable_segment_timestamps;
       
            },
            ASRVariant::StreamingZipformer | ASRVariant::NemotronStreaming => {
                return Err(VocomError::AsrConfig(
                    "streaming ASR variant requires online recognizer builder".to_string(),
                ));
            }
            ASRVariant::Unknown => {
                return Err(VocomError::AsrConfig("ASR model is not recognized".to_string()));
            }
        }

        

        if let Some(recognizer) = OfflineRecognizer::create(&config)  {
            Ok(recognizer)
        } else {
            Err(VocomError::AsrConfig("Failed to create recognizer from config".to_string()))
        }

    }

    pub fn build_online(self) -> Result<OnlineRecognizer, VocomError> {
        let encoder = self.encoder;
        let decoder = self.decoder;
        let joiner = self.joiner;
        let tokens = self.tokens;

        Self::assert_path_exists(encoder.as_deref())?;
        Self::assert_path_exists(decoder.as_deref())?;
        Self::assert_path_exists(joiner.as_deref())?;
        Self::assert_path_exists(tokens.as_deref())?;

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.tokens = tokens;
        config.model_config.num_threads = self.num_threads;
        config.model_config.provider = self.provider;
        config.decoding_method = self.online_decoding_method;
        config.enable_endpoint = self.online_enable_endpoint;

        match self.variant {
            ASRVariant::StreamingZipformer | ASRVariant::NemotronStreaming => {
                config.model_config.transducer.encoder = encoder;
                config.model_config.transducer.decoder = decoder;
                config.model_config.transducer.joiner = joiner;
            }
            _ => {
                return Err(VocomError::AsrConfig(
                    "online recognizer currently supports only streaming_zipformer".to_string(),
                ));
            }
        }

        OnlineRecognizer::create(&config).ok_or_else(|| {
            VocomError::AsrConfig("Failed to create online recognizer from config".to_string())
        })
    }

    pub fn transcribe(
        offline_recognizer: Result<OfflineRecognizer, VocomError>,
        file_path: &str,
    ) -> Result<String, VocomError> {
        let recognizer = offline_recognizer?;
        let stream = recognizer.create_stream();

        let wave = Wave::read(file_path)
            .ok_or_else(|| VocomError::AsrConfig(format!("Failed to read wave path: {file_path}")))?;
        stream.accept_waveform(wave.sample_rate(), wave.samples());
        recognizer.decode(&stream);

        if let Some(result) = stream.get_result() {
            Ok(result.text)
        } else {
            Err(VocomError::AsrConfig("Failed to get transcription result".to_string()))
        }
    }

    pub fn transcribe_samples(
        recognizer: &OfflineRecognizer,
        sample_rate: i32,
        samples: &[f32],
    ) -> Result<String, VocomError> {
        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate, samples);
        recognizer.decode(&stream);

        stream
            .get_result()
            .map(|result| result.text)
            .ok_or_else(|| VocomError::AsrConfig("Failed to get transcription result".to_string()))
    }

    pub fn transcribe_samples_online(
        recognizer: &OnlineRecognizer,
        stream: &OnlineStream,
        sample_rate: i32,
        samples: &[f32],
    ) -> Result<String, VocomError> {
        stream.accept_waveform(sample_rate, samples);
        stream.input_finished();

        while recognizer.is_ready(stream) {
            recognizer.decode(stream);
        }

        let text = recognizer
            .get_result(stream)
            .map(|result| result.text)
            .ok_or_else(|| VocomError::AsrConfig("Failed to get online transcription result".to_string()))?;

        recognizer.reset(stream);
        Ok(text)
    }

    fn assert_path_exists(path: Option<&str>) -> Result<(), VocomError> {
        if let Some(p) = path {
            if !Path::new(p).exists() {
                return Err(VocomError::MissingModelPath(p.to_string()));
            }
        }

        Ok(())
    }



}
