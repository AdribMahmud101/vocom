
use std::ffi::{CStr, CString};
use std::mem;
use std::path::Path;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig,
    OnlineStream, Wave,
};
use sherpa_onnx_sys::offline_asr::{
    OfflineRecognizerConfig as COfflineRecognizerConfig, SherpaOnnxAcceptWaveformOffline,
    SherpaOnnxCreateOfflineRecognizer, SherpaOnnxCreateOfflineStream,
    SherpaOnnxDecodeOfflineStream, SherpaOnnxDestroyOfflineRecognizer,
    SherpaOnnxDestroyOfflineStream, SherpaOnnxDestroyOfflineStreamResultJson,
    SherpaOnnxGetOfflineStreamResultAsJson,
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
    model_type: Option<String>,

    // whisper specific
    language: Option<String>,
    task: Option<String>,
    tail_paddings: i32,
    enable_token_timestamps: bool,
    enable_segment_timestamps: bool,

    // online specific
    online_decoding_method: Option<String>,
    online_enable_endpoint: bool,
    online_rule1_min_trailing_silence: f32,
    online_rule2_min_trailing_silence: f32,
    online_rule3_min_utterance_length: f32,
    
    // hotwords
    hotwords_path: Option<String>,
    hotwords_score: f32,
}

pub enum OfflineAsrBackend {
    Sherpa(OfflineRecognizer),
    CApi(CApiOfflineRecognizer),
}

pub struct CApiOfflineRecognizer {
    ptr: *const sherpa_onnx_sys::offline_asr::OfflineRecognizer,
    _encoder: CString,
    _decoder: CString,
    _tokens: CString,
    _provider: CString,
    _model_type: CString,
    _language: CString,
    _task: CString,
    _decoding_method: CString,
}

unsafe impl Send for CApiOfflineRecognizer {}
unsafe impl Sync for CApiOfflineRecognizer {}

impl CApiOfflineRecognizer {
    fn create_whisper(
        encoder: String,
        decoder: String,
        tokens: String,
        provider: String,
        num_threads: i32,
        language: String,
        task: String,
        tail_paddings: i32,
        hotwords_score: f32,
    ) -> Result<Self, VocomError> {
        let encoder_c = CString::new(encoder)
            .map_err(|err| VocomError::AsrConfig(format!("invalid encoder path: {err}")))?;
        let decoder_c = CString::new(decoder)
            .map_err(|err| VocomError::AsrConfig(format!("invalid decoder path: {err}")))?;
        let tokens_c = CString::new(tokens)
            .map_err(|err| VocomError::AsrConfig(format!("invalid tokens path: {err}")))?;
        let provider_c = CString::new(provider)
            .map_err(|err| VocomError::AsrConfig(format!("invalid provider: {err}")))?;
        let model_type_c = CString::new("whisper").expect("literal has no nul");
        let language_c = CString::new(language)
            .map_err(|err| VocomError::AsrConfig(format!("invalid whisper language: {err}")))?;
        let task_c = CString::new(task)
            .map_err(|err| VocomError::AsrConfig(format!("invalid whisper task: {err}")))?;
        let decoding_method_c = CString::new("greedy_search").expect("literal has no nul");

        let ptr = unsafe {
            let mut cfg: COfflineRecognizerConfig = mem::zeroed();
            cfg.feat_config.sample_rate = 16_000;
            cfg.feat_config.feature_dim = 80;
            cfg.model_config.whisper.encoder = encoder_c.as_ptr();
            cfg.model_config.whisper.decoder = decoder_c.as_ptr();
            cfg.model_config.whisper.language = language_c.as_ptr();
            cfg.model_config.whisper.task = task_c.as_ptr();
            cfg.model_config.whisper.tail_paddings = tail_paddings;
            cfg.model_config.whisper.enable_token_timestamps = 0;
            cfg.model_config.whisper.enable_segment_timestamps = 0;
            cfg.model_config.tokens = tokens_c.as_ptr();
            cfg.model_config.num_threads = num_threads;
            cfg.model_config.provider = provider_c.as_ptr();
            cfg.model_config.model_type = model_type_c.as_ptr();
            cfg.decoding_method = decoding_method_c.as_ptr();
            cfg.max_active_paths = 4;
            cfg.hotwords_score = hotwords_score;
            SherpaOnnxCreateOfflineRecognizer(&cfg)
        };

        if ptr.is_null() {
            return Err(VocomError::AsrConfig(
                "failed to create Whisper recognizer through C API".to_string(),
            ));
        }

        Ok(Self {
            ptr,
            _encoder: encoder_c,
            _decoder: decoder_c,
            _tokens: tokens_c,
            _provider: provider_c,
            _model_type: model_type_c,
            _language: language_c,
            _task: task_c,
            _decoding_method: decoding_method_c,
        })
    }

    fn transcribe_samples(&self, sample_rate: i32, samples: &[f32]) -> Result<String, VocomError> {
        unsafe {
            let stream = SherpaOnnxCreateOfflineStream(self.ptr);
            if stream.is_null() {
                return Err(VocomError::AsrConfig(
                    "failed to create Whisper offline stream".to_string(),
                ));
            }

            SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
            SherpaOnnxDecodeOfflineStream(self.ptr, stream);
            let result_ptr = SherpaOnnxGetOfflineStreamResultAsJson(stream);
            let result = if result_ptr.is_null() {
                Err(VocomError::AsrConfig(
                    "failed to get Whisper transcription result".to_string(),
                ))
            } else {
                let json = CStr::from_ptr(result_ptr).to_string_lossy().into_owned();
                SherpaOnnxDestroyOfflineStreamResultJson(result_ptr);
                let value: serde_json::Value = serde_json::from_str(&json).map_err(|err| {
                    VocomError::AsrConfig(format!("invalid Whisper result JSON: {err}"))
                })?;
                Ok(value
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string())
            };
            SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

impl Drop for CApiOfflineRecognizer {
    fn drop(&mut self) {
        unsafe {
            SherpaOnnxDestroyOfflineRecognizer(self.ptr);
        }
    }
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
            model_type: None,
            // whisper specific
            language: Some(String::new()),
            task: Some("transcribe".to_string()),
            tail_paddings: 0,
            enable_token_timestamps: false, // false by default
            enable_segment_timestamps: false,
            online_decoding_method: Some("greedy_search".to_string()),
            online_enable_endpoint: true,
            online_rule1_min_trailing_silence: 0.35,
            online_rule2_min_trailing_silence: 0.8,
            online_rule3_min_utterance_length: 8.0,
            hotwords_path: None,
            hotwords_score: 1.5,
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

    pub fn model_type(mut self, value: &str) -> Self {
        self.model_type = Some(value.to_string());
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

    pub fn online_rule1_min_trailing_silence(mut self, value: f32) -> Self {
        self.online_rule1_min_trailing_silence = value;
        self
    }

    pub fn online_rule2_min_trailing_silence(mut self, value: f32) -> Self {
        self.online_rule2_min_trailing_silence = value;
        self
    }

    pub fn online_rule3_min_utterance_length(mut self, value: f32) -> Self {
        self.online_rule3_min_utterance_length = value;
        self
    }


    pub fn hotwords_path(mut self, path: Option<String>) -> Self {
        self.hotwords_path = path;
        self
    }

    pub fn hotwords_score(mut self, score: f32) -> Self {
        self.hotwords_score = score;
        self
    }

    pub fn build(self)-> Result<OfflineAsrBackend, VocomError>{
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
        config.model_config.model_type = self.model_type;
        config.decoding_method = Some("greedy_search".to_string());
        config.hotwords_score = self.hotwords_score;
   
        match self.variant {
            ASRVariant::Moonshinev2 => {
                config.model_config.model_type = Some("moonshine".to_string());
                config.model_config.moonshine.encoder = encoder;
                config.model_config.moonshine.merged_decoder = decoder;
                
            },
            ASRVariant::Whisper => {
                let encoder = encoder.ok_or_else(|| {
                    VocomError::AsrConfig("Whisper encoder path is missing".to_string())
                })?;
                let decoder = decoder.ok_or_else(|| {
                    VocomError::AsrConfig("Whisper decoder path is missing".to_string())
                })?;
                let tokens = config.model_config.tokens.clone().ok_or_else(|| {
                    VocomError::AsrConfig("Whisper tokens path is missing".to_string())
                })?;
                let provider = config
                    .model_config
                    .provider
                    .clone()
                    .unwrap_or_else(|| "cpu".to_string());
                let language = self.language.unwrap_or_default();
                let task = self.task.unwrap_or_else(|| "transcribe".to_string());
                config.model_config.model_type = Some("whisper".to_string());
                config.model_config.whisper.encoder = Some(encoder.clone());
                config.model_config.whisper.decoder = Some(decoder.clone());
                config.model_config.whisper.language = Some(language.clone());
                config.model_config.whisper.task = Some(task.clone());
                config.model_config.whisper.tail_paddings = self.tail_paddings;
                config.model_config.whisper.enable_token_timestamps =
                    self.enable_token_timestamps;
                config.model_config.whisper.enable_segment_timestamps =
                    self.enable_segment_timestamps;
                #[cfg(target_os = "android")]
                {
                    // Android Whisper path: avoid hotword knobs in offline config.
                    // Some sherpa Android runtimes have shown ABI-sensitive crashes
                    // around offline recognizer config hotword fields.
                    config.hotwords_score = 0.0;
                }

                // Prefer the high-level sherpa wrapper to avoid C-ABI drift issues
                // (observed as SIGSEGV on Android during Whisper recognizer create).
                if let Some(recognizer) = OfflineRecognizer::create(&config) {
                    return Ok(OfflineAsrBackend::Sherpa(recognizer));
                }

                #[cfg(target_os = "android")]
                {
                    return Err(VocomError::AsrConfig(
                        "failed to create Whisper recognizer (android safe path)".to_string(),
                    ));
                }

                #[cfg(not(target_os = "android"))]
                {
                    let recognizer = CApiOfflineRecognizer::create_whisper(
                        encoder,
                        decoder,
                        tokens,
                        provider,
                        config.model_config.num_threads,
                        language,
                        task,
                        self.tail_paddings,
                        self.hotwords_score,
                    )?;
                    return Ok(OfflineAsrBackend::CApi(recognizer));
                }
       
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
            Ok(OfflineAsrBackend::Sherpa(recognizer))
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
        
        // Sherpa-onnx requires modified_beam_search for hotwords
        if self.hotwords_path.is_some() {
            config.decoding_method = Some("modified_beam_search".to_string());
        } else {
            config.decoding_method = self.online_decoding_method;
        }
        config.hotwords_file = self.hotwords_path;
        config.hotwords_score = self.hotwords_score;
        
        config.enable_endpoint = self.online_enable_endpoint;
        config.rule1_min_trailing_silence = self.online_rule1_min_trailing_silence;
        config.rule2_min_trailing_silence = self.online_rule2_min_trailing_silence;
        config.rule3_min_utterance_length = self.online_rule3_min_utterance_length;

        match self.variant {
            ASRVariant::StreamingZipformer | ASRVariant::NemotronStreaming => {
                config.model_config.transducer.encoder = encoder;
                config.model_config.transducer.decoder = decoder;
                config.model_config.transducer.joiner = joiner;
            }
            _ => {
                return Err(VocomError::AsrConfig(
                    "online recognizer currently supports streaming_zipformer or nemotron_streaming"
                        .to_string(),
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
        recognizer: &OfflineAsrBackend,
        sample_rate: i32,
        samples: &[f32],
    ) -> Result<String, VocomError> {
        match recognizer {
            OfflineAsrBackend::Sherpa(recognizer) => {
                let stream = recognizer.create_stream();
                stream.accept_waveform(sample_rate, samples);
                recognizer.decode(&stream);

                stream.get_result().map(|result| result.text).ok_or_else(|| {
                    VocomError::AsrConfig("Failed to get transcription result".to_string())
                })
            }
            OfflineAsrBackend::CApi(recognizer) => {
                recognizer.transcribe_samples(sample_rate, samples)
            }
        }
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
